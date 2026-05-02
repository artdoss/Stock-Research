import os
import re
import json
import time
import requests
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request
from markupsafe import Markup, escape
from anthropic import Anthropic

app = Flask(__name__)


# Simple in-memory TTL cache. Keyed by string. Entries expire after `ttl` seconds.
# Goal: repeat searches of the same ticker return instantly instead of re-running
# every Claude call. First load is still slow; subsequent loads within the TTL
# pull from memory.
_cache = {}
_cache_lock = Lock()


def cached_or_fetch(key, ttl, fn, *args, **kwargs):
    """Return cached value for `key` if fresh, else call fn(*args, **kwargs) and cache it."""
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and now - entry["t"] < ttl:
            return entry["v"]
    result = fn(*args, **kwargs)
    with _cache_lock:
        _cache[key] = {"v": result, "t": now}
    return result


@app.template_filter("bold_md")
def bold_md(text):
    """Convert **markdown bold** to <strong> tags. Escapes any other HTML
    in the input to keep AI-generated content safe to render."""
    if not text:
        return ""
    escaped = str(escape(text))
    bolded = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return Markup(bolded)

FMP_API_KEY = os.environ.get("FMP_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

claude = Anthropic(api_key=ANTHROPIC_API_KEY)

CLAUDE_MODEL = "claude-sonnet-4-6"


def get_plain_english_summary(company):
    """Ask Claude to rewrite the company's official description in plain English."""
    prompt = f"""You're explaining a publicly traded company to someone who is new to investing. The official description below is written in dense, corporate-style language. Rewrite it as 3-4 conversational sentences that:

1. Start with what the company actually does in plain terms (e.g., "Apple makes iPhones, Macs, and other consumer electronics" — not "designs, manufactures, and markets smartphones, personal computers...").
2. Explain clearly how the company primarily makes money.
3. Note any obvious strengths, dependencies, or risks visible in the description.
4. Avoid corporate jargon like "diversified", "leverages", "ecosystem", "solutions". Use words a high schooler would use.

Company: {company.get('companyName')}
Industry: {company.get('industry')}
Official description: {company.get('description')}

Output ONLY the rewritten summary. No preamble, headers, or explanations."""

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"(AI summary unavailable: {e})"


def get_news(company_name, limit=4):
    """Fetch recent news articles about the company, deduped by title and content.
    Fetches more than needed because NewsAPI returns syndicated duplicates
    (same article on Yahoo, Mercury News, MSN, etc.)."""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{company_name}"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return []
    if data.get("status") != "ok":
        return []

    raw = data.get("articles", [])

    seen_titles = set()
    seen_descriptions = set()
    seen_urls = set()
    unique = []
    for art in raw:
        title = (art.get("title") or "").strip()
        url_ = art.get("url") or ""
        desc = (art.get("description") or "")[:200].strip().lower()

        # Skip articles that NewsAPI marks as removed
        if not title or title.lower() in ("[removed]", "removed"):
            continue

        # Normalized title key: first 80 chars lowercased, ignore source suffixes
        title_key = title.lower()[:80]

        if title_key in seen_titles or url_ in seen_urls or (desc and desc in seen_descriptions):
            continue

        seen_titles.add(title_key)
        seen_urls.add(url_)
        if desc:
            seen_descriptions.add(desc)
        unique.append(art)
        if len(unique) >= limit:
            break

    return unique


def analyze_article(article, company):
    """Use Claude to summarize the article, classify sentiment, and identify what
    substantively got lost or downplayed. Claude should use its knowledge of the
    company to provide actual missing context — not just critique word choice."""
    title = article.get("title") or ""
    source = (article.get("source") or {}).get("name") or "Unknown"
    snippet = article.get("description") or article.get("content") or ""
    company_name = company.get("companyName") or ""
    industry = company.get("industry") or ""

    prompt = f"""You are analyzing a news headline + snippet about {company_name} ({industry}). A typical reader sees only this snippet on a stock app, not the full article. Your audience is a BEGINNER-to-INTERMEDIATE investor — someone who knows what a stock is and roughly what "market cap" means, but who does NOT have a finance background. They may not know what "antitrust," "gross margin," "revenue exposure," or "upgrade cycle" mean unless you explain.

Your job:

1. Write a neutral 2-sentence summary of what the snippet actually says. Plain English, no jargon.
2. Classify sentiment toward the company: positive, negative, or neutral.
3. Identify what's missing or downplayed — but go beyond word-choice critique. Tell an investor what they SHOULD KNOW that this snippet leaves out, using your background knowledge of {company_name} where confident.

Categories (pick the best fit per item):

- buried_risk: Risks mentioned but glossed over, OR known significant risks/headwinds for {company_name} that the article doesn't mention but a careful investor should weigh.
- missing_context: Specific facts, recent events, financial trends, or competitive dynamics an investor would want to know to interpret this article. Provide the actual context.
- removed_uncertainty: Hedging language in the snippet ("may", "could", "alleged", "reportedly") presented as fact, OR strong claims that aren't sourced.
- framing_shift: Whose viewpoint dominates; selective emphasis; what the headline foregrounds vs. what gets buried.
- source_dodging: What the company, executives, or officials in the article appear to avoid stating directly.

WRITING RULES for each "what_got_lost":
- LEAD WITH A BOLD TAKEAWAY that DELIVERS the answer or insight, wrapped in markdown asterisks like **this**. The bold sentence should give the substantive answer to the gap — not pose a question and walk away.
- Follow the bold takeaway with 1-2 sentences of concrete context: specific numbers, recent events, or trends. When you don't know exact numbers, give a directional answer ("iPhone sales have been roughly flat for several years") rather than fabricating precise figures.
- Briefly define any financial or industry term you use, inline. (Example: "antitrust scrutiny — meaning the government is investigating whether the company is competing unfairly")
- Spell out acronyms (DOJ -> "the U.S. Department of Justice"; SEC -> "the Securities and Exchange Commission, the agency that regulates U.S. stock markets").
- Use everyday words: "money" or "sales" instead of "revenue" or "exposure" where possible; "fewer customers upgrading" instead of "weaker upgrade cycle".
- Total length per item: 2-3 sentences. Be tight.
- Do NOT fabricate specific numbers or events. If you're not confident, hedge with a range or general trend, but DO answer — never just flag a missing number without supplying any answer.
- Skip items where you can't supply at least a directional answer. Better to omit than to leave the user with more questions.

GOOD example items:

{{"category": "buried_risk", "what_got_lost": "**iPhone sales have basically stopped growing — they've been roughly flat for the past several years**, after long stretches of double-digit growth. The iPhone is still over half of Apple's total sales, so when it stalls, the whole company's growth slows. A new CEO would inherit this problem on day one."}}

{{"category": "missing_context", "what_got_lost": "**About 18 cents of every dollar Apple makes comes from selling things in China**, which makes Apple unusually exposed to U.S.-China political tensions. If trade restrictions tighten or Chinese consumers shift to local brands like Huawei, that revenue is at risk."}}

BAD example (poses a question without answering — DO NOT do this):
{{"category": "missing_context", "what_got_lost": "The snippet doesn't say whether iPhone sales were better than last year — which would be the key comparison for judging growth."}}

The BAD example asks a question and leaves the user to find the answer themselves. That defeats the purpose of this tool. ANSWER THE QUESTION using your knowledge. If you can't, skip the item.

Output 0-4 items. Quality and answer-completeness over quantity.

Article title: {title}
Source: {source}
Snippet: {snippet}

Respond in this exact JSON format. No preamble, no markdown fences, no explanation outside the JSON:
{{
  "summary": "...",
  "sentiment": "positive" or "negative" or "neutral",
  "lost_in_summary": [
    {{"category": "removed_uncertainty|buried_risk|framing_shift|source_dodging|missing_context", "what_got_lost": "specific substantive thing in plain English"}}
  ]
}}"""

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        return {
            "summary": snippet[:200],
            "sentiment": "neutral",
            "lost_in_summary": [],
            "_error": str(e),
        }


def get_bull_bear(company):
    """Generate balanced bull case (reasons to own) and bear case (reasons to avoid)
    arguments for the stock, written for a beginner-to-intermediate investor."""
    company_name = company.get("companyName") or ""
    industry = company.get("industry") or ""
    description = (company.get("description") or "")[:1500]
    price = company.get("price")
    market_cap = company.get("marketCap")

    metric_bits = []
    if price:
        metric_bits.append(f"current price ${price:.2f}")
    if market_cap and market_cap >= 1e9:
        metric_bits.append(f"market cap ~${market_cap / 1e9:.0f}B")
    elif market_cap:
        metric_bits.append(f"market cap ~${market_cap / 1e6:.0f}M")
    metrics_str = ", ".join(metric_bits) if metric_bits else "(metrics not available)"

    prompt = f"""You are generating a balanced bull case (reasons to own) and bear case (reasons NOT to own) for {company_name}, a publicly traded company in the {industry} industry. The audience is a beginner-to-intermediate investor trying to decide if this stock is worth deeper research — not someone looking for a final buy/sell decision.

Company: {company_name} ({industry})
Recent metrics: {metrics_str}
Description: {description}

Generate:
- 3-4 strong bull case points (reasons someone might want to own this)
- 3-4 strong bear case points (reasons someone might NOT want to own this)

WRITING RULES (same as elsewhere in this app):
- Each point: lead with **a bold takeaway sentence** that delivers the substantive claim, wrapped in markdown asterisks. Then 1-2 sentences of supporting reasoning or context.
- Be specific to THIS company. Avoid generic claims ("they have a strong brand"); explain WHICH brands or assets and WHY that matters.
- Plain English. Define financial terms inline. Spell out acronyms (DOJ -> "the U.S. Department of Justice"; SEC -> "the Securities and Exchange Commission").
- Don't fabricate specific numbers. If you're not confident, give direction ("revenue has grown roughly 5-10% per year recently") rather than precise figures.
- Be balanced. Don't make one side weak just because the other is strong — beginner investors need to see both perspectives clearly.
- Total length per point: 2-3 sentences.

Respond in this exact JSON format. No preamble, no markdown fences, no explanation outside the JSON:
{{
  "bull_case": [
    {{"point": "**bold takeaway**. supporting reasoning."}},
    {{"point": "**bold takeaway**. supporting reasoning."}}
  ],
  "bear_case": [
    {{"point": "**bold takeaway**. supporting reasoning."}},
    {{"point": "**bold takeaway**. supporting reasoning."}}
  ]
}}"""

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        return {
            "bull_case": [],
            "bear_case": [],
            "_error": str(e),
        }


def fetch_company_profile(ticker):
    """Fetch company profile from FMP. Returns the company dict or None."""
    profile_url = (
        f"https://financialmodelingprep.com/stable/profile"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        response = requests.get(profile_url, timeout=10)
        data = response.json()
    except Exception:
        return None
    if not data or not isinstance(data, list):
        return None
    return data[0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stock")
def stock():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")

    # Cache TTLs — short for fast-moving data (price), longer for slow-moving
    # data (description, AI analysis of fixed text)
    company = cached_or_fetch(f"profile:{ticker}", 300, fetch_company_profile, ticker)
    if not company:
        return render_template(
            "index.html",
            error=f"Could not find data for '{ticker}'. Double-check the ticker symbol.",
        )

    company_name = company.get("companyName") or ticker
    raw_articles = cached_or_fetch(f"news:{company_name}", 600, get_news, company_name, 4)

    # Run every Claude call in parallel, each individually cached. First search
    # is slow; repeat searches of the same ticker hit the cache and return fast.
    with ThreadPoolExecutor(max_workers=8) as executor:
        plain_future = executor.submit(
            cached_or_fetch, f"plain:{ticker}", 1800, get_plain_english_summary, company
        )
        bullbear_future = executor.submit(
            cached_or_fetch, f"bb:{ticker}", 1800, get_bull_bear, company
        )
        article_futures = [
            executor.submit(
                cached_or_fetch,
                f"art:{art.get('url')}",
                3600,
                analyze_article,
                art,
                company,
            )
            for art in raw_articles
        ]

        plain_english = plain_future.result()
        bull_bear = bullbear_future.result()
        analyses = [f.result() for f in article_futures]

    articles = []
    for art, analysis in zip(raw_articles, analyses):
        articles.append({
            "title": art.get("title"),
            "source": (art.get("source") or {}).get("name") or "Unknown",
            "url": art.get("url"),
            "published_at": art.get("publishedAt"),
            "analysis": analysis,
        })

    return render_template(
        "stock.html",
        company=company,
        plain_english=plain_english,
        bull_bear=bull_bear,
        articles=articles,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)