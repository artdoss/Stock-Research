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


# SEC EDGAR requires a real-looking User-Agent header per their fair-access policy
SEC_HEADERS = {"User-Agent": "Bowdoin Stock Research arthurdossantos@bowdoin.edu"}


def strip_html(html):
    """Strip HTML tags and common entities from text. Crude but adequate for SEC
    filings, which are mostly plain text inside tables and paragraphs."""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&#160;": " ", "&#8217;": "'", "&#8220;": '"', "&#8221;": '"',
        "&#8211;": "-", "&#8212;": "-", "&#39;": "'",
    }
    for ent, repl in entities.items():
        html = html.replace(ent, repl)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_sec_filing(ticker):
    """Fetch the most recent 10-Q or 10-K filing for the ticker from SEC EDGAR.
    Returns dict with form, filing_date, report_date, url, text (MD&A section)."""
    # Step 1: ticker -> CIK
    try:
        tickers_resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=10,
        )
        tickers_data = tickers_resp.json()
    except Exception:
        return None

    ticker_upper = ticker.upper()
    cik = None
    for entry in tickers_data.values():
        if entry.get("ticker") == ticker_upper:
            cik = entry.get("cik_str")
            break
    if cik is None:
        return None

    # Step 2: Get list of recent filings
    cik_padded = str(cik).zfill(10)
    try:
        sub_resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
            headers=SEC_HEADERS,
            timeout=10,
        )
        sub_data = sub_resp.json()
    except Exception:
        return None

    recent = sub_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    # Step 3: Find the most recent 10-Q or 10-K (10-Q is quarterly, 10-K is annual)
    target_idx = None
    for i, form in enumerate(forms):
        if form in ("10-Q", "10-K"):
            target_idx = i
            break
    if target_idx is None:
        return None

    accession = accession_numbers[target_idx].replace("-", "")
    primary_doc = primary_docs[target_idx]
    form_type = forms[target_idx]
    filing_date = filing_dates[target_idx] if target_idx < len(filing_dates) else ""
    report_date = report_dates[target_idx] if target_idx < len(report_dates) else ""

    # Step 4: Fetch the primary document
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    try:
        doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=20)
        html = doc_resp.text
    except Exception:
        return None

    text = strip_html(html)

    # Step 5: Find the MD&A section. Filings have a "Discussion and Analysis"
    # header somewhere after the financial tables. Take a 60K-char window from
    # there. If we can't find it, fall back to the first 60K chars.
    lower = text.lower()
    mda_keywords = [
        "management's discussion and analysis",
        "managements discussion and analysis",
        "management discussion and analysis",
    ]
    mda_start = -1
    for kw in mda_keywords:
        # Skip the first occurrence (usually in the table of contents) and
        # take the second, which is the actual section header.
        first = lower.find(kw)
        if first == -1:
            continue
        second = lower.find(kw, first + len(kw))
        mda_start = second if second != -1 else first
        break

    if mda_start == -1:
        text_slice = text[:60000]
    else:
        text_slice = text[mda_start:mda_start + 60000]

    return {
        "form": form_type,
        "filing_date": filing_date,
        "report_date": report_date,
        "url": doc_url,
        "text": text_slice,
    }


def analyze_sec_filing(filing, company):
    """Use Claude to analyze the SEC filing's MD&A section. Returns structured
    growth / margins / outlook / risks / downplaying analysis."""
    if not filing or not filing.get("text"):
        return None

    company_name = company.get("companyName") or ""
    industry = company.get("industry") or ""
    form = filing.get("form")
    filing_date = filing.get("filing_date")
    text = filing.get("text", "")

    prompt = f"""You are analyzing a {form} filing from {company_name} ({industry}), filed on {filing_date}. The text below is from the Management's Discussion and Analysis (MD&A) section — a part of every quarterly/annual SEC filing where executives explain their business results, outlook, and risks. Your audience is a beginner-to-intermediate investor.

Generate a structured plain-English analysis with the sections below. Each section should:
- Lead with **a bold takeaway sentence** wrapped in markdown asterisks (**like this**).
- Use everyday words. Define financial terms inline (example: "operating margin — meaning the share of each dollar of sales the company keeps after running the business but before interest and taxes").
- Spell out acronyms (DOJ -> "the U.S. Department of Justice"; SEC -> "the Securities and Exchange Commission").
- Quote specific numbers from the filing when discussing them. Don't fabricate.
- 2-4 sentences per section. Be tight.

1. growth — How did revenue/sales change versus the prior period? Quote actual numbers when present.

2. margins — Did profitability improve or worsen? Why?

3. outlook — What did management say about the future? If they declined to give specifics, note that.

4. risks — What concerns did management acknowledge? Be specific to this company.

5. downplaying — This is the most important section. Look for things management seems to be glossing over or carefully framing:
   - Strong claims with no supporting numbers (vague reassurances)
   - Topics raised then quickly skipped past
   - Risks listed in technical or boilerplate language to dilute their impact
   - Numbers given in unusual frames ("constant currency", "adjusted", "normalized") that may obscure what really happened
   - Industry-wide tailwinds claimed as company achievements
   - Be specific. Quote phrases or describe the evasion concretely. Don't manufacture if it's not there. If the filing was unusually transparent, that's fine — output 0 items.

For the downplaying items, each entry has:
- "topic": short 2-5 word label of the issue
- "what_happened": 1-2 sentences explaining the dodge in plain English with the bold-takeaway pattern (**bold lead** then context)

MD&A text (truncated to ~60K chars):
{text}

Respond in this exact JSON format. No preamble, no markdown fences, no explanation outside the JSON:
{{
  "growth": "...",
  "margins": "...",
  "outlook": "...",
  "risks": "...",
  "downplaying": [
    {{"topic": "...", "what_happened": "..."}}
  ]
}}"""

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_response = message.content[0].text.strip()
        if text_response.startswith("```"):
            text_response = text_response.split("\n", 1)[1] if "\n" in text_response else text_response
            text_response = text_response.rsplit("```", 1)[0].strip()
        return json.loads(text_response)
    except Exception as e:
        return {"_error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


VALID_TABS = ("overview", "news", "cases", "filing", "financials")


def fetch_income_statement(ticker, limit=5):
    """Annual income statements (revenue, net income, etc.) from FMP."""
    if not FMP_API_KEY:
        return []
    url = (
        f"https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={ticker}&apikey={FMP_API_KEY}&limit={limit}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def fetch_historical_prices(ticker):
    """Historical end-of-day prices from FMP. Returns up to ~1 year."""
    if not FMP_API_KEY:
        return []
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/light"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception:
        return []
    # Endpoint may return either a list or a wrapper dict
    if isinstance(data, dict):
        data = data.get("historical") or data.get("data") or []
    if not isinstance(data, list):
        return []
    # Trim to last 252 trading days (~1 year). FMP returns newest first.
    return data[:252]


def analyze_financials(income_statement, price_history, company):
    """Use Claude to generate 1-2 sentence plain-English commentary under each
    chart. Compresses raw data to key numbers before prompting to keep cost low."""
    if not income_statement and not price_history:
        return None

    company_name = company.get("companyName") or ""

    # Compress income statement to key numbers per year
    income_summary = ""
    if income_statement:
        sorted_income = sorted(
            income_statement,
            key=lambda d: d.get("calendarYear") or d.get("fiscalYear") or d.get("date", ""),
        )
        lines = []
        for d in sorted_income:
            year = d.get("calendarYear") or d.get("fiscalYear") or (d.get("date") or "")[:4]
            rev = d.get("revenue")
            ni = d.get("netIncome")
            if rev is None:
                continue
            rev_b = rev / 1e9
            if ni is not None:
                lines.append(f"{year}: revenue ${rev_b:.1f}B, net income ${ni / 1e9:.1f}B")
            else:
                lines.append(f"{year}: revenue ${rev_b:.1f}B")
        income_summary = "\n".join(lines)

    # Compress price history to start/end/range
    price_summary = ""
    if price_history:
        sorted_prices = sorted(price_history, key=lambda d: d.get("date", ""))
        first = sorted_prices[0]
        last = sorted_prices[-1]
        first_price = first.get("price") if first.get("price") is not None else first.get("close")
        last_price = last.get("price") if last.get("price") is not None else last.get("close")
        all_prices = [
            (p.get("price") if p.get("price") is not None else p.get("close"))
            for p in sorted_prices
            if (p.get("price") is not None or p.get("close") is not None)
        ]
        if first_price and last_price and all_prices:
            pct = ((last_price - first_price) / first_price) * 100.0
            price_summary = (
                f"Price went from ${first_price:.2f} on {first.get('date')} "
                f"to ${last_price:.2f} on {last.get('date')} ({pct:+.1f}% over the year). "
                f"Range: ${min(all_prices):.2f} – ${max(all_prices):.2f}."
            )

    prompt = f"""You are explaining a publicly traded company's financial charts to a beginner-to-intermediate investor. Your job is to add a 1-2 sentence plain-English commentary under each chart that contextualizes whether the trend is good or bad and why.

Company: {company_name}

Annual financials (oldest to newest):
{income_summary or '(not available)'}

Stock price over the last year:
{price_summary or '(not available)'}

Generate three short commentaries:

1. revenue_commentary: Is the revenue trend strong or weak? Has growth accelerated or slowed? Compare to what's typical for this company or industry when confident.

2. net_income_commentary: Is profit stable, growing, or shrinking? Are margins (the share of revenue that becomes profit) improving or compressing? Connect to the revenue trend if relevant.

3. price_commentary: How has the stock done over the year? If you're confident, mention whether it outperformed or underperformed the broader market (the S&P 500 grew roughly 10–15% per year on average historically). If you're not confident about the comparison, just describe the trend without making a benchmark claim.

WRITING RULES:
- Lead each commentary with **a bold takeaway** wrapped in markdown asterisks (e.g., "**Revenue growth has slowed sharply.**").
- Plain English. Define financial terms inline (e.g., "net income — meaning profit after all costs and taxes").
- 1-2 sentences max per commentary. Be tight.
- Don't fabricate numbers beyond what's given above.

Respond in this exact JSON format. No preamble, no markdown fences:
{{
  "revenue_commentary": "...",
  "net_income_commentary": "...",
  "price_commentary": "..."
}}"""

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"_error": str(e)}


@app.route("/stock")
def stock():
    ticker = request.args.get("ticker", "").upper().strip()
    tab = request.args.get("tab", "overview")
    if tab not in VALID_TABS:
        tab = "overview"

    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")

    # Always fetch the company profile — header bar (name + stats) is on every tab
    company = cached_or_fetch(f"profile:{ticker}", 300, fetch_company_profile, ticker)
    if not company:
        return render_template(
            "index.html",
            error=f"Could not find data for '{ticker}'. Double-check the ticker symbol.",
        )

    # Each tab loads only what it needs. First time on a tab = slow; cached subsequent.
    plain_english = None
    bull_bear = None
    articles = []

    if tab == "overview":
        plain_english = cached_or_fetch(
            f"plain:{ticker}", 1800, get_plain_english_summary, company
        )
    elif tab == "news":
        company_name = company.get("companyName") or ticker
        raw_articles = cached_or_fetch(f"news:{company_name}", 600, get_news, company_name, 4)
        if raw_articles:
            with ThreadPoolExecutor(max_workers=4) as executor:
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
                analyses = [f.result() for f in article_futures]
            for art, analysis in zip(raw_articles, analyses):
                articles.append({
                    "title": art.get("title"),
                    "source": (art.get("source") or {}).get("name") or "Unknown",
                    "url": art.get("url"),
                    "published_at": art.get("publishedAt"),
                    "analysis": analysis,
                })
    elif tab == "cases":
        bull_bear = cached_or_fetch(f"bb:{ticker}", 1800, get_bull_bear, company)
    elif tab == "filing":
        filing = cached_or_fetch(f"sec:{ticker}", 86400, fetch_sec_filing, ticker)
        sec_analysis = None
        if filing:
            sec_analysis = cached_or_fetch(
                f"sec_analysis:{ticker}:{filing.get('filing_date', '')}",
                86400,
                analyze_sec_filing,
                filing,
                company,
            )
        return render_template(
            "stock.html",
            company=company,
            tab=tab,
            filing=filing,
            sec_analysis=sec_analysis,
            plain_english=None,
            bull_bear=None,
            articles=[],
            income_statement=[],
            price_history=[],
            financials_commentary=None,
        )
    elif tab == "financials":
        income = cached_or_fetch(f"income:{ticker}", 86400, fetch_income_statement, ticker, 5)
        prices = cached_or_fetch(f"prices:{ticker}", 3600, fetch_historical_prices, ticker)
        # Cache key includes most-recent income date so commentary refreshes when new
        # data lands; otherwise we'd serve stale interpretation against new charts.
        latest_marker = (income[0].get("date") if income else "") + (
            prices[0].get("date") if prices else ""
        )
        financials_commentary = None
        if income or prices:
            financials_commentary = cached_or_fetch(
                f"fin_analysis:{ticker}:{latest_marker}",
                86400,
                analyze_financials,
                income,
                prices,
                company,
            )
        return render_template(
            "stock.html",
            company=company,
            tab=tab,
            income_statement=income,
            price_history=prices,
            financials_commentary=financials_commentary,
            plain_english=None,
            bull_bear=None,
            articles=[],
            filing=None,
            sec_analysis=None,
        )

    return render_template(
        "stock.html",
        company=company,
        tab=tab,
        plain_english=plain_english,
        bull_bear=bull_bear,
        articles=articles,
        filing=None,
        sec_analysis=None,
        income_statement=[],
        price_history=[],
        financials_commentary=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)