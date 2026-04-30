import os
import requests
from flask import Flask, render_template, request
from anthropic import Anthropic

app = Flask(__name__)

FMP_API_KEY = os.environ.get("FMP_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

claude = Anthropic(api_key=ANTHROPIC_API_KEY)


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
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"(AI summary unavailable: {e})"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stock")
def stock():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")

    profile_url = (
        f"https://financialmodelingprep.com/stable/profile"
        f"?symbol={ticker}&apikey={FMP_API_KEY}"
    )
    try:
        response = requests.get(profile_url, timeout=10)
        data = response.json()
    except Exception as e:
        return render_template("index.html", error=f"Network error: {e}")

    if not data or not isinstance(data, list):
        return render_template(
            "index.html",
            error=f"Could not find data for '{ticker}'. Double-check the ticker symbol.",
        )

    company = data[0]
    plain_english = get_plain_english_summary(company)
    return render_template("stock.html", company=company, plain_english=plain_english)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)