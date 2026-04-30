import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# These pull from Replit Secrets — never hardcode keys here.
FMP_API_KEY = os.environ.get("FMP_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stock")
def stock():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")

    # Pull basic company profile from Financial Modeling Prep
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
    return render_template("stock.html", company=company)


if __name__ == "__main__":
    # 0.0.0.0 + port 8080 is what Replit expects for web apps
    app.run(host="0.0.0.0", port=8080, debug=True)
