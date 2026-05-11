# Stock Research

An AI-powered stock research web app built for beginner-to-intermediate investors who don't have time to parse Bloomberg articles, SEC filings, or earnings calls on their own. Translates dense financial information into plain English and explicitly flags what AI summaries usually leave out.

Built as the final project for Bowdoin College's "AI in the World" course, Spring 2026.

> > **Live demo:** [stock-research-gv9e.onrender.com](https://stock-research-gv9e.onrender.com)

## What makes this different

Most stock-research apps either give you raw data (Yahoo Finance) or give you confident-sounding AI summaries (newer "AI investor" tools). The first leaves beginners overwhelmed and the second hides the fact that AI summarization loses information.

This app does both and adds a third layer the others don't: **bias detection**. Every news article and SEC filing gets analyzed for what the AI summary *dropped* or *downplayed* compared to the source. Buried risks, removed uncertainties, framing shifts, and what management seemed to be dodging are surfaced as explicit, labeled items so the reader can see them.

It practices the project's thesis on itself: when AI is going to lose information, the honest design is to tell the user what got lost.

## Features

- **Search by ticker or company name.** Type "Apple" or "AAPL" — both work. For private companies (OpenAI, SpaceX), the app returns a Claude-generated explanation of why they're not searchable and suggests related public proxies.
- **Plain-English company overview.** Every key stat (P/E, market cap, 52-week range, dividend yield) gets a click-to-expand AI explanation specific to that company.
- **News feed with bias detection.** Each article is summarized AND labeled with what an AI summary would lose — categorized as buried risk, removed uncertainty, framing shift, source dodging, or missing context.
- **Bull case vs. bear case.** Side-by-side balanced arguments for and against owning the stock, written in beginner-friendly language with definitions inline.
- **Latest SEC filing analysis.** Pulls the most recent 10-Q or 10-K from SEC EDGAR, runs Claude over the Management Discussion & Analysis section, surfaces growth, margins, outlook, risks, and what management seemed to be downplaying.
- **Financials charts with AI commentary.** Annual revenue and net income charts plus 1-year stock price, each with a 1-2 sentence interpretation of what the trend means.
- **Side-by-side comparison.** Compare any two stocks with a Claude-synthesized headline difference, three category breakdowns, and "who's this for" framing.
- **Discover mode.** Don't know what to research? Pick an interest (AI & technology, healthcare, dividends, etc.) or describe your interest in free text — get 5 AI-suggested publicly-traded companies that fit.
- **Browse ETFs.** Curated list of 6 beginner-friendly broad-market ETFs (VOO, VTI, QQQ, VT, VIG, BND) with plain-English descriptions, because for most beginners these tend to outperform individual stock picking.
- **Dark and light mode.** Toggleable, persisted, respects system preference on first visit.
- **Recently viewed stocks** sidebar for fast navigation between researched tickers.
- **First-time-user guide** dismissible overlays on the home and stock pages, reopenable via the help button in the header.

## Tech stack

- **Backend:** Python, Flask
- **AI:** Claude API (Anthropic) — Sonnet 4.6 — for all natural-language generation, bias detection, and synthesis
- **Stock data:** Financial Modeling Prep API (free tier)
- **News data:** NewsAPI (free tier)
- **SEC filings:** SEC EDGAR (free, public-domain)
- **Charts:** Chart.js (CDN)
- **Typography:** Inter via Google Fonts
- **Deployment:** Render
- **Caching:** In-memory TTL cache, keyed per ticker and per Claude prompt, so repeat searches are near-instant

## Running locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/stock-research.git
cd stock-research

# Install dependencies
pip install -r requirements.txt

# Set environment variables (get keys from the providers below)
export ANTHROPIC_API_KEY="your-key-here"
export FMP_API_KEY="your-key-here"
export NEWSAPI_KEY="your-key-here"

# Run the dev server
python main.py
```

Open `http://localhost:8080` in your browser.

API keys are free (besides Anthropic, which requires loading credits):
- Anthropic: [console.anthropic.com](https://console.anthropic.com) (paid, $5)
- Financial Modeling Prep: [financialmodelingprep.com](https://financialmodelingprep.com) (free tier)
- NewsAPI: [newsapi.org](https://newsapi.org) (free tier)

## Project context

This was the artifact for our final project in Bowdoin College's "AI in the World" course (Spring 2026). The accompanying maker's-statement essay documents the build process, the prompt-engineering iterations, the technical pivots, and what the experience revealed about AI's capabilities and limitations in a consumer-facing tool.

Some honest notes on what we learned:
- AI summaries lose information. The bias-detection prompt itself had to be iterated multiple times to overcome generic "word-choice critique" failure modes and produce substantive analysis with inline definitions for beginner readers.
- Free APIs aren't always free for long. FMP deprecated the v3 profile endpoint mid-project; SEC EDGAR is the more durable, mandatory-disclosure data source.
- Tooling pivots happen. Replit replaced its template flow with an AI-agent flow during our build; we moved to GitHub Codespaces.
- Claude's training cutoff matters when the tool is positioned as "real-time research." We added hedging language ("as of last reporting") for time-sensitive claims like IPO plans.

## Credits

- **Arthur Dos Santos & Masai Gordon** — product direction, prompt engineering, build sessions, maker's-statement essay, and user testing.


Built with substantial help from Claude (Anthropic), which generated and iteratively refined the code over four phases across two weeks.

## Disclaimer

This app is for educational and research purposes only. Nothing here is investment advice. AI-generated content may be inaccurate or outdated — verify any specific claim before making investment decisions.
