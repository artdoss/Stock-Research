# Stock Research

AI-powered stock research web app. Built as a final project for Bowdoin's "AI in the World" course.

## Features (in progress)

- Search any publicly traded stock by ticker
- Plain-English company overview with key stats
- News feed with bias-detection layer that flags what AI summaries leave out

## Stack

- Python / Flask
- Claude API for summarization, bias detection, and plain-English explanations
- NewsAPI for article aggregation
- Financial Modeling Prep API for stock data

## Running

```bash
pip install -r requirements.txt
python main.py
```

Requires environment variables: `ANTHROPIC_API_KEY`, `NEWSAPI_KEY`, `FMP_API_KEY`.
