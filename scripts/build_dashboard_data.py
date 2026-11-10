"""Build dashboard/data.json: daily prices and Wikipedia page views.

Run by the GitHub Action in .github/workflows/dashboard-data.yml every
weekday, so the dashboard always has fresh numbers. You can also run it
yourself: `python scripts/build_dashboard_data.py`.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant.data import load_prices  # noqa: E402
from wikiquant.pageviews import load_pageviews  # noqa: E402

START = "2019-01-01"
INDEX = "SPY"  # ETF that tracks the S&P 500

# ticker, display name, English Wikipedia article(s). When a company page was
# renamed (Facebook -> Meta), list every title: views are summed, because
# readers of the old title land on a redirect that is counted separately.
UNIVERSE = [
    ("AAPL", "Apple", "Apple_Inc."),
    ("MSFT", "Microsoft", "Microsoft"),
    ("NVDA", "Nvidia", "Nvidia"),
    ("AMZN", "Amazon", "Amazon_(company)"),
    ("GOOGL", "Alphabet", "Alphabet_Inc."),
    ("META", "Meta", ["Meta_Platforms", "Facebook,_Inc."]),
    ("TSLA", "Tesla", "Tesla,_Inc."),
    ("NFLX", "Netflix", "Netflix"),
    ("JPM", "JPMorgan Chase", "JPMorgan_Chase"),
    ("DIS", "Disney", "The_Walt_Disney_Company"),
    ("BA", "Boeing", "Boeing"),
    ("GME", "GameStop", "GameStop"),
    ("AMD", "AMD", ["AMD", "Advanced_Micro_Devices"]),
    ("PFE", "Pfizer", "Pfizer"),
    ("NKE", "Nike", "Nike,_Inc."),
    ("INTC", "Intel", "Intel"),
]


def build_page() -> None:
    """Inline data.json into template.html to make one self-contained page."""
    template = (ROOT / "dashboard" / "template.html").read_text()
    data = (ROOT / "dashboard" / "data.json").read_text()
    page = template.replace("/*__DATA__*/", data.replace("</", "<\\/"))
    (ROOT / "dashboard" / "index.html").write_text(page)
    print("Wrote dashboard/index.html")


def main() -> None:
    end = (date.today() - timedelta(days=1)).isoformat()
    tickers = [t for t, _, _ in UNIVERSE] + [INDEX]
    prices = load_prices(tickers, start=START, refresh=True).dropna(how="all")
    prices = prices.ffill()

    stocks = []
    for ticker, name, title in UNIVERSE:
        titles = [title] if isinstance(title, str) else title
        parts = [load_pageviews(t, START, end, refresh=True) for t in titles]
        views = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        title = titles[0]
        # Before an article existed under any listed title (e.g. Meta before
        # "Facebook, Inc." was split out), counts are near zero. Treat that
        # stretch as missing rather than as a sudden attention explosion later.
        real = views >= 0.1 * views.median()
        if real.any():
            views = views.where(views.index >= real.idxmax())
        views = views.reindex(pd.date_range(START, views.index.max(), freq="D"))
        stocks.append({
            "ticker": ticker,
            "name": name,
            "wiki": title,
            "close": [None if pd.isna(v) else round(float(v), 2) for v in prices[ticker]],
            "views": [None if pd.isna(v) else int(v) for v in views],
        })
        print(f"{ticker}: {len(views)} days of views, last price {prices[ticker].iloc[-1]:.2f}")

    out = {
        "built": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "sources": {
            "views": "Wikimedia Pageviews API (en.wikipedia, human traffic)",
            "prices": "Yahoo Finance via yfinance (split and dividend adjusted close)",
        },
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
        "viewsStart": START,
        "index": {"ticker": INDEX, "name": "S&P 500 (SPY)",
                  "close": [round(float(v), 2) for v in prices[INDEX]]},
        "stocks": stocks,
    }
    path = ROOT / "dashboard" / "data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    build_page()


if __name__ == "__main__":
    main()
