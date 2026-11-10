"""Build dashboard/data.json: daily prices plus attention from every source.

Run by the GitHub Action in .github/workflows/dashboard-data.yml every
weekday, so the dashboard always has fresh numbers. You can also run it
yourself: `python scripts/build_dashboard_data.py`.

Which companies and how each is looked up: src/wikiquant/universe.py.
Which data sources: src/wikiquant/sources/ (add a plugin, list it in SOURCES).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant.data import load_prices  # noqa: E402
from wikiquant.sources import get_source  # noqa: E402
from wikiquant.universe import BENCHMARK, UNIVERSE  # noqa: E402

START = "2019-01-01"
# The first source is required; the others are skipped (with a warning) if
# their API is down, so one flaky source cannot stop the daily refresh.
SOURCES = ["wikipedia", "hackernews"]


def build_page() -> None:
    """Inline data.json into template.html to make one self-contained page."""
    template = (ROOT / "dashboard" / "template.html").read_text()
    data = (ROOT / "dashboard" / "data.json").read_text()
    page = template.replace("/*__DATA__*/", data.replace("</", "<\\/"))
    (ROOT / "dashboard" / "index.html").write_text(page)
    print("Wrote dashboard/index.html")


def to_list(series: pd.Series, digits: int | None = None) -> list:
    if digits is None:
        return [None if pd.isna(v) else int(round(v)) for v in series]
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--page-only", action="store_true", help="only rebuild index.html")
    args = parser.parse_args()
    if args.page_only:
        return build_page()

    end = (date.today() - timedelta(days=1)).isoformat()
    tickers = [c.ticker for c in UNIVERSE] + [BENCHMARK]
    prices = load_prices(tickers, start=START, refresh=True).dropna(how="all").ffill()
    calendar = pd.date_range(START, end, freq="D")

    attention: dict[str, dict[str, list]] = {c.ticker: {} for c in UNIVERSE}
    meta = []
    for i, name in enumerate(args.sources):
        source = get_source(name)
        try:
            for c in UNIVERSE:
                series = source.load(c.keys_for(name), START, end).reindex(calendar)
                attention[c.ticker][name] = to_list(series)
                last = series.dropna()
                print(f"[{name}] {c.ticker}: {int(last.sum()):,} {source.unit}, "
                      f"last {last.index.max().date() if len(last) else '-'}")
        except Exception as err:  # noqa: BLE001
            if i == 0:
                raise
            print(f"WARNING: skipping source {name!r}: {err!r}")
            for c in UNIVERSE:
                attention[c.ticker].pop(name, None)
            continue
        meta.append(source.describe())

    # Trim trailing calendar days where no source has data yet.
    last_day = max(
        (calendar[max(i for i, v in enumerate(vals) if v is not None)]
         for per in attention.values() for vals in per.values() if any(v is not None for v in vals)),
        default=calendar[-1],
    )
    n = len(pd.date_range(START, last_day, freq="D"))

    out = {
        "built": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "sources": {"prices": "Yahoo Finance via yfinance (split and dividend adjusted close)"},
        "attention": meta,
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
        "viewsStart": START,
        "index": {"ticker": BENCHMARK, "name": "S&P 500 (SPY)", "close": to_list(prices[BENCHMARK], 2)},
        "stocks": [
            {
                "ticker": c.ticker,
                "name": c.name,
                "wiki": c.keys_for("wikipedia")[0],
                "close": to_list(prices[c.ticker], 2),
                "attention": {k: v[:n] for k, v in attention[c.ticker].items()},
            }
            for c in UNIVERSE
        ],
    }
    path = ROOT / "dashboard" / "data.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    build_page()


if __name__ == "__main__":
    main()
