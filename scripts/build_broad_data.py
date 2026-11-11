"""Build the broad dataset: ~500 stocks picked as of 1 Jan 2016, with prices and page views.

Run by the GitHub Action in .github/workflows/broad-study.yml (this needs the
internet: Wikipedia and Yahoo Finance). You can also run it yourself:

    python scripts/build_broad_data.py            # uses data/broad/universe.csv if present
    python scripts/build_broad_data.py --rebuild-universe

Writes, under data/broad/:
- universe.csv     every company in the S&P 500 list on the start date
- universe.json    which old version of the Wikipedia list was used
- prices.csv.gz    daily adjusted close, one column per company (old ticker)
- views.csv.gz     daily Wikipedia page views, one column per company
- coverage.csv     per company: where its prices came from, or why they are missing

Prices: Yahoo Finance first. For companies Yahoo no longer has (most were
bought out), we try Stooq, which keeps some delisted US stocks. Whatever is
still missing is listed in coverage.csv, so the report can say exactly how
much of the universe we could not test.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant import broad_universe as U  # noqa: E402
from wikiquant.sources import get_source  # noqa: E402

OUT = ROOT / "data" / "broad"
VIEWS_START = "2015-07-01"     # first day of the Wikimedia Pageviews API
PRICE_START = "2015-07-01"
BENCHMARK = "SPY"
# A stock that was in the index on the start date must have traded then.
# Anything Yahoo returns that starts later is a different company that reused
# the ticker (e.g. "SE" was Spectra Energy, now it is Sea Ltd), so we reject it.
MAX_LATE_START = pd.Timedelta(days=10)


# ---------------------------------------------------------------- prices
def yahoo_many(tickers: list[str], start: str) -> dict[str, pd.Series]:
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        if i:
            time.sleep(2)
        df = None
        for attempt in range(4):
            try:
                df = yf.download(batch, start=start, auto_adjust=True, progress=False,
                                 group_by="ticker", threads=True)
                break
            except Exception as err:  # noqa: BLE001  (rate limits, timeouts)
                print(f"  yahoo batch failed ({err!r}), retrying")
                time.sleep(10 * (attempt + 1))
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                s = df[t]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
            except KeyError:
                continue
            s = s.dropna()
            if len(s):
                out[t] = s.astype(float)
    return out


def yahoo_one(ticker: str, start: str) -> pd.Series | None:
    """Single-ticker retry. Big batch downloads sometimes come back empty or
    truncated for a few tickers (rate limits), so anything missing or suspiciously
    short is asked for again on its own."""
    import yfinance as yf

    best = None
    for attempt in range(3):
        got = yahoo_many([ticker], start).get(ticker)
        if attempt:
            # A different request shape: Yahoo sometimes serves only recent
            # months to start= queries for a ticker but the full history to period="max".
            try:
                h = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"].dropna()
                h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                h = h.loc[start:].astype(float)
                if len(h) and (got is None or h.index.min() < got.index.min()):
                    got = h
            except Exception:  # noqa: BLE001
                pass
        if got is not None and (best is None or got.index.min() < best.index.min()):
            best = got
        if best is not None and best.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=200):
            break
        time.sleep(3 * (attempt + 1))
    return best


def stooq_one(ticker: str, start: str) -> pd.Series | None:
    import requests

    sym = ticker.lower().replace(".", "-") + ".us"
    try:
        r = requests.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d"}, timeout=30,
                         headers={"User-Agent": U.USER_AGENT})
        if r.status_code != 200 or not r.text.startswith("Date"):
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        s = df["Close"].loc[start:].dropna().astype(float)
        return s if len(s) else None
    except Exception:  # noqa: BLE001
        return None


def load_all_prices(members, start_date: pd.Timestamp, use_stooq: bool = True):
    cands = {m.ticker: m.yahoo.split("|") for m in members}
    first_try = sorted({c[0] for c in cands.values()} | {BENCHMARK})
    got = yahoo_many(first_try, PRICE_START)
    later = sorted({t for c in cands.values() for t in c[1:]} - set(got))
    if later:
        got.update(yahoo_many(later, PRICE_START))

    prices, rows = {}, []
    for m in members:
        row = {"ticker": m.ticker, "name": m.name, "sector": m.sector,
               "source": "", "symbol": "", "first": "", "last": "", "note": ""}
        rejected = []
        for sym in cands[m.ticker]:
            s = got.get(sym)
            if s is None or s.index.min() > start_date + MAX_LATE_START:
                retry = yahoo_one(sym, PRICE_START)
                if retry is not None and (s is None or retry.index.min() < s.index.min()):
                    s = got[sym] = retry
            if s is None:
                continue
            if s.index.min() > start_date + MAX_LATE_START:
                rejected.append(f"{sym} starts {s.index.min().date()}")
                continue
            prices[m.ticker] = s
            row.update(source="yahoo", symbol=sym)
            break
        if m.ticker not in prices and use_stooq:
            s = stooq_one(m.ticker, PRICE_START)
            time.sleep(0.3)
            if s is not None and s.index.min() <= start_date + MAX_LATE_START:
                prices[m.ticker] = s
                row.update(source="stooq", symbol=m.ticker)
        if m.ticker in prices:
            s = prices[m.ticker]
            row.update(first=str(s.index.min().date()), last=str(s.index.max().date()))
        else:
            row["note"] = "; ".join(rejected) or "no price data (most likely bought out or delisted)"
        rows.append(row)
    bench = got.get(BENCHMARK)
    if bench is None:
        raise RuntimeError("Could not download the benchmark (SPY)")
    return prices, bench, pd.DataFrame(rows)


# ---------------------------------------------------------------- views
def load_all_views(members, end: str) -> tuple[dict[str, pd.Series], dict[str, str]]:
    src = get_source("wikipedia")
    views, errors = {}, {}
    for i, m in enumerate(members):
        try:
            views[m.ticker] = src.load(m.keys, VIEWS_START, end)
        except Exception as err:  # noqa: BLE001
            errors[m.ticker] = repr(err)[:200]
        if i % 50 == 0:
            print(f"  views {i}/{len(members)}")
        time.sleep(0.2)    # be polite to the Wikimedia API
    return views, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=U.START_DATE)
    ap.add_argument("--rebuild-universe", action="store_true")
    ap.add_argument("--no-stooq", action="store_true")
    ap.add_argument("--refresh-former", action="store_true", help="re-scan redirects for old article titles")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.rebuild_universe or not (OUT / "universe.csv").exists():
        members, meta = U.build_universe(args.as_of)
        U.save_universe(members, OUT / "universe.csv")
        (OUT / "universe.json").write_text(json.dumps(meta, indent=2))
        print(f"Universe: {len(members)} companies from {meta['url']} ({meta['timestamp']})")
    else:
        members = U.load_universe(OUT / "universe.csv")
        # Pick up ticker renames added since the list was saved, and re-follow
        # redirects (articles get renamed over time).
        current = U.resolve_redirects([m.article for m in members])
        for m in members:
            m.yahoo = "|".join(U.yahoo_tickers(m.ticker))
            m.current = current.get(m.article, m.article)
        U.save_universe(members, OUT / "universe.csv")
        print(f"Universe: {len(members)} companies (cached in data/broad/universe.csv)")

    end = (date.today() - timedelta(days=1)).isoformat()
    if args.refresh_former or not (OUT / "former_checked").exists():
        # Moved articles keep their pre-move views under the old title.
        print("Looking for former article titles (busy redirects)...")
        former = U.former_titles([m.current for m in members], VIEWS_START, end)
        for m in members:
            m.former = "|".join(t for t in former.get(m.current, []) if t != m.article)
        print(f"  {sum(bool(m.former) for m in members)} articles had earlier titles with real traffic")
        U.save_universe(members, OUT / "universe.csv")
        (OUT / "former_checked").write_text(end + "\n")

    start_date = pd.Timestamp(args.as_of)

    print("Downloading prices...")
    prices, bench, cov = load_all_prices(members, start_date, use_stooq=not args.no_stooq)
    print("Downloading page views...")
    views, errors = load_all_views(members, end)

    px = pd.DataFrame(prices).sort_index()
    px[BENCHMARK] = bench
    px = px.loc[PRICE_START:].dropna(how="all", subset=[BENCHMARK])
    px.index.name = "date"
    vw = pd.DataFrame(views).reindex(pd.date_range(VIEWS_START, end, freq="D"))
    vw.index.name = "date"
    # Trim trailing days the API has not published yet.
    vw = vw.loc[:vw.dropna(how="all").index.max()]

    cov["views_median"] = [float(np.nanmedian(views[t].to_numpy())) if t in views and views[t].notna().any()
                           else np.nan for t in cov["ticker"]]
    cov["views_error"] = [errors.get(t, "") for t in cov["ticker"]]
    cov["pages"] = [" + ".join(m.keys) for m in members]
    try:
        today = U.current_articles()
        cov["in_index_today"] = [m.current in today for m in members]
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: could not read today's S&P 500 list: {err!r}")
        cov["in_index_today"] = ""
    cov.to_csv(OUT / "coverage.csv", index=False)

    # 5 significant figures is plenty for daily returns and keeps the files small.
    px.apply(lambda s: s.map(lambda v: float(f"{v:.5g}") if pd.notna(v) else v)).to_csv(
        OUT / "prices.csv.gz", compression={"method": "gzip", "mtime": 0})
    vw.round(0).astype("Int64").to_csv(OUT / "views.csv.gz", compression={"method": "gzip", "mtime": 0})

    n_px = (cov["source"] != "").sum()
    print(f"Prices for {n_px}/{len(cov)} companies "
          f"(yahoo {(cov['source'] == 'yahoo').sum()}, stooq {(cov['source'] == 'stooq').sum()}); "
          f"views for {len(views)}/{len(members)}; errors {len(errors)}")
    print(f"Wrote {OUT}/prices.csv.gz, views.csv.gz, coverage.csv")
    if len(errors) > 0.02 * len(members):
        # Usually rate limiting. Fail so the workflow does not commit a half-empty dataset.
        sys.exit(f"Page views failed for {len(errors)} companies, e.g. {next(iter(errors.values()))}")


if __name__ == "__main__":
    main()
