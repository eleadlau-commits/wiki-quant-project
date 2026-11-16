"""Download earnings 8-K filings from SEC EDGAR (free, no key).

SEC asks every client to send a User-Agent with contact details and to stay
under ~10 requests a second. The contact comes from the SEC_USER_AGENT
environment variable (e.g. "Your Name your@email"), never from code.

    events = load_earnings_events()       # one row per earnings 8-K, all 23 names
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from wikiquant.earnings_cycle import UNIVERSE, is_earnings_filing

BASE = "https://data.sec.gov/submissions/"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "edgar"
COLUMNS = ["ticker", "cik", "form", "items", "filing_date", "accepted", "accession"]


def user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT to your name and email (SEC requires it), "
                         'e.g. export SEC_USER_AGENT="Jane Doe jane@example.com"')
    return ua


def _get_json(url: str, session: requests.Session) -> dict:
    for attempt in range(4):
        r = session.get(url, timeout=30)
        if r.status_code in (429, 503) and attempt < 3:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.15)  # stay well under SEC's 10 requests a second
        return r.json()
    raise RuntimeError(f"SEC kept refusing {url}")


def _rows(block: dict, ticker: str, cik: str) -> list[dict]:
    n = len(block.get("form", []))
    return [{"ticker": ticker, "cik": cik, "form": block["form"][i], "items": block.get("items", [""] * n)[i],
             "filing_date": block["filingDate"][i], "accepted": block["acceptanceDateTime"][i],
             "accession": block["accessionNumber"][i]} for i in range(n)]


def fetch_filings(ticker: str, cik: str, session: requests.Session, older: bool = True) -> list[dict]:
    """Every filing of one CIK: the recent block plus (optionally) the older pages."""
    data = _get_json(f"{BASE}CIK{cik}.json", session)
    rows = _rows(data["filings"]["recent"], ticker, cik)
    if older:
        for f in data["filings"].get("files", []):
            rows += _rows(_get_json(BASE + f["name"], session), ticker, cik)
    return rows


def load_earnings_events(cache_dir: Path | str = DEFAULT_CACHE, refresh: bool = True,
                         since: str = "2014-01-01", session: requests.Session | None = None) -> pd.DataFrame:
    """Earnings 8-Ks (Item 2.02 + the three mistagged 2.01s) for the whole universe.

    The older filing pages never change, so they are cached; the recent block
    is re-downloaded when refresh=True (every evening).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
    rows = []
    for ticker, ciks in UNIVERSE.items():
        for cik in ciks:
            path = cache_dir / f"{cik}.json"
            old_path = cache_dir / f"{cik}_older.json"
            if refresh or not path.exists():
                data = _get_json(f"{BASE}CIK{cik}.json", session)
                path.write_text(json.dumps(data["filings"]["recent"]))
                if not old_path.exists():
                    older = []
                    for f in data["filings"].get("files", []):
                        older += _rows(_get_json(BASE + f["name"], session), ticker, cik)
                    old_path.write_text(json.dumps(older))
            rows += _rows(json.loads(path.read_text()), ticker, cik)
            rows += json.loads(old_path.read_text()) if old_path.exists() else []
    return earnings_only(pd.DataFrame(rows, columns=COLUMNS), since)


def earnings_only(filings: pd.DataFrame, since: str = "2014-01-01") -> pd.DataFrame:
    keep = [is_earnings_filing(r.ticker, r.form, r.items, r.filing_date) for r in filings.itertuples()]
    out = filings[keep].drop_duplicates("accession")
    out = out[out["filing_date"] >= since]
    return out.sort_values(["ticker", "accepted"]).reset_index(drop=True)


def gap_report(events: pd.DataFrame, max_days: int = 120) -> list[str]:
    """Names whose gap between earnings filings exceeds max_days (how the mistagged ones were found).

    Known and expected: MRVL in 2016 (audit review, no 2.02 filing).
    """
    out = []
    for tk, g in events.groupby("ticker"):
        d = pd.to_datetime(g["filing_date"]).sort_values()
        gaps = d.diff().dt.days
        for day, gap in gaps[gaps > max_days].items():
            out.append(f"{tk}: {int(gap)} days without an earnings 8-K before {d[day].date()}")
    return out


def timing_report(events: pd.DataFrame) -> dict:
    """Share of filings after the close, before the open and during the session (New York time)."""
    from wikiquant.earnings_cycle import parse_acceptance
    ny = events["accepted"].map(lambda a: parse_acceptance(a).tz_convert("America/New_York"))
    minutes = ny.map(lambda t: t.hour * 60 + t.minute)
    return {
        "after_close": float((minutes >= 16 * 60).mean()),
        "before_open": float((minutes < 9 * 60 + 30).mean()),
        "during_session": float(((minutes >= 9 * 60 + 30) & (minutes < 16 * 60)).mean()),
        "n": int(len(events)),
    }
