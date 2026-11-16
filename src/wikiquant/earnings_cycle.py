"""Pooled earnings-cycle overlay: hold a stock only just after earnings and late in its cycle.

The rule (frozen, do not tune; full write-up in docs/earnings-cycle.md):

* Universe: 23 US chip and large-tech stocks, fixed in advance (UNIVERSE).
* Earnings event: every SEC Form 8-K with Item 2.02, plus three filings that
  were tagged Item 2.01 by mistake (MISTAGGED).
* Reaction day: the first NYSE session whose close comes after the filing's
  acceptance time (read as true UTC). After-close filings map to the next day.
* dse(t) = sessions since the most recent reaction day <= t (reaction day = 0).
* Bucket map: dse 0-2 long, 3-45 cash, 46+ long.
* Weight: 1/n for each long name, n = names with a price today and yesterday.
  The rest sits in cash; long names are never scaled up.
* Timing: weights from the close of t, trade at the close of t+1, first return
  on t+2. So net return on day s = sum w(s-2) r(s) - cost * sum |w(s-2) - w(s-3)|.

Everything here is pure functions on DataFrames, so it can be tested offline.
Downloading lives in wikiquant.edgar (filings) and scripts/earnings_bot.py (prices).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Ticker -> SEC CIK(s). Several CIKs mean the company changed legal entity;
# its earnings history is stitched across all of them.
# META is called META here (the plan's files call it FB).
UNIVERSE: dict[str, tuple[str, ...]] = {
    "AAPL": ("0000320193",),
    "ADI": ("0000006281",),
    "AMAT": ("0000006951",),
    "AMZN": ("0001018724",),
    "AVGO": ("0001441634", "0001649338", "0001730168"),
    "CSCO": ("0000858877",),
    "META": ("0001326801",),
    "GOOGL": ("0001288776", "0001652044"),
    "IBM": ("0000051143",),
    "INTC": ("0000050863",),
    "KLAC": ("0000319201",),
    "LRCX": ("0000707549",),
    "MCHP": ("0000827054",),
    "MRVL": ("0001058057", "0001835632"),
    "MSFT": ("0000789019",),
    "MU": ("0000723125",),
    "NVDA": ("0001045810",),
    "ON": ("0001097864",),
    "ORCL": ("0001341439",),
    "QCOM": ("0000804328",),
    "SWKS": ("0000004127",),
    "TER": ("0000097210",),
    "TXN": ("0000097476",),
}
TICKERS = sorted(UNIVERSE)

# Earnings 8-Ks filed under Item 2.01 by mistake: (ticker, filing date).
MISTAGGED = {("AMAT", "2024-05-16"), ("LRCX", "2016-04-20"), ("SWKS", "2015-11-05")}

# Frozen bucket map: edges of the dse buckets and whether each one is long.
BUCKET_EDGES = (0, 3, 21, 46)      # buckets [0,2], [3,20], [21,45], [46, inf)
BUCKET_LONG = (1, 0, 0, 1)
COST_PER_SIDE = 0.0005             # 5 bps on |change in weight|


@lru_cache(maxsize=1)
def nyse():
    import exchange_calendars as xc
    return xc.get_calendar("XNYS", start="2005-01-01")


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    return nyse().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


# --- Earnings events and reaction days --------------------------------------

def is_earnings_filing(ticker: str, form: str, items: str, filing_date: str) -> bool:
    """True for an 8-K carrying Item 2.02, or one of the three known mistagged filings."""
    if form != "8-K":
        return False
    codes = {c.strip() for c in str(items or "").split(",")}
    if "2.02" in codes:
        return True
    return "2.01" in codes and (ticker, str(filing_date)[:10]) in MISTAGGED


def parse_acceptance(value: str) -> pd.Timestamp:
    """SEC's acceptanceDateTime, e.g. '2026-11-12T21:06:13.000Z', read as TRUE UTC.

    (Checked on NVDA and AMD: read as UTC, ~95% of releases land just after the
    16:00 New York close. Read as Eastern time almost none do.) Do not
    auto-detect: that breaks for companies that report before the open.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def reaction_day(accepted_utc: pd.Timestamp) -> pd.Timestamp:
    """First NYSE session whose close is strictly after the acceptance time.

    16:05 ET on a Tuesday -> Wednesday. 08:00 ET -> same day. Saturday -> Monday.
    13:05 ET on a 13:00 half-day -> the next session.
    """
    closes = nyse().closes  # Series: session -> close time (UTC)
    close_utc = closes.values.astype("datetime64[ns]")  # tz-naive UTC
    t = accepted_utc.tz_convert("UTC").tz_localize(None).to_datetime64()
    i = int(np.searchsorted(close_utc, t, side="right"))
    return closes.index[i]


def reaction_days(events: pd.DataFrame) -> pd.DataFrame:
    """Add a `reaction_day` column to events with columns ticker, accepted (UTC strings)."""
    out = events.copy()
    out["accepted_utc"] = out["accepted"].map(parse_acceptance)
    out["reaction_day"] = out["accepted_utc"].map(reaction_day)
    return out


# --- Sessions since earnings, buckets, weights ------------------------------

def sessions_since_earnings(events: pd.DataFrame, sessions: pd.DatetimeIndex,
                            tickers=TICKERS) -> pd.DataFrame:
    """dse(t) for each ticker and session (NaN before the first reaction day).

    `events` needs columns ticker and reaction_day. `sessions` must be NYSE
    sessions; counting is in sessions, so weekends and holidays don't count.
    """
    all_sessions = nyse().sessions
    pos = pd.Series(np.arange(len(all_sessions)), index=all_sessions)
    t_idx = pos.reindex(sessions).to_numpy()
    out = {}
    for tk in tickers:
        days = events.loc[events["ticker"] == tk, "reaction_day"]
        r_idx = np.unique(pos.reindex(pd.DatetimeIndex(days)).dropna().to_numpy())
        k = np.searchsorted(r_idx, t_idx, side="right") - 1   # latest reaction day <= t
        dse = np.where(k >= 0, t_idx - r_idx[np.clip(k, 0, None)] if len(r_idx) else np.nan, np.nan)
        out[tk] = dse.astype(float)
    return pd.DataFrame(out, index=sessions)


def bucket_long(dse: pd.DataFrame) -> pd.DataFrame:
    """1.0 where the frozen map says long, 0.0 where cash, NaN where dse is undefined."""
    b = np.searchsorted(np.array(BUCKET_EDGES), dse.to_numpy(), side="right") - 1
    long_ = np.array(BUCKET_LONG, dtype=float)[np.clip(b, 0, len(BUCKET_LONG) - 1)]
    return pd.DataFrame(np.where(dse.isna(), np.nan, long_), index=dse.index, columns=dse.columns)


def live_mask(prices: pd.DataFrame) -> pd.DataFrame:
    """A name is live on t if it has a valid adjusted close on t and the session before."""
    ok = prices.notna() & (prices > 0)
    return ok & ok.shift(1, fill_value=False)


def target_weights(dse: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """1/n for each live, long name; 0 otherwise. No renormalising: the rest is cash."""
    live = live_mask(prices.reindex(index=dse.index, columns=dse.columns))
    n = live.sum(axis=1).replace(0, np.nan)
    long_ = bucket_long(dse).fillna(0.0).astype(bool)
    w = (long_ & live).astype(float).div(n, axis=0)
    return w.fillna(0.0)


# --- Backtest ---------------------------------------------------------------

def backtest(weights: pd.DataFrame, prices: pd.DataFrame, cost: float = COST_PER_SIDE) -> pd.DataFrame:
    """Daily gross/net returns of the rule, plus an equal-weight benchmark.

    net(s) = sum w(s-2) r(s) - cost * sum |w(s-2) - w(s-3)|
    (signal at close of t, trade at close of t+1, first return on t+2).
    The benchmark holds 1/n of every live name, rebalanced daily, no costs.
    """
    rets = prices.pct_change(fill_method=None)
    w_held = weights.shift(2).reindex(rets.index).fillna(0.0)
    gross = (w_held * rets.fillna(0.0)).sum(axis=1)
    turnover = w_held.diff().abs().sum(axis=1).fillna(w_held.abs().sum(axis=1))
    net = gross - cost * turnover
    bench = rets.where(live_mask(prices)).mean(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": turnover, "net": net,
                         "invested": w_held.sum(axis=1), "benchmark": bench.fillna(0.0)})


def summary(returns: pd.Series) -> dict:
    """Sharpe (daily, x sqrt 252, no risk-free), total return and max drawdown."""
    r = returns.dropna()
    nav = (1 + r).cumprod()
    return {
        "days": int(len(r)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "total_return": float(nav.iloc[-1] - 1) if len(r) else 0.0,
        "max_drawdown": float((nav / nav.cummax() - 1).min()) if len(r) else 0.0,
    }


# --- Price checks -----------------------------------------------------------

def price_problems(prices: pd.DataFrame, signal_date: pd.Timestamp) -> list[str]:
    """The evening checks from the plan. An empty list means the data is usable for signal_date."""
    problems = []
    if prices.empty or prices.index.max() < signal_date:
        problems.append(f"latest price date {prices.index.max() if not prices.empty else None} "
                        f"is before the signal date {signal_date.date()}")
        return problems
    window = prices.loc[:signal_date].tail(260)
    rets = window.pct_change(fill_method=None)
    crashes = rets[rets < -0.5].stack()
    for (day, tk), r in crashes.items():
        problems.append(f"{tk} fell {r:.0%} on {day.date()}: an unadjusted split?")
    expected = nyse_sessions(window.index.min(), signal_date)
    for tk in window.columns:
        s = window[tk]
        first = s.first_valid_index()
        if first is None:
            problems.append(f"{tk}: no prices at all")
            continue
        missing = expected[(expected >= first)].difference(s.dropna().index)
        if len(missing):
            problems.append(f"{tk}: {len(missing)} missing session(s), e.g. {missing[0].date()}")
    return problems


# --- Hash-chained signal log ------------------------------------------------

def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def record_hash(prev: str, body: dict) -> str:
    """hash = SHA-256(previous hash + JSON of the body with sorted keys)."""
    return sha256_text(prev + json.dumps(body, sort_keys=True))


def read_log(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_record(path: Path | str, body: dict) -> dict:
    """Append one record to the append-only log, chained to the one before."""
    path = Path(path)
    records = read_log(path)
    verify_chain(records)
    prev = records[-1]["hash"] if records else ""
    body = {k: v for k, v in body.items() if k not in ("prev", "hash")}
    record = {**body, "prev": prev, "hash": record_hash(prev, body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def verify_chain(records: list[dict]) -> None:
    """Raise ValueError if any record was edited, removed or reordered."""
    prev = ""
    for i, rec in enumerate(records):
        body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
        if rec.get("prev") != prev or rec.get("hash") != record_hash(prev, body):
            raise ValueError(f"signal log broken at record {i} ({rec.get('signal_date')})")
        prev = rec["hash"]


@dataclass
class Signal:
    signal_date: str
    weights: dict[str, float]
    n_names: int
    invested: float
    dse: dict[str, float | None]
    stale: bool = False
    problems: tuple[str, ...] = ()

    def record_body(self, **hashes: str) -> dict:
        return {
            "signal_date": self.signal_date, "trade_at": "next session close",
            "n_names": self.n_names, "invested": round(self.invested, 6),
            "weights": {k: round(v, 6) for k, v in sorted(self.weights.items())},
            "dse": self.dse, "stale": self.stale, "problems": list(self.problems),
            "ticker_note": "META (FB in the reference files)",
            **hashes, "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def signal_for(day, events: pd.DataFrame, prices: pd.DataFrame) -> Signal:
    """Target weights for one signal day, from reaction days and prices up to that day."""
    day = pd.Timestamp(day)
    prices = prices.loc[:day]
    sessions = nyse_sessions(prices.index.min(), day)
    dse = sessions_since_earnings(events[events["reaction_day"] <= day], sessions)
    w = target_weights(dse, prices.reindex(sessions)).iloc[-1]
    n = int(live_mask(prices.reindex(sessions)).iloc[-1].sum())
    last = dse.iloc[-1]
    return Signal(str(day.date()), {k: float(v) for k, v in w.items()}, n, float(w.sum()),
                  {k: (None if pd.isna(v) else int(v)) for k, v in last.items()})
