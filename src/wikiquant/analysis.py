"""Source-agnostic tests: does attention from *any* source predict returns?

Every function takes plain pandas objects:

- `attention`: DataFrame, one row per calendar day, one column per ticker
  (whatever the source measures: views, stories, searches...).
- `prices`: DataFrame, one row per trading day, one column per ticker.
- `bench`: Series of benchmark prices (SPY) on the same trading days.

So a new data source never needs new analysis code. These functions mirror
the maths in the dashboard (dashboard/template.html), so the page and the
written report agree.

The one rule that keeps it honest: attention dated day D is only acted on at
the close of the first trading day *strictly after* D (`first_trade_after`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 60      # days of history that define "normal" attention
COOLDOWN = 10    # after a spike, ignore further spikes for this many trading days
HOLD = 20        # trading days we hold after a spike
TOP_K = 4        # stocks in the Buzz and Quiet portfolios


# ---------------------------------------------------------------- attention -> signal
def spike_ratio(series: pd.Series | pd.DataFrame, window: int = WINDOW, smooth: int = 1):
    """Attention divided by its usual level (median of the previous `window` days).

    1.0 is a normal day, 3.0 is three times the usual attention. With
    smooth > 1 we first add up the last `smooth` days, which steadies sparse
    counts (2 stories vs a usual 1 is not a real "2x spike"). Only past days
    are used, so there is no look-ahead. A usual level of 0 gives NaN.
    """
    s = series.astype(float)
    if smooth > 1:
        s = s.rolling(smooth, min_periods=smooth).sum()
    baseline = s.shift(1).rolling(window, min_periods=window // 2).median()
    return s / baseline.where(baseline > 0)


def first_trade_after(calendar_dates: pd.DatetimeIndex, trading_dates: pd.DatetimeIndex) -> np.ndarray:
    """Position in `trading_dates` of the first trading day strictly after each date."""
    return np.searchsorted(trading_dates.values, pd.DatetimeIndex(calendar_dates).values, side="right")


def find_spikes(ratio: pd.Series, trading_dates: pd.DatetimeIndex,
                threshold: float = 3.0, cooldown: int = COOLDOWN) -> pd.DataFrame:
    """Days where the ratio reaches `threshold`, mapped to the day we could trade.

    Columns: attention_date, entry (position in trading_dates), entry_date, ratio.
    """
    hot = ratio[ratio >= threshold].dropna()
    entries = first_trade_after(hot.index, trading_dates)
    rows, last = [], -10**9
    for day, r, e in zip(hot.index, hot.values, entries):
        if e >= len(trading_dates) or e - last < cooldown:
            continue
        last = e
        rows.append((day, int(e), trading_dates[e], float(r)))
    return pd.DataFrame(rows, columns=["attention_date", "entry", "entry_date", "ratio"])


# ---------------------------------------------------------------- event study
def _excess(p: np.ndarray, b: np.ndarray, i: int, j: int) -> float:
    if i < 0 or j >= len(p) or not (p[i] > 0 and p[j] > 0):
        return np.nan
    return (p[j] / p[i] - 1) - (b[j] / b[i] - 1)


def event_study(attention: pd.DataFrame, prices: pd.DataFrame, bench: pd.Series,
                threshold: float = 3.0, smooth: int = 1, hold: int = HOLD,
                window: int = WINDOW, cooldown: int = COOLDOWN) -> pd.DataFrame:
    """One row per spike, for every ticker.

    excess:   stock minus S&P 500 from the close of the entry day to `hold`
              trading days later. This is the return you could have traded.
    reaction: stock minus S&P 500 from the day before entry to the entry day.
              Usually the news itself, already priced in before you can act.
    """
    b = bench.to_numpy(float)
    out = []
    for ticker in attention.columns.intersection(prices.columns):
        ratio = spike_ratio(attention[ticker], window, smooth)
        ev = find_spikes(ratio, prices.index, threshold, cooldown)
        p = prices[ticker].to_numpy(float)
        ev["excess"] = [_excess(p, b, e, e + hold) for e in ev["entry"]]
        ev["reaction"] = [_excess(p, b, e - 1, e) for e in ev["entry"]]
        ev.insert(0, "ticker", ticker)
        out.append(ev)
    cols = ["ticker", "attention_date", "entry", "entry_date", "ratio", "excess", "reaction"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def summarise(x) -> dict:
    """Count, average, share above zero and t-stat (mean / standard error).

    A |t| below about 2 means the average is within the range luck could produce.
    """
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "hit": np.nan, "t": np.nan}
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return {"n": n, "mean": x.mean(), "hit": (x > 0).mean(), "t": t}


def baseline_excess(prices: pd.DataFrame, bench: pd.Series, hold: int = HOLD) -> pd.Series:
    """Average `hold`-day excess return over *all* days: what "normal" looks like."""
    fwd = prices.shift(-hold) / prices - 1
    bfwd = bench.shift(-hold) / bench - 1
    return fwd.sub(bfwd, axis=0).mean()


# ---------------------------------------------------------------- weekly strategy
def rebalance_days(trading_dates: pd.DatetimeIndex) -> list[int]:
    """Positions of the first trading day of each week."""
    d = pd.DatetimeIndex(trading_dates)
    return [i for i in range(1, len(d))
            if d[i].weekday() < d[i - 1].weekday() or (d[i] - d[i - 1]).days > 3]


def weekly_scores(ratios: pd.DataFrame, trading_dates: pd.DatetimeIndex,
                  lookback: int = 7, min_obs: int = 4) -> pd.DataFrame:
    """For each rebalance day: average log attention ratio over the previous week.

    Uses calendar days up to the day *before* the rebalance day only.
    """
    logr = np.log(ratios.where(ratios > 0))
    avg = logr.rolling(lookback, min_periods=min_obs).mean()
    rows = {}
    for i in rebalance_days(trading_dates):
        day_before = trading_dates[i] - pd.Timedelta(days=1)
        if day_before in avg.index:
            rows[trading_dates[i]] = avg.loc[day_before]
    return pd.DataFrame(rows).T.reindex(columns=ratios.columns)


def buzz_portfolios(scores: pd.DataFrame, prices: pd.DataFrame, k: int = TOP_K,
                    exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Daily returns of Buzz (top k by attention), Quiet (bottom k) and equal weight.

    Picks are made at the close of each rebalance day and earn returns from the
    next day on, so the strategy never uses information it would not have had.
    """
    tickers = [t for t in prices.columns if t not in exclude and t in scores.columns]
    rets = prices[tickers].pct_change()
    held = {"buzz": None, "quiet": None}
    out, started = [], False
    for day in prices.index[1:]:
        r = rets.loc[day]
        if started:
            out.append((day, r[held["buzz"]].mean(), r[held["quiet"]].mean(), r.mean()))
        if day in scores.index:
            sc = scores.loc[day, tickers]
            sc = sc[prices.loc[day, tickers].notna()].dropna().sort_values(ascending=False)
            if len(sc) >= 2 * k:
                held = {"buzz": list(sc.index[:k]), "quiet": list(sc.index[-k:])}
                started = True
    return pd.DataFrame(out, columns=["date", "buzz", "quiet", "ew"]).set_index("date")


def annual_edge(daily: pd.DataFrame, a: str = "buzz", b: str = "ew") -> dict:
    """Average yearly gap between two portfolios (mean daily gap x 252) and its t-stat."""
    s = summarise(daily[a] - daily[b])
    return {"edge": s["mean"] * 252, "t": s["t"], "days": s["n"]}


def leave_one_out(scores: pd.DataFrame, prices: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """How much of the Buzz-minus-equal-weight edge disappears without each company."""
    full = annual_edge(buzz_portfolios(scores, prices, k))["edge"]
    rows = []
    for t in scores.columns:
        e = annual_edge(buzz_portfolios(scores, prices, k, exclude=(t,)))["edge"]
        rows.append({"ticker": t, "edge_without": e, "drop": full - e})
    return pd.DataFrame(rows).sort_values("drop", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- comparing sources
@dataclass
class SourceResult:
    name: str
    events: pd.DataFrame
    daily: pd.DataFrame
    loo: pd.DataFrame


def run_source(name: str, attention: pd.DataFrame, prices: pd.DataFrame, bench: pd.Series,
               smooth: int = 1, threshold: float = 3.0) -> SourceResult:
    ev = event_study(attention, prices, bench, threshold=threshold, smooth=smooth)
    ratios = spike_ratio(attention, smooth=smooth)
    scores = weekly_scores(ratios, prices.index)
    daily = buzz_portfolios(scores, prices)
    loo = leave_one_out(scores, prices) if len(daily) else pd.DataFrame()
    return SourceResult(name, ev, daily, loo)


def spike_overlap(a: pd.DataFrame, b: pd.DataFrame, tolerance: int = 2) -> float:
    """Share of spikes in `a` with a spike in `b` for the same stock within
    `tolerance` trading days. High = the two sources see the same news."""
    if a.empty:
        return np.nan
    hits = 0
    for ticker, grp in a.groupby("ticker"):
        other = b.loc[b["ticker"] == ticker, "entry"].to_numpy()
        hits += sum(bool(len(other)) and np.abs(other - e).min() <= tolerance for e in grp["entry"])
    return hits / len(a)


def ratio_correlation(att_a: pd.DataFrame, smooth_a: int, att_b: pd.DataFrame, smooth_b: int) -> float:
    """Average (over stocks) correlation of daily log attention ratios."""
    ra, rb = spike_ratio(att_a, smooth=smooth_a), spike_ratio(att_b, smooth=smooth_b)
    ra, rb = np.log(ra.where(ra > 0)), np.log(rb.where(rb > 0))
    cors = [ra[t].corr(rb[t]) for t in ra.columns.intersection(rb.columns)]
    cors = [c for c in cors if pd.notna(c)]
    return float(np.mean(cors)) if cors else np.nan
