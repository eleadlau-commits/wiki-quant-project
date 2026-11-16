"""Classic rule-based strategies. Each returns target positions for run_backtest.

A strategy here is just a function: prices in, positions out (1 = invested,
0 = cash), using only data up to each day's close. The backtester handles the
lag, costs and compounding, so strategies stay short and easy to check.
"""

from __future__ import annotations

import pandas as pd


def sma_crossover(prices: pd.DataFrame | pd.Series, fast: int = 50, slow: int = 200):
    """Moving-average crossover ("golden cross"): invested while the fast average
    is above the slow one, in cash otherwise.

    The idea is trend following: prices that have been rising tend to keep
    rising for a while (momentum), and long falls (2008, 2020, 2022) usually
    start with the price sliding under its long-run average. The rule tries to
    sit out the big falls, at the price of reacting late and sometimes
    whipsawing in and out in choppy markets.

    Until `slow` days of history exist the averages are undefined, and we stay
    in cash. The backtest comparison should start after that warm-up so
    buy-and-hold is measured over the same days.
    """
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float).where(slow_ma.notna(), 0.0)


def buy_and_hold(prices: pd.DataFrame | pd.Series):
    """Always fully invested: the benchmark every strategy must beat."""
    return prices * 0 + 1.0


def vol_target(prices: pd.Series, target: float = 0.15, window: int = 20,
               cap: float = 1.0, band: float = 0.0) -> pd.Series:
    """Volatility targeting: hold less when the market is jumpy.

    Weight = target volatility / recent volatility, capped at `cap` (1 = no
    borrowing). If the market has been moving 30% a year lately and the
    target is 15%, hold 50%. Crashes nearly always begin with a jump in
    volatility, so this shrinks the position *during* a fall instead of
    months later, as a 200-day average does.

    `band` avoids tiny daily trades: the weight only changes when the new
    value is more than `band` away from the current one (e.g. 0.1 = 10 points).
    """
    vol = prices.pct_change().rolling(window).std() * (252 ** 0.5)
    raw = (target / vol).clip(upper=cap).fillna(0.0)
    return apply_band(raw, band) if band > 0 else raw


def big_drop_exit(prices: pd.Series, drop: float = 0.04, days: int = 2,
                  reentry: int = 10) -> pd.Series:
    """Sell after an unusually big fall, buy back once the price recovers.

    Out if the price fell more than `drop` over the last `days` days; back in
    when the price closes above its `reentry`-day average. Ordinary wobbles
    don't trigger it, only sharp drops.
    """
    fall = prices / prices.shift(days) - 1
    avg = prices.rolling(reentry).mean()
    out, invested = [], 1.0
    for f, p, a in zip(fall.to_numpy(), prices.to_numpy(), avg.to_numpy()):
        if invested and f < -drop:
            invested = 0.0
        elif not invested and p > a:
            invested = 1.0
        out.append(invested)
    return pd.Series(out, index=prices.index)


def apply_band(weights: pd.Series, band: float) -> pd.Series:
    """Only move the weight when it drifts more than `band` from the last traded value."""
    out, cur = [], 0.0
    for w in weights.to_numpy():
        if abs(w - cur) > band or (w == 0 and cur != 0):
            cur = w
        out.append(cur)
    return pd.Series(out, index=weights.index)
