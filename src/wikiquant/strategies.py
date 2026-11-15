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
