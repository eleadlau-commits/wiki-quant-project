"""Performance metrics. YOUR WEEK 1 EXERCISE.

Each function below raises NotImplementedError. Replace the body with your own
code until `pytest tests/test_metrics.py` passes. Every one fits in 1 to 5
lines of pandas. Try it yourself before asking AI: you'll be asked to explain
these in interviews.

Conventions used throughout the project:
- `prices` is a DataFrame (dates x tickers) or a Series.
- `returns` are simple returns: 0.01 means +1% that period.
- Daily data has 252 trading days per year.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Period-on-period returns: price_t / price_{t-1} - 1, first row dropped.

    Hint: look up DataFrame.pct_change, and remember to drop the NaN first row.
    """
    raise NotImplementedError


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

    Grow 1 by every return, take the final value, then raise it to the power
    (periods_per_year / number_of_periods) and subtract 1.
    Hint: (1 + returns).prod()
    """
    raise NotImplementedError


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of returns, scaled to a year.

    Volatility grows with the square root of time, so multiply the per-period
    standard deviation by sqrt(periods_per_year).
    """
    raise NotImplementedError


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Return per unit of risk, assuming a 0% risk-free rate.

    Use the arithmetic mean here: mean(returns) * periods_per_year divided by
    annualised_volatility(returns). Roughly: above 1 is good, above 2 is
    suspicious in a backtest (check for bugs).
    """
    raise NotImplementedError


def max_drawdown(returns: pd.Series) -> float:
    """Worst fall from a previous peak, as a negative number (-0.25 = -25%).

    Steps: build the equity curve with (1 + returns).cumprod(), track its
    running peak with .cummax(), compute equity / peak - 1, take the minimum.
    Subtle bit: the starting value of 1 counts as a peak too, so a loss on
    day one is still a drawdown.
    """
    raise NotImplementedError
