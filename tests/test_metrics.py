"""Tests for your metric functions. They fail until you fill in metrics.py.

Run with:  pytest tests/test_metrics.py -v
"""

import math

import pandas as pd
import pytest

from wikiquant.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
)


def test_simple_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = simple_returns(prices)
    assert list(result.round(6)) == [0.10, -0.10]


def test_simple_returns_works_on_dataframes():
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 25.0]})
    result = simple_returns(prices)
    assert result.shape == (1, 2)
    assert result["B"].iloc[0] == pytest.approx(-0.5)


def test_annualised_return_one_year_of_10_percent():
    # 12 monthly returns that compound to exactly +10% over the year.
    monthly = 1.10 ** (1 / 12) - 1
    returns = pd.Series([monthly] * 12)
    assert annualised_return(returns, periods_per_year=12) == pytest.approx(0.10)


def test_annualised_return_two_years():
    # +21% over two years is +10% a year, compounded.
    returns = pd.Series([0.21])
    assert annualised_return(returns, periods_per_year=0.5) == pytest.approx(0.10)


def test_annualised_volatility_scales_with_sqrt_time():
    returns = pd.Series([0.01, -0.01] * 50)
    daily_std = returns.std()
    assert annualised_volatility(returns) == pytest.approx(daily_std * math.sqrt(252))


def test_sharpe_ratio():
    returns = pd.Series([0.02, 0.00, 0.01, 0.03, -0.01])
    expected = returns.mean() * 252 / (returns.std() * math.sqrt(252))
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_max_drawdown():
    # Equity: 1 -> 1.5 -> 0.75 -> 1.5. Worst fall is 1.5 to 0.75 = -50%.
    returns = pd.Series([0.5, -0.5, 1.0])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_max_drawdown_counts_a_first_day_loss():
    returns = pd.Series([-0.2, 0.1])
    assert max_drawdown(returns) == pytest.approx(-0.2)


def test_max_drawdown_is_zero_when_only_rising():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert max_drawdown(returns) == pytest.approx(0.0)
