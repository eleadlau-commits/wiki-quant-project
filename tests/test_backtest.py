"""Tests for the backtester and the moving-average strategy (no internet needed)."""

import pandas as pd
import pytest

from wikiquant.backtest import count_trades, run_backtest
from wikiquant.strategies import buy_and_hold, sma_crossover


def test_worked_example_from_the_docstring():
    prices = pd.Series([100.0, 102.0, 101.0, 99.0])
    signal = pd.Series([1.0, 1.0, 0.0, 0.0])
    bt = run_backtest(prices, signal, cost_bps=10)
    assert list(bt["position"]) == [1.0, 1.0, 0.0]
    assert bt["net_ret"].iloc[0] == pytest.approx(0.02 - 0.001)
    assert bt["net_ret"].iloc[1] == pytest.approx(101 / 102 - 1)
    assert bt["net_ret"].iloc[2] == pytest.approx(-0.001)


def test_no_look_ahead():
    # A signal that "knows" tomorrow's move must not earn it: the position is lagged.
    prices = pd.Series([100.0, 100.0, 150.0])
    cheat = pd.Series([0.0, 0.0, 1.0])  # only turns on at the close of the big day
    bt = run_backtest(prices, cheat, cost_bps=0)
    assert bt["equity"].iloc[-1] == pytest.approx(1.0)


def test_buy_and_hold_matches_the_asset_without_costs():
    prices = pd.Series([100.0, 110.0, 99.0, 120.0])
    bt = run_backtest(prices, buy_and_hold(prices), cost_bps=0)
    assert bt["equity"].iloc[-1] == pytest.approx(1.2)  # buys at the first close, so earns every move after it


def test_costs_are_charged_on_each_change():
    prices = pd.Series([100.0] * 6)
    flip = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    bt = run_backtest(prices, flip, cost_bps=100)  # 1% per unit traded
    assert bt["cost"].sum() == pytest.approx(0.04)  # in, out, in, out
    assert count_trades(flip) == 3  # changes after the first day


def test_lag_two_waits_an_extra_day():
    prices = pd.Series([100.0, 100.0, 110.0, 121.0])
    signal = pd.Series([1.0, 1.0, 1.0, 1.0])
    one = run_backtest(prices, signal, cost_bps=0, lag=1)["equity"].iloc[-1]
    two = run_backtest(prices, signal, cost_bps=0, lag=2)["equity"].iloc[-1]
    assert one == pytest.approx(1.21)
    assert two == pytest.approx(1.21)  # day 2's +0% is the one skipped
    assert run_backtest(prices, signal, cost_bps=0, lag=3)["equity"].iloc[-1] == pytest.approx(1.1)


def test_dataframe_splits_capital_equally():
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 90.0]})
    bt = run_backtest(prices, buy_and_hold(prices), cost_bps=0)
    assert bt["net_ret"].iloc[0] == pytest.approx(0.0)


def test_sma_crossover_turns_on_in_an_uptrend_and_off_in_a_downtrend():
    up = list(range(100, 130))
    down = list(range(130, 100, -1))
    prices = pd.Series([float(x) for x in up + down])
    pos = sma_crossover(prices, fast=3, slow=10)
    assert (pos.iloc[:9] == 0).all()     # warm-up: not enough history yet
    assert pos.iloc[20] == 1.0           # rising: fast average above slow
    assert pos.iloc[-1] == 0.0           # falling: fast average below slow


def test_sma_crossover_rejects_bad_windows():
    with pytest.raises(ValueError):
        sma_crossover(pd.Series([1.0, 2.0]), fast=200, slow=50)
