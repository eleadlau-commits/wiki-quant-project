"""Small hand-checkable cases for the broad backtest: costs, timing, delistings, luck test."""

import numpy as np
import pandas as pd
import pytest

from wikiquant import backtest as B


def _prices(rets: dict[str, list[float]], start="2024-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(next(iter(rets.values()))) + 1)
    return pd.DataFrame({t: 100 * np.cumprod(1 + np.array([0.0] + list(r))) for t, r in rets.items()}, index=idx)


def test_picks_earn_from_the_next_day_only():
    # A jumps 50% on the rebalance day itself: a portfolio picked at that close must not get it.
    px = _prices({"A": [0.5, 0.0, 0.0], "B": [0.0, 0.1, 0.1]})
    scores = pd.DataFrame({"A": [2.0], "B": [1.0]}, index=[px.index[1]])
    bt = B.run_portfolio(scores, px, "top", frac=0.5, cost_bps=0, min_names=2)
    assert list(bt.gross.index) == list(px.index[2:])
    assert bt.gross.tolist() == pytest.approx([0.0, 0.0])


def test_costs_are_charged_on_both_legs_of_each_trade():
    px = _prices({"A": [0.0] * 6, "B": [0.0] * 6})
    idx = px.index
    # Week 1 hold A, week 2 switch to B.
    scores = pd.DataFrame({"A": [2.0, 1.0], "B": [1.0, 2.0]}, index=[idx[0], idx[3]])
    bt = B.run_portfolio(scores, px, "top", frac=0.5, cost_bps=10, min_names=2)
    assert bt.turnover.tolist() == pytest.approx([0.5, 1.0])     # buy in from cash (one leg), then full switch
    # Entering costs 10 bps (buy only); switching costs 20 bps (sell A, buy B).
    assert bt.net.loc[idx[1]] == pytest.approx(-0.0010)
    assert bt.net.loc[idx[4]] == pytest.approx(-0.0020)
    assert bt.net.sum() == pytest.approx(-0.0030)


def test_delisted_stock_counts_as_cash_until_next_rebalance():
    px = _prices({"A": [0.1, 0.0, 0.0, 0.0], "B": [0.1, 0.1, 0.1, 0.1]})
    px.iloc[2:, 0] = np.nan                       # A stops trading after day 1
    scores = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=[px.index[0]])
    bt = B.run_portfolio(scores, px, "all", cost_bps=0, min_names=2)
    assert np.isfinite(bt.gross).all()
    assert bt.gross.iloc[0] == pytest.approx(0.1)
    # Afterwards only B's share of the money moves.
    w_b = 0.5 * 1.1 / (0.5 * 1.1 + 0.5 * 1.1)
    assert bt.gross.iloc[1] == pytest.approx(w_b * 0.1)


def test_shuffled_scores_are_reproducible_and_change_picks():
    rng = np.random.default_rng(0)
    n = 40
    px = _prices({f"S{i}": list(rng.normal(0, 0.01, 30)) for i in range(n)})
    scores = pd.DataFrame(rng.normal(size=(3, n)), index=px.index[[0, 10, 20]], columns=px.columns)
    a = B.placebo(scores, px, B.run_portfolio(scores, px, "all").gross, runs=5, seed=1)
    b = B.placebo(scores, px, B.run_portfolio(scores, px, "all").gross, runs=5, seed=1)
    assert np.allclose(a, b)
    assert len(set(np.round(a, 10))) > 1


def test_breakeven_cost():
    # 2% a year of extra return, replacing the portfolio 10 more times a year than the benchmark:
    # 10 replacements x 2 legs x c = 2%  ->  c = 10 bps.
    assert B.breakeven_cost_bps(0.02, 10) == pytest.approx(10)


def test_month_clustered_t_counts_each_month_once():
    ev = pd.DataFrame({"entry_date": pd.to_datetime(["2020-03-02"] * 50 + ["2020-04-01", "2020-05-01"]),
                       "excess": [0.05] * 50 + [-0.01, 0.0]})
    s = B.clustered_summary(ev)
    assert s["n"] == 52 and s["months"] == 3
    assert abs(s["t_month"]) < abs(s["t_naive"])
    net = B.clustered_summary(ev, cost_bps=10)
    assert net["mean"] == pytest.approx(s["mean"] - 0.002)
