"""Tests for the earnings-cycle rule and its trader. Offline: no SEC, no Yahoo, no orders."""

import json

import numpy as np
import pandas as pd
import pytest

from wikiquant import earnings_cycle as ec
from wikiquant.earnings_trader import SafetyStop, execute, plan_rebalance


# --- Reaction days ------------------------------------------------------------

@pytest.mark.parametrize("accepted, expected", [
    ("2026-11-12T21:06:13.000Z", "2026-11-13"),   # 16:06 ET Thursday -> Friday
    ("2026-11-12T20:59:00.000Z", "2026-11-12"),   # 15:59 ET -> same day
    ("2026-11-12T12:30:00.000Z", "2026-11-12"),   # 07:30 ET, before the open -> same day
    ("2026-11-14T15:00:00.000Z", "2026-11-16"),   # Saturday -> Monday
    ("2024-11-29T18:05:00.000Z", "2024-12-02"),   # 13:05 ET on a half-day -> next session
    ("2024-11-29T17:55:00.000Z", "2024-11-29"),   # 12:55 ET on a half-day -> same day
    ("2024-07-03T20:10:00.000Z", "2024-07-05"),   # after the close before 4 July
])
def test_reaction_day(accepted, expected):
    assert ec.reaction_day(ec.parse_acceptance(accepted)) == pd.Timestamp(expected)


def test_acceptance_is_read_as_utc_not_eastern():
    # 21:06Z is 16:06 in New York (after the close). Read as Eastern it would be mid-session.
    t = ec.parse_acceptance("2026-11-12T21:06:13.000Z").tz_convert("America/New_York")
    assert (t.hour, t.minute) == (16, 6)


def test_earnings_filter():
    assert ec.is_earnings_filing("AAPL", "8-K", "2.02,9.01", "2026-10-30")
    assert not ec.is_earnings_filing("AAPL", "8-K", "5.02,9.01", "2026-10-30")
    assert not ec.is_earnings_filing("AAPL", "10-Q", "", "2026-10-30")
    assert not ec.is_earnings_filing("AAPL", "8-K/A", "2.02", "2026-10-30")
    assert ec.is_earnings_filing("AMAT", "8-K", "2.01,9.01", "2024-05-16")      # known mistag
    assert not ec.is_earnings_filing("AMAT", "8-K", "2.01,9.01", "2024-05-17")


# --- dse, buckets, weights ------------------------------------------------------

def events(**days):
    return pd.DataFrame([{"ticker": t, "reaction_day": pd.Timestamp(d)} for t, ds in days.items() for d in ds])


def test_sessions_since_earnings_counts_sessions_not_days():
    sessions = ec.nyse_sessions("2026-11-10", "2026-11-17")   # Tue..Tue, weekend in between
    dse = ec.sessions_since_earnings(events(AAA=["2026-11-12"]), sessions, tickers=["AAA", "BBB"])
    assert dse["AAA"].tolist()[:2] == [None, None] or dse["AAA"].iloc[:2].isna().all()
    assert dse.loc["2026-11-12", "AAA"] == 0
    assert dse.loc["2026-11-13", "AAA"] == 1
    assert dse.loc["2026-11-16", "AAA"] == 2      # Monday: the weekend doesn't count
    assert dse.loc["2026-11-17", "AAA"] == 3
    assert dse["BBB"].isna().all()                # no earnings yet -> undefined


def test_new_filing_resets_the_counter():
    sessions = ec.nyse_sessions("2026-01-02", "2026-03-31")
    dse = ec.sessions_since_earnings(events(AAA=["2026-01-05", "2026-02-02"]), sessions, tickers=["AAA"])
    assert dse.loc["2026-01-30", "AAA"] > 15
    assert dse.loc["2026-02-02", "AAA"] == 0


def test_bucket_map_is_frozen():
    dse = pd.DataFrame({"x": [0, 1, 2, 3, 20, 21, 45, 46, 200, np.nan]})
    assert ec.bucket_long(dse)["x"].tolist()[:9] == [1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert np.isnan(ec.bucket_long(dse)["x"].iloc[-1])


def test_weights_are_one_over_n_and_not_renormalised():
    idx = ec.nyse_sessions("2026-11-02", "2026-11-04")
    dse = pd.DataFrame({"A": [0, 1, 2], "B": [10, 11, 12], "C": [50, 51, 52], "D": [np.nan] * 3}, index=idx)
    prices = pd.DataFrame(100.0, index=idx, columns=list("ABCD"))
    w = ec.target_weights(dse, prices)
    assert w.iloc[-1].to_dict() == pytest.approx({"A": 0.25, "B": 0.0, "C": 0.25, "D": 0.0})
    assert w.iloc[-1].sum() == pytest.approx(0.5)       # the rest stays in cash


def test_a_name_without_yesterdays_price_is_not_live():
    idx = ec.nyse_sessions("2026-11-02", "2026-11-04")
    dse = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2]}, index=idx)
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, np.nan, 1.0]}, index=idx)
    w = ec.target_weights(dse, prices)
    assert w.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0}   # n = 1


def test_after_close_filing_does_not_change_that_days_signal():
    # Filed 16:05 ET on 2026-11-12: reaction day 11-13, so the signal for 11-12 ignores it.
    ev = ec.reaction_days(pd.DataFrame({"ticker": ["AAPL"], "accepted": ["2026-11-12T21:05:00.000Z"]}))
    idx = ec.nyse_sessions("2026-10-01", "2026-11-13")
    prices = pd.DataFrame(100.0, index=idx, columns=ec.TICKERS)
    assert ec.signal_for("2026-11-12", ev, prices).weights["AAPL"] == 0.0
    assert ec.signal_for("2026-11-13", ev, prices).weights["AAPL"] == pytest.approx(1 / 23)


# --- Backtest timing -------------------------------------------------------------

def test_backtest_uses_weights_from_two_sessions_earlier():
    idx = ec.nyse_sessions("2026-11-02", "2026-11-06")
    prices = pd.DataFrame({"A": [100, 100, 100, 110, 121.0]}, index=idx)
    w = pd.DataFrame({"A": [0, 1.0, 1.0, 1.0, 1.0]}, index=idx)   # signal on day 1
    bt = ec.backtest(w, prices, cost=0.0005)
    assert bt["gross"].tolist() == pytest.approx([0, 0, 0, 0.10, 0.10])  # first return on day 3 = t+2
    assert bt.loc[idx[3], "net"] == pytest.approx(0.10 - 0.0005)       # entry cost paid with it


# --- Hash chain -------------------------------------------------------------------

def test_hash_chain_detects_edits(tmp_path):
    log = tmp_path / "signals.jsonl"
    ec.append_record(log, {"signal_date": "2026-11-13", "weights": {"A": 0.5}})
    ec.append_record(log, {"signal_date": "2026-11-16", "weights": {"A": 0.0}})
    ec.verify_chain(ec.read_log(log))
    lines = log.read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["prev"] == "" and rec["hash"] == ec.record_hash("", {"signal_date": "2026-11-13", "weights": {"A": 0.5}})
    rec["weights"]["A"] = 0.9
    log.write_text(json.dumps(rec) + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError):
        ec.verify_chain(ec.read_log(log))
    with pytest.raises(ValueError):
        ec.append_record(log, {"signal_date": "2026-11-17"})   # refuses to extend a broken log


# --- Trader -------------------------------------------------------------------------

W = {"AVGO": 1 / 23, "NVDA": 1 / 23, "AAPL": 0.0, "MSFT": 1 / 23}


def test_rebalance_buys_slots_and_sells_out():
    orders = {o.ticker: o for o in plan_rebalance(
        W, 23, 50, holdings={"AAPL": 0.01, "MSFT": 0.0}, prices={"AVGO": 200, "NVDA": 150, "AAPL": 180, "MSFT": 300},
        skip={"NVDA"})}
    assert orders["AVGO"].action == "buy" and orders["AVGO"].order_value == pytest.approx(50 / 23, abs=0.02)
    assert orders["AAPL"].action == "sell" and orders["AAPL"].order_qty == -0.01   # sell out exactly
    assert orders["NVDA"].action == "skip (yours)" and orders["NVDA"].order_qty == 0


def test_nvda_is_never_sold_even_if_held():
    orders = plan_rebalance({"NVDA": 0.0}, 23, 50, {"NVDA": 30.0}, {"NVDA": 150}, skip={"NVDA"})
    assert orders[0].action == "skip (yours)" and orders[0].order_qty == 0


def test_sells_come_before_buys_and_tiny_top_ups_are_skipped():
    orders = plan_rebalance({"AAPL": 0.0, "AVGO": 1 / 23, "MSFT": 1 / 23}, 23, 50,
                            {"AAPL": 0.01, "MSFT": 2.0 / 300}, {"AAPL": 180, "AVGO": 200, "MSFT": 300})
    assert orders[0].ticker == "AAPL" and orders[0].action == "sell"
    assert {o.ticker: o.action for o in orders}["MSFT"] == "skip (below minimum)"


def test_stops_if_account_holds_far_more_than_the_bot_could_have_bought():
    with pytest.raises(SafetyStop):
        plan_rebalance({"AAPL": 0.0}, 23, 50, {"AAPL": 1.0}, {"AAPL": 180})


def test_order_cap():
    with pytest.raises(SafetyStop):
        plan_rebalance({"AAPL": 1 / 23}, 23, 1000, {}, {"AAPL": 180}, max_order=10)


def test_execute_skips_buys_without_cash_and_keeps_going_after_an_error():
    class Broker:
        sent = []

        def market_order(self, ticker, qty):
            if ticker.startswith("AMZN"):
                raise RuntimeError("rejected")
            self.sent.append((ticker, qty))
            return {"status": "FILLED"}

    orders = plan_rebalance({"AVGO": 1 / 23, "MSFT": 1 / 23, "AMZN": 1 / 23}, 23, 50, {},
                            {"AVGO": 200, "MSFT": 300, "AMZN": 150})
    execute(orders, Broker(), free_cash=3.0)
    by = {o.ticker: o for o in orders}
    assert by["AMZN"].reply["status"] == "ERROR"      # failed, so its cash is still free
    assert by["AVGO"].reply["status"] == "FILLED"
    assert by["MSFT"].action == "skip (not enough cash)"


# --- Price checks -------------------------------------------------------------------

def test_price_checks_pass_on_clean_data_and_catch_problems():
    idx = ec.nyse_sessions("2026-10-01", "2026-11-13")
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    assert ec.price_problems(prices, pd.Timestamp("2026-11-13")) == []
    bad = prices.copy()
    bad.loc["2026-11-10":, "A"] = 10.0             # a 10:1 split left unadjusted
    bad = bad.drop(pd.Timestamp("2026-11-05"))     # a missing session
    problems = ec.price_problems(bad, pd.Timestamp("2026-11-13"))
    assert any("A fell" in p for p in problems) and any("missing" in p for p in problems)
    assert ec.price_problems(prices, pd.Timestamp("2026-11-16"))   # data not in yet
