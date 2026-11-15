"""Tests for the trading bot logic and brokers. No internet, no real orders."""

import base64
import json

import pandas as pd
import pytest

from wikiquant.brokers import PaperBroker
from wikiquant.brokers.trading212 import Trading212Broker, Trading212Error
from wikiquant.trader import OrderTooLarge, plan_order, run_once


def trend(up: bool, n: int = 260) -> pd.Series:
    vals = [100 + i if up else 400 - i for i in range(n)]
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=n), dtype=float, name="SPY")


def test_plan_order_worked_example():
    assert plan_order(1000, 50, 0, 1.0, allocation=0.95) == pytest.approx((19.0, 19.0, "buy"))
    assert plan_order(1000, 50, 19, 0.0) == (0.0, -19.0, "sell")


def test_small_orders_are_skipped_but_selling_out_never_is():
    assert plan_order(1000, 50, 18.9, 1.0, allocation=0.95, min_trade_value=20)[2] == "skip (below minimum)"
    assert plan_order(1000, 50, 0.1, 0.0, min_trade_value=20)[2] == "sell"


def test_paper_round_trip(tmp_path):
    prices = {"FUND": 50.0}
    broker = PaperBroker(tmp_path / "acct.json", price_of=lambda t: prices[t], cost_bps=10)
    log = tmp_path / "log.jsonl"

    plan = run_once(broker, trend(up=True), "FUND", dry_run=False, log_path=log)
    assert plan.action == "buy" and plan.target_weight == 1.0
    assert broker.holding("FUND")[0] == pytest.approx(19.0)       # 95% of 1,000 at 50
    assert broker.state["cash"] == pytest.approx(50 - 950 * 0.001)  # minus 10 bps cost

    again = run_once(broker, trend(up=True), "FUND", dry_run=False, log_path=log)
    assert again.action in ("hold", "skip (below minimum)")       # already there: no churn

    prices["FUND"] = 55.0
    out = run_once(broker, trend(up=False), "FUND", dry_run=False, log_path=log)
    assert out.action == "sell" and broker.holding("FUND")[0] == 0.0
    assert len(log.read_text().splitlines()) == 3
    assert json.loads(log.read_text().splitlines()[-1])["action"] == "sell"


def test_dry_run_changes_nothing(tmp_path):
    broker = PaperBroker(tmp_path / "acct.json", price_of=lambda t: 50.0)
    plan = run_once(broker, trend(up=True), "FUND", dry_run=True)
    assert plan.action == "buy" and plan.broker_reply == {}
    assert broker.holding("FUND")[0] == 0.0 and not (tmp_path / "acct.json").exists()


def test_order_cap_stops_before_sending(tmp_path):
    broker = PaperBroker(tmp_path / "acct.json", price_of=lambda t: 50.0)
    with pytest.raises(OrderTooLarge):
        run_once(broker, trend(up=True), "FUND", dry_run=False, max_order_value=500)
    assert broker.holding("FUND")[0] == 0.0


# --- Trading 212, with a fake HTTP session -------------------------------

class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.headers = data, status, {}
        self.ok, self.text = status < 400, json.dumps(data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, positions=(), pending=()):
        self.headers, self.posts = {}, []
        self.positions, self.pending = list(positions), list(pending)

    def get(self, url, params=None, timeout=None):
        if url.endswith("/equity/account/summary"):
            return FakeResponse({"totalValue": 1000.0, "currency": "GBP"})
        if url.endswith("/equity/positions"):
            return FakeResponse(self.positions)
        if url.endswith("/equity/orders"):
            return FakeResponse(self.pending)
        return FakeResponse({"error": "unknown"}, 404)

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        return FakeResponse({"id": 1, "status": "NEW", **json})


def test_trading212_uses_basic_auth_and_demo_by_default():
    s = FakeSession()
    b = Trading212Broker("KEY", "SECRET", session=s, price_fallback=lambda t: 80.0)
    assert b.base.startswith("https://demo.")
    assert s.headers["Authorization"] == "Basic " + base64.b64encode(b"KEY:SECRET").decode()
    assert b.account_value() == 1000.0
    assert b.holding("VUAGl_EQ") == (0.0, 80.0)  # not held: price from fallback


def test_trading212_buy_then_sell_payloads():
    s = FakeSession()
    b = Trading212Broker("K", "S", session=s, price_fallback=lambda t: 100.0)
    run_once(b, trend(up=True), "VUAGl_EQ", dry_run=False)
    assert s.posts[-1][1] == {"ticker": "VUAGl_EQ", "quantity": 9.5, "extendedHours": False}

    s.positions = [{"instrument": {"ticker": "VUAGl_EQ"}, "quantity": 9.5, "currentPrice": 101.0}]
    run_once(b, trend(up=False), "VUAGl_EQ", dry_run=False)
    assert s.posts[-1][1]["quantity"] == -9.5  # sells use a negative quantity


def test_trading212_waits_while_an_order_is_pending():
    s = FakeSession(pending=[{"id": 7, "ticker": "VUAGl_EQ", "status": "NEW"}])
    b = Trading212Broker("K", "S", session=s, price_fallback=lambda t: 100.0)
    plan = run_once(b, trend(up=True), "VUAGl_EQ", dry_run=False)
    assert plan.action.startswith("wait") and s.posts == []
    with pytest.raises(Trading212Error):
        b.market_order("VUAGl_EQ", 1.0)


def test_trading212_refuses_bad_setup():
    with pytest.raises(ValueError):
        Trading212Broker("", "", session=FakeSession())
    with pytest.raises(ValueError):
        Trading212Broker("K", "S", env="real", session=FakeSession())
