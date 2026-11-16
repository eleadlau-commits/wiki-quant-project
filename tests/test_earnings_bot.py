"""End-to-end test of one trade run with a fake broker: no internet, no real orders."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import earnings_bot as eb  # noqa: E402
import wikiquant.brokers.trading212 as t212  # noqa: E402
from wikiquant import earnings_cycle as ec  # noqa: E402


class FakeBroker:
    sent: list = []

    def __init__(self, *a, **k):
        self.name = "trading212-fake"
        FakeBroker.sent = []

    def account_currency(self):
        return "GBP"

    def all_pending_orders(self):
        return []

    def positions(self):
        # Your own NVDA and the VUAG bot's fund must never be touched.
        return {"AAPL_US_EQ": 0.01, "NVDA_US_EQ": 3.0, "VUAGl_EQ": 0.5}

    def free_cash(self):
        raise RuntimeError("unexpected reply shape")

    def market_order(self, ticker, qty):
        FakeBroker.sent.append((ticker, qty))
        return {"status": "FILLED"}


def test_trade_run(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "SIGNALS", tmp_path / "signals.jsonl")
    monkeypatch.setattr(eb, "TRADES", tmp_path / "trades.jsonl")
    monkeypatch.setattr(eb, "LEDGER", tmp_path)
    monkeypatch.setattr(t212, "Trading212Broker", FakeBroker)
    monkeypatch.setattr(eb, "latest_quotes", lambda ts: {t: 100.0 for t in ts})
    monkeypatch.setattr(eb, "in_trade_window", lambda now: (True, pd.Timestamp("2026-11-16")))
    w = {t: (1 / 23 if t in ("ADI", "AMAT", "AVGO", "CSCO", "NVDA") else 0.0) for t in ec.TICKERS}
    ec.append_record(eb.SIGNALS, {"signal_date": "2026-11-13", "n_names": 23, "invested": 5 / 23, "weights": w})
    args = argparse.Namespace(execute=True, live=False, budget=None, anytime=False, hide_amounts=True, log=True)

    assert eb.cmd_trade(args) == 0
    sent = dict(FakeBroker.sent)
    assert FakeBroker.sent[0] == ("AAPL_US_EQ", -0.01)            # sells first, all of it
    assert set(sent) == {"AAPL_US_EQ", "ADI_US_EQ", "AMAT_US_EQ", "AVGO_US_EQ", "CSCO_US_EQ"}
    assert "NVDA_US_EQ" not in sent and "VUAGl_EQ" not in sent
    assert eb.TRADES.exists()                                      # logged even without free cash

    FakeBroker.sent = []
    assert eb.cmd_trade(args) == 0 and FakeBroker.sent == []       # never twice in a day
