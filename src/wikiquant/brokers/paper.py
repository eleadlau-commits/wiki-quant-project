"""A pretend broker that keeps its account in a JSON file. No sign-up, no money.

Orders fill instantly at the price you give it, minus a cost in basis points,
so you can watch the bot run for weeks before trusting it with a real account.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class PaperBroker:
    name = "paper"

    def __init__(self, path: Path | str, price_of: Callable[[str], float],
                 starting_cash: float = 1000.0, cost_bps: float = 10.0):
        self.path = Path(path)
        self.price_of = price_of
        self.cost_bps = cost_bps
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": starting_cash, "holdings": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    def account_value(self) -> float:
        held = sum(q * self.price_of(t) for t, q in self.state["holdings"].items())
        return self.state["cash"] + held

    def holding(self, ticker: str) -> tuple[float, float]:
        return float(self.state["holdings"].get(ticker, 0.0)), self.price_of(ticker)

    def market_order(self, ticker: str, quantity: float) -> dict:
        price = self.price_of(ticker)
        value = quantity * price
        cost = abs(value) * self.cost_bps / 10_000
        if value + cost > self.state["cash"] + 1e-9:
            raise ValueError(f"Not enough cash: need {value + cost:.2f}, have {self.state['cash']:.2f}")
        held = self.state["holdings"].get(ticker, 0.0)
        if held + quantity < -1e-9:
            raise ValueError(f"Can't sell {-quantity} {ticker}: only {held} held")
        self.state["cash"] -= value + cost
        new = held + quantity
        if abs(new) < 1e-9:
            self.state["holdings"].pop(ticker, None)
        else:
            self.state["holdings"][ticker] = new
        self._save()
        return {"status": "FILLED", "ticker": ticker, "quantity": quantity,
                "price": price, "cost": round(cost, 4)}
