"""Trade the earnings-cycle signal on Trading 212: many stocks, one small budget.

This is separate from the one-fund bot in wikiquant.trader. It only ever
touches the US stocks of the earnings-cycle universe, minus any you keep for
yourself (NVDA by default, see config/earnings_cycle.json), so it can't sell
the VUAG bot's fund or your own NVDA.

How it sizes trades: the budget (say 50) is split into equal slots of
budget / n. A stock the signal holds should be worth one slot; a stock it
doesn't hold should be worth 0. Orders smaller than `min_trade` are skipped,
except selling out completely, which always goes through.

Safety: dry run unless told otherwise, demo account unless --live and
EARNINGS_CONFIRM_LIVE=yes, a cap on any single order, and a refusal to run if
the account holds far more of a stock than the bot could have bought (that
means the shares are yours, not the bot's).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


class SafetyStop(RuntimeError):
    """Raised before any order is sent when something looks wrong."""


def t212_ticker(ticker: str) -> str:
    """Trading 212's code for a US stock, e.g. AAPL -> AAPL_US_EQ."""
    return f"{ticker}_US_EQ"


@dataclass
class Order:
    ticker: str
    held_qty: float
    price: float            # in the account currency
    target_value: float
    order_qty: float
    action: str             # buy, sell, hold, skip (below minimum), skip (yours)
    reply: dict = field(default_factory=dict)

    @property
    def order_value(self) -> float:
        return self.order_qty * self.price

    def public(self) -> dict:
        """What's safe to write in a public log: no amounts."""
        return {"ticker": self.ticker, "action": self.action,
                **({"status": self.reply.get("status", "")} if self.reply else {})}

    def full(self) -> dict:
        d = asdict(self)
        d["order_value"] = self.order_value
        return d


def plan_rebalance(weights: dict[str, float], n_names: int, budget: float,
                   holdings: dict[str, float], prices: dict[str, float],
                   skip: set[str] = frozenset(), min_trade: float = 1.0,
                   max_order: float = 10.0, decimals: int = 4) -> list[Order]:
    """Orders that move each bot stock to its target value. Sells come first.

    weights:  ticker -> target weight from the signal (1/n or 0)
    holdings: ticker -> quantity held in the account
    prices:   ticker -> price in the account currency

    Worked example: budget 50, n = 23, so one slot is 50/23 = 2.17. AVGO is
    long, priced 200, holding 0 -> buy 2.17/200 = 0.0108 shares. Next week AVGO
    leaves its window (weight 0) -> sell all 0.0108.
    """
    slot = budget / n_names if n_names else 0.0
    orders = []
    for tk in sorted(weights):
        held = float(holdings.get(tk, 0.0))
        if tk in skip:
            orders.append(Order(tk, held, prices.get(tk, float("nan")), 0.0, 0.0, "skip (yours)"))
            continue
        price = prices[tk]
        if not (price > 0 and math.isfinite(price)):
            raise SafetyStop(f"No usable price for {tk}: {price}")
        if held * price > 2 * slot + 5:
            raise SafetyStop(f"The account holds {tk} worth {held * price:.2f}, far more than one slot "
                             f"({slot:.2f}). Those shares look like yours, so the bot stops. Add {tk} "
                             f"to 'skip' in config/earnings_cycle.json or sell them yourself.")
        target_value = slot if weights[tk] > 0 else 0.0
        if target_value == 0:
            qty = -held
        else:
            qty = math.trunc((target_value / price - held) * 10**decimals) / 10**decimals
        value = qty * price
        if qty == 0:
            action = "hold"
        elif target_value == 0:
            action = "sell"          # selling out is never skipped
        elif abs(value) < min_trade:
            qty, action = 0.0, "skip (below minimum)"
        else:
            action = "buy" if qty > 0 else "sell"
        if abs(qty * price) > max_order:
            raise SafetyStop(f"{tk} order worth {abs(qty * price):.2f} is above the cap of {max_order:.2f}; "
                             "nothing was sent.")
        orders.append(Order(tk, held, price, target_value, qty, action))
    return sorted(orders, key=lambda o: (o.action != "sell", o.ticker))


def execute(orders: list[Order], broker, free_cash: float | None = None) -> list[Order]:
    """Send the orders (sells first). One failed order doesn't stop the others.

    Never retries an order: Trading 212's market orders aren't idempotent.
    """
    for o in orders:
        if o.action not in ("buy", "sell"):
            continue
        if o.action == "buy" and free_cash is not None and o.order_value > free_cash:
            o.action, o.reply = "skip (not enough cash)", {}
            continue
        try:
            o.reply = broker.market_order(t212_ticker(o.ticker), o.order_qty) or {}
            o.reply.setdefault("status", "SENT")
            if free_cash is not None:
                free_cash -= o.order_value
        except Exception as e:  # keep going with the other stocks
            o.reply = {"status": "ERROR", "error": str(e)[:200]}
    return orders
