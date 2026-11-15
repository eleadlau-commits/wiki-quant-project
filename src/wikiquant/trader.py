"""Run a strategy for real: decide today's target, compare with what you hold, trade the gap.

One run of the bot does five things, in this order:

1. Load recent prices of the *signal* asset (SPY by default) and ask the
   strategy for today's target weight (1 = fully invested, 0 = cash).
2. Ask the broker how much the account is worth and how much of the *traded*
   asset it holds.
3. Work out the order that moves the holding to the target.
4. Skip it if it's too small to bother with (avoids paying costs for noise).
5. Place the order (unless it's a dry run) and append everything to a log.

The strategy only sees prices up to the last close, and the order is placed
after that, so the bot can't use information from the future: the same
"lag" rule as the backtester.

The signal and traded assets can differ. UK retail investors can't buy SPY
(a US fund without a UK key-information document), so the bot reads SPY's
trend and trades a UK-listed fund that tracks the same S&P 500 index.

Brokers live in `wikiquant.brokers`. The only thing a broker must do is
answer three questions and place one kind of order (see `Broker`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from wikiquant.strategies import buy_and_hold, sma_crossover

# Every strategy the bot can run: prices -> positions (one per day). The bot
# uses the last value. Add a new one here and choose it with --strategy.
STRATEGIES: dict[str, Callable[[pd.Series], pd.Series]] = {
    "sma_50_200": lambda p: sma_crossover(p, 50, 200),
    "buy_and_hold": buy_and_hold,
}


class OrderTooLarge(RuntimeError):
    pass


class Broker(Protocol):
    """What the bot needs from a broker. Values are in the account currency."""

    name: str

    def account_value(self) -> float:
        """Cash plus the value of everything held."""

    def holding(self, ticker: str) -> tuple[float, float]:
        """(quantity held, current price) of one instrument. Quantity 0 if none."""

    def market_order(self, ticker: str, quantity: float) -> dict:
        """Buy (quantity > 0) or sell (quantity < 0) at market. Returns the broker's reply."""


@dataclass
class Plan:
    """Everything the bot decided in one run, in plain numbers (this is what gets logged)."""

    time: str
    strategy: str
    signal_ticker: str
    signal_date: str
    signal_close: float
    target_weight: float
    trade_ticker: str
    account_value: float
    price: float
    held_qty: float
    target_qty: float
    order_qty: float
    order_value: float
    action: str  # "buy", "sell", "hold", "skip (below minimum)" or "wait (...)"
    broker: str = ""
    dry_run: bool = True
    broker_reply: dict = field(default_factory=dict)


def target_weight(prices: pd.Series, strategy: str) -> tuple[float, pd.Timestamp]:
    """The strategy's position for the most recent close, and that close's date."""
    prices = prices.dropna()
    positions = STRATEGIES[strategy](prices)
    return float(positions.iloc[-1]), prices.index[-1]


def plan_order(account_value: float, price: float, held_qty: float, weight: float,
               allocation: float = 1.0, min_trade_value: float = 20.0,
               fractional: bool = True) -> tuple[float, float, str]:
    """How much to buy or sell to hold `weight x allocation` of the account.

    Worked example: account worth 1,000, allocation 0.95 (keep 5% as a cash
    buffer for fees and price moves), target weight 1, price 50, holding 0.
    Target value = 950, so target quantity = 19, order = +19 (buy).
    Next day the signal turns off (weight 0): target 0, order = -19 (sell all).

    Returns (target quantity, order quantity, action).
    """
    target_qty = account_value * allocation * weight / price
    if not fractional:
        target_qty = math.floor(target_qty)
    if weight == 0:
        target_qty = 0.0  # selling out means all of it, not "nearly all"
    order_qty = target_qty - held_qty
    if abs(order_qty * price) < min_trade_value and not (weight == 0 and held_qty > 0):
        return target_qty, 0.0, "hold" if order_qty == 0 else "skip (below minimum)"
    return target_qty, order_qty, "buy" if order_qty > 0 else "sell"


def run_once(broker: Broker, signal_prices: pd.Series, trade_ticker: str,
             strategy: str = "sma_50_200", allocation: float = 0.95,
             min_trade_value: float = 20.0, dry_run: bool = True,
             log_path: Path | str | None = None,
             max_order_value: float = float("inf"),
             budget: float | None = None) -> Plan:
    """One full cycle: signal -> holdings -> order -> (place) -> log.

    Raises OrderTooLarge, before anything is sent, if the order is worth more
    than `max_order_value`: a guard against a bug or a bad price buying far
    more than intended.

    `budget` limits how much of the account the bot manages: with a budget of
    50 in a 1,000 account, it sizes trades as if the account were worth 50
    and leaves the rest alone.
    """
    weight, sig_date = target_weight(signal_prices, strategy)
    pending = getattr(broker, "pending_orders", None)
    waiting = bool(pending and pending(trade_ticker))
    value = broker.account_value()
    if budget is not None:
        value = min(value, budget)
    held, price = broker.holding(trade_ticker)
    target_qty, order_qty, action = plan_order(value, price, held, weight, allocation, min_trade_value)
    if target_qty == 0:
        order_qty = -held  # sell out exactly, leaving no dust
    else:
        # Fractional shares are fine, but brokers accept limited decimals.
        # Rounding towards zero never buys more than the account can afford.
        order_qty = math.trunc(order_qty * 1e4) / 1e4

    if waiting:
        # Yesterday's order hasn't filled yet (e.g. sent while the exchange was
        # closed). Holdings are out of date, so don't trade on them.
        order_qty, action = 0.0, "wait (earlier order still pending)"
    plan = Plan(
        time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        strategy=strategy, signal_ticker=str(signal_prices.name or ""),
        signal_date=str(sig_date.date()), signal_close=float(signal_prices.dropna().iloc[-1]),
        target_weight=weight, trade_ticker=trade_ticker, account_value=value, price=price,
        held_qty=held, target_qty=target_qty, order_qty=order_qty,
        order_value=order_qty * price, action=action, broker=broker.name, dry_run=dry_run,
    )
    if abs(plan.order_value) > max_order_value:
        raise OrderTooLarge(f"Order worth {plan.order_value:,.2f} is above the cap of "
                            f"{max_order_value:,.2f}; nothing was sent.")
    if order_qty != 0 and not dry_run:
        plan.broker_reply = broker.market_order(trade_ticker, order_qty)
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(asdict(plan)) + "\n")
    return plan
