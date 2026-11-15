"""Run the trading bot once: today's signal -> compare with holdings -> order.

Safe by default: it uses the pretend paper broker and only *prints* what it
would do. Each step towards real money needs its own explicit switch.

    # 1. Dry run on a pretend 1,000 account (no keys needed)
    python scripts/run_bot.py

    # 2. Paper trading: the pretend account in trades/paper_account.json changes
    python scripts/run_bot.py --execute

    # 3. Trading 212 practice account (keys from the demo app, in env vars)
    export T212_API_KEY=... T212_API_SECRET=...
    python scripts/run_bot.py --broker trading212 --find "S&P 500"   # find the ticker code
    python scripts/run_bot.py --broker trading212                     # dry run
    python scripts/run_bot.py --broker trading212 --execute           # real demo orders

    # 4. Real money: live keys, plus --live, plus BOT_CONFIRM_LIVE=yes
    BOT_CONFIRM_LIVE=yes python scripts/run_bot.py --broker trading212 --live --execute

Full guide: docs/trading-bot.md
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant.trader import STRATEGIES, OrderTooLarge, run_once  # noqa: E402


@lru_cache(maxsize=None)
def yahoo_close_history(ticker: str):
    """Daily adjusted closes from Yahoo Finance, about 2 years (enough for a 200-day average)."""
    import yfinance as yf

    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No prices from Yahoo for {ticker}")
    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def yahoo_last_price(ticker: str) -> float:
    """Latest price in the main currency unit. London prices quoted in pence (GBp) become pounds."""
    import yfinance as yf

    price = float(yahoo_close_history(ticker).iloc[-1])
    try:
        if yf.Ticker(ticker).fast_info.get("currency") == "GBp":
            price /= 100
    except Exception:
        pass
    return price


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", choices=["paper", "trading212"], default="paper")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="sma_50_200")
    ap.add_argument("--signal", default="SPY", help="Yahoo ticker the strategy reads (default SPY)")
    ap.add_argument("--trade-ticker", default=os.getenv("BOT_TRADE_TICKER", "VUAGl_EQ"),
                    help="the broker's code for what to buy (Trading 212 format, e.g. VUAGl_EQ)")
    ap.add_argument("--yahoo-ticker", default=os.getenv("BOT_YAHOO_TICKER", "VUAG.L"),
                    help="the same fund on Yahoo, for its price when you hold none")
    ap.add_argument("--allocation", type=float, default=0.95, help="share of the account to invest when the signal is on")
    ap.add_argument("--min-trade", type=float, default=20.0, help="skip orders smaller than this value")
    ap.add_argument("--max-order", type=float, default=float(os.getenv("BOT_MAX_ORDER", "2000")),
                    help="refuse any single order bigger than this value (a safety cap)")
    ap.add_argument("--budget", type=float, default=float(os.getenv("BOT_BUDGET") or 0) or None,
                    help="only manage this much of the account (e.g. 50); the rest is left alone")
    ap.add_argument("--execute", action="store_true", help="actually place the order (default: dry run)")
    ap.add_argument("--live", action="store_true", help="use the real-money Trading 212 account")
    ap.add_argument("--hide-amounts", action="store_true", help="don't print money amounts (for public CI logs)")
    ap.add_argument("--log", default=str(ROOT / "trades" / "bot_log.jsonl"))
    ap.add_argument("--paper-file", default=str(ROOT / "trades" / "paper_account.json"))
    ap.add_argument("--find", help="search Trading 212 instruments by name and exit")
    args = ap.parse_args()

    if args.broker == "paper":
        from wikiquant.brokers import PaperBroker
        broker = PaperBroker(args.paper_file, price_of=lambda t: yahoo_last_price(args.yahoo_ticker))
    else:
        from wikiquant.brokers.trading212 import Trading212Broker
        env = "live" if args.live else "demo"
        if env == "live" and os.getenv("BOT_CONFIRM_LIVE") != "yes":
            raise SystemExit("Refusing to use the live account: set BOT_CONFIRM_LIVE=yes as well as --live.")
        broker = Trading212Broker(os.getenv("T212_API_KEY", ""), os.getenv("T212_API_SECRET", ""), env,
                                  price_fallback=lambda t: yahoo_last_price(args.yahoo_ticker))
        if args.find:
            for i in broker.find_instruments(args.find):
                print(f"{i['ticker']:<16} {i.get('type', ''):<6} {i.get('currencyCode', ''):<4} {i.get('isin', '')}  {i['name']}")
            return

    signal = yahoo_close_history(args.signal)
    dry = not args.execute
    try:
        plan = run_once(broker, signal, args.trade_ticker, args.strategy, args.allocation,
                        args.min_trade, dry_run=dry, log_path=args.log, max_order_value=args.max_order,
                        budget=args.budget)
    except OrderTooLarge as e:
        raise SystemExit(f"{e} Raise --max-order (or BOT_MAX_ORDER) if this is intended.")

    on = "INVESTED" if plan.target_weight > 0 else "CASH"
    print(f"[{plan.broker}{' DRY RUN' if dry else ''}] {plan.strategy}: {plan.signal_ticker} close "
          f"{plan.signal_date} says {on}")
    if args.hide_amounts:
        print(f"  {plan.trade_ticker}: action = {plan.action}")
    else:
        print(f"  managing {plan.account_value:,.2f} | {plan.trade_ticker} at {plan.price:,.2f} | "
              f"hold {plan.held_qty:g} -> target {plan.target_qty:.4f}")
        print(f"  action = {plan.action} {abs(plan.order_qty):g} ({abs(plan.order_value):,.2f})")
    if plan.broker_reply:
        print(f"  broker: status {plan.broker_reply.get('status')}, id {plan.broker_reply.get('id', '-')}")


if __name__ == "__main__":
    main()
