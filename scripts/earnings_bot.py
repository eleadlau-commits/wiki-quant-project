"""Earnings-cycle overlay: backtest, nightly signal, and next-day trade.

    export SEC_USER_AGENT="Your Name your@email"      # SEC requires contact details

    python scripts/earnings_bot.py verify      # rebuild history, check the plan's test dates and numbers
    python scripts/earnings_bot.py signal      # after the US close: log tomorrow's targets
    python scripts/earnings_bot.py trade       # near the next close: dry run on the demo account
    python scripts/earnings_bot.py trade --execute                      # demo orders
    EARNINGS_CONFIRM_LIVE=yes python scripts/earnings_bot.py trade --live --execute   # real money

Needs T212_API_KEY / T212_API_SECRET for `trade`. Full guide: docs/earnings-cycle.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant import earnings_cycle as ec  # noqa: E402
from wikiquant.earnings_trader import SafetyStop, execute, plan_rebalance, t212_ticker  # noqa: E402

CONFIG = ROOT / "config" / "earnings_cycle.json"
LEDGER = ROOT / "ledger" / "earnings"
SIGNALS = LEDGER / "signals.jsonl"
TRADES = LEDGER / "trades.jsonl"
EVENTS_CSV = ROOT / "data" / "earnings" / "earnings_events.csv"

# From the plan (reference_weights_2022_onward.csv). FB is META here.
TEST_VECTORS = {
    "2022-03-01": ["AVGO", "MRVL", "MU", "ORCL"],
    "2022-10-13": ["AAPL", "AMZN", "GOOGL", "IBM", "INTC", "KLAC", "LRCX", "MCHP", "META", "MSFT",
                   "ON", "QCOM", "SWKS", "TER", "TXN"],
    "2023-05-25": ["ADI", "AVGO", "MRVL", "NVDA", "ORCL"],
    "2024-08-29": ["AVGO", "MRVL", "NVDA", "ORCL"],
    "2026-11-13": ["ADI", "AMAT", "AVGO", "CSCO", "NVDA"],
}


def config() -> dict:
    return json.loads(CONFIG.read_text())


# --- Data -------------------------------------------------------------------

def load_prices(start: str = "2015-01-01") -> pd.DataFrame:
    """Split- and dividend-adjusted closes for the 23 names, from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ec.TICKERS, start=start, auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[[c for c in ec.TICKERS if c in close.columns]].sort_index()
    # Keep NYSE sessions only (Yahoo sometimes adds a stray row).
    return close[close.index.isin(ec.nyse().sessions)]


def load_events(refresh: bool = True) -> pd.DataFrame:
    from wikiquant.edgar import load_earnings_events

    events = ec.reaction_days(load_earnings_events(refresh=refresh))
    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.drop(columns=["accepted_utc"]).to_csv(EVENTS_CSV, index=False)
    return events


def last_closed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or pd.Timestamp.now(tz="UTC")
    closes = ec.nyse().closes
    return closes[closes <= now].index[-1]


# --- Commands ---------------------------------------------------------------

def cmd_verify(args) -> int:
    """Rebuild 2022 onward and compare with the plan's test vectors and headline numbers."""
    from wikiquant.edgar import gap_report, timing_report

    events = load_events()
    prices = load_prices()
    print(f"{len(events)} earnings filings 2014+; timing (New York): {timing_report(events)}")
    for g in gap_report(events[events['filing_date'] >= '2015-01-01']):
        print("  gap:", g)

    sessions = ec.nyse_sessions("2015-01-02", prices.index.max())
    dse = ec.sessions_since_earnings(events, sessions)
    weights = ec.target_weights(dse, prices.reindex(sessions))
    ok = True
    print("\nTest vectors (signal date: n live, invested, long names)")
    for day, expected in TEST_VECTORS.items():
        if pd.Timestamp(day) not in weights.index:
            print(f"  {day}: no data yet")
            continue
        w = weights.loc[day]
        got = sorted(w[w > 0].index)
        n = int(ec.live_mask(prices.reindex(sessions)).loc[day].sum())
        match = got == sorted(expected) and n == 23
        ok &= match
        print(f"  {day}: n={n} invested={w.sum():.4f} {'OK' if match else 'MISMATCH'}  {got}")
        if not match:
            print(f"      expected {sorted(expected)}")
            print(f"      dse: {dse.loc[day].to_dict()}")

    bt = ec.backtest(weights, prices.reindex(sessions))
    rows = []
    for label, a, b in [("2022-2023 (out of sample)", "2022-01-01", "2023-12-31"),
                        ("2017-2021 (in sample)", "2017-01-01", "2021-12-31"),
                        ("2024 onward", "2024-01-01", None)]:
        part = bt.loc[a:b]
        s, bm = ec.summary(part["net"]), ec.summary(part["benchmark"])
        rows.append(f"| {label} | {s['sharpe']:.2f} | {s['total_return']:+.1%} | {s['max_drawdown']:.1%} | "
                    f"{part['invested'].mean():.0%} | {bm['sharpe']:.2f} | {bm['total_return']:+.1%} | "
                    f"{bm['max_drawdown']:.1%} |")
    table = ("| Period | Sharpe | Return | Max DD | Avg invested | EW Sharpe | EW return | EW max DD |\n"
             "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    print("\nNet of 5 bps per side:\n" + table)
    print("Plan says 2022-2023: Sharpe 0.58, +15.1%, max drawdown -10.5%.")
    turnover = bt.loc["2022":"2023", "turnover"].sum() / 2   # sum of |change in weight|, per year
    print(f"Turnover 2022-2023: {turnover:.1f}x capital a year, sum of |change in weight| (plan: about 8.3x)")

    out = ROOT / "reports" / "earnings_cycle.md"
    out.write_text(
        "# Earnings-cycle overlay: rebuilt backtest\n\n"
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `scripts/earnings_bot.py verify` from SEC EDGAR "
        "filings and Yahoo adjusted closes. Equal-weight (EW) benchmark: buy-and-hold of the same live names.\n\n"
        + table + "\n\nPlan's figures for 2022-2023: Sharpe 0.58, +15.1%, max drawdown -10.5%.\n\n"
        "Test vectors: " + ("all match." if ok else "**some do not match**, see the workflow log.") + "\n")
    LEDGER.mkdir(parents=True, exist_ok=True)
    weights.loc["2022":].round(6).to_csv(ROOT / "data" / "earnings" / "weights_2022_onward.csv")
    bt.loc["2022":].round(8).to_csv(ROOT / "data" / "earnings" / "daily_returns_2022_onward.csv")
    print("\nAll test vectors match." if ok else "\nSome test vectors do NOT match.")
    return 0 if ok else 1


def cmd_signal(args) -> int:
    """After the close of t: compute and log the targets for signal day t (once per day)."""
    day = pd.Timestamp(args.date) if args.date else last_closed_session()
    records = ec.read_log(SIGNALS)
    ec.verify_chain(records)
    if any(r["signal_date"] == str(day.date()) for r in records):
        print(f"Signal for {day.date()} already logged; nothing to do.")
        return 0
    events = load_events()
    prices = load_prices(start=str((day - pd.Timedelta(days=400)).date()))
    problems = ec.price_problems(prices, day)
    sig = ec.signal_for(day, events, prices)
    if problems:
        # Keep yesterday's targets and log the gap. Never back-fill later.
        prev = records[-1] if records else None
        sig.stale, sig.problems = True, tuple(problems)
        if prev:
            sig.weights, sig.n_names, sig.invested = prev["weights"], prev["n_names"], prev["invested"]
        else:
            sig.weights = {k: 0.0 for k in sig.weights}
            sig.invested = 0.0
    hashes = {
        "earnings_sha256": ec.sha256_file(EVENTS_CSV),
        "plan_sha256": ec.sha256_file(CONFIG),
        "code_sha256": ec.sha256_file(ROOT / "src" / "wikiquant" / "earnings_cycle.py"),
    }
    rec = ec.append_record(SIGNALS, sig.record_body(**hashes))
    longs = [k for k, v in rec["weights"].items() if v > 0]
    print(f"Signal {rec['signal_date']}: n={rec['n_names']} invested={rec['invested']:.4f} long={longs}"
          + (" (STALE: kept previous targets)" if sig.stale else ""))
    for p in problems:
        print("  problem:", p)
    return 0


def in_trade_window(now: pd.Timestamp, earliest: int = 60, latest: int = 3) -> tuple[bool, pd.Timestamp | None]:
    """True if now is a session, between `earliest` and `latest` minutes before its close."""
    cal = ec.nyse()
    today = now.tz_convert("America/New_York").normalize().tz_localize(None)
    if not cal.is_session(today):
        return False, None
    close = cal.session_close(today)
    left = (close - now).total_seconds() / 60
    return latest <= left <= earliest, today


def latest_quotes(tickers: list[str]) -> dict[str, float]:
    """Latest prices in GBP: Yahoo's latest USD price divided by GBP/USD."""
    import yfinance as yf

    df = yf.download(tickers + ["GBPUSD=X"], period="5d", interval="5m", auto_adjust=True,
                     progress=False, prepost=False)["Close"].ffill()
    last = df.iloc[-1]
    fx = float(last["GBPUSD=X"])
    return {t: float(last[t]) / fx for t in tickers}


def cmd_trade(args) -> int:
    cfg = config()["trading"]
    now = pd.Timestamp.now(tz="UTC")
    ok, today = in_trade_window(now)
    if not ok and not args.anytime:
        print(f"Not within the last hour before an NYSE close ({now:%Y-%m-%d %H:%M} UTC); nothing to do.")
        return 0
    today = today or last_closed_session(now)
    records = ec.read_log(SIGNALS)
    ec.verify_chain(records)
    if not records:
        print("No signal logged yet; run `signal` after a close first.")
        return 0
    rec = records[-1]
    prev_session = ec.nyse().previous_session(today) if ec.nyse().is_session(today) else today
    if rec["signal_date"] != str(prev_session.date()) and not args.anytime:
        print(f"Latest signal is for {rec['signal_date']}, not {prev_session.date()}; holding (no trades).")
        return 0
    done = [t for t in ec.read_log(TRADES) if t.get("trade_date") == str(today.date()) and not t.get("dry_run")]
    if done and args.execute:
        print(f"Already traded on {today.date()}; not trading twice.")
        return 0

    if args.live and os.getenv("EARNINGS_CONFIRM_LIVE") != "yes":
        raise SystemExit("Refusing the live account: set EARNINGS_CONFIRM_LIVE=yes as well as --live.")
    from wikiquant.brokers.trading212 import Trading212Broker
    broker = Trading212Broker(os.getenv("T212_API_KEY", ""), os.getenv("T212_API_SECRET", ""),
                              "live" if args.live else "demo")
    if broker.account_currency() not in ("GBP", ""):
        print(f"Note: account currency is {broker.account_currency()}, sizes assume GBP.")
    tickers = sorted(rec["weights"])
    pending = {o.get("ticker") or o.get("instrument", {}).get("ticker") for o in broker.all_pending_orders()}
    if pending & {t212_ticker(t) for t in tickers}:
        print("An earlier order for one of these stocks is still pending; not trading today.")
        return 0
    held_all = broker.positions()
    holdings = {t: held_all.get(t212_ticker(t), 0.0) for t in tickers}
    prices = latest_quotes(tickers)
    try:
        orders = plan_rebalance(rec["weights"], rec["n_names"], float(args.budget or cfg["budget_gbp"]),
                                holdings, prices, skip=set(cfg["skip"]), min_trade=cfg["min_trade_gbp"],
                                max_order=cfg["max_order_gbp"])
    except SafetyStop as e:
        raise SystemExit(f"STOPPED, nothing sent: {e}")

    if args.execute:
        execute([o for o in orders if o.action == "sell"], broker)
        try:
            cash = broker.free_cash()
        except Exception as e:  # unknown reply shape: let the broker reject what it can't afford
            print(f"Couldn't read free cash ({str(e)[:80]}); sending buys anyway.")
            cash = None
        execute([o for o in orders if o.action == "buy"], broker, free_cash=cash)
    for o in orders:
        if o.action in ("hold",) or o.action.startswith("skip (yours"):
            continue
        detail = "" if args.hide_amounts else f"  qty {o.order_qty:+.4f} (~{o.order_value:+.2f})"
        status = f"  [{o.reply.get('status')}]" if o.reply else ""
        print(f"  {o.ticker:<6} {o.action}{detail}{status}")
    print(f"[{broker.name}{'' if args.execute else ' DRY RUN'}] signal {rec['signal_date']} -> "
          f"{sum(o.action in ('buy', 'sell') for o in orders)} order(s)")
    entry = {"trade_date": str(today.date()), "signal_date": rec["signal_date"], "signal_hash": rec["hash"],
             "broker": broker.name, "dry_run": not args.execute,
             "time": now.isoformat(timespec="seconds"), "orders": [o.public() for o in orders]}
    if args.log:
        LEDGER.mkdir(parents=True, exist_ok=True)
        with TRADES.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    return 1 if any(o.reply.get("status") == "ERROR" for o in orders) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="rebuild history and check the plan's test vectors")
    s = sub.add_parser("signal", help="log the targets for the latest closed session")
    s.add_argument("--date", help="signal date (default: latest closed NYSE session)")
    t = sub.add_parser("trade", help="rebalance to the latest logged targets")
    t.add_argument("--execute", action="store_true", help="actually send orders (default: dry run)")
    t.add_argument("--live", action="store_true", help="real-money account (also needs EARNINGS_CONFIRM_LIVE=yes)")
    t.add_argument("--budget", type=float, help="override the budget in config/earnings_cycle.json")
    t.add_argument("--anytime", action="store_true", help="ignore the trading window (dry runs and testing)")
    t.add_argument("--hide-amounts", action="store_true", help="don't print amounts (public CI logs)")
    t.add_argument("--no-log", dest="log", action="store_false", help="don't append to ledger/earnings/trades.jsonl")
    args = ap.parse_args()
    if args.cmd == "trade" and args.anytime and args.execute:
        raise SystemExit("--anytime is for dry runs only.")
    sys.exit({"verify": cmd_verify, "signal": cmd_signal, "trade": cmd_trade}[args.cmd](args))


if __name__ == "__main__":
    main()
