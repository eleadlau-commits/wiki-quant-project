"""Can a simple rule dodge crashes? A fair test on SPY since 1993.

Two families of rules, each tried with many settings:
  - volatility targeting: hold less when the market is jumpy
  - big-drop exit: sell after an unusually large 1-3 day fall, buy back on recovery

The honest part: settings are chosen using 1994-2009 only (the dot-com crash
and 2008), then judged on 2010-2026, which the choice never saw (2011, 2015,
2018, 2020, 2022, 2025). Cash earns the T-bill rate, trades cost 10 bps, and
signals act from the next day.

    python scripts/crash_defence.py

Needs data/long/spy_tbill.csv (from scripts/fetch_long_history.py) and your
metrics.py. Writes reports/crash_defence.md and a chart.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

from wikiquant import metrics as M
from wikiquant.backtest import count_trades, run_backtest
from wikiquant.strategies import big_drop_exit, buy_and_hold, sma_crossover, vol_target

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "long" / "spy_tbill.csv"
REPORT = ROOT / "reports" / "crash_defence.md"
CHART = ROOT / "reports" / "crash_defence.png"

COST_BPS = 10
TRAIN = ("1994-01-01", "2009-12-31")
TEST = ("2010-01-01", None)

FAMILIES = {
    "Volatility targeting": [
        (f"target {t:.0%}, {w}-day vol, band {b:.0%}", lambda p, t=t, w=w, b=b: vol_target(p, t, w, band=b))
        for t, w, b in product([0.10, 0.12, 0.15, 0.20], [10, 20, 60], [0.0, 0.1])
    ],
    "Big-drop exit": [
        (f"fall >{d:.0%} in {n}d, back above {r}d avg", lambda p, d=d, n=n, r=r: big_drop_exit(p, d, n, r))
        for d, n, r in product([0.03, 0.04, 0.05, 0.06], [1, 2, 3], [5, 10, 20])
    ],
}
BASELINES = [
    ("Buy and hold", buy_and_hold),
    ("50/200 crossover", lambda p: sma_crossover(p, 50, 200)),
    ("Price above 200-day avg", lambda p: (p > p.rolling(200).mean()).astype(float)),
]


def check_metrics_done() -> None:
    try:
        M.max_drawdown(pd.Series([0.1, -0.1]))
        M.sharpe_ratio(pd.Series([0.1, -0.05, 0.02]))
    except NotImplementedError:
        sys.exit("Finish src/wikiquant/metrics.py first (pytest tests/test_metrics.py).")


def load():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    spy = df["spy"].rename("SPY")
    # Yield is % a year; a day's cash return is known from the previous close.
    cash = (df["tbill_pct"].clip(lower=0) / 100 / 252).shift(1).fillna(0.0)
    return spy, cash


def evaluate(spy, cash, positions, period, lag=1):
    bt = run_backtest(spy, positions, COST_BPS, lag=lag, cash_ret=cash)
    a, b = period
    bt = bt.loc[a:b] if b else bt.loc[a:]
    r = bt["net_ret"]
    return {
        "CAGR": M.annualised_return(r),
        "Sharpe": M.sharpe_ratio(r),
        "MaxDD": M.max_drawdown(r),
        "Exposure": bt["position"].mean(),
        "Trades/yr": count_trades(positions.loc[r.index]) / (len(r) / 252),
        "equity": bt["equity"] / bt["equity"].iloc[0],
    }


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%"


def row(name, s):
    return (f"| {name} | {pct(s['CAGR'])} | {s['Sharpe']:.2f} | {pct(s['MaxDD'], 0)} | "
            f"{s['Exposure'] * 100:.0f}% | {s['Trades/yr']:.1f} |")


HEAD = ("| Rule | CAGR | Sharpe | Worst fall | Avg invested | Trades / yr |\n"
        "| --- | --- | --- | --- | --- | --- |")


def crash_table(spy, cash, rules):
    """Return of each rule through each big fall (peak to trough of SPY)."""
    crashes = [("Dot-com", "2000-03-24", "2002-10-09"), ("2008", "2007-10-09", "2009-03-09"),
               ("2011", "2011-04-29", "2011-10-03"), ("2018 Q4", "2018-09-20", "2018-12-24"),
               ("Covid 2020", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-10-12"),
               ("Tariffs 2025", "2025-02-19", "2025-04-08")]
    lines = ["| Fall | Days | SPY | " + " | ".join(n for n, _ in rules) + " |",
             "| --- | --- | --- |" + " --- |" * len(rules)]
    bts = {n: run_backtest(spy, pos, COST_BPS, cash_ret=cash) for n, pos in rules}
    for label, a, b in crashes:
        if pd.Timestamp(b) > spy.index[-1]:
            continue
        days = len(spy.loc[a:b]) - 1
        cells = [pct(spy.loc[:b].iloc[-1] / spy.loc[:a].iloc[-1] - 1, 0)]
        for n, _ in rules:
            r = bts[n]["net_ret"].loc[a:b].iloc[1:]
            cells.append(pct((1 + r).prod() - 1, 0))
        lines.append(f"| {label} | {days} | " + " | ".join(cells) + " |")
    return lines


def make_chart(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a6fdb", "#e07b00", "#1a9e77", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, eq), c in zip(curves.items(), colors):
        ax.plot(eq.index, eq, lw=1.5, color=c, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Out of sample, 2010-2026: settings chosen on 1994-2009 only (after costs)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART, dpi=130)
    plt.close(fig)


def main() -> None:
    check_metrics_done()
    spy, cash = load()
    L = []
    w = L.append
    w("# Can a simple rule dodge crashes? SPY 1993-2026\n")
    w(f"_Generated by `scripts/crash_defence.py`. Data {spy.index[0]:%b %Y} to {spy.index[-1]:%d %b %Y}. "
      f"Costs {COST_BPS} bps per unit traded; cash earns the 13-week T-bill rate; signals act from "
      "the next day. Explanation: [`docs/crash-defence.md`](../docs/crash-defence.md)._\n")

    picked, oos_curves = {}, {}
    base_train = {n: evaluate(spy, cash, f(spy), TRAIN) for n, f in BASELINES}
    base_test = {n: evaluate(spy, cash, f(spy), TEST) for n, f in BASELINES}
    bh_test_sharpe = base_test["Buy and hold"]["Sharpe"]
    bh_test_dd = base_test["Buy and hold"]["MaxDD"]

    fam_summary = []
    for fam, rules in FAMILIES.items():
        train = [(n, f, evaluate(spy, cash, f(spy), TRAIN)) for n, f in rules]
        best_name, best_f, best_train = max(train, key=lambda x: x[2]["Sharpe"])
        test = [(n, evaluate(spy, cash, f(spy), TEST)) for n, f, _ in train]
        best_test = dict(test)[best_name]
        picked[fam] = (best_name, best_f, best_train, best_test)
        sh = pd.Series([t["Sharpe"] for _, t in test])
        dd = pd.Series([t["MaxDD"] for _, t in test])
        cg = pd.Series([t["CAGR"] for _, t in test])
        fam_summary.append((fam, len(rules), (sh > bh_test_sharpe).mean(), sh.median(),
                            (dd > bh_test_dd).mean(), cg.median()))

    # 1. Headline: out of sample
    w("## 1. The fair test: 2010-2026, with settings picked on 1994-2009\n")
    w(HEAD)
    for n in ("Buy and hold", "50/200 crossover", "Price above 200-day avg"):
        w(row(n, base_test[n]))
    for fam, (name, _, _, t) in picked.items():
        w(row(f"**{fam}** ({name})", t))
    w("")
    w(f"![Out-of-sample equity curves]({CHART.name})\n")
    oos_curves["Buy and hold"] = base_test["Buy and hold"]["equity"]
    oos_curves["50/200 crossover"] = base_test["50/200 crossover"]["equity"]
    for fam, (_, _, _, t) in picked.items():
        oos_curves[fam] = t["equity"]
    make_chart(oos_curves)

    # 2. Training period
    w("## 2. Training period, 1994-2009 (where the settings were chosen)\n")
    w(HEAD)
    for n in ("Buy and hold", "50/200 crossover", "Price above 200-day avg"):
        w(row(n, base_train[n]))
    for fam, (name, _, tr, _) in picked.items():
        w(row(f"{fam} ({name})", tr))
    w("")

    # 3. Robustness: every setting
    w("## 3. Is it the idea or the lucky setting? Every setting, out of sample\n")
    w(f"Buy and hold 2010-2026: Sharpe **{bh_test_sharpe:.2f}**, worst fall **{pct(bh_test_dd, 0)}**.\n")
    w("| Family | Settings tried | Beat B&H Sharpe | Median Sharpe | Smaller worst fall | Median CAGR |\n"
      "| --- | --- | --- | --- | --- | --- |")
    for fam, n, beat, med, dd_share, med_cagr in fam_summary:
        w(f"| {fam} | {n} | {beat * 100:.0f}% | {med:.2f} | {dd_share * 100:.0f}% | {pct(med_cagr)} |")
    w("")

    # 4. Crash by crash
    w("## 4. Crash by crash\n")
    w("Return from SPY's peak to its bottom in each fall (whole history, chosen settings).\n")
    rules = [("50/200", sma_crossover(spy, 50, 200))]
    rules += [(fam.split()[0] if fam.startswith("Vol") else "Big drop", f(spy)) for fam, (_, f, _, _) in picked.items()]
    L.extend(crash_table(spy, cash, rules))
    w("")

    # 5. One more day of delay
    w("## 5. Stricter timing: act one day later\n")
    w("| Rule | CAGR | Sharpe | Worst fall |\n| --- | --- | --- | --- |")
    for fam, (name, f, _, _) in picked.items():
        t = evaluate(spy, cash, f(spy), TEST, lag=2)
        w(f"| {fam} | {pct(t['CAGR'])} | {t['Sharpe']:.2f} | {pct(t['MaxDD'], 0)} |")
    w("")

    REPORT.write_text("\n".join(L))
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
