"""The broad, honest test: ~500 stocks fixed in 2016, trading costs, out-of-sample, luck test.

Works offline from data/broad/ (built by scripts/build_broad_data.py):

    python scripts/broad_study.py              # -> reports/broad_study.md, dashboard/broad.html
    python scripts/broad_study.py --runs 50    # fewer shuffles, faster

The main test was fixed *before* looking at the results, so it cannot have been
tuned to them:

    Every Monday, score each stock by how unusual its Wikipedia attention was
    over the last 7 days. Hold the top 10% ("Buzz") until next Monday.
    Compare with holding every stock equally ("equal weight").
    Charge 0.10% per trade, each way.

Everything else in the report (other settings, other costs, the split by
period) is there to show how fragile or sturdy that one number is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikiquant import analysis as A  # noqa: E402
from wikiquant import backtest as B  # noqa: E402
from wikiquant import metrics  # noqa: E402
from wikiquant.broad_dataset import load_broad  # noqa: E402

START = "2016-01-01"         # the date the universe was fixed
COSTS = [0, 5, 10, 20, 40]   # bps per trade, for the sensitivity table
GRID_FRAC = [0.05, 0.10, 0.20]
GRID_LOOKBACK = [3, 7, 14]
FAMOUS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "FB", "TSLA", "NFLX", "JPM",
          "DIS", "BA", "GME", "AMD", "PFE", "NKE", "INTC"]


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def num(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def share(x):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def clean(x):
    """JSON-safe number."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(x) else round(x, 6)


def try_metric(fn, r):
    try:
        return fn(r)
    except NotImplementedError:
        return None


def verdict(oos_t: float, p_value: float, net_edge: float) -> str:
    if np.isfinite(oos_t) and oos_t >= 2 and p_value < 0.05 and net_edge > 0:
        return "Survives: positive after costs, out of sample, and rarely matched by luck."
    if np.isfinite(net_edge) and net_edge > 0 and p_value < 0.10:
        return "Weak hint: positive, but not strong enough out of sample to trust."
    return "Does not survive: after costs and out of sample, it is indistinguishable from luck."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "broad"))
    ap.add_argument("--runs", type=int, default=200, help="shuffled-signal runs for the luck test")
    ap.add_argument("--out", default=str(ROOT / "reports" / "broad_study.md"))
    args = ap.parse_args()

    bd = load_broad(args.data)
    prices = bd.prices.loc[START:]
    bench = bd.bench.loc[START:]
    views = bd.views
    cov = bd.coverage
    split = pd.Timestamp(B.SPLIT)

    # ------------------------------------------------------------ coverage
    n_univ = len(cov)
    has_px = cov["source"].astype(str) != ""
    n_yahoo = int((cov["source"] == "yahoo").sum())
    n_stooq = int((cov["source"] == "stooq").sum())
    left = cov[cov["in_index_today"].astype(str).str.lower() == "false"] if "in_index_today" in cov else cov.iloc[0:0]
    left_with_px = int((left["source"].astype(str) != "").sum())
    missing = cov.loc[~has_px, ["ticker", "name"]]
    stopped = [t for t in prices.columns if prices[t].last_valid_index() is not None
               and prices[t].last_valid_index() < prices.index[-1] - pd.Timedelta(days=10)]

    # ------------------------------------------------------------ main test
    print("Scoring...")
    scores = B.attention_scores(views, prices.index)
    scores = scores.loc[START:]
    eligible = scores.notna().sum(axis=1)
    print("Backtesting...")
    buzz = B.run_portfolio(scores, prices, "top")
    quiet = B.run_portfolio(scores, prices, "bottom")
    ew = B.run_portfolio(scores, prices, "all")
    spy = bench.pct_change().reindex(ew.net.index)

    e_gross = B.edge_stats(buzz.gross, ew.gross)
    e_net = B.edge_stats(buzz.net, ew.net)
    e_bq = B.edge_stats(buzz.net, quiet.net)
    periods = B.period_edges(buzz.net, ew.net)
    periods_bq = B.period_edges(buzz.net, quiet.net)
    yearly = B.yearly_edges(buzz.net, ew.net)
    turn_gap = buzz.annual_turnover - ew.annual_turnover
    be_cost = B.breakeven_cost_bps(e_gross["edge"], turn_gap)

    cost_rows = []
    for c in COSTS:
        bz = B.run_portfolio(scores, prices, "top", cost_bps=c)
        e = B.run_portfolio(scores, prices, "all", cost_bps=c)
        cost_rows.append({"bps": c, **B.edge_stats(bz.net, e.net)})

    # Which stocks made the money?
    gap_contrib = (buzz.contrib.reindex(ew.contrib.index, fill_value=0) - ew.contrib).sort_values(ascending=False)
    total_gap = buzz.gross.sum() - ew.gross.sum()
    top5 = gap_contrib.head(5)
    no_top5 = B.run_portfolio(scores, prices, "top", exclude=tuple(top5.index))
    ew_no_top5 = B.run_portfolio(scores, prices, "all", exclude=tuple(top5.index))
    e_no_top5 = B.edge_stats(no_top5.net, ew_no_top5.net)

    print(f"Luck test ({args.runs} shuffles)...")
    shuffled = B.placebo(scores, prices, ew.gross, runs=args.runs)
    p_value = float((shuffled >= e_gross["edge"]).mean()) if len(shuffled) else np.nan

    # ------------------------------------------------------------ settings chosen in-sample only
    print("Settings search on 2016-2020 only...")
    grid = []
    for lb in GRID_LOOKBACK:
        sc = B.attention_scores(views, prices.index, lookback=lb).loc[START:]
        base = B.run_portfolio(sc, prices, "all")
        for f in GRID_FRAC:
            bt = B.run_portfolio(sc, prices, "top", frac=f)
            pe = B.period_edges(bt.net, base.net)
            grid.append({"lookback": lb, "frac": f, "in": pe["in_sample"], "out": pe["out_of_sample"]})
    best = max(grid, key=lambda g: g["in"]["edge"] if np.isfinite(g["in"]["edge"]) else -np.inf)

    # ------------------------------------------------------------ spike events
    print("Spike events...")
    ev = B.spike_events(views, prices, bench)
    ev = ev[ev["entry_date"] >= START]
    ev_all = B.clustered_summary(ev)
    ev_net = B.clustered_summary(ev, cost_bps=B.COST_BPS)
    ev_in = B.clustered_summary(ev[ev["entry_date"] < split], cost_bps=B.COST_BPS)
    ev_out = B.clustered_summary(ev[ev["entry_date"] >= split], cost_bps=B.COST_BPS)
    react = B.clustered_summary(ev, "reaction")
    base_excess = float(A.baseline_excess(prices.ffill(), bench.ffill()).mean())
    by_size = []
    for lo, hi, label in [(3, 5, "3-5x"), (5, 10, "5-10x"), (10, np.inf, "10x+")]:
        sub = ev[(ev["ratio"] >= lo) & (ev["ratio"] < hi)]
        by_size.append({"label": label, **B.clustered_summary(sub, cost_bps=B.COST_BPS)})
    by_year = ev.groupby(ev["entry_date"].dt.year).apply(
        lambda g: pd.Series(B.clustered_summary(g, cost_bps=B.COST_BPS))).reset_index()

    # Famous-16 inside the broad data, same rules: how much did picking them flatter things?
    famous = [t for t in FAMOUS if t in prices.columns]
    ev_famous = B.clustered_summary(ev[ev["ticker"].isin(famous)], cost_bps=B.COST_BPS)

    cagr = try_metric(metrics.annualised_return, buzz.net)
    sharpe = try_metric(metrics.sharpe_ratio, buzz.net)
    mdd = try_metric(metrics.max_drawdown, buzz.net)
    v = verdict(periods["out_of_sample"]["t"], p_value, e_net["edge"])

    # ------------------------------------------------------------ report
    first, last = prices.index[0].date(), prices.index[-1].date()
    um = bd.universe_meta
    L = ["# The broad test: does attention still work on ~500 stocks, after costs?", "",
         f"Generated by `scripts/broad_study.py` from `data/broad/` · prices {first} to {last}.", "",
         f"**Verdict: {v}**", "",
         "## What changed from the 16-stock dashboard", "",
         f"- **Universe fixed in advance.** Every company in Wikipedia's *List of S&P 500 companies* "
         f"as it stood on {um.get('as_of', START)} ([that exact version]({um.get('url', '')})): "
         f"{n_univ} companies. Nobody in 2016 knew which of them would boom, shrink or be bought.",
         f"- **Delisted companies kept where free data allows.** {len(left)} of the {n_univ} have since left the "
         f"index; we have prices for {left_with_px} of those. Prices found for {int(has_px.sum())} companies in total "
         f"(Yahoo {n_yahoo}, Stooq {n_stooq}); **{len(missing)} are missing**, mostly companies that were bought out "
         f"(Yahoo deletes their history). {len(stopped)} of the stocks we do have stopped trading during the test.",
         f"- **Trading costs.** {B.COST_BPS:g} bps (0.10%) per trade each way, charged on the actual change in weights.",
         f"- **Out of sample.** {START[:4]}-2020 vs 2021-{last.year}. Settings may only be chosen on the first part.",
         f"- **Luck test.** The same strategy with scores shuffled between stocks, {args.runs} times.",
         f"- **Month-clustered t-stats** for spike events, because spikes bunch up on market-wide news.",
         "",
         "## 1. The pre-registered test: Buzz 10% vs equal weight", "",
         f"On a typical Monday {int(eligible.median())} stocks have enough Wikipedia traffic to score "
         f"(usual level ≥ {B.MIN_VIEWS} views a day), so Buzz holds about {int(buzz.holdings.median())} stocks.", "",
         "| | Per year | t-stat | Years positive |", "| --- | ---: | ---: | ---: |",
         f"| Buzz − equal weight, before costs | {pct(e_gross['edge'])} | {num(e_gross['t'])} | {share(e_gross['years_positive'])} |",
         f"| Buzz − equal weight, after costs | **{pct(e_net['edge'])}** | {num(e_net['t'])} | {share(e_net['years_positive'])} |",
         f"| Buzz − Quiet (bottom 10%), after costs | {pct(e_bq['edge'])} | {num(e_bq['t'])} | {share(e_bq['years_positive'])} |",
         "",
         f"Buzz replaces its whole portfolio about {num(buzz.annual_turnover, 0)} times a year "
         f"(equal weight: {num(ew.annual_turnover, 1)}), and every replacement pays the cost twice (sell, then buy). "
         (f"The edge before costs would be wiped out at **{num(be_cost, 0)} bps per trade**." if be_cost and be_cost > 0
          else "There is no edge before costs to protect, so costs only make it worse."), "",
         "### Before and after 2021 (the out-of-sample check)", "",
         "| Period | Buzz − equal weight, after costs | t-stat | Buzz − Quiet | t-stat |", "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in [("in_sample", f"{START[:4]}-2020"), ("out_of_sample", f"2021-{last.year}"), ("all", "All")]:
        L.append(f"| {label} | {pct(periods[key]['edge'])} | {num(periods[key]['t'])} "
                 f"| {pct(periods_bq[key]['edge'])} | {num(periods_bq[key]['t'])} |")
    L += ["", "### Year by year (Buzz − equal weight, after costs)", "",
          "| " + " | ".join(str(y) for y in yearly.index) + " |",
          "|" + " ---: |" * len(yearly),
          "| " + " | ".join(pct(x) for x in yearly.values) + " |", "",
          "### Cost sensitivity", "", "| Cost per trade | Buzz − equal weight per year | t-stat |", "| ---: | ---: | ---: |"]
    for r in cost_rows:
        L.append(f"| {r['bps']} bps | {pct(r['edge'])} | {num(r['t'])} |")
    L += ["", "### Is it luck?", "",
          f"With the attention scores shuffled between stocks, the before-cost edge averaged "
          f"{pct(np.mean(shuffled))} and ranged {pct(np.percentile(shuffled, 5))} to {pct(np.percentile(shuffled, 95))} "
          f"(5th to 95th percentile). The real signal's {pct(e_gross['edge'])} was matched or beaten by "
          f"**{share(p_value)}** of shuffles (p = {num(p_value, 3)}).", "",
          "### Who made the money?", "",
          "| Stock | Share of Buzz's gap vs equal weight |", "| --- | ---: |"]
    for t, c in top5.items():
        L.append(f"| {t} | {share(c / total_gap) if total_gap else '–'} |")
    L += ["", f"Without these five: Buzz − equal weight after costs = {pct(e_no_top5['edge'])} (t {num(e_no_top5['t'])}).", "",
          f"Buzz CAGR {pct(cagr)}, Sharpe {num(sharpe, 2)}, worst drop {pct(mdd, 0)} "
          "_(from your `metrics.py`; – until you write it)_.", "",
          "## 2. Settings chosen on 2016-2020 only, then tested on 2021+", "",
          "Nine combinations of look-back and portfolio size. The best on 2016-2020 is the one a researcher "
          "would have picked; its 2021+ number is the honest estimate.", "",
          "| Look-back days | Buzz size | 2016-2020 (after costs) | t | 2021+ (after costs) | t |",
          "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in grid:
        mark = " **← picked**" if g is best else ""
        L.append(f"| {g['lookback']} | {share(g['frac'])} | {pct(g['in']['edge'])} | {num(g['in']['t'])} "
                 f"| {pct(g['out']['edge'])} | {num(g['out']['t'])}{mark} |")
    L += ["", "## 3. Spike events (3x usual attention) across all stocks", "",
          f"Stock minus S&P 500 over the next {A.HOLD} trading days after the first day you could trade, "
          f"after a {2 * B.COST_BPS:g} bps round trip. A normal {A.HOLD}-day stretch averages {pct(base_excess)} for these stocks. "
          "The *month* t-stat averages spikes within each month first.", "",
          "| | Spikes | Months | Next 20 days vs S&P | Beat S&P | t (naive) | t (by month) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, s in [("All, before costs", ev_all), ("All, after costs", ev_net),
                     (f"{START[:4]}-2020", ev_in), (f"2021-{last.year}", ev_out),
                     ("Only the famous 16 (those in the 2016 list)", ev_famous)]:
        L.append(f"| {label} | {s['n']} | {s['months']} | {pct(s['mean'])} | {share(s['hit'])} "
                 f"| {num(s['t_naive'])} | {num(s['t_month'])} |")
    for s in by_size:
        L.append(f"| Spike size {s['label']} | {s['n']} | {s['months']} | {pct(s['mean'])} | {share(s['hit'])} "
                 f"| {num(s['t_naive'])} | {num(s['t_month'])} |")
    L += ["", f"Day-0 move (the news itself, before you can trade): {pct(react['mean'], 2)} on average.", "",
          "## Limits that remain", "",
          f"- {len(missing)} companies have no free price history (mostly buyouts). Buyouts usually come at a premium, "
          "so leaving them out probably makes *any* long strategy look slightly worse, not better; but we cannot test it.",
          "- Wikipedia page titles come from links in the 2016 list; a few may point at the wrong page or a "
          "parent company. Pages with under 100 views a day are ignored.",
          "- Costs are a flat estimate. Real costs are higher for small, volatile stocks, which is where attention spikes cluster.",
          "- The luck test shuffles stocks within each week, so it keeps the market's ups and downs and only breaks the link "
          "between attention and which stocks were picked.", ""]
    if len(missing):
        L += ["<details><summary>Companies with no price data</summary>", "",
              ", ".join(f"{r.ticker} ({r.name})" for r in missing.itertuples()), "", "</details>", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))

    # ------------------------------------------------------------ dashboard data
    weekly = pd.DataFrame({"buzz": buzz.net, "quiet": quiet.net, "ew": ew.net, "spy": spy}).fillna(0)
    curve = (1 + weekly).cumprod().resample("W-FRI").last()
    hist, edges = np.histogram(shuffled, bins=24)
    res = {
        "built": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "period": [str(first), str(last)],
        "verdict": v,
        "universe": {"n": n_univ, "as_of": um.get("as_of", START), "url": um.get("url", ""),
                     "with_prices": int(has_px.sum()), "yahoo": n_yahoo, "stooq": n_stooq,
                     "missing": len(missing), "left_index": len(left), "left_with_prices": left_with_px,
                     "stopped_trading": len(stopped), "eligible_median": int(eligible.median()),
                     "buzz_size": int(buzz.holdings.median()), "min_views": B.MIN_VIEWS},
        "settings": {"cost_bps": B.COST_BPS, "frac": B.FRACTION, "lookback": B.LOOKBACK, "split": B.SPLIT},
        "edge": {"gross": {k: clean(x) for k, x in e_gross.items()},
                 "net": {k: clean(x) for k, x in e_net.items()},
                 "buzz_quiet": {k: clean(x) for k, x in e_bq.items()},
                 "periods": {k: {kk: clean(x) for kk, x in d.items()} for k, d in periods.items()},
                 "periods_bq": {k: {kk: clean(x) for kk, x in d.items()} for k, d in periods_bq.items()},
                 "turnover": clean(buzz.annual_turnover), "turnover_ew": clean(ew.annual_turnover),
                 "breakeven_bps": clean(be_cost),
                 "without_top5": clean(e_no_top5["edge"]),
                 "top5": [{"ticker": t, "share": clean(c / total_gap if total_gap else np.nan)} for t, c in top5.items()]},
        "yearly": [{"year": int(y), "edge": clean(x)} for y, x in yearly.items()],
        "costs": [{"bps": r["bps"], "edge": clean(r["edge"]), "t": clean(r["t"])} for r in cost_rows],
        "luck": {"runs": args.runs, "p": clean(p_value), "real": clean(e_gross["edge"]),
                 "mean": clean(np.mean(shuffled)), "hist": hist.tolist(), "edges": [clean(x) for x in edges]},
        "grid": [{"lookback": g["lookback"], "frac": g["frac"], "in": clean(g["in"]["edge"]), "in_t": clean(g["in"]["t"]),
                  "out": clean(g["out"]["edge"]), "out_t": clean(g["out"]["t"]), "picked": g is best} for g in grid],
        "events": {k: {kk: clean(x) for kk, x in s.items()} for k, s in
                   [("all", ev_all), ("net", ev_net), ("in", ev_in), ("out", ev_out), ("famous", ev_famous),
                    ("reaction", react)]},
        "events_by_size": [{kk: clean(x) for kk, x in s.items()} for s in by_size],
        "events_by_year": [{kk: clean(x) for kk, x in r.items()} for r in by_year.to_dict("records")],
        "baseline_excess": clean(base_excess),
        "curve": {"dates": [d.strftime("%Y-%m-%d") for d in curve.index],
                  **{k: [clean(x) for x in curve[k]] for k in curve.columns}},
    }
    (ROOT / "dashboard" / "broad_results.json").write_text(json.dumps(res, separators=(",", ":")))
    build_page()
    print(f"\nWrote {out}, dashboard/broad_results.json, dashboard/broad.html")


def build_page() -> None:
    template = (ROOT / "dashboard" / "broad_template.html").read_text()
    data = (ROOT / "dashboard" / "broad_results.json").read_text()
    (ROOT / "dashboard" / "broad.html").write_text(template.replace("/*__DATA__*/", data.replace("</", "<\\/")))


if __name__ == "__main__":
    if "--page-only" in sys.argv:
        build_page()
    else:
        main()
