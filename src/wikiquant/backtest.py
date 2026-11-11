"""Honest backtests for a broad universe: trading costs, out-of-sample checks, luck tests.

The 16-stock dashboard asks "did attention spikes come before good returns?".
This module asks the harder questions a sceptical interviewer would:

1. **Costs.** Every trade costs money (the bid-ask spread plus a little market
   impact). We charge `cost_bps` basis points (1 bp = 0.01%) on every pound
   traded, each way. A weekly strategy that replaces most of its stocks every
   Monday pays this a lot, so a small edge can vanish.
2. **Out of sample.** Choose the settings using only 2016-2020, then look at
   2021 onwards exactly once. If the edge only exists in the period used to
   pick the settings, it was probably fitted to noise.
3. **Luck.** Run the same strategy with the attention scores shuffled between
   stocks (so the signal is meaningless) hundreds of times. If the real result
   is no better than most shuffled ones, luck explains it.
4. **Concentration.** Which stocks made the money? If a handful (GameStop...)
   explain most of it, it is a story about those stocks, not a pattern.

Timing rules are the same as `analysis.py`: attention dated day D is acted on at
the close of the first trading day after D, and a portfolio picked at a close
earns returns from the next day.

Delisted stocks: a stock's price series stops when it is bought out or
delisted. A portfolio still holding it earns 0% on that money (as if paid out in
cash) until the next rebalance, when it is no longer eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analysis as A

COST_BPS = 10.0      # one-way cost per trade, in basis points (0.10%)
FRACTION = 0.10      # Buzz = top 10% of stocks by abnormal attention, Quiet = bottom 10%
LOOKBACK = 7         # days of attention averaged for the weekly score
MIN_VIEWS = 100      # ignore pages with fewer usual daily views (too noisy)
SPLIT = "2021-01-01" # settings may only be chosen on data before this date
TRADING_DAYS = 252


# ---------------------------------------------------------------- signal
def attention_scores(views: pd.DataFrame, trading_dates: pd.DatetimeIndex,
                     lookback: int = LOOKBACK, min_views: float = MIN_VIEWS,
                     window: int = A.WINDOW) -> pd.DataFrame:
    """Weekly score per stock: average log attention ratio over the last `lookback` days.

    A stock whose usual level (median of the previous `window` days) is below
    `min_views` gets no score that day, because 30 views vs a usual 10 is noise,
    not news. Only past data is used.
    """
    ratio = A.spike_ratio(views, window=window)
    usual = views.shift(1).rolling(window, min_periods=window // 2).median()
    ratio = ratio.where(usual >= min_views)
    return A.weekly_scores(ratio, trading_dates, lookback=lookback,
                           min_obs=min(4, lookback))


# ---------------------------------------------------------------- portfolios
@dataclass
class Backtest:
    """Daily results of one portfolio rule."""
    gross: pd.Series                 # daily return before costs
    net: pd.Series                   # daily return after costs
    turnover: pd.Series              # per rebalance day: share of the portfolio replaced (0 to 1)
    holdings: pd.Series              # per rebalance day: number of stocks held
    contrib: pd.Series = field(default_factory=pd.Series)  # per stock: summed daily contribution

    @property
    def annual_turnover(self) -> float:
        """How many times a year the whole portfolio is replaced (1.0 = once)."""
        years = len(self.gross) / TRADING_DAYS
        return float(self.turnover.sum() / years) if years else np.nan


def _select(scores: np.ndarray, side: str, frac: float, min_names: int) -> np.ndarray:
    """Boolean mask of chosen stocks from one row of scores (NaN = not eligible)."""
    ok = np.isfinite(scores)
    n = int(ok.sum())
    pick = np.zeros(len(scores), bool)
    if n < min_names:
        return pick
    if side == "all":
        return ok
    k = max(1, int(round(frac * n)))
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(scores[idx], kind="stable")]
    pick[order[-k:] if side == "top" else order[:k]] = True
    return pick


def run_portfolio(scores: pd.DataFrame, prices: pd.DataFrame, side: str = "top",
                  frac: float = FRACTION, cost_bps: float = COST_BPS,
                  min_names: int = 20, exclude: tuple[str, ...] = (),
                  rng: np.random.Generator | None = None) -> Backtest:
    """Equal-weight portfolio of the top / bottom `frac` of stocks by score, or all of them.

    side: "top" (Buzz), "bottom" (Quiet) or "all" (equal weight, the fair benchmark).
    Rebalanced on each row of `scores`. Weights drift with prices between
    rebalances, and the cost is charged on the actual change in weights.
    rng: if given, scores are shuffled between eligible stocks on every
    rebalance (the "luck" test).
    """
    cols = [c for c in prices.columns if c in scores.columns and c not in exclude]
    px = prices[cols]
    R = px.pct_change(fill_method=None).to_numpy(float)
    has_px = px.notna().to_numpy()
    dates = px.index
    pos = dates.get_indexer(scores.index)
    sc = scores[cols].to_numpy(float)
    keep = pos >= 0
    pos, sc = pos[keep], sc[keep]
    if len(pos) == 0:
        empty = pd.Series(dtype=float)
        return Backtest(empty, empty, empty, empty)

    T, N = R.shape
    gross = np.full(T, np.nan)
    net = np.full(T, np.nan)
    contrib = np.zeros(N)
    turn, held = [], []
    w = np.zeros(N)                          # current (drifted) weights
    c = cost_bps / 1e4
    for k, i in enumerate(pos):
        row = np.where(has_px[i], sc[k], np.nan)
        if rng is not None:
            ok = np.flatnonzero(np.isfinite(row))
            row[ok] = rng.permutation(row[ok])
        pick = _select(row, side, frac, min_names)
        if pick.any():
            new_w = pick / pick.sum()
        else:
            new_w = np.zeros(N)
        traded = np.abs(new_w - w).sum()
        if k == 0 and not pick.any():
            continue
        # Replacing stock A with stock B sells 10% and buys 10%: `traded` = 0.2,
        # the portfolio replaced = 0.1. Costs are paid on both legs.
        turn.append((dates[i], traded / 2))
        held.append((dates[i], int(pick.sum())))
        w = new_w
        end = pos[k + 1] if k + 1 < len(pos) else T - 1
        first = True
        for t in range(i + 1, end + 1):
            r = np.nan_to_num(R[t])           # missing price (delisted) = 0% (cash)
            day = float(w @ r)
            contrib += w * r
            gross[t] = day
            net[t] = day - (c * traded if first else 0.0)
            first = False
            grown = w * (1 + r)
            total = grown.sum()
            w = grown / total if total > 0 else w
    idx = pd.DatetimeIndex(dates)
    g = pd.Series(gross, idx).dropna()
    n = pd.Series(net, idx).dropna()
    to = pd.Series(dict(turn), dtype=float)
    hd = pd.Series(dict(held), dtype=float)
    return Backtest(g, n, to, hd, pd.Series(contrib, index=cols))


# ---------------------------------------------------------------- statistics
def edge_stats(a: pd.Series, b: pd.Series | None = None) -> dict:
    """Yearly average of a (minus b), its t-stat, and the share of years it was positive."""
    d = a if b is None else (a - b).dropna()
    s = A.summarise(d)
    yearly = d.groupby(d.index.year).sum()
    return {"edge": s["mean"] * TRADING_DAYS if s["n"] else np.nan, "t": s["t"], "days": s["n"],
            "years_positive": float((yearly > 0).mean()) if len(yearly) else np.nan}


def period_edges(a: pd.Series, b: pd.Series, split: str = SPLIT) -> dict:
    d = (a - b).dropna()
    return {"in_sample": edge_stats(d[d.index < split]),
            "out_of_sample": edge_stats(d[d.index >= split]),
            "all": edge_stats(d)}


def yearly_edges(a: pd.Series, b: pd.Series) -> pd.Series:
    """Sum of daily gaps per calendar year (roughly the yearly gap in returns)."""
    d = (a - b).dropna()
    return d.groupby(d.index.year).sum()


def placebo(scores: pd.DataFrame, prices: pd.DataFrame, bench_ew: pd.Series,
            runs: int = 200, frac: float = FRACTION, seed: int = 0) -> np.ndarray:
    """Yearly gross edge (Buzz minus equal weight) with shuffled scores, `runs` times."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        bt = run_portfolio(scores, prices, "top", frac, cost_bps=0.0, rng=rng)
        out.append(edge_stats(bt.gross, bench_ew)["edge"])
    return np.array(out)


def breakeven_cost_bps(gross_edge: float, annual_turnover_gap: float) -> float:
    """Cost per trade (bps) at which the yearly edge is fully eaten by extra trading.

    Replacing the portfolio once means selling everything and buying everything,
    so each unit of turnover pays the cost twice.
    """
    if not np.isfinite(gross_edge) or annual_turnover_gap <= 0:
        return np.nan
    return gross_edge / (2 * annual_turnover_gap) * 1e4


# ---------------------------------------------------------------- spike events
def spike_events(views: pd.DataFrame, prices: pd.DataFrame, bench: pd.Series,
                 threshold: float = 3.0, hold: int = A.HOLD, min_views: float = MIN_VIEWS,
                 window: int = A.WINDOW, cooldown: int = A.COOLDOWN) -> pd.DataFrame:
    """Every spike (attention >= threshold x usual) for every stock, with its outcome.

    Like `analysis.event_study`, plus: pages with few views are skipped, a spike
    only counts if the stock actually traded on the entry day, and if the stock
    stops trading during the holding period we use its last price (a buyout is
    paid in cash at roughly that price).
    """
    b = bench.ffill().to_numpy(float)
    filled = prices.ffill()
    rows = []
    for t in views.columns.intersection(prices.columns):
        v = views[t]
        ratio = A.spike_ratio(v, window=window)
        usual = v.shift(1).rolling(window, min_periods=window // 2).median()
        ratio = ratio.where(usual >= min_views)
        ev = A.find_spikes(ratio, prices.index, threshold, cooldown)
        if ev.empty:
            continue
        raw = prices[t].to_numpy(float)
        p = filled[t].to_numpy(float)
        for e, d, r, a in zip(ev["entry"], ev["entry_date"], ev["ratio"], ev["attention_date"]):
            if not np.isfinite(raw[e]):
                continue
            j = min(e + hold, len(p) - 1)
            if e + hold >= len(p):
                continue
            rows.append({"ticker": t, "attention_date": a, "entry_date": d, "ratio": r,
                         "excess": A._excess(p, b, e, j), "reaction": A._excess(p, b, e - 1, e)})
    return pd.DataFrame(rows, columns=["ticker", "attention_date", "entry_date", "ratio", "excess", "reaction"])


def clustered_summary(events: pd.DataFrame, col: str = "excess", cost_bps: float = 0.0) -> dict:
    """Average outcome with a t-stat that counts each *month* once.

    Spikes bunch up on market-wide news (March 2020: hundreds of pages spiked
    together), and those events all move together. Treating them as hundreds
    of independent bets overstates confidence. Averaging within each month
    first, then testing across months, is the simple fix.
    """
    if events.empty:
        return {"n": 0, "months": 0, "mean": np.nan, "t_naive": np.nan, "t_month": np.nan, "hit": np.nan}
    x = events[col] - 2 * cost_bps / 1e4      # round trip: buy then sell
    naive = A.summarise(x)
    monthly = x.groupby(events["entry_date"].dt.to_period("M")).mean()
    m = A.summarise(monthly)
    return {"n": naive["n"], "months": m["n"], "mean": naive["mean"], "hit": naive["hit"],
            "t_naive": naive["t"], "t_month": m["t"]}
