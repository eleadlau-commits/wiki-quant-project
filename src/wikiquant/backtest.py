"""A small, honest backtester: target positions in, daily strategy returns out.

The whole engine is three ideas, each one line of pandas:

1. **Lag the position.** A signal computed from today's close can only be
   traded at (or after) today's close, so it earns *tomorrow's* return.
   `positions.shift(lag)` enforces that. Forgetting it is look-ahead bias,
   the most common bug in beginner backtests.
2. **Charge for trading.** Every change in position costs money (bid-ask
   spread, slippage, commission). We charge `cost_bps` basis points on the
   size of the change: going 0% -> 100% invested costs 1.0 x cost, a full
   round trip (in and out) costs 2.0 x cost.
3. **Compound.** Strategy return each day = position x asset return - cost.
   The equity curve is (1 + returns).cumprod().

Worked example, 10 bps cost, position decided at each close:

    day   price  signal  position held  asset ret  trade  cost    strategy ret
    1     100    1       0 (not yet)    -          -      -       -
    2     102    1       1              +2.0%      1.0    0.10%   +1.90%
    3     101    0       1              -0.98%     0.0    0       -0.98%
    4     99     0       0              -1.98%     1.0    0.10%   -0.10%

On day 2 we hold because of day 1's signal and pay to get in. On day 4 we are
out (day 3's signal) so the fall does not hurt us, but we pay to get out.

Assumptions to keep in mind (all deliberately simple):
- You trade at the closing price. Real fills differ a little.
- Uninvested cash earns 0%. In 2023-25 cash paid ~5%, so this is unfair
  to strategies that spend time in cash.
- No leverage, no shorting unless positions go negative, no taxes.
"""

from __future__ import annotations

import pandas as pd


def run_backtest(
    prices: pd.DataFrame | pd.Series,
    positions: pd.DataFrame | pd.Series,
    cost_bps: float = 10.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Daily results of holding `positions` (fraction of capital, decided at each close).

    Args:
        prices: adjusted close prices (dates x tickers, or one Series).
        positions: target weight per date and ticker, same shape as prices.
            1 = fully invested, 0 = cash, -1 = short. Weights are decided
            using information up to that day's close.
        cost_bps: cost per unit of turnover, in basis points (10 bps = 0.10%).
        lag: days between deciding and earning. 1 = trade at the same close
            you observed; 2 = trade one day later (a stricter test).

    Returns:
        DataFrame with columns
          asset_ret    return of the asset(s), equally weighted if several
          position     weight actually held during each day
          turnover     how much the position changed (what we pay costs on)
          cost         cost paid that day, as a return
          gross_ret    strategy return before costs
          net_ret      strategy return after costs
          equity       growth of 1 after costs
    """
    asset_ret = prices.pct_change()
    held = positions.shift(lag).fillna(0.0)
    # Turnover is the change in what we hold. The first day counts as buying in.
    turnover = held.diff().abs()
    turnover.iloc[0] = held.iloc[0].abs() if isinstance(held, pd.DataFrame) else abs(held.iloc[0])
    cost = turnover * cost_bps / 10_000

    gross = held * asset_ret.fillna(0.0)
    net = gross - cost

    if isinstance(prices, pd.DataFrame):
        # Several assets: split capital equally between them (each column is
        # one sleeve), so the totals are simple averages across columns.
        n = prices.shape[1]
        out = pd.DataFrame({
            "asset_ret": asset_ret.mean(axis=1),
            "position": held.sum(axis=1) / n,
            "turnover": turnover.sum(axis=1) / n,
            "cost": cost.sum(axis=1) / n,
            "gross_ret": gross.sum(axis=1) / n,
            "net_ret": net.sum(axis=1) / n,
        })
    else:
        out = pd.DataFrame({
            "asset_ret": asset_ret, "position": held, "turnover": turnover,
            "cost": cost, "gross_ret": gross, "net_ret": net,
        })

    out = out.iloc[1:]  # day one has no return
    out["equity"] = (1 + out["net_ret"]).cumprod()
    return out


def count_trades(positions: pd.Series) -> int:
    """Number of times the position changes (each entry or exit is one trade)."""
    return int((positions.diff().fillna(0) != 0).sum())
