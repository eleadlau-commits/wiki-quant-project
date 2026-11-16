# Can a simple rule dodge crashes?

The 50/200 crossover failed in 2020 and 2025 because those crashes took only
23 and 34 trading days, and a 200-day average needs months to turn. This study
asks whether faster rules do better, and tests them fairly enough to trust the answer.

Code: `vol_target` and `big_drop_exit` in [`src/wikiquant/strategies.py`](../src/wikiquant/strategies.py),
the study in [`scripts/crash_defence.py`](../scripts/crash_defence.py), data from
[`scripts/fetch_long_history.py`](../scripts/fetch_long_history.py).
Full results: [`reports/crash_defence.md`](../reports/crash_defence.md).

## The two ideas

- **Volatility targeting.** Hold `target / recent volatility` of the fund, at
  most 100%. When the market has been swinging 30% a year and the target is
  10%, hold a third. Crashes nearly always start with a jump in volatility,
  so the position shrinks *during* the fall.
- **Big-drop exit.** Sell after an unusually large fall over 1-3 days (3-6%),
  and buy back when the price closes above a short average. Ordinary wobbles
  never trigger it.

## How the test avoids fooling itself

1. **Long history:** SPY from 1993, with seven big falls, some slow
   (dot-com, 2008, 2022) and some fast (2020, 2025).
2. **Choose, then judge:** 60 settings were tried. The best of each family was
   picked using **1994-2009 only**, then judged on **2010-2026**, which the
   choice never saw. That is the same as choosing in 2009 and living with it.
3. **The whole family, not just the winner:** we also report how *all* the
   settings did out of sample. An idea that only works with one lucky setting
   isn't an idea.
4. **Realistic frictions:** 10 bps per trade, signals act the next day (and a
   test with one more day of delay), and cash earns the T-bill rate. Earning
   that rate matters: in 2023-25 cash paid about 5%.

## Results (2010-2026, out of sample)

| Rule | CAGR | Sharpe | Worst fall |
| --- | --- | --- | --- |
| Buy and hold | 14.0% | 0.85 | -34% |
| 50/200 crossover | 9.1% | 0.67 | -34% |
| Volatility targeting | 9.1% | 0.87 | -17% |
| Big-drop exit | 13.8% | 0.93 | -24% |

**Volatility targeting is the robust one.** 23 of its 24 settings beat
buy-and-hold's Sharpe out of sample, and every one of them had a smaller worst
fall. It cushioned every crash on record, fast and slow (2008: -22% vs -55%;
2020: -17% vs -34%). The price: it is only 73% invested on average, so it
earns less in total. It is a *risk reducer*, not a return booster.

**Big-drop exit is the fast-crash specialist, and it's fragile.** Out of
sample it kept nearly all of the market's return while cutting 2020 to -12%.
But that success rests on two events (2020 and 2025). In the training period
it barely helped (Sharpe 0.52 vs 0.46) and did nothing in slow crashes
(dot-com -50%, 2008 -41%), because a slow slide never has a big enough day.

**The crossover's story depends on the period.** In 1994-2009 it was
excellent: Sharpe 0.89 vs 0.46, worst fall -19% vs -55%, because both crashes
were slow. In 2010-2026 it lost. The lesson from the first backtest still
holds, but seven years of data were not enough to judge it.

**Keep the differences in proportion.** Over 16 years, a Sharpe ratio is only
known to about ±0.25, so 0.87 vs 0.85 is a tie. The clear, repeatable effect
is on the **size of the falls**, not on returns.

## Questions to think about

1. No single rule handles both slow and fast crashes. What would you expect
   from combining the crossover (slow) with the big-drop exit (fast)? Why is
   testing that combination riskier, statistically, than testing each alone?
2. Volatility targeting at 10% holds about 73% on average. If you scaled it up
   so it averaged 100% invested, what would happen to its CAGR and its worst fall?
3. Why does a high T-bill rate (like 2023-25) make every "sometimes in cash"
   strategy look better? Look for it in the year-by-year numbers.
