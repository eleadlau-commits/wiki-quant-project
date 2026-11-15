# A first trading strategy: the moving-average crossover

The attention study asked a hard question (does a new kind of data predict
returns?). This one asks an easy one on purpose: **can the oldest rule in the
book beat simply buying and holding the market?** It is a good first strategy
because every step is visible, and the ways it fails teach most of the traps
that matter later.

Code: [`src/wikiquant/strategies.py`](../src/wikiquant/strategies.py) (the rule),
[`src/wikiquant/backtest.py`](../src/wikiquant/backtest.py) (the engine),
[`scripts/ma_crossover.py`](../scripts/ma_crossover.py) (the experiment).
Results: [`reports/ma_crossover.md`](../reports/ma_crossover.md).

## 1. The rule

Each day, compute two averages of the closing price:

- the **fast** average of the last 50 days (where the price has been lately)
- the **slow** average of the last 200 days (the long-run level)

If fast > slow, hold the stock. Otherwise hold cash. That's all.

Why might this work? Prices trend: a market that has been rising for months
tends to keep rising for a while (this is *momentum*, one of the most studied
effects in finance). Big bear markets such as 2008 and 2022 were slow slides,
and a slow slide pulls the 50-day average under the 200-day one, so the rule
steps aside. The cost: it always reacts late, because averages lag the price.

Small example with a 3-day and 5-day average:

| Day | Price | 3-day avg | 5-day avg | Hold? |
| --- | --- | --- | --- | --- |
| 5 | 104 | 103.0 | 102.0 | yes (103 > 102) |
| 6 | 99 | 101.7 | 102.0 | no (101.7 < 102) |

## 2. The backtest, and the three ways to fool yourself

A backtest replays history as if you had followed the rule. Three things
decide whether it is honest:

1. **Look-ahead bias.** The averages use today's close, so the earliest you
   can act is at today's close, and you earn tomorrow's return. In code that is
   `positions.shift(1)`. Without it, you'd be "trading" on prices you couldn't
   have seen yet, and almost any rule looks brilliant. There is a test
   (`test_no_look_ahead`) that fails if this ever breaks. We also try trading
   one day *later* (`lag=2`), which is stricter.
2. **Costs.** Each trade loses a little to the bid-ask spread and slippage. We
   charge 10 basis points (0.10%) every time the position changes by 100%.
   That is generous for SPY (closer to 1-2 bps) and fair for big single stocks.
3. **A fair benchmark over the same days.** The 200-day average needs 200 days
   of history before the rule can start, so buy-and-hold starts on that same
   day too. Otherwise one of them gets a head start.

Two simplifications to remember when reading results: cash earns 0% (in
2023-25 it actually earned about 5%, so this is unfair to the crossover), and
prices are adjusted for dividends, so buy-and-hold includes them.

## 3. How to read the results

The report compares the two with the metrics from your Week 1 exercise:

- **CAGR**: average yearly growth. The headline number, but not the whole story.
- **Volatility** and **Sharpe**: how bumpy the ride was, and return per unit
  of bumpiness. A strategy with lower return but much lower risk can still win
  on Sharpe, because you could borrow to scale it up.
- **Max drawdown**: the worst fall from a peak. This is what makes real people
  sell at the bottom, so trend rules are mostly sold on reducing it.
- **Trades** and **costs per year**: how much the rule churns.

## 4. What we found (SPY, Dec 2019 to Nov 2026, 10 bps costs)

| | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- |
| Buy and hold | +14.8% | 0.79 | -33.7% |
| 50/200 crossover | +9.6% | 0.63 | -33.7% |

**The crossover lost, on return and on risk.** The reasons are the lessons:

- **Too slow for fast crashes.** The Covid crash of Feb-Mar 2020 took 23
  trading days. The rule was still invested on the way down, so it took the
  full -34%, then sold near the bottom and missed most of the rebound (2020:
  -4% vs +18%). April 2025 repeated this: sell after the fall, buy back higher.
  Every trade the rule made on SPY:

  | Date | Action | SPY price | Note |
  | --- | --- | --- | --- |
  | 31 Mar 2020 | sell | 236 | a week after the bottom, already down 23% |
  | 6 Jul 2020 | buy | 291 | 24% higher than where it sold |
  | 16 Mar 2022 | sell | 409 | early in the 2022 slide, down 9% |
  | 26 Jan 2023 | buy | 386 | lower than it sold: the one good round trip |
  | 16 Apr 2025 | sell | 517 | after the tariff crash, down 14% |
  | 27 Jun 2025 | buy | 607 | 17% higher, at a new all-time high |

- **It did help in a slow bear market.** In 2022 it sat in cash for 200 days
  and lost 8% while the market lost 18%. Trend rules protect against slides,
  not shocks.
- **Costs barely matter here.** Six trades in seven years means even 50 bps
  costs move CAGR by only 0.5%. The damage is all from bad timing, not fees.
- **The parameter grid is a warning.** One pair (10/100) shows a Sharpe of
  0.92, beating buy-and-hold. Its neighbours (20/100 = 0.70, 10/150 = 0.75)
  don't. If you had tested 16 pairs and reported only the best, you'd have
  "found" a strategy that is really just luck. This is *data snooping*, the
  same trap the attention study guarded against.
- **On single stocks it's an insurance policy.** It cut the worst drawdown on
  15 of 16 stocks (META: -40% instead of -77%), but raised the Sharpe on only
  5 of 16. You pay for the protection in lost return.

Honest summary for a CV or interview: *"I built a backtester with lagged
signals and transaction costs and tested the classic 50/200 crossover. Over
2020-26 it underperformed buy-and-hold on SPY (Sharpe 0.63 vs 0.79), because
the 2020 and 2025 crashes were too fast for a 200-day filter. It reduced
drawdowns on single stocks but at a large cost in return, and the best
parameter pair didn't survive a neighbourhood check."* A negative result,
explained, is worth more than a suspicious positive one.

Caveat: seven years with two sharp V-shaped crashes is one market regime.
Over 1990-2020 studies find the crossover roughly matches the market's
return with smaller drawdowns, largely thanks to 2001 and 2008. To test that
you'd need longer history (`load_prices(["SPY"], start="1993-01-01")` on your
own machine).

## 5. Try it yourself

1. Finish `src/wikiquant/metrics.py`, then run `python scripts/ma_crossover.py`.
   You should reproduce the numbers above exactly. If you don't, either your
   metrics or mine have a bug, which is worth finding.
2. Predict before you run: if cash earned 4% a year, how much would the
   crossover's CAGR rise? (Hint: it was in cash 19% of the time.) Then add a
   `cash_rate` argument to `run_backtest` and check.
3. Make the rule faster: hold only if the price is above its 100-day average
   (one average, not two). Does it handle 2020 better? What happens to trades
   and costs?
4. Why does the crossover have *lower* volatility but the *same* max drawdown
   as buy-and-hold on SPY? Explain it from the chart.
