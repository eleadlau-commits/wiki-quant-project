# Does public attention predict stock returns?

A Wikipedia page-view signal for ~100 large US stocks (2015 onwards), tested
with a backtesting engine built from scratch, out-of-sample testing and
realistic transaction costs.

**Status:** Week 1 of 4: data and metrics. A first interactive dashboard,
**Attention Tape**, is live (see below).

## Attention Tape dashboard

One self-contained web page (`dashboard/index.html`) that answers the core
question with real data for 16 large US stocks since 2019:

1. **Explore:** share price against daily Wikipedia views, with spikes marked
   (views at least N times the median of the previous 60 days; you choose N).
2. **The test:** the average return *minus the S&P 500* in the days around a
   spike, a per-company league table, and a t-stat so noise is not mistaken for signal.
3. **Strategy:** a weekly "Buzz 4" portfolio of the most-read companies vs the
   least-read, an equal-weight basket and the S&P 500, plus leave-one-out and
   year-by-year checks.
4. **Replay:** scroll through time and watch attention move with the market.

**First findings (Nov 2026).** After a spike, most companies' next-20-day
returns are within noise (|t| < 2). The Buzz 4 beats equal weight by about
21% a year, but removing GameStop alone cuts that to about 8%, and most of
it comes from 2020-21. So: an interesting lead, not a reliable signal yet.

**How the data stays fresh.** `.github/workflows/dashboard-data.yml` runs every
weekday after the US close: it calls the free Wikimedia Pageviews API and Yahoo
Finance, writes `dashboard/data.json`, and rebuilds `dashboard/index.html`
from `dashboard/template.html`. To rebuild locally:
`python scripts/build_dashboard_data.py`.

Not included yet: trading costs, a larger universe (the 16 names are famous
today, which flatters the results), and out-of-sample testing. Those are Weeks 2-4.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes `import wikiquant` work everywhere
pytest                           # loader tests pass; metric tests fail until you write them
```

## Layout

| Path | What lives there |
| --- | --- |
| `src/wikiquant/data.py` | Price downloader with an on-disk cache |
| `src/wikiquant/pageviews.py` | Wikipedia page-view downloader and abnormal-attention measure |
| `dashboard/` | Attention Tape page, its template and data |
| `src/wikiquant/metrics.py` | Returns, volatility, Sharpe, drawdown (your Week 1 exercise) |
| `tests/` | Unit tests, run with `pytest` |
| `notebooks/` | Exploration and charts |
| `data/cache/` | Downloaded data (git-ignored) |

## Quick start

```python
from wikiquant.data import load_prices

prices = load_prices(["AAPL", "MSFT", "SPY"])
prices.tail()
```

## Plan

1. **Data and basics:** price loader, returns and risk metrics.
2. **Backtester:** signal in, weights out, returns out, with costs and a no-look-ahead test. Validated on 12-1 momentum.
3. **Attention signal:** abnormal Wikipedia page views as a weekly signal.
4. **Break it:** out-of-sample, walk-forward, cost stress tests, then the write-up.
