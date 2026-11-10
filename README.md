# Does public attention predict stock returns?

A Wikipedia page-view signal for ~100 large US stocks (2015 onwards), tested
with a backtesting engine built from scratch, out-of-sample testing and
realistic transaction costs.

**Status:** Week 1 of 4: data and metrics. A first interactive dashboard,
**Attention Tape**, is live (see below).

## Plug-in attention sources

Wikipedia is just the first source. Any attention data (Hacker News, news
volume, Google Trends, ...) can be plugged in as a small class, and every
source then goes through **the same** spike, forward-return and strategy tests,
with one look-ahead rule enforced in one place. Two sources are plugged in:
Wikipedia page views and Hacker News stories, both free with no sign-up.

```bash
python scripts/compare_sources.py      # offline, writes reports/attention_comparison.md
```

The latest comparison is in [`reports/attention_comparison.md`](reports/attention_comparison.md)
(refreshed every weekday). An honest evaluation of the idea, the pitfalls, and
how to add your own source: [`docs/attention-framework.md`](docs/attention-framework.md).

## The broad test: ~500 stocks, after costs

The 16-stock dashboard flatters any signal (those companies are famous *because*
they did well). The broad test fixes that:

- **Universe picked in advance:** every company in Wikipedia's *List of S&P 500
  companies* exactly as it stood on 1 Jan 2016 (read from the page history),
  including companies later bought out, delisted or dropped from the index.
  Wikipedia article titles come from that list's own links.
- **Trading costs:** 0.10% per trade each way, charged on the real change in weights.
- **Out of sample:** settings may only be chosen on 2016-2020; 2021 onwards is the check.
- **Luck test:** the same strategy with attention scores shuffled between stocks, 200 times.
- **Month-clustered t-stats** for spike events, since spikes bunch on market-wide news.

```bash
python scripts/build_broad_data.py   # needs internet; the GitHub Action does this
python scripts/broad_study.py        # offline from data/broad/ -> reports/broad_study.md
```

Results: [`reports/broad_study.md`](reports/broad_study.md) and the page
`dashboard/broad.html` (linked from the main dashboard). Rebuilt by
`.github/workflows/broad-study.yml`. Companies Yahoo no longer has prices for
are listed in `data/broad/coverage.csv`: free data cannot fully remove
survivorship bias, but it can measure how much is left.

## Attention Tape dashboard

One self-contained web page (`dashboard/index.html`) that answers the core
question with real data for 16 large US stocks since 2019. A switch at the top
chooses the attention source (Wikipedia or Hacker News):

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
22% a year, but removing GameStop alone cuts that to about 9%, and most of
it comes from 2020-21. So: an interesting lead, not a reliable signal yet.

**How the data stays fresh.** `.github/workflows/dashboard-data.yml` runs every
weekday after the US close: it calls every attention source and Yahoo
Finance, writes `dashboard/data.json`, runs the source comparison, and rebuilds `dashboard/index.html`
from `dashboard/template.html`. To rebuild locally:
`python scripts/build_dashboard_data.py`.

Trading costs, a larger universe and out-of-sample testing are in the broad test above.

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
| `src/wikiquant/sources/` | Attention plugins (Wikipedia, Hacker News) behind one interface |
| `src/wikiquant/universe.py` | The companies and their search keys per source |
| `src/wikiquant/analysis.py` | Source-agnostic tests: spikes, event study, Buzz 4, overlap |
| `src/wikiquant/dataset.py` | Load `dashboard/data.json` into pandas, offline |
| `src/wikiquant/broad_universe.py` | S&P 500 as listed on Wikipedia on a past date |
| `src/wikiquant/backtest.py` | Costs, turnover, out-of-sample split, luck test, clustered t-stats |
| `src/wikiquant/broad_dataset.py` | Load `data/broad/` into pandas, offline |
| `data/broad/` | Broad dataset (~500 stocks, prices + views), built by the Action |
| `reports/` | Generated comparison of sources |
| `docs/` | Framework guide and evaluation |
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
