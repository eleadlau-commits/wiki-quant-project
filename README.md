# Does public attention predict stock returns?

A Wikipedia page-view signal for ~100 large US stocks (2015 onwards), tested
with a backtesting engine built from scratch, out-of-sample testing and
realistic transaction costs.

**Status:** Week 1 of 4: data and metrics.

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
