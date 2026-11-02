"""Download daily stock prices and cache them on disk.

The first call for a ticker downloads from Yahoo Finance (via yfinance) and
saves a CSV under data/cache/. Later calls read the CSV, so they are fast and
work offline. Delete the cache folder, or pass refresh=True, to re-download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "prices"
DEFAULT_START = "2015-01-01"

# A fetcher takes (ticker, start, end) and returns a Series of adjusted close
# prices indexed by date. Passing a fake fetcher lets tests run without internet.
Fetcher = Callable[[str, str, str | None], pd.Series]


def _yfinance_fetcher(ticker: str, start: str, end: str | None) -> pd.Series:
    import yfinance as yf

    # auto_adjust=True corrects prices for splits and dividends, so a 2-for-1
    # split does not look like a 50% crash.
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.rename(ticker)


def load_prices(
    tickers: Iterable[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    fetcher: Fetcher = _yfinance_fetcher,
) -> pd.DataFrame:
    """Return adjusted close prices: one row per trading day, one column per ticker.

    Example:
        prices = load_prices(["AAPL", "MSFT"])
        prices.tail()
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    columns = []
    for ticker in tickers:
        path = cache_dir / f"{ticker}.csv"
        if path.exists() and not refresh:
            series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            series = fetcher(ticker, start, end)
            series.to_frame(ticker).to_csv(path)
        columns.append(series.rename(ticker))

    prices = pd.concat(columns, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Trim to the requested window, since the cache may hold a wider range.
    prices = prices.loc[start:end] if end else prices.loc[start:]
    return prices
