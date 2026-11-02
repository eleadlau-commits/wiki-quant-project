"""Tests for the price loader. These use a fake fetcher, so no internet needed."""

import pandas as pd

from wikiquant.data import load_prices


def make_fake_fetcher(calls):
    def fake_fetcher(ticker, start, end):
        calls.append(ticker)
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        return pd.Series(range(1, 6), index=dates, dtype=float, name=ticker)

    return fake_fetcher


def test_returns_one_column_per_ticker(tmp_path):
    calls = []
    prices = load_prices(["AAA", "BBB"], start="2020-01-01", cache_dir=tmp_path,
                         fetcher=make_fake_fetcher(calls))
    assert list(prices.columns) == ["AAA", "BBB"]
    assert len(prices) == 5
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_second_call_uses_cache(tmp_path):
    calls = []
    fetcher = make_fake_fetcher(calls)
    first = load_prices(["AAA"], start="2020-01-01", cache_dir=tmp_path, fetcher=fetcher)
    second = load_prices(["AAA"], start="2020-01-01", cache_dir=tmp_path, fetcher=fetcher)
    assert calls == ["AAA"]  # downloaded once only
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_refresh_redownloads(tmp_path):
    calls = []
    fetcher = make_fake_fetcher(calls)
    load_prices(["AAA"], start="2020-01-01", cache_dir=tmp_path, fetcher=fetcher)
    load_prices(["AAA"], start="2020-01-01", cache_dir=tmp_path, fetcher=fetcher, refresh=True)
    assert calls == ["AAA", "AAA"]


def test_trims_to_requested_window(tmp_path):
    prices = load_prices(["AAA"], start="2020-01-03", cache_dir=tmp_path,
                         fetcher=make_fake_fetcher([]))
    assert prices.index.min() == pd.Timestamp("2020-01-03")
