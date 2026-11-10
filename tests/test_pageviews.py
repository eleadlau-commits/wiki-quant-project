"""Tests for the page-view loader and the abnormal-attention measure (offline)."""

import pandas as pd

from wikiquant.pageviews import abnormal_attention, load_pageviews


def test_abnormal_attention_flags_a_spike_without_look_ahead():
    views = pd.Series([100.0] * 70 + [500.0] + [100.0] * 5,
                      index=pd.date_range("2024-01-01", periods=76))
    ratio = abnormal_attention(views, window=60)
    assert ratio.iloc[70] == 5.0          # five times the usual level
    assert ratio.iloc[69] == 1.0          # the day before is unaffected by the spike
    assert ratio.iloc[:30].isna().all()   # not enough history yet


def test_load_pageviews_uses_cache(tmp_path):
    calls = []

    def fake(title, start, end):
        calls.append(title)
        return pd.Series([1.0, 2.0, 3.0], index=pd.date_range(start, periods=3), name=title)

    a = load_pageviews("Foo", "2024-01-01", "2024-01-03", cache_dir=tmp_path, fetcher=fake)
    b = load_pageviews("Foo", "2024-01-01", "2024-01-03", cache_dir=tmp_path, fetcher=fake)
    assert calls == ["Foo"]
    assert list(b) == list(a) == [1.0, 2.0, 3.0]
