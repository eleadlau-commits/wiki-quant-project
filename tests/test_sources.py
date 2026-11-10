"""Tests for the plug-in source interface and the two built-in sources (offline)."""

import re

import pandas as pd
import pytest

from wikiquant.sources import AttentionSource, available_sources, get_source, register
from wikiquant.sources.hackernews import HackerNewsSource, headline_matcher


def _window(params):
    return map(int, re.findall(r"\d+", params["numericFilters"]))


class CountingSource(AttentionSource):
    """A fake source: attention on day D = day of month. Records each download."""

    name = "counting-test"
    missing_is_zero = True

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.calls = []

    def fetch(self, key, start, end):
        self.calls.append((key, start.date().isoformat(), end.date().isoformat()))
        days = pd.date_range(start, end, freq="D")
        return pd.Series([float(d.day) for d in days], index=days)


def test_builtin_sources_are_registered():
    assert {"wikipedia", "hackernews"} <= set(available_sources())
    assert get_source("wikipedia").unit == "views"
    with pytest.raises(KeyError):
        get_source("nope")


def test_register_makes_a_new_source_available(tmp_path):
    @register
    class Tiny(CountingSource):
        name = "tiny-test"

    assert "tiny-test" in available_sources()
    assert get_source("tiny-test", cache_dir=tmp_path).name == "tiny-test"


def test_load_caches_then_only_tops_up_recent_days(tmp_path):
    src = CountingSource(tmp_path)
    a = src.load("X", "2024-01-01", "2024-01-31")
    assert src.calls == [("X", "2024-01-01", "2024-01-31")]
    assert len(a) == 31 and a.iloc[-1] == 31

    src.load("X", "2024-01-01", "2024-01-31")      # fully cached: no download
    assert len(src.calls) == 1

    b = src.load("X", "2024-01-01", "2024-02-10")  # re-download the last 7 cached days onwards
    assert src.calls[-1] == ("X", "2024-01-24", "2024-02-10")
    assert len(b) == 41 and b.index.is_unique and b.loc["2024-02-10"] == 10


def test_load_sums_several_keys_and_fills_gaps(tmp_path):
    class Gappy(CountingSource):
        def fetch(self, key, start, end):
            return pd.Series({pd.Timestamp("2024-01-02"): 5.0})

    src = Gappy(tmp_path)
    s = src.load(["A", "B"], "2024-01-01", "2024-01-03")
    assert list(s) == [0.0, 10.0, 0.0]     # missing days count as zero; keys are added


def test_headline_matcher_is_whole_word_and_case_sensitive():
    m = headline_matcher("Intel")
    assert m.search("Intel cuts 15% of staff")
    assert m.search("Why Intel's foundry matters")
    assert not m.search("Artificial Intelligence is here")
    assert not m.search("intel on the enemy")
    assert headline_matcher("JP Morgan").search("JP Morgan buys a bank")


def test_hackernews_counts_matching_stories_per_utc_day(tmp_path):
    def ts(s):
        return int(pd.Timestamp(s, tz="UTC").timestamp())

    stories = [
        ("Nvidia beats estimates", ts("2024-03-01 10:00")),
        ("Nvidia stock falls", ts("2024-03-01 23:30")),            # still 1 March in UTC
        ("Ask HN: Nvidia drivers on Linux?", ts("2024-03-02 01:00")),
        ("Nvidiaesque startup", ts("2024-03-02 02:00")),           # not a whole word
    ]
    seen = []

    def fake_get_json(params):
        seen.append(params)
        lo, hi = _window(params)
        hits = [{"title": t, "created_at_i": c} for t, c in stories if lo <= c < hi]
        return {"hits": hits, "nbHits": len(hits), "nbPages": 1}

    src = HackerNewsSource(tmp_path, get_json=fake_get_json, pause=0)
    s = src.load("Nvidia", "2024-02-28", "2024-03-03")
    assert s.loc["2024-03-01"] == 2 and s.loc["2024-03-02"] == 1 and s.loc["2024-02-28"] == 0
    assert len(seen) == 2                                         # one request per month
    assert all(p["tags"] == "story" for p in seen)


def test_hackernews_splits_windows_that_hit_the_cap(tmp_path):
    calls = []

    def fake_get_json(params):
        lo, hi = _window(params)
        calls.append((lo, hi))
        n = (hi - lo) // 1800                                     # a story every 30 minutes
        shown = min(n, 1000)
        hits = [{"title": "Apple news", "created_at_i": lo + 1800 * i} for i in range(shown)]
        return {"hits": hits, "nbHits": n, "nbPages": 1}

    src = HackerNewsSource(tmp_path, get_json=fake_get_json, pause=0)
    s = src.load("Apple", "2024-01-01", "2024-01-31")
    assert (s == 48).all()                                         # nothing lost to the cap
    assert len(calls) > 1
