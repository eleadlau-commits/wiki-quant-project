"""The common interface every attention source implements.

A source only has to answer one question: "for this search key, how much
attention was there on each calendar day between start and end?" That is the
`fetch` method. Everything else (caching, topping up the cache each day,
combining several keys for one company, filling gaps) lives here once, so a
new source is usually 30 to 60 lines.

Timing rule, which keeps the whole project free of look-ahead bias:
a value dated day D must describe attention *during* day D (any time zone that
ends at or after the US close, such as UTC). The analysis then only lets you
trade from the close of the first trading day strictly after D.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "attention"

_REGISTRY: dict[str, type["AttentionSource"]] = {}


class AttentionSource(ABC):
    """Base class for a daily attention data source.

    Subclasses set the class attributes below and implement `fetch`.
    """

    #: short id used in code, file names and data.json, e.g. "wikipedia"
    name: str = ""
    #: human-readable name, e.g. "Wikipedia page views"
    label: str = ""
    #: what one unit is, e.g. "views" or "stories"
    unit: str = ""
    #: one sentence on where the data comes from (shown on the dashboard)
    about: str = ""
    #: True if a day with no data means zero attention (counts of posts).
    #: False if it means "unknown" (e.g. the Wikipedia article did not exist yet).
    missing_is_zero: bool = True
    #: Sum this many trailing days before looking for spikes. Use > 1 for
    #: sparse sources where most days are 0, 1 or 2.
    smooth: int = 1
    #: When topping up the cache, re-download this many recent days, because
    #: the last few days are often revised.
    refetch_days: int = 7

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR / self.name

    # ------------------------------------------------------------------ to implement
    @abstractmethod
    def fetch(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Download daily attention for one key, start and end inclusive.

        Return a float Series indexed by calendar date (midnight timestamps).
        Days may be missing; `load` fills them according to `missing_is_zero`.
        """

    def postprocess(self, series: pd.Series) -> pd.Series:
        """Optional clean-up applied to a company's combined series."""
        return series

    # ------------------------------------------------------------------ provided
    def load(
        self,
        keys: str | Iterable[str],
        start: str,
        end: str,
        refresh: bool = False,
    ) -> pd.Series:
        """Daily attention for one company, summed over all its keys.

        Downloads are cached in data/cache/attention/<source>/. A later call
        only downloads the days the cache does not have yet (plus the last
        `refetch_days`). Pass refresh=True to re-download everything.
        """
        keys = [keys] if isinstance(keys, str) else list(keys)
        start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        parts = [self._load_one(k, start_ts, end_ts, refresh) for k in keys]
        combined = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        combined = self.postprocess(combined.loc[start_ts:end_ts])
        return combined.rename("+".join(keys))

    def _cache_path(self, key: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_") or "key"
        return self.cache_dir / f"{slug}.csv"

    def _download(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        raw = self.fetch(key, start, end)
        raw = pd.Series(raw, dtype=float)
        raw.index = pd.to_datetime(raw.index).normalize()
        raw = raw.groupby(level=0).sum(min_count=1)
        days = pd.date_range(start, end, freq="D")
        out = raw.reindex(days)
        return out.fillna(0.0) if self.missing_is_zero else out

    def _load_one(self, key: str, start: pd.Timestamp, end: pd.Timestamp, refresh: bool) -> pd.Series:
        path = self._cache_path(key)
        cached = None
        if path.exists() and not refresh:
            cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)

        if cached is None or cached.empty or cached.index.min() > start:
            series = self._download(key, start, end)
        elif cached.index.max() < end:
            top_up_from = max(start, cached.index.max() - pd.Timedelta(days=self.refetch_days))
            fresh = self._download(key, top_up_from, end)
            series = pd.concat([cached[cached.index < top_up_from], fresh])
        else:
            series = cached

        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename(key).to_frame().to_csv(path)
        return series.loc[start:end]

    def describe(self) -> dict:
        """Metadata written into data.json for the dashboard."""
        return {"id": self.name, "label": self.label, "unit": self.unit,
                "about": self.about, "smooth": self.smooth}


def register(cls: type[AttentionSource]) -> type[AttentionSource]:
    """Class decorator: make a source available by name."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a `name`")
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str, **kwargs) -> AttentionSource:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Available: {available_sources()}") from None


def available_sources() -> list[str]:
    return sorted(_REGISTRY)
