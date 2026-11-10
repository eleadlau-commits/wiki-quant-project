"""Plugin 2: Hacker News stories whose title names the company.

Hacker News is a busy tech news forum. Its free search API (run by Algolia,
no key or sign-up) has every story since 2007:
https://hn.algolia.com/api
Each story has a UTC timestamp, so we count stories per UTC day.

This measures a different crowd from Wikipedia: engineers and tech investors
rather than the general public, so it is a useful contrast. It is sparse for
non-tech companies (Nike might get one story a week), which is why the
spike detector sums the last 7 days first (`smooth = 7`).

Search keys are names as they appear in headlines, matched as whole words and
case-sensitive: "Intel" does not match "intelligence", "Apple" does not match
"apple pie". If a company has two keys, a headline naming both counts twice.
"""

from __future__ import annotations

import re
import time
from typing import Callable

import pandas as pd

from .base import AttentionSource, register

API = "https://hn.algolia.com/api/v1/search_by_date"
PAGE_SIZE = 1000          # the API returns at most 1,000 hits per query
USER_AGENT = "wikiquant-research/0.1 (https://github.com/eleadlau-commits/wiki-quant-project)"

GetJson = Callable[[dict], dict]


def _http_get_json(params: dict) -> dict:
    import requests

    for attempt in range(5):
        resp = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)   # rate-limited or server hiccup: back off and retry
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def headline_matcher(key: str) -> re.Pattern:
    """Whole-word, case-sensitive match: 'Intel' but not 'Intelligence'."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])")


@register
class HackerNewsSource(AttentionSource):
    name = "hackernews"
    label = "Hacker News stories"
    unit = "stories"
    about = "Hacker News search API by Algolia (stories whose headline names the company, UTC days)"
    missing_is_zero = True
    smooth = 7

    def __init__(self, cache_dir=None, get_json: GetJson = _http_get_json, pause: float = 0.2):
        super().__init__(cache_dir)
        self.get_json = get_json
        self.pause = pause

    def fetch(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        match = headline_matcher(key)
        stamps: list[int] = []
        # Work month by month so each request stays under the 1,000-hit cap.
        edges = pd.date_range(start.to_period("M").to_timestamp(), end + pd.Timedelta(days=1), freq="MS")
        edges = sorted({start, *[e for e in edges if start < e <= end], end + pd.Timedelta(days=1)})
        for a, b in zip(edges[:-1], edges[1:]):
            stamps += self._window(key, match, int(a.timestamp()), int(b.timestamp()))
        if not stamps:
            return pd.Series(dtype=float)
        days = pd.to_datetime(stamps, unit="s", utc=True).tz_localize(None).normalize()
        return pd.Series(1.0, index=days).groupby(level=0).sum()

    def _window(self, key: str, match: re.Pattern, lo: int, hi: int) -> list[int]:
        """Timestamps of matching stories in [lo, hi), splitting busy windows."""
        params = {
            "query": key,
            "tags": "story",
            "restrictSearchableAttributes": "title",
            "typoTolerance": "false",
            "hitsPerPage": PAGE_SIZE,
            "attributesToRetrieve": "title,created_at_i",
            "attributesToHighlight": "",
            "numericFilters": f"created_at_i>={lo},created_at_i<{hi}",
        }
        hits, page, total = [], 0, 0
        while True:
            data = self.get_json({**params, "page": page})
            if self.pause:
                time.sleep(self.pause)
            hits += data.get("hits", [])
            total = data.get("nbHits", len(hits))
            page += 1
            if page >= data.get("nbPages", 1) or len(hits) >= PAGE_SIZE:
                break
        if total > len(hits) and hi - lo > 86400:
            mid = lo + (hi - lo) // 2   # too many to list in one go: split in half
            return self._window(key, match, lo, mid) + self._window(key, match, mid, hi)
        return [h["created_at_i"] for h in hits
                if lo <= h.get("created_at_i", -1) < hi and match.search(h.get("title") or "")]
