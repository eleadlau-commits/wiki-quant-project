"""Plugin 1: daily human page views of an English Wikipedia article.

Free Wikimedia Pageviews API, no key or sign-up:
https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
Days are UTC, so a day's count includes the hours after the US close.
Search keys are article titles, e.g. "Apple_Inc.".
"""

from __future__ import annotations

import pandas as pd

from .base import AttentionSource, register


@register
class WikipediaSource(AttentionSource):
    name = "wikipedia"
    label = "Wikipedia page views"
    unit = "views"
    about = "Wikimedia Pageviews API (en.wikipedia, human readers, UTC days)"
    missing_is_zero = False   # no row = the article did not exist yet
    smooth = 1

    def fetch(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        import requests

        from ..pageviews import _api_fetcher

        try:
            return _api_fetcher(key, start, end)
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                return pd.Series(dtype=float)   # no views in this range
            raise

    def postprocess(self, series: pd.Series) -> pd.Series:
        # Before an article existed under any listed title (e.g. Meta before
        # "Facebook, Inc." was split out), counts are near zero. Treat that
        # stretch as missing rather than as a sudden attention explosion later.
        real = series >= 0.1 * series.median()
        if real.any():
            series = series.where(series.index >= real.idxmax())
        return series
