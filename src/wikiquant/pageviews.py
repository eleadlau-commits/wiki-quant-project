"""Download daily Wikipedia page views and cache them on disk.

Data comes from the free Wikimedia Pageviews API (no key or sign-up needed):
https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews

We count only human traffic ("user" agent), across desktop and mobile.
Daily counts are in UTC, so a day's views include hours after the US market
closes. Any trading signal must therefore act on the *next* trading day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "pageviews"
API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/daily/{start}/{end}"
)
# Wikimedia asks automated clients to identify themselves.
USER_AGENT = "wikiquant-research/0.1 (https://github.com/eleadlau-commits/wiki-quant-project)"

Fetcher = Callable[[str, str, str], pd.Series]


def _api_fetcher(title: str, start: str, end: str) -> pd.Series:
    import requests

    url = API.format(
        title=requests.utils.quote(title, safe=""),
        start=pd.Timestamp(start).strftime("%Y%m%d"),
        end=pd.Timestamp(end).strftime("%Y%m%d"),
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    items = resp.json()["items"]
    dates = pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d")
    return pd.Series([i["views"] for i in items], index=dates, name=title, dtype=float)


def load_pageviews(
    title: str,
    start: str,
    end: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    fetcher: Fetcher = _api_fetcher,
) -> pd.Series:
    """Daily human page views for one English Wikipedia article."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{title.replace('/', '_')}.csv"
    if path.exists() and not refresh:
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        series = fetcher(title, start, end)
        series.to_frame(title).to_csv(path)
    series.index = pd.to_datetime(series.index)
    return series.loc[start:end].rename(title)


def abnormal_attention(views: pd.Series, window: int = 60) -> pd.Series:
    """Today's views divided by the median of the previous `window` days.

    1.0 means a normal day, 3.0 means three times the usual attention.
    The median only uses past days, so there is no look-ahead.
    """
    baseline = views.shift(1).rolling(window, min_periods=window // 2).median()
    return views / baseline
