"""Load dashboard/data.json (prices + every source's attention) into pandas.

data.json is rebuilt every weekday by the GitHub Action, so this gives you all
the data offline, with no API calls:

    from wikiquant.dataset import load_dataset
    ds = load_dataset()
    ds.prices.tail()                  # trading days x tickers
    ds.attention["wikipedia"].tail()  # calendar days x tickers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "data.json"


@dataclass
class Dataset:
    prices: pd.DataFrame
    bench: pd.Series
    attention: dict[str, pd.DataFrame]
    sources: dict[str, dict]
    names: dict[str, str] = field(default_factory=dict)
    built: str = ""


def load_dataset(path: Path | str = DEFAULT_PATH) -> Dataset:
    d = json.loads(Path(path).read_text())
    dates = pd.to_datetime(d["dates"])
    prices = pd.DataFrame({s["ticker"]: s["close"] for s in d["stocks"]}, index=dates, dtype=float)
    bench = pd.Series(d["index"]["close"], index=dates, dtype=float, name=d["index"]["ticker"])
    sources = {m["id"]: m for m in d["attention"]}
    start = pd.Timestamp(d["viewsStart"])
    attention = {}
    for sid in sources:
        cols = {s["ticker"]: s["attention"][sid] for s in d["stocks"] if sid in s["attention"]}
        n = max(len(v) for v in cols.values())
        cal = pd.date_range(start, periods=n, freq="D")
        attention[sid] = pd.DataFrame({t: pd.Series(v, index=cal[:len(v)], dtype=float)
                                       for t, v in cols.items()})
    names = {s["ticker"]: s["name"] for s in d["stocks"]}
    return Dataset(prices, bench, attention, sources, names, d.get("built", ""))
