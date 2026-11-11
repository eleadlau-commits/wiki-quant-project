"""Load the broad dataset written by scripts/build_broad_data.py (works offline).

    from wikiquant.broad_dataset import load_broad
    b = load_broad()
    b.prices.shape     # trading days x ~500 companies (NaN after a stock stops trading)
    b.views.tail()     # calendar days x companies
    b.coverage.head()  # where each company's prices came from, or why missing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "broad"
BENCHMARK = "SPY"


@dataclass
class BroadData:
    prices: pd.DataFrame
    bench: pd.Series
    views: pd.DataFrame
    coverage: pd.DataFrame
    universe_meta: dict


def load_broad(folder: Path | str = DEFAULT_DIR) -> BroadData:
    folder = Path(folder)
    px = pd.read_csv(folder / "prices.csv.gz", index_col=0, parse_dates=True)
    bench = px.pop(BENCHMARK)
    views = pd.read_csv(folder / "views.csv.gz", index_col=0, parse_dates=True).astype(float)
    cov = pd.read_csv(folder / "coverage.csv", keep_default_na=False)
    meta_path = folder / "universe.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return BroadData(px, bench, views, cov, meta)
