"""Download long daily history for the crash-defence study into data/long/.

- SPY adjusted close since 1993 (its launch), dividends included.
- ^IRX: the 13-week US Treasury bill yield, in percent a year. This is what
  uninvested cash could earn, so strategies that sit in cash are judged fairly.

Runs on GitHub Actions (.github/workflows/fetch-long-history.yml) because the
data source isn't reachable from every machine. The output is small and
committed, so the study itself runs offline.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parents[1] / "data" / "long"
OUT.mkdir(parents=True, exist_ok=True)


def close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="1993-01-01", auto_adjust=True, progress=False)
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    if c.dropna().empty:
        raise SystemExit(f"No data for {ticker}")
    return c.rename(ticker)


spy, irx = close("SPY"), close("^IRX")
df = pd.concat([spy.rename("spy"), irx.rename("tbill_pct")], axis=1).dropna(subset=["spy"])
df["tbill_pct"] = df["tbill_pct"].ffill()
df.index.name = "date"
df.round(6).to_csv(OUT / "spy_tbill.csv")
print(f"Saved {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
