# Plug-in attention framework

Any source of "how much are people paying attention to this company?" can be
plugged in, and it is then put through exactly the same tests against stock
returns as every other source. Wikipedia page views and Hacker News stories
are the first two plugins.

## 1. Is this a good idea? An honest evaluation

**What it is good for.** For an internship CV, the value is not "I found a
signal that beats the market". (Almost nobody's student project does, and
interviewers know it.) The value is showing you can do research the way a quant
team does:

- **Data plumbing that scales.** A new data set is a 30 to 60 line plugin, not
  a rewrite. Teams that buy alternative data work like this.
- **One timing rule, enforced in one place and tested.** That is how you avoid
  look-ahead bias, the most common way backtests lie.
- **The same tests on every source.** A fair comparison is a research result
  in itself: "Wikipedia and Hacker News spikes mostly happen on different days,
  and neither predicts 20-day returns once GameStop is removed" is a real,
  defensible finding you can talk about for ten minutes.

**Is there anything to find?** There is published work suggesting attention
matters a little. Da, Engelberg and Gao (2011, *Journal of Finance*) found
that a jump in Google searches for a stock is followed by higher prices for
about two weeks and a partial reversal later, mostly in smaller stocks. Moat
et al. (2013, *Scientific Reports*) linked Wikipedia views to later market
moves, though later work questioned how robust that was. Expect weak, noisy,
short-lived effects, strongest in small, retail-heavy stocks, and weakest in
the mega-caps we start with.

**What could fool us** (and what the framework does about it):

| Pitfall | What it means | What we do |
| --- | --- | --- |
| Look-ahead bias | Using information before you could have had it. Wikipedia days end at midnight UTC, *after* the US close. | `analysis.first_trade_after`: attention dated day D is only traded from the close of the first trading day *after* D. A test scrambles future attention and checks no past decision changes. |
| Already priced in | The news that causes the spike usually moves the price the same day. | The report shows the "day-0 move" separately from the return you could actually trade. |
| Survivorship bias | Our 16 companies are famous *today*, so they mostly did well in the past. | Flagged everywhere. The fix is a universe fixed in advance (past S&P 500 members), planned for Week 3. |
| Data snooping | Try 5 sources x 6 thresholds x 4 holding periods and something will look great by luck. | The report uses one fixed setting (3x spike, 20-day hold, top 4). Decide settings before looking; check winners on unseen data (Week 4). |
| One company carries it | GameStop's 2021 squeeze dominates any attention strategy. | Leave-one-out check, and first-half vs second-half results. |
| Overlapping returns | 20-day returns that overlap are not independent, which makes t-stats look better than they are. | Spikes have a 10-day cooldown; treat t around 2 as "worth a closer look", not proof. |
| Source quirks | Wikipedia pages get renamed; "Meta" and "Apple" are ordinary words; Google Trends rescales every download to 0-100. | Per-source keys in `universe.py`, whole-word headline matching, and notes on each plugin. |
| Costs | Trading weekly or after every spike costs money. | Not modelled yet (Week 4). |

**Which free sources are realistic?**

| Source | Free, no sign-up? | History | Verdict |
| --- | --- | --- | --- |
| Wikipedia page views | Yes | 2015 on | Plugged in |
| Hacker News stories (Algolia search) | Yes | 2007 on | Plugged in |
| Wikipedia edit counts | Yes (same API) | 2001 on | Easy next plugin, good exercise |
| GDELT news volume | Yes | 2017 on | Possible; the API is slow and strict on rate |
| Google Trends | No official API; unofficial libraries break and are blocked from cloud servers; data rescaled per request | 2004 on | Fine for a one-off download by hand, not for a daily automatic refresh |
| Reddit | Needs a registered app since 2023, and free history (Pushshift) is gone | Recent only | Not feasible for free backtests |

## 2. How it fits together

```
sources/            plugins: key -> daily attention (one file each)
  base.py           AttentionSource: caching, daily top-ups, gap filling
  wikipedia.py
  hackernews.py
universe.py         the companies, and each one's search keys per source
scripts/build_dashboard_data.py   runs every source -> dashboard/data.json
dataset.py          loads data.json back into pandas, offline
analysis.py         source-agnostic tests (spikes, event study, Buzz 4, overlap)
scripts/compare_sources.py        -> reports/attention_comparison.md
dashboard/          the web page, with a switch between sources
```

The GitHub Action runs the build and the comparison every weekday, so the
dashboard and `reports/attention_comparison.md` stay current.

## 3. Adding a source

Write one class with a `fetch` method that returns daily numbers for one key,
register it, and list it in `SOURCES` in `scripts/build_dashboard_data.py`.
Here is a complete plugin for Wikipedia **edits** (how many times the article
was edited each day), as an example you could build yourself:

```python
# src/wikiquant/sources/wikiedits.py
import pandas as pd
import requests

from .base import AttentionSource, register

API = ("https://wikimedia.org/api/rest_v1/metrics/edits/per-page/en.wikipedia/"
       "{title}/user/daily/{start}/{end}")


@register
class WikipediaEditsSource(AttentionSource):
    name = "wikiedits"
    label = "Wikipedia edits"
    unit = "edits"
    about = "Wikimedia edits API (human editors, UTC days)"
    missing_is_zero = True   # no row = no edits that day
    smooth = 7               # edits are sparse, so compare weekly totals

    def fetch(self, key, start, end):
        url = API.format(title=requests.utils.quote(key, safe=""),
                         start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
        resp = requests.get(url, headers={"User-Agent": "wikiquant-research/0.1"}, timeout=30)
        resp.raise_for_status()
        rows = resp.json()["items"][0]["results"]
        return pd.Series({pd.Timestamp(r["timestamp"][:10]): r["edits"] for r in rows})
```

Then:

1. Import it in `src/wikiquant/sources/__init__.py` so it registers.
2. Add keys per company in `universe.py` if the company name is not the
   right key (Wikipedia sources need article titles, e.g. `"Apple_Inc."`).
3. Add `"wikiedits"` to `SOURCES` in `scripts/build_dashboard_data.py`.
4. Write an offline test with a fake `fetch` (see `tests/test_sources.py`).

Rules every source must follow:

- **Date = the day the attention happened**, in UTC or any time zone whose day
  ends at or after the US close. Never date something by when you downloaded it.
- **Use the data as it was known at the time, if you can.** Some sources revise
  history (Google Trends rescales; a deleted post vanishes). Revised data can
  quietly leak the future into a backtest.
- **Say what a missing day means** (`missing_is_zero`), and set `smooth` if
  most days are 0, 1 or 2.

## 4. Using it from Python

```python
from wikiquant.dataset import load_dataset
from wikiquant import analysis as A

ds = load_dataset()                       # offline, from dashboard/data.json
hn = ds.attention["hackernews"]           # calendar days x tickers
events = A.event_study(hn, ds.prices, ds.bench, threshold=3, smooth=7)
A.summarise(events["excess"])             # n, mean, hit rate, t-stat
```
