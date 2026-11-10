"""A broad, fixed-in-advance universe: the S&P 500 as Wikipedia listed it on a past date.

Why: the original 16 stocks were picked in 2026 because they are famous *now*.
Companies that are famous now mostly did well in the past, so any strategy that
holds them looks better than it should (survivorship bias). The fix is to pick
the stocks the way you would have on the start date, before knowing how they
turned out. Here that means: every company in Wikipedia's "List of S&P 500
companies" article as it stood on START_DATE (default 1 Jan 2016), including
the ones that were later bought out, went bust or shrank out of the index.

Everything is free and needs no sign-up:
- the old version of the list comes from the MediaWiki API (page history),
- each row links to the company's Wikipedia article, which gives us the page
  title for the page-view data,
- redirects are resolved so renamed articles (Facebook -> Meta) are followed.

The one thing free data cannot fully fix: Yahoo Finance drops the price history
of companies that no longer trade. `scripts/build_broad_data.py` records which
companies have no prices, so the report can say how big that gap is.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

LIST_TITLE = "List of S&P 500 companies"
START_DATE = "2016-01-01"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "wikiquant-research/0.1 (https://github.com/eleadlau-commits/wiki-quant-project)"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "broad" / "universe.csv"

# Tickers that changed since 2016 while the company kept trading. Yahoo keeps the
# history under the new ticker. Old ticker -> Yahoo tickers to try, in order.
# Companies that were bought out are *not* listed: they have no current ticker,
# and the build script records them as "no price data".
RENAMED: dict[str, tuple[str, ...]] = {
    "FB": ("META",),          # Facebook -> Meta Platforms (2022)
    "ANTM": ("ELV",),         # Anthem -> Elevance Health (2022)
    "PCLN": ("BKNG",),        # Priceline -> Booking Holdings (2018)
    "KORS": ("CPRI",),        # Michael Kors -> Capri Holdings (2018)
    "COH": ("TPR",),          # Coach -> Tapestry (2017)
    "DLPH": ("APTV",),        # Delphi Automotive -> Aptiv (2017)
    "UTX": ("RTX",),          # United Technologies -> Raytheon Technologies -> RTX
    "HRS": ("LHX",),          # Harris -> L3Harris (2019)
    "BBT": ("TFC",),          # BB&T -> Truist (2019)
    "JEC": ("J",),            # Jacobs Engineering (2019)
    "SYMC": ("GEN", "NLOK"),  # Symantec -> NortonLifeLock -> Gen Digital
    "WLTW": ("WTW",),         # Willis Towers Watson (2022)
    "HCN": ("WELL",),         # Health Care REIT -> Welltower
    "HCP": ("DOC", "PEAK"),   # HCP -> Healthpeak
    "ABC": ("COR",),          # AmerisourceBergen -> Cencora (2023)
    "PX": ("LIN",),           # Praxair -> Linde (2018 merger)
    "DISCA": ("WBD",),        # Discovery -> Warner Bros. Discovery (2022)
    "CBS": ("PSKY", "PARA"),  # CBS -> ViacomCBS -> Paramount -> Paramount Skydance
    "LUK": ("JEF",),          # Leucadia -> Jefferies Financial (2018)
    "FISV": ("FISV", "FI"),   # Fiserv (traded as FI 2023-2025)
    "COG": ("CTRA",),         # Cabot Oil & Gas -> Coterra (2021)
    "Q": ("IQV",),            # Quintiles -> IQVIA (2017)
    "WYN": ("TNL",),          # Wyndham Worldwide -> Travel + Leisure (2021)
    "ADS": ("BFH",),          # Alliance Data -> Bread Financial (2022)
    "DPS": ("KDP",),          # Dr Pepper Snapple -> Keurig Dr Pepper (2018)
    "MHFI": ("SPGI",),        # McGraw Hill Financial -> S&P Global (2016)
    "MYL": ("VTRS",),         # Mylan -> Viatris (2020)
    "ESV": ("VAL",),          # Ensco -> Valaris (after bankruptcy; history may be short)
    "WRK": ("SW",),           # WestRock -> Smurfit WestRock (2024)
    "TYC": ("JCI",),          # Tyco merged into Johnson Controls (2016)
    "GOOG": ("GOOGL", "GOOG"),
    "CBG": ("CBRE",),         # CBRE Group (2018)
    "BLL": ("BALL",),         # Ball Corp (2022)
    "CHK": ("EXE",),          # Chesapeake -> Expand Energy (2024)
    "LB": ("BBWI",),          # L Brands -> Bath & Body Works (2021)
    "PKI": ("RVTY",),         # PerkinElmer -> Revvity (2023)
    "GPS": ("GAP",),          # Gap (2024)
    "IR": ("TT",),            # Ingersoll-Rand plc -> Trane Technologies (2020)
    "UA": ("UAA",),           # Under Armour class A (UA is now the newer class C)
}


@dataclass
class Member:
    ticker: str         # as written in the list on START_DATE
    yahoo: str          # tickers to try at Yahoo Finance, separated by "|"
    name: str
    sector: str
    article: str        # article the list linked to on START_DATE
    current: str        # same article after following redirects today

    @property
    def keys(self) -> list[str]:
        """Page titles whose views we add up (old and new names of a renamed article)."""
        return list(dict.fromkeys([self.current, self.article]))


# ---------------------------------------------------------------- parsing the old list
class _TableParser(HTMLParser):
    """Collect every wikitable: rows of cells, each cell = (text, first /wiki/ link)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[tuple[str, str | None]]]] = []
        self._depth = 0          # nesting depth of tables we are inside
        self._row = None
        self._cell = None        # [text parts, link]
        self._skip = 0           # inside <sup> (footnotes): ignore text

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
            if self._depth == 1 and "wikitable" in (a.get("class") or ""):
                self.tables.append([])
            elif self._depth == 1:
                self.tables.append(None)   # placeholder for non-wikitables
        elif not self.tables or self.tables[-1] is None or self._depth != 1:
            return
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = [[], None]
        elif tag == "sup":
            self._skip += 1
        elif tag == "a" and self._cell is not None and self._cell[1] is None:
            href = a.get("href") or ""
            if href.startswith("/wiki/") and ":" not in href[6:]:
                self._cell[1] = unquote(href[6:].split("#")[0])

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
            return
        if not self.tables or self.tables[-1] is None or self._depth != 1:
            return
        if tag == "sup":
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell[0])).strip()
            self._row.append((text, self._cell[1]))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None and not self._skip:
            self._cell[0].append(data)


def _find_col(header: list[str], *words: str) -> int | None:
    for i, h in enumerate(header):
        if any(w in h.lower() for w in words):
            return i
    return None


def parse_constituents(html: str) -> list[dict]:
    """Rows of the first table that has a ticker column and a company column."""
    p = _TableParser()
    p.feed(html)
    for table in p.tables:
        if not table:
            continue
        header = [t for t, _ in table[0]]
        i_tick = _find_col(header, "symbol", "ticker")
        i_name = _find_col(header, "security", "company")
        i_sect = _find_col(header, "gics sector", "sector")
        if i_tick is None or i_name is None:
            continue
        rows = []
        for row in table[1:]:
            if len(row) <= max(i_tick, i_name):
                continue
            ticker = row[i_tick][0].strip().upper()
            name, link = row[i_name]
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", ticker) or not link:
                continue
            sector = row[i_sect][0] if i_sect is not None and i_sect < len(row) else ""
            rows.append({"ticker": ticker, "name": name, "sector": sector, "article": link})
        if len(rows) >= 50:
            return rows
    raise ValueError("Could not find the constituents table in the page")


def yahoo_tickers(ticker: str) -> list[str]:
    """Ticker in the old list -> tickers to try at Yahoo today (BRK.B -> BRK-B, FB -> META).

    The old ticker itself is always tried last, in case Yahoo still has it.
    """
    cands = [*RENAMED.get(ticker, ()), ticker]
    return list(dict.fromkeys(t.replace(".", "-") for t in cands))


def dedupe_share_classes(rows: list[dict]) -> list[dict]:
    """Some companies have two share classes in the index (GOOG and GOOGL, FOX and
    FOXA). They share one Wikipedia page, so keep only the first row per article."""
    seen, out = set(), []
    for r in rows:
        if r["article"] in seen:
            continue
        seen.add(r["article"])
        out.append(r)
    return out


# ---------------------------------------------------------------- talking to Wikipedia
def _get(params: dict, session=None) -> dict:
    import requests

    s = session or requests
    params = {"format": "json", "formatversion": 2, **params}
    for attempt in range(4):
        resp = s.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def revision_at(date: str, title: str = LIST_TITLE) -> dict:
    """The last revision of `title` saved on or before `date`."""
    ts = pd.Timestamp(date).strftime("%Y-%m-%dT00:00:00Z")
    d = _get({"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
              "rvdir": "older", "rvstart": ts, "rvprop": "ids|timestamp"})
    rev = d["query"]["pages"][0]["revisions"][0]
    return {"revid": rev["revid"], "timestamp": rev["timestamp"]}


def revision_html(revid: int) -> str:
    return _get({"action": "parse", "oldid": revid, "prop": "text"})["parse"]["text"]


def _resolve_once(titles: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(titles), 50):
        batch = [t.replace("_", " ") for t in titles[i:i + 50]]
        d = _get({"action": "query", "titles": "|".join(batch), "redirects": 1})["query"]
        hop = {}
        for key in ("normalized", "redirects"):
            for x in d.get(key, []):
                hop[x["from"]] = x["to"]
        for t in batch:
            cur, steps = t, 0
            while cur in hop and steps < 5:
                cur, steps = hop[cur], steps + 1
            out[t.replace(" ", "_")] = cur.replace(" ", "_")
    return out


def resolve_redirects(titles: list[str], rounds: int = 4) -> dict[str, str]:
    """Map each article title to the real page it ends up at today.

    The API follows only one redirect per request, and after a rename the old
    title often points at another redirect ("Netflix Inc." -> "Netflix, Inc."
    -> "Netflix"). So we keep resolving until nothing changes.
    """
    result = {t: t for t in titles}
    for _ in range(rounds):
        todo = sorted(set(result.values()))
        step = _resolve_once(todo)
        new = {t: step.get(cur, cur) for t, cur in result.items()}
        if new == result:
            break
        result = new
    return result


def build_universe(date: str = START_DATE) -> tuple[list[Member], dict]:
    """Download the S&P 500 list as it was on `date` and return its members."""
    rev = revision_at(date)
    rows = dedupe_share_classes(parse_constituents(revision_html(rev["revid"])))
    current = resolve_redirects([r["article"] for r in rows])
    members = [Member(r["ticker"], "|".join(yahoo_tickers(r["ticker"])), r["name"], r["sector"],
                      r["article"], current.get(r["article"], r["article"])) for r in rows]
    meta = {"list": LIST_TITLE, "as_of": date, **rev,
            "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"}
    return members, meta


def save_universe(members: list[Member], path: Path | str = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(m) for m in members]).to_csv(path, index=False)


def load_universe(path: Path | str = DEFAULT_PATH) -> list[Member]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [Member(**r) for r in df.to_dict("records")]


def current_articles(title: str = LIST_TITLE) -> set[str]:
    """Articles linked from today's version of the list (redirects followed).

    Used to mark which 2016 members are still in the index, so the report can
    show how many of the original companies later dropped out.
    """
    html = _get({"action": "parse", "page": title, "prop": "text"})["parse"]["text"]
    rows = parse_constituents(html)
    resolved = resolve_redirects([r["article"] for r in rows])
    return {resolved.get(r["article"], r["article"]) for r in rows}
