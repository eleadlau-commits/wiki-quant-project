"""Which companies we study, and how to look each one up in every source.

Each company lists its search keys per source. A source that is not listed
falls back to the company's name, so a new plugin works straight away and you
only add keys where the name is not good enough.

Survivorship warning: these 16 were picked in 2026 because they are famous
*now*. Famous-now companies mostly did well in the past, which flatters any
strategy that holds them. A fair test needs a universe fixed in advance, such
as the S&P 500 members on each past date.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    keys: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def keys_for(self, source: str) -> list[str]:
        return list(self.keys.get(source, (self.name,)))


def _c(ticker, name, wikipedia, hackernews=None):
    keys = {"wikipedia": tuple([wikipedia] if isinstance(wikipedia, str) else wikipedia)}
    if hackernews:
        keys["hackernews"] = tuple([hackernews] if isinstance(hackernews, str) else hackernews)
    return Company(ticker, name, keys)


BENCHMARK = "SPY"   # ETF that tracks the S&P 500

# When a Wikipedia page was renamed (Facebook -> Meta) list every title: views
# are summed, because readers of the old title land on a redirect that is
# counted separately.
UNIVERSE: list[Company] = [
    _c("AAPL", "Apple", "Apple_Inc."),
    _c("MSFT", "Microsoft", "Microsoft"),
    _c("NVDA", "Nvidia", "Nvidia", ["Nvidia", "NVIDIA"]),
    _c("AMZN", "Amazon", "Amazon_(company)"),
    _c("GOOGL", "Alphabet", "Alphabet_Inc.", ["Google", "Alphabet"]),
    _c("META", "Meta", ["Meta_Platforms", "Facebook,_Inc."], ["Meta", "Facebook"]),
    _c("TSLA", "Tesla", "Tesla,_Inc."),
    _c("NFLX", "Netflix", "Netflix"),
    _c("JPM", "JPMorgan Chase", "JPMorgan_Chase", ["JPMorgan", "JP Morgan"]),
    _c("DIS", "Disney", "The_Walt_Disney_Company"),
    _c("BA", "Boeing", "Boeing"),
    _c("GME", "GameStop", "GameStop"),
    _c("AMD", "AMD", ["AMD", "Advanced_Micro_Devices"]),
    _c("PFE", "Pfizer", "Pfizer"),
    _c("NKE", "Nike", "Nike,_Inc.", "Nike"),
    _c("INTC", "Intel", "Intel"),
]
