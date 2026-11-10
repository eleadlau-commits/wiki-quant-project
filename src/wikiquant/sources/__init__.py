"""Attention data sources ("plugins").

Every source turns a search key (a Wikipedia title, a company name, ...) into a
daily time series of attention. They all share one interface, so the analysis
code never needs to know where the numbers came from.

    from wikiquant.sources import get_source, available_sources
    available_sources()                    # ['hackernews', 'wikipedia']
    wiki = get_source("wikipedia")
    views = wiki.load(["Nvidia"], "2024-01-01", "2024-06-30")

To add your own source, see docs/attention-framework.md.
"""

from .base import AttentionSource, available_sources, get_source, register

# Importing a plugin module registers it.
from . import hackernews, wikipedia  # noqa: E402,F401

__all__ = ["AttentionSource", "available_sources", "get_source", "register"]
