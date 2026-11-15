"""Trading 212 broker, through its public API (beta). Demo by default.

Docs: https://docs.trading212.com/api

- Two environments with separate keys: demo (practice money) and live (real
  money). Create a key in the app under Settings > API (Beta). Give it these
  permissions: account, portfolio, orders:read, orders:execute.
- Authentication is HTTP Basic with the API key as username and the API
  secret as password.
- Only Invest and Stocks ISA accounts work with the API.
- The market-order endpoint is NOT idempotent: sending the same request twice
  can buy twice. So this class never retries an order, and the bot refuses to
  trade while an earlier order for the same instrument is still pending.
- A market order sent while the exchange is closed waits until it opens.
"""

from __future__ import annotations

import base64
import time
from typing import Callable

import requests

BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}


class Trading212Error(RuntimeError):
    pass


class Trading212Broker:
    def __init__(self, api_key: str, api_secret: str, env: str = "demo",
                 price_fallback: Callable[[str], float] | None = None,
                 session: requests.Session | None = None, timeout: float = 20.0):
        if env not in BASE_URLS:
            raise ValueError(f"env must be 'demo' or 'live', not {env!r}")
        if not api_key or not api_secret:
            raise ValueError("Trading 212 needs both an API key and an API secret")
        self.env = env
        self.name = f"trading212-{env}"
        self.base = BASE_URLS[env]
        token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        self.http = session or requests.Session()
        self.http.headers.update({"Authorization": f"Basic {token}"})
        self.timeout = timeout
        # Used for the price of something you don't hold yet (the positions
        # endpoint only gives prices of what you own).
        self.price_fallback = price_fallback

    # --- HTTP -------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None, retries: int = 3):
        """GET with a polite retry on rate limits (429). Reads are safe to repeat."""
        for attempt in range(retries + 1):
            r = self.http.get(self.base + path, params=params, timeout=self.timeout)
            if r.status_code == 429 and attempt < retries:
                reset = r.headers.get("x-ratelimit-reset")
                wait = max(1.0, float(reset) - time.time()) if reset else 5.0
                time.sleep(min(wait, 60.0))
                continue
            if not r.ok:
                raise Trading212Error(f"GET {path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        raise Trading212Error(f"GET {path}: still rate-limited")

    # --- Broker interface -------------------------------------------------
    def account_value(self) -> float:
        return float(self._get("/equity/account/summary")["totalValue"])

    def account_currency(self) -> str:
        return self._get("/equity/account/summary").get("currency", "")

    def holding(self, ticker: str) -> tuple[float, float]:
        positions = self._get("/equity/positions", {"ticker": ticker})
        for p in positions:
            if p.get("instrument", {}).get("ticker") == ticker:
                return float(p["quantity"]), float(p["currentPrice"])
        if self.price_fallback is None:
            raise Trading212Error(f"No position in {ticker} and no fallback price source")
        return 0.0, float(self.price_fallback(ticker))

    def pending_orders(self, ticker: str) -> list[dict]:
        return [o for o in self._get("/equity/orders") if o.get("ticker") == ticker]

    def market_order(self, ticker: str, quantity: float) -> dict:
        if self.pending_orders(ticker):
            raise Trading212Error(f"An order for {ticker} is still pending; not sending another")
        r = self.http.post(self.base + "/equity/orders/market",
                           json={"ticker": ticker, "quantity": quantity, "extendedHours": False},
                           timeout=self.timeout)
        # Deliberately no retry: a repeated market order could execute twice.
        if not r.ok:
            raise Trading212Error(f"Order rejected ({r.status_code}): {r.text[:300]}")
        return r.json()

    def find_instruments(self, text: str) -> list[dict]:
        """Search tradable instruments by name, ticker or ISIN (slow: 1 call per 50s allowed)."""
        text = text.lower()
        return [i for i in self._get("/equity/metadata/instruments")
                if text in (i.get("name", "") + " " + i.get("ticker", "") + " " + i.get("isin", "")).lower()]
