"""Brokers the trading bot can use. Each one implements `wikiquant.trader.Broker`."""

from wikiquant.brokers.paper import PaperBroker

__all__ = ["PaperBroker"]
