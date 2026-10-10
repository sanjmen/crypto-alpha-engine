"""
Unit Tests for AggTradesStreamer.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data.agg_trades_streamer import AggTradesStreamer


def test_url_construction():
    """Verifies daily and monthly Binance Vision URL construction."""
    streamer = AggTradesStreamer(cache_dir="/tmp/test_agg_trades")
    daily_url = streamer.build_daily_url("BTC/USDT", "2024-03-15")
    monthly_url = streamer.build_monthly_url("ETH/USDT", 2024, 2)

    assert "daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-03-15.zip" in daily_url
    assert "monthly/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-02.zip" in monthly_url


def test_trade_enrichment():
    """Verifies calculation of notional and aggressor side flags."""
    raw_df = pd.DataFrame({
        "agg_trade_id": [1, 2, 3],
        "price": [90_000.0, 90_100.0, 89_900.0],
        "quantity": [0.5, 1.0, 2.0],
        "transact_time": [1700000000000, 1700000001000, 1700000002000],
        "is_buyer_maker": [True, False, True],
    })
    streamer = AggTradesStreamer()
    enriched = streamer._enrich_trades_df(raw_df)

    assert "notional" in enriched.columns
    assert enriched["notional"].iloc[0] == 45_000.0
    assert enriched["notional"].iloc[1] == 90_100.0
    assert enriched["notional"].iloc[2] == 179_800.0

    # is_buyer_maker True -> Buyer was maker (seller took) -> tick_sign -1
    # is_buyer_maker False -> Buyer took -> tick_sign +1
    assert enriched["tick_sign"].iloc[0] == -1
    assert enriched["tick_sign"].iloc[1] == 1
    assert enriched["tick_sign"].iloc[2] == -1
    assert "timestamp" in enriched.columns
