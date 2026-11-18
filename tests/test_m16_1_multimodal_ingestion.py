"""
Unit Tests for Issue #67 (M16-1): Multi-Modal Vision Ingestor, Spot-Perp Multiplier Parity,
Dynamic Funding Interval Detection (`1h/2h/4h/8h`), `5m` Derivatives Metrics Normalization,
and Point-in-Time Survivorship-Free Universe Catalog Generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.multimodal_vision_ingestor import (
    build_universe_catalog_row,
    epoch_to_utc_seconds,
    is_valid_crypto_usdt_perp,
    normalize_funding_df,
    normalize_kline_df,
    normalize_metrics_5m_df,
    read_symbol_from_shard,
    resolve_spot_candidates,
    verify_spot_price_parity,
    write_symbol_rowgroup_shard,
)


def test_perp_universe_filter_excludes_tradfi_leveraged_and_dated_contracts() -> None:
    assert is_valid_crypto_usdt_perp("BTCUSDT") is True
    assert is_valid_crypto_usdt_perp("1000PEPEUSDT") is True
    assert is_valid_crypto_usdt_perp("1000SATSUSDT") is True
    assert is_valid_crypto_usdt_perp("LOOMUSDT") is True  # delisted historical coin preserved!

    # Exclusions: dated quarterly, TradFi equities/commodities, leveraged tokens, canonical-aliased RNDRUSDT
    assert is_valid_crypto_usdt_perp("BTCUSDT_240628") is False
    assert is_valid_crypto_usdt_perp("NVDAUSDT") is False
    assert is_valid_crypto_usdt_perp("XAUUSDT") is False
    assert is_valid_crypto_usdt_perp("BTCUPUSDT") is False
    assert is_valid_crypto_usdt_perp("ETHDOWNUSDT") is False
    assert is_valid_crypto_usdt_perp("BTCDOMUSDT") is False
    assert is_valid_crypto_usdt_perp("USDCUSDT") is False
    assert is_valid_crypto_usdt_perp("RNDRUSDT") is False


def test_spot_multiplier_resolution_handles_1000x_vs_literal_1000sats() -> None:
    spot_set = {"BTCUSDT", "PEPEUSDT", "SHIBUSDT", "1000SATSUSDT", "RNDRUSDT", "RENDERUSDT"}

    # 1000PEPEUSDT Perp -> PEPEUSDT Spot with 1000x price multiplier
    cands_pepe = resolve_spot_candidates("1000PEPEUSDT", spot_set)
    assert cands_pepe == [("PEPEUSDT", 1000.0)]
    assert verify_spot_price_parity(perp_close=0.0125, spot_close=0.00001248, multiplier=1000.0) is True
    assert verify_spot_price_parity(perp_close=0.0125, spot_close=0.00001248, multiplier=1.0) is False

    # 1000SATSUSDT Perp -> 1000SATSUSDT Spot with 1.0x multiplier (since Binance Spot lists 1000SATSUSDT directly!)
    cands_sats = resolve_spot_candidates("1000SATSUSDT", spot_set)
    assert cands_sats[0] == ("1000SATSUSDT", 1.0)
    assert verify_spot_price_parity(perp_close=0.00028, spot_close=0.000279, multiplier=1.0) is True


def test_spot_microsecond_and_1000x_kline_normalization() -> None:
    # Simulate >=2025 Binance Spot Vision CSV with 16-digit microsecond timestamps for PEPEUSDT
    ts_us_1 = 1_740_000_000_000_000  # 1,740,000,000 sec in microseconds
    ts_ms_2 = 1_740_000_060_000  # +60s in milliseconds
    raw_spot = pd.DataFrame(
        [
            [
                ts_us_1,
                0.000010,
                0.000011,
                0.000009,
                0.0000105,
                2_000_000_000.0,
                0,
                21000.0,
                140,
                1_200_000_000.0,
                12600.0,
                0,
            ],
            [
                ts_ms_2,
                0.0000105,
                0.000012,
                0.000010,
                0.0000110,
                3_000_000_000.0,
                0,
                33000.0,
                210,
                1_800_000_000.0,
                19800.0,
                0,
            ],
        ]
    )
    norm = normalize_kline_df(raw_spot, symbol="1000PEPEUSDT", price_multiplier=1000.0, spot_symbol="PEPEUSDT")
    assert norm is not None
    assert list(norm["timestamp"]) == [1_740_000_000, 1_740_000_060]
    assert pytest.approx(norm["close"].iloc[0], rel=1e-9) == 0.0105
    # Base volume scaled down by 1000x to match 1000PEPE contract units, while USD quote_volume stays exact!
    assert pytest.approx(norm["volume"].iloc[0], rel=1e-9) == 2_000_000.0
    assert pytest.approx(norm["quote_volume"].iloc[0], rel=1e-9) == 21_000.0
    assert pytest.approx(norm["taker_buy_quote_volume"].iloc[0], rel=1e-9) == 12_600.0
    assert norm["spot_symbol"].iloc[0] == "PEPEUSDT"


def test_metrics_5m_and_dynamic_funding_interval_normalization() -> None:
    raw_metrics = pd.DataFrame(
        {
            "create_time": ["2025-05-01 00:05:00", "2025-05-01 00:00:00", "2025-05-01 00:05:00"],
            "symbol": ["SOLUSDT", "SOLUSDT", "SOLUSDT"],
            "sum_open_interest": [10100.0, 10000.0, 10150.0],
            "sum_open_interest_value": [1_515_000.0, 1_500_000.0, 1_522_500.0],
            "count_toptrader_long_short_ratio": [1.4, 1.35, 1.42],
            "sum_toptrader_long_short_ratio": [1.8, 1.75, 1.85],
            "count_long_short_ratio": [0.85, 0.88, 0.84],
            "sum_taker_long_short_vol_ratio": [1.22, 1.10, 1.25],
        }
    )
    m_df = normalize_metrics_5m_df(raw_metrics, symbol="SOLUSDT")
    assert m_df is not None
    assert len(m_df) == 2  # deduplicated & sorted ascending
    assert m_df["timestamp"].iloc[1] - m_df["timestamp"].iloc[0] == 300
    assert pytest.approx(m_df["sum_open_interest"].iloc[1]) == 10150.0

    # Funding transitioning dynamically from 8h -> 4h -> 2h during an extreme negative squeeze!
    t0 = 1_750_000_000
    raw_fnd = pd.DataFrame(
        {
            "calc_time": [t0 * 1000, (t0 + 8 * 3600) * 1000, (t0 + 12 * 3600) * 1000, (t0 + 14 * 3600) * 1000],
            "funding_interval_hours": [np.nan, 8, np.nan, 2],
            "last_funding_rate": [-0.0005, -0.0030, -0.0075, -0.0120],
        }
    )
    f_df = normalize_funding_df(raw_fnd, symbol="ALPACAUSDT")
    assert f_df is not None
    assert list(f_df["interval_hours"]) == [8, 8, 4, 2]

    cat = build_universe_catalog_row(
        symbol="ALPACAUSDT",
        bars_1m_df=pd.DataFrame({"timestamp": [t0, t0 + 86400], "quote_volume": [5_000_000.0, 8_000_000.0]}),
        metrics_5m_df=m_df,
        spot_1m_df=None,
        funding_df=f_df,
        active_cutoff_ts=t0 + 30 * 86400,
    )
    assert cat["status"] == "HISTORICAL_DELISTED"
    assert cat["min_funding_interval_hours"] == 2
    assert epoch_to_utc_seconds(np.array([1_700_000_000_000]))[0] == 1_700_000_000


def test_rowgroup_sharded_parquet_predicate_pushdown(tmp_path) -> None:
    import pyarrow.parquet as pq

    shard_file = tmp_path / "metrics_5m" / "shard_00.parquet"
    df_btc = pd.DataFrame(
        {"timestamp": [100, 400], "symbol": ["BTCUSDT", "BTCUSDT"], "sum_open_interest": [10.0, 11.0]}
    )
    df_eth = pd.DataFrame(
        {
            "timestamp": [100, 400, 700],
            "symbol": ["ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "sum_open_interest": [50.0, 52.0, 55.0],
        }
    )

    total = write_symbol_rowgroup_shard({"BTCUSDT": df_btc, "ETHUSDT": df_eth}, shard_file)
    assert total == 5
    pf = pq.ParquetFile(shard_file)
    assert pf.num_row_groups == 2

    eth_only = read_symbol_from_shard(shard_file, "ETHUSDT")
    assert len(eth_only) == 3
    assert set(eth_only["symbol"]) == {"ETHUSDT"}


def test_compact_rowgroup_shard_and_default_pit_220_universe(tmp_path) -> None:
    from src.data.multimodal_vision_ingestor import (
        compact_kline_df,
        load_default_pit_220_plan,
        resample_1m_to_1h,
        write_compact_rowgroup_shard,
    )

    plan = load_default_pit_220_plan()
    assert len(plan) == 220
    assert "BTCUSDT" in plan
    assert "AGIXUSDT" in plan  # historical delisted symbol preserved (0 survivorship bias)
    assert plan["1000PEPEUSDT"]["spot_symbol"] == "PEPEUSDT"
    assert plan["1000PEPEUSDT"]["spot_multiplier"] == 1000.0

    raw_1m = pd.DataFrame(
        {
            "timestamp": [1_700_002_800 + 60 * i for i in range(60)],
            "symbol": ["BTCUSDT"] * 60,
            "open": np.linspace(40000.0, 40059.0, 60),
            "high": np.linspace(40010.0, 40070.0, 60),
            "low": np.linspace(39990.0, 40045.0, 60),
            "close": np.linspace(40001.123456789, 40060.987654321, 60),
            "volume": np.ones(60) * 2.5,
            "quote_volume": np.ones(60) * 100_000.0,
            "trade_count": np.ones(60, dtype=np.int64) * 50,
            "taker_buy_vol": np.ones(60) * 1.3,
            "taker_buy_quote_volume": np.ones(60) * 52_000.0,
        }
    )
    comp = compact_kline_df(raw_1m)
    assert comp["close"].dtype == np.float64  # exact sub-bps Basis precision preserved
    assert comp["open"].dtype == np.float32

    res_1h = resample_1m_to_1h(comp, "BTCUSDT")
    assert len(res_1h) == 1

    out_shard = tmp_path / "bars_1m" / "shard_00.parquet"
    n_written = write_compact_rowgroup_shard({"BTCUSDT": comp}, out_shard)
    assert n_written == 60
    loaded = read_symbol_from_shard(out_shard, "BTCUSDT")
    assert pytest.approx(loaded["close"].iloc[-1], rel=1e-12) == 40060.987654321
