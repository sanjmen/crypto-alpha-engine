from pathlib import Path
import pytest
import pandas as pd

from src.data.binance_vision_downloader import BinanceVisionDownloader
from src.domain.enums import Timeframe


def test_vision_downloader_save_parquet(tmp_path):
    downloader = BinanceVisionDownloader(output_dir=tmp_path)

    sample_data = {
        "open_time": [1704067200000, 1704070800000],
        "open": ["42000.0", "42100.0"],
        "high": ["42200.0", "42300.0"],
        "low": ["41900.0", "42050.0"],
        "close": ["42100.0", "42250.0"],
        "volume": ["150.5", "200.1"],
        "trade_count": [1200, 1500],
        "taker_buy_base_vol": ["75.2", "110.0"],
    }
    df = pd.DataFrame(sample_data)

    count = downloader.save_klines_to_parquet("BTCUSDT", Timeframe.H1, df)
    assert count == 2

    # Check file exists
    parquet_file = tmp_path / "bars" / "1h" / "BTCUSDT.parquet"
    assert parquet_file.exists()

    # Append overlapping data (dedup test)
    df_overlapping = pd.DataFrame({
        "open_time": [1704070800000, 1704074400000],
        "open": ["42100.0", "42250.0"],
        "high": ["42300.0", "42400.0"],
        "low": ["42050.0", "42100.0"],
        "close": ["42250.0", "42350.0"],
        "volume": ["200.1", "180.0"],
        "trade_count": [1500, 1300],
        "taker_buy_base_vol": ["110.0", "95.0"],
    })
    count2 = downloader.save_klines_to_parquet("BTCUSDT", Timeframe.H1, df_overlapping)
    assert count2 == 3  # 2 + 1 new = 3
