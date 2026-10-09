from datetime import datetime, timedelta, timezone
import pytest

from src.data.parquet_storage import ParquetBarStorage
from src.domain.entities import Bar
from src.domain.enums import Timeframe


def test_parquet_storage_roundtrip_and_dedup(tmp_path):
    storage = ParquetBarStorage(base_dir=tmp_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    bars_batch1 = [
        Bar(timestamp=now + timedelta(hours=i), symbol="BTC/USDT", open=50000+i, high=50100+i, low=49900+i, close=50050+i, volume=10.0)
        for i in range(5)
    ]
    count1 = storage.store_bars("BTC/USDT", Timeframe.H1, bars_batch1)
    assert count1 == 5

    # Second batch with 2 overlapping bars and 3 new bars
    bars_batch2 = [
        Bar(timestamp=now + timedelta(hours=i), symbol="BTC/USDT", open=50000+i, high=50100+i, low=49900+i, close=50050+i, volume=10.0)
        for i in range(3, 8)
    ]
    count2 = storage.store_bars("BTC/USDT", Timeframe.H1, bars_batch2)
    assert count2 == 8  # Deduplicated from 5 + 5 to 8 unique timestamps

    # Load bars back
    loaded = storage.load_bars("BTC/USDT", Timeframe.H1)
    assert len(loaded) == 8
    assert loaded[0].timestamp == now
    assert loaded[-1].timestamp == now + timedelta(hours=7)
