#!/usr/bin/env python3
"""
CLI script to download free crypto market data via CCXT and persist to Parquet.
Usage:
    python scripts/download_market_data.py --preset top10 --timeframe 1h --days 30
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ccxt_provider import CCXTMarketDataProvider, TOP_30_SYMBOLS
from src.data.parquet_storage import ParquetBarStorage
from src.domain.enums import Timeframe

PRESETS = {
    "top5": TOP_30_SYMBOLS[:5],
    "top10": TOP_30_SYMBOLS[:10],
    "top30": TOP_30_SYMBOLS[:30],
}


def main():
    parser = argparse.ArgumentParser(description="Download free historical crypto market data via CCXT.")
    parser.add_argument("--preset", choices=["top5", "top10", "top30"], default="top10", help="Symbol universe preset.")
    parser.add_argument("--exchange", default="binance", help="Exchange ID (default: binance).")
    parser.add_argument("--timeframe", choices=["1m", "5m", "15m", "1h", "4h", "1d"], default="1h", help="Bar timeframe.")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to fetch.")
    args = parser.parse_args()

    symbols = PRESETS[args.preset]
    timeframe = Timeframe(args.timeframe)
    since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)

    print("=" * 70)
    print(f"  CRYPTO ALPHA ENGINE: MARKET DATA INGESTION PIPELINE")
    print("=" * 70)
    print(f"  Exchange:  {args.exchange.upper()} (Free Public Endpoints)")
    print(f"  Timeframe: {timeframe.value} | Lookback: {args.days} days")
    print(f"  Universe:  {len(symbols)} assets ({', '.join(symbols[:5])}...)")
    print("-" * 70)

    provider = CCXTMarketDataProvider(exchange_id=args.exchange)
    storage = ParquetBarStorage()

    total_bars = 0
    for sym in symbols:
        print(f"  Fetching {sym:<12} ... ", end="", flush=True)
        bars = provider.fetch_bars(symbol=sym, timeframe=timeframe, since=since, limit=1000)
        if bars:
            count = storage.store_bars(symbol=sym, timeframe=timeframe, bars=bars)
            total_bars += len(bars)
            print(f"OK ({len(bars)} bars downloaded, {count} stored in Parquet)")
        else:
            print("FAILED or EMPTY")

    print("-" * 70)
    print(f"  Completed! Total bars ingested: {total_bars:,}")
    print(f"  Stored at: {storage.base_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
