"""
Downloads sub-hourly data (15m from Google Drive + top-up to 2026-10-20, plus 5m and 1m from Binance Futures)
into /tmp/crypto_alpha_backtest_cache/bars/{15m,5m,1m}/.
"""

from pathlib import Path
import subprocess
import time
import ccxt
import pandas as pd

CACHE_DIR = Path("/tmp/crypto_alpha_backtest_cache")
TOP15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "FETUSDT",
]

END_TS_MS = int(pd.Timestamp("2026-10-20 18:00:00", tz="UTC").timestamp() * 1000)


def parse_klines_to_df(rows: list, symbol: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_vol", "taker_buy_quote_vol", "ignore",
        ],
    )
    df["timestamp"] = (df["open_time"].astype("int64") // 1000).astype("float64")
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume", "trade_count", "taker_buy_vol"]:
        df[col] = df[col].astype("float64")
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "taker_buy_vol"]]


def fetch_klines_range(exchange: ccxt.binance, symbol: str, interval: str, start_ms: int, end_ms: int, step_ms: int) -> pd.DataFrame:
    all_rows = []
    cur_ms = start_ms
    while cur_ms < end_ms:
        try:
            rows = exchange.fapiPublicGetKlines({
                "symbol": symbol,
                "interval": interval,
                "startTime": cur_ms,
                "limit": 1500,
            })
            if not rows:
                break
            all_rows.extend(rows)
            last_open = int(rows[-1][0])
            if last_open >= end_ms or len(rows) < 1500:
                break
            cur_ms = last_open + step_ms
        except Exception as e:
            print(f"  Warning {symbol} {interval} at {cur_ms}: {e}")
            time.sleep(0.5)
            break
    df = parse_klines_to_df(all_rows, symbol)
    if not df.empty:
        df = df[df["timestamp"] <= end_ms / 1000.0]
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    t0 = time.time()
    bars_15m_dir = CACHE_DIR / "bars" / "15m"
    bars_5m_dir = CACHE_DIR / "bars" / "5m"
    bars_1m_dir = CACHE_DIR / "bars" / "1m"
    for d in [bars_15m_dir, bars_5m_dir, bars_1m_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Sync 15m from Google Drive via rclone
    print("1) Syncing 15m Parquet files from Google Drive...")
    pattern = "{" + ",".join([f"{s}.parquet" for s in TOP15]) + "}"
    subprocess.run(
        [
            "rclone", "copy",
            "gdrive:trading/crypto-alpha-engine/data/bars/15m/",
            str(bars_15m_dir),
            "--include", pattern,
            "--transfers", "16",
        ],
        check=False,
    )

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})

    # 2. Top up 15m from 2026-09-30 to 2026-10-20 18:00 UTC
    print("2) Topping up 15m bars up to 2026-10-20 18:00 UTC...")
    since_15m_ms = int(pd.Timestamp("2026-09-30 00:00:00", tz="UTC").timestamp() * 1000)
    for sym in TOP15:
        p = bars_15m_dir / f"{sym}.parquet"
        new_df = fetch_klines_range(exchange, sym, "15m", since_15m_ms, END_TS_MS, 15 * 60 * 1000)
        if p.exists():
            old_df = pd.read_parquet(p)
            if "timestamp" not in old_df.columns:
                old_df = old_df.reset_index()
            if pd.api.types.is_datetime64_any_dtype(old_df["timestamp"]):
                old_df["timestamp"] = old_df["timestamp"].astype("int64") // 10**9
            # Keep last 7 months (from 2026-03-15) to make parquet reads super fast
            old_df = old_df[old_df["timestamp"] >= pd.Timestamp("2026-03-15", tz="UTC").timestamp()]
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        else:
            # Fallback: fetch 60 days of 15m
            s60 = END_TS_MS - 60 * 86400 * 1000
            combined = fetch_klines_range(exchange, sym, "15m", s60, END_TS_MS, 15 * 60 * 1000)
        combined.to_parquet(p, index=False)
        print(f"  15m {sym:<12}: {len(combined):6d} bars")

    # 3. Download 5m bars for the last 31 days (2026-09-19 to 2026-10-20)
    print("3) Downloading 5m bars (last 31 days: 2026-09-19 to 2026-10-20)...")
    since_5m_ms = int(pd.Timestamp("2026-09-19 00:00:00", tz="UTC").timestamp() * 1000)
    for sym in TOP15:
        p = bars_5m_dir / f"{sym}.parquet"
        df_5m = fetch_klines_range(exchange, sym, "5m", since_5m_ms, END_TS_MS, 5 * 60 * 1000)
        df_5m.to_parquet(p, index=False)
        print(f"  5m  {sym:<12}: {len(df_5m):6d} bars")

    # 4. Download 1m bars for the last 7.5 days (2026-10-13 to 2026-10-20)
    print("4) Downloading 1m bars (last 7.5 days: 2026-10-13 to 2026-10-20)...")
    since_1m_ms = int(pd.Timestamp("2026-10-13 06:00:00", tz="UTC").timestamp() * 1000)
    for sym in TOP15:
        p = bars_1m_dir / f"{sym}.parquet"
        df_1m = fetch_klines_range(exchange, sym, "1m", since_1m_ms, END_TS_MS, 60 * 1000)
        df_1m.to_parquet(p, index=False)
        print(f"  1m  {sym:<12}: {len(df_1m):6d} bars")

    print(f"Done in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
