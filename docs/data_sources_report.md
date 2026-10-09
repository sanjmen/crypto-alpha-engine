# Crypto Market Data Sources & Ingestion Report

## 1. Executive Summary

This report establishes the data acquisition strategy for the **Crypto Alpha Engine**. Following an empirical audit of institutional endpoints, **we have 100% free, permanent access to all raw and derivative datasets required for professional quantitative trading**, without requiring paid subscriptions (e.g., Kaiko, Tardis, Glassnode) or private exchange API keys.

We audited four primary channels:
1. **Binance Public Data Vision (`data.binance.vision`)**: S3-backed public bulk archive. Zero rate limits, zero keys, historical archives back to 2020.
2. **Binance Futures REST API (`fapi.binance.com`)**: Real-time snapshots, funding history, and market sentiment metrics.
3. **CCXT Unified Framework**: Python integration layer for live sync, pagination, and multi-exchange order execution.
4. **Bybit Public Archive (`public.bybit.com`)**: Multi-exchange validation data for cross-market statistical arbitrage.

---

## 2. Comprehensive Data Taxonomy

| Data Category | Resolution | Coverage | Direct Source | Storage Format | Quantitative Alpha Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OHLCV Candlesticks** | 1s, 1m, 5m, 15m, 1h, 4h, 1d | Jan 2020 – Present | Binance Vision / CCXT | Parquet (ZSTD) | Price momentum, Garman-Klass, Parkinson, Hurst exponent |
| **Aggregated Trades (`aggTrades`)** | Tick-by-tick (ms) | Complete History | Binance Vision / WS | Parquet (ZSTD) | Cumulative Volume Delta (CVD), order flow imbalance, Roll spread |
| **Funding Rates (`fundingRate`)** | 8-hour settlements | Complete History | Binance Vision / CCXT | DuckDB / Parquet | Carry harvesting, cost-of-carry arbitrage, basis trading |
| **Market Metrics (`metrics`)** | 5-minute intervals | Sep 2020 – Present | Binance Vision / REST | Parquet (ZSTD) | Open Interest shocks, Top-Trader vs Global Long/Short divergences |
| **Order Book Quotes (`bookTicker`)** | Continuous (ms) | Complete History | Binance Vision / WS | Parquet (ZSTD) | Effective bid-ask spread, slippage estimation, micro-liquidity |
| **Order Book Depth (`bookDepth`)** | 50 levels | Daily Snapshots | Binance Vision / WS | Parquet (ZSTD) | Market depth, queue position simulation, order book imbalance (OBI) |

---

## 3. Deep Dive by Source

### 3.1 Binance Public Data Vision (`data.binance.vision`)
* **Underlying Architecture**: Hosted on Amazon S3 (`s3-ap-northeast-1.amazonaws.com/data.binance.vision`).
* **Authentication & Rate Limits**: Zero authentication required. No rate limiting; downloads sustain 50–100 MB/s.
* **Available Directory Trees**:
  * `data/futures/um/monthly/klines/{SYMBOL}/{TIMEFRAME}/`: Monthly ZIP archives containing standardized CSVs.
  * `data/futures/um/daily/metrics/{SYMBOL}/`: Daily 5-minute metrics files.
  * `data/futures/um/monthly/fundingRate/{SYMBOL}/`: Historical funding settlement logs.
  * `data/futures/um/daily/aggTrades/{SYMBOL}/`: Full tick execution logs with buyer-maker tags.
  * `data/futures/um/daily/bookTicker/{SYMBOL}/`: L1 top-of-book best bid/ask updates.

#### Metrics Schema Verified Empirically:
* `create_time`: Timestamp at 5-minute granularity (`YYYY-MM-DD HH:MM:SS`).
* `symbol`: Asset ticker (e.g. `BTCUSDT`).
* `sum_open_interest`: Aggregate open contract positions in coin units.
* `sum_open_interest_value`: Aggregate open interest in USDT.
* `count_toptrader_long_short_ratio`: Top trader accounts Long / Short ratio.
* `sum_toptrader_long_short_ratio`: Top trader positions Long / Short ratio.
* `count_long_short_ratio`: Overall retail / global Long / Short account ratio.
* `sum_taker_long_short_vol_ratio`: Net aggressive taker buy vs sell volume ratio.

### 3.2 Binance Futures REST API (`fapi.binance.com`)
* **Role**: Near real-time synchronization of recent intervals (last 1–30 days) that have not yet been rolled into the monthly S3 archives.
* **Key Public Endpoints**:
  * `/fapi/v1/klines`: Recent candlesticks (limit 1500).
  * `/fapi/v1/fundingRate`: Historical funding rates (limit 1000).
  * `/futures/data/openInterestHist`: Rolling Open Interest history.
  * `/futures/data/takerlongshortRatio`: Aggressive buyer vs seller flow.
  * `/futures/data/topLongShortPositionRatio`: Institutional leverage bias.

### 3.3 CCXT Integration Layer
* **Role**: Provides a uniform, object-oriented facade across Binance, Bybit, and OKX.
* **Verified Capabilities**:
  * `fetchOHLCV()`: Standardized bar retrieval with automatic timestamp normalization.
  * `fetchFundingRateHistory()`: Normalized funding timestamps and rates.
  * `fetchOpenInterestHistory()`: Real-time and historical open contract statistics.
  * `fetchOrderBook()`: Snapshot of L2 bids and asks.

### 3.4 Bybit Public Archive (`public.bybit.com`)
* **Role**: Independent out-of-sample data source and cross-exchange basis reference.
* **Available**: Direct CSV downloads for trade ticks, order book snapshots, and klines for all Bybit Linear Perpetual pairs.

---

## 4. Recommended Ingestion Pipeline Architecture

```
                                [ DATA SOURCES ]
                                       |
          +----------------------------+----------------------------+
          |                                                         |
  [ HISTORICAL BULK ]                                     [ LIVE & RECENT ]
  Binance Data Vision                                     CCXT / REST / WS
  - Multi-year Klines (1m/1h)                             - Rolling 30d Klines
  - 5-min Metrics (OI & L/S)                              - Real-time Funding
  - Funding Rate History                                  - Live WebSocket Ticks
          |                                                         |
          v                                                         v
  [ INGESTION ENGINE ]                                    [ STREAMING ENGINE ]
  - Direct HTTP/S3 stream                                 - Circular memory buffer
  - Unzip on-the-fly                                      - Incremental bar builder
          |                                                         |
          +----------------------------+----------------------------+
                                       |
                                       v
                       [ DUCKDB / PARQUET STORAGE ]
                       - Partitioned by Symbol / Timeframe
                       - Timestamp Deduplication (rn = 1)
                       - Sub-millisecond Analytical Queries
                                       |
                                       v
                             [ QUANT ALPHA ENGINE ]
                             - Cross-Sectional Ranking
                             - Volatility Estimators (GK/RS)
                             - GMM Jump / Tail Risk Stops
```

---

## 5. Storage & Bandwidth Budget

For our target universe of the **Top 30 Crypto Assets**:
* **1-Hour OHLCV (5 Years)**: $\approx 15 \text{ MB}$ total in Parquet (extremely compact).
* **15-Minute OHLCV (3 Years)**: $\approx 120 \text{ MB}$ total in Parquet.
* **5-Minute Metrics (Open Interest + Long/Short) (3 Years)**: $\approx 250 \text{ MB}$ in Parquet.
* **Funding Rate History (Complete)**: $\approx 10 \text{ MB}$.
* **Total Local Disk Footprint**: Less than **1.0 GB** for a complete institutional multi-asset quantitative research warehouse.
