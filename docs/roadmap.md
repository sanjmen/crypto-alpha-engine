# Development Roadmap: Crypto Alpha Engine

This roadmap structures the development of the **Crypto Alpha Engine** into six progressive, test-driven milestones. Each milestone is decoupled, adheres to Clean Architecture and SOLID principles, and is verified by comprehensive unit tests.

---

## Milestone 1: Ingestion & Market Data Pipeline (M1)
**Objective**: Build a high-throughput, free crypto market data pipeline bridging bulk S3 streaming from Binance Vision (multi-year history) with CCXT (rolling recent sync), persisting to partitioned Parquet files with DuckDB querying.

* **Issue #1**: Define Core Domain Entities & Data Ports
  * `Bar`, `Tick`, `OrderBook`, `FundingRate`, `DerivativesMetrics`, `SymbolInfo` domain entities.
  * `IDataProvider`, `IBarStorage`, `IFundingStorage`, `IMetricsStorage` port abstractions.
* **Issue #2**: Binance Vision Bulk S3 Downloader (Historical Backfill)
  * Direct streaming download and on-the-fly decompression of monthly/daily ZIPs from `data.binance.vision`.
  * Support for `klines` (1m, 15m, 1h), `fundingRate` (8h), and `metrics` (5m OI + Long/Short ratios).
* **Issue #3**: CCXT Rolling Sync Adapter (Near Real-Time)
  * Incremental synchronization of recent hours/days not yet rolled into monthly archives.
  * Integration with Binance USDT-M Futures & Bybit Linear Futures.
* **Issue #4**: Parquet Storage & DuckDB Cache Engine
  * Fast partitioned storage (`data/cache/bars/`, `data/cache/metrics/`, `data/cache/funding/`).
  * Sub-millisecond analytical queries and timestamp deduplication via DuckDB.
* **Issue #5**: CLI Data Management & Audit Tool
  * `scripts/download_market_data.py` supporting presets (`top10`, `top30`) across Binance Vision and CCXT.

---

## Milestone 2: Feature Engineering & Alpha Porting (M2)
**Objective**: Port mathematical estimators from `crunch-synth` and `datacrunch-2`, enriched with crypto derivatives metrics (Open Interest, Long/Short ratios) and L2 book depth.

* **Issue #6**: Intraday Volatility Estimators
  * Parkinson, Garman-Klass, Rogers-Satchell, Online GARCH(1,1), Online EWMA.
* **Issue #7**: Market Microstructure & L2 Book Depth Estimators
  * Corwin-Schultz bid-ask spread, Amihud illiquidity, Roll spread, and L2 order book depth imbalance (from `bookDepth`).
* **Issue #8**: Fractal Hurst Exponent Analysis
  * Rescaled Range (R/S) and Detrended Fluctuation Analysis (DFA) calibrator for anomalous diffusion.
* **Issue #9**: Derivatives & Sentiment Feature Extractors
  * Open Interest velocity & acceleration ($\Delta \text{OI}$), Top-Trader vs Retail Long/Short divergence, Taker Buy/Sell volume imbalance.
* **Issue #10**: Cross-Sectional Normalization & Beta Neutralization
  * Gaussian rank transformation, cross-sectional z-score, orthogonal market beta projection.

---

## Milestone 3: Regime Detection & Risk Sizing Engine (M3)
**Objective**: Port density and regime trackers from `synth`, converting probabilistic distributions into capital sizing and circuit breakers.

* **Issue #9**: Student-$t$ Heavy Tail Modeling
  * Maximum Likelihood estimation of degrees of freedom $\nu$, tail risk assessment (VaR / CVaR).
* **Issue #10**: Gaussian Mixture Jump Detector
  * 2-state GMM separating normal diffusion from tail jumps / flash crashes.
* **Issue #11**: Market Regime Classifier
  * Categorization into `TRENDING`, `RANGING`, `VOLATILE` with state confidence.
* **Issue #12**: Volatility-Targeted & Fractional Kelly Position Sizer
  * Position sizing scaling inversely with Garman-Klass volatility and proportional to expected drift.

---

## Milestone 4: Strategy Engines & Signal Orchestration (M4)
**Objective**: Build production-grade trading strategies bridging cross-sectional ranking and real-time regime signals.

* **Issue #13**: Cross-Sectional Market-Neutral Long/Short Strategy
  * Long top quantiles, Short bottom quantiles across Top 30 altcoins.
* **Issue #14**: Fractal Hurst Regime-Switching Strategy
  * Momentum trend-following when $H > 0.55$, mean-reversion oscillator when $H < 0.45$, cash when volatile.
* **Issue #15**: Perpetual Funding Rate Harvesting Strategy
  * Delta-neutral carry collection with GMM tail jump safety brake.
* **Issue #16**: Meta-Strategy Ensemble Allocator
  * Dynamic weight blending across strategies based on rolling Sharpe / CRPS metrics.

---

## Milestone 5: Backtesting Engine with Real Slippage & Fees (M5)
**Objective**: Build a realistic backtest simulation engine preventing lookahead bias, accounting for maker/taker fees, slippage, and funding cashflows.

* **Issue #17**: Event-Driven & Vectorized Backtest Simulator
  * Strict time-step simulation preventing data leakage.
* **Issue #18**: Cost & Execution Impact Models
  * Binance Futures tier fee schedules (0.02% maker, 0.05% taker) + L2 order book depth execution simulator (using `bookDepth`).
* **Issue #19**: Performance Analytics & Plotly Tear Sheets
  * Cumulative PnL, Annualized Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio.

---

## Milestone 6: Paper Trading & Live Execution Bot (M6)
**Objective**: Build paper trading simulation on testnet and live execution capabilities with automated safety stops.

* **Issue #20**: Automated Trading Daemon & Multi-Strategy Rebalancer

---

## Milestone 7: Quantitative Research & Production Hardening (M7)
**Objective**: Empirically calibrate machine learning models, sector cointegration, non-linear market impact, and run full-universe backtests and paper trading.

* **Issue #21**: ML: Purged Walk-Forward Cross-Validation & LightGBM Hyperparameter Tuning in Colab GPU
  * Prevent temporal leakage and train models across 83 symbols utilizing Colab GPU compute credits.
* **Issue #22**: Stat-Arb: Sector Clustering & Multi-Asset Cointegration Testing for Pair Baskets
  * Cluster assets (L1, DeFi, Meme, AI) and model stationary cointegrated spreads.
* **Issue #23**: Execution Alpha: Rebalance Frequency Optimization & Passive Maker Order Execution
  * Quantify fee drag and model post-only limit orders to capture VIP maker rebates.
* **Issue #24**: Microstructure: Non-Linear Market Impact & Slippage Modeling via L2 Book Depth
  * Square-root market impact law ($\Delta P \propto \sigma \sqrt{Q/V}$) calibrated with L2 book depth.
* **Issue #25**: Backtest: Full Universe Multi-Asset Simulation (2023–2026) on 83 Symbols via Google Drive
  * Benchmark multi-year performance across the 12.45 million bar dataset stored in Drive.
* **Issue #26**: Execution: Live Paper Trading Deployment & Telemetry Dashboard with Real-Time Prices
  * Connect `TradingDaemon` to CCXT public data feed and monitor live rebalancing via terminal dashboard.

---

## Milestone 8: Operational Resilience & Cross-Exchange Infrastructure (M8)
**Objective**: Build high-availability exchange connectivity, shadow accounting, real-time alerts, and cross-exchange funding arbitrage.

* **Issue #27**: Infra: Exchange Rate Limiting, WebSocket Heartbeats & Auto-Reconnect Resilience
  * Leaky-bucket weight limiter, 24h forced-disconnect recovery, and REST fallback.
* **Issue #28**: Risk: Shadow Accounting & Real-Time Position Reconciliation Engine
  * Periodic position sync, drift detection, and automatic circuit-breaker lockout upon mismatch.
* **Issue #29**: Alerts: Real-Time Incident Response & Telemetry Webhooks (Telegram / Discord)
  * Priority webhook alerting for risk breaches, trade notifications, and daily EOD PnL tear sheets.
* **Issue #30**: Arbitrage: Multi-Exchange Perp-Perp Funding Spread Arbitrage (Binance vs Bybit)
  * Dual-exchange delta-neutral funding rate harvesting without spot margin fees.
* **Issue #31**: Security: Zero-Trust API Key Vault & Read/Trade Permission Validator
  * Automated pre-flight check guaranteeing withdrawal permissions are disabled on trading keys.

---

## Milestone 9: Information-Driven Bars & aggTrades Synthesis (M9) [COMPLETED]
**Objective**: Ingest tick aggTrades from Binance Vision, synthesize Dollar Bars and Dollar Imbalance Bars for market microstructure modeling.

* **Issue #32**: Research: Information-Driven & Dollar Volume Bars Synthesis from aggTrades
  * Statistical sampling over dynamic dollar thresholds, statistical properties normalization.
* **Issue #33**: Data: Market-Wide Aggregated Trades Collector & S3 Streaming Pipeline
  * Low-latency streaming collector with rolling tick buffer and memory-safe batch flush.
* **Issue #34**: Pipeline: Binance Vision Bulk Historical aggTrades Ingestion
  * High-throughput downloader for tick archives with parquet synthesis.

---

## Milestone 10: Heterogeneous Meta-Model Ensembling & Meta-Labeling (M10) [COMPLETED]
**Objective**: Build secondary meta-labeling filter to improve primary alpha precision and allocate capital via Hierarchical Risk Parity (HRP).

* **Issue #35**: ML: Meta-Labeling Engine with Triple-Barrier Method & Bet Sizing
  * Triple barrier labeling, Secondary Random Forest / LightGBM meta-model, and probability-based position sizing.
* **Issue #36**: Meta: Hierarchical Risk Parity (HRP) Portfolio Allocation Engine
  * Quasi-diagonalization, dendrogram bisection, and tree-based risk allocation eliminating correlation inversion instability.

---

## Milestone 11 & 12: 7-Factor Causal Online Engine, Stepped Leverage & Mainnet Bridge (M11–M12) [COMPLETED]
**Objective**: Implement the 7-Factor Causal Online Meta-Allocator (`36h/96h/240h` net t-statistics), 72h Volatility Targeting, Hysteresis Stepped Leverage (`2.5x/3.0x -> 5.0x`), operational shields (Open Interest Squeeze Guard, L2 OBI Adaptive Maker Routing, CVD Absorption, Spot-Perp Basis), and the Binance USDS-M Futures Mainnet live execution bridge.

* **Issues #37–#50**:
  * `SevenFactorOnlineEngine` (`F1_Macro_Idio_PB`, `F2_Med_Idio_PB`, `F3_Fast_Swing_PB`, `F4_Rev_24_72h`, `F5_Carry_Neg`, `F6_Liq_Surge`, `F7_HighLiq_Size`).
  * `SteppedLeverageController` with compound HWM step-ups (`+0.5x` per `+15%` gain) and drawdown de-escalation (`-0.5x` per `6%` DD).
  * `BinanceLiveBroker` with `Post-Only (GTX)` Maker Limit-Chase, L1 top-of-book pricing, and automatic One-Way / Cross-Margin / Leverage configuration.

---

## Milestone 13: Parallel Multi-Mode Engine, Yield Acceleration & Smart Money Integration (M13) [COMPLETED]
**Objective**: Build unified multi-mode execution (`--shadow`, `--testnet`, `--mainnet`), implement the 4 Direct Yield Levers, and integrate Binance institutional positioning (`fapiData`), extreme funding harvesting, yield-bearing collateral (`BFUSD`/`LDUSDT`), and quarterly delivery basis regime scaling.

* **Issue #51**: Unified Multi-Mode Trading Launcher (`scripts/run_trading.py`) with isolated SQLite WAL databases.
* **Issue #52**: Factor #6 (`F6_Liq_Surge`) High-Conviction Gating (`t-stat >= +0.75`).
* **Issue #53**: Post-Only (`GTX`) Maker Limit-Chase & Dynamic `% Equity` Sizing for Anomaly Strategies.
* **Issue #54**: Market-Neutral `3.0x` Base Leverage Calibration & Multi-Slice Order Chunking (`$5k–$50k`).
* **Issue #55**: Tier 1A Institutional Positioning Factor (`F8_Smart_Money_Flow` via `fapiData`).
* **Issue #56**: Tier 1B Extreme Funding Rate Harvester (`300+` Perps with Beta-Neutral `BTC/USDT` Hedge).
* **Issue #57**: Tier 1C Yield-Bearing Futures Collateral Integration (`BFUSD` & `LDUSDT` Cross-Margin Accounting).
* **Issue #58**: Tier 2 Quarterly Delivery Basis Regime Indicator (`fapiDataGetBasis`) & Ecosystem Guardrails.

---

## Milestone 14: Storage/SQLite TTL Pruning & WebSocket Event-Loop Optimization (M14) [COMPLETED]
**Objective**: Optimize SQLite WAL disk growth and WebSocket thread CPU consumption on the Google Cloud Tokyo (`asia-northeast1-a`) production VM.

* **Issue #59 [CLOSED]**: Native SQLite TTL Pruning (`retention_days=7` for `accounts`/`orders`/`reconciliation_audit_logs` while preserving 100% of `trades`, `funding_settlements`, and `rebalance_snapshots`), `60s` Account Snapshot Throttling, L1 Tick Deduplication, and `PRAGMA wal_checkpoint(TRUNCATE)` (reduced DB from `25 MB` to `2.8 MB` and CPU from `21%` to `<4%`).
* **Issue #60 [OPEN — Reserve]**: Optional Redis 7 Ephemeral Streams for high-frequency tick fan-out if scaling to multiple consumers.

---

## Milestone 15: Selective Production Promotion & Core Bracket Optimization (M15) [COMPLETED]
**Objective**: Promote empirically validated M13 features to `--mainnet` while keeping experimental anomaly modules quarantined in `--shadow`, and optimize Core exit barriers based on the 22-day live database audit and 14-configuration backtest matrix.

* **Issue #61 [CLOSED]**: Promote Dynamic Online IC (`priors=None`) and `F6_Liq_Surge` Gate (`t-stat >= +0.75`) to `--mainnet`.
* **Issue #62 [CLOSED]**: Promote `F8_Smart_Money_Flow`, `Quarterly Delivery Basis` Regime Scalar, and `BFUSD/LDUSDT` Yield Collateral Accounting to `--mainnet`.
* **Issue #63 [CLOSED]**: Implement `60s` Live Wallet Balance Sync (`sync_from_live_exchange`) in `--mainnet`.
* **Issue #64 [OPEN — Quarantine in `--shadow`]**: Quarantine `Extreme Funding Harvester` in `--shadow` pending post-settlement stop-loss refinement.
* **Issue #65 [OPEN — Quarantine in `--shadow`]**: Quarantine `4.5% Dynamic Anomaly Equity Sizing` in `--shadow` (keeping `$45` cap in `--mainnet`).
* **Core Bracket Optimization (`commits cfdb24b` & `4cdbb2a`)**: Unify Core 7-Factor exit brackets to **Passive Maker Take-Profit `+5.0%` (`0.02%` fee) + Profit-Lock (`+3.5% -> +1.5%`) with Trailing Stop disabled (`0.0`) and initial negative Stop-Loss disabled (`None`)** across both `--shadow` and `--mainnet` (documented in [`docs/SYSTEM_OPERATION_AND_DECISIONS_2026.md`](SYSTEM_OPERATION_AND_DECISIONS_2026.md)).




