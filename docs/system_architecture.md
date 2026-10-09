# System Architecture: Crypto Alpha Engine

## 1. Architectural Philosophy & Clean Architecture

The **Crypto Alpha Engine** is designed following **Clean Architecture (Hexagonal / Ports & Adapters)** and strict **SOLID** software engineering principles.

The core motivation is complete decoupling:
- The **Domain** contains pure quantitative logic, financial entities, mathematical representations, and risk invariants. It has zero dependencies on external exchanges, databases, or frameworks.
- The **Ports** define strict interfaces (`Protocols` and abstract base classes) for data ingestion, feature generation, model inference, risk management, and order execution.
- The **Adapters** implement specific external integrations (e.g., `CCXT` for exchange connectivity, `DuckDB`/`PyArrow` for high-throughput Parquet storage, `Plotly` for backtest tear sheets).

```
                      +---------------------------------------+
                      |         External Systems               |
                      |   (Binance / Bybit / CCXT / Parquet)  |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |               Adapters                |
                      |   - CCXTMarketDataProvider            |
                      |   - ParquetBarStorage                 |
                      |   - CCXTExecutionEngine               |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |                Ports                  |
                      |   - IDataProvider                     |
                      |   - IFeatureExtractor                 |
                      |   - IAlphaModel                       |
                      |   - IRiskManager                      |
                      |   - IExecutionEngine                  |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |             Use Cases                 |
                      |   - MarketDataIngestionPipeline       |
                      |   - AlphaSignalGenerator              |
                      |   - PortfolioRebalancer               |
                      |   - BacktestSimulationEngine          |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |              Domain                   |
                      |   - Bar, Tick, OrderBook              |
                      |   - Signal, Order, Position           |
                      |   - Portfolio, Trade, Regime          |
                      +---------------------------------------+
```

---

## 2. SOLID Design Principles Applied

### 2.1 Single Responsibility Principle (SRP)
- **Features**: Dedicated classes for specific feature families:
  - `VolatilityEstimator`: Parkinson, Garman-Klass, Rogers-Satchell, GARCH.
  - `MicrostructureEstimator`: Corwin-Schultz spread, Amihud illiquidity, Roll spread.
  - `HurstEstimator`: R/S and DFA fractal Hurst exponents.
- **Risk**: `RiskManager` only sizes positions and checks circuit breakers. It does not predict signals or execute orders.
- **Models**: `IAlphaModel` only computes forward expectations / rankings. It does not decide capital allocation.

### 2.2 Open/Closed Principle (OCP)
- New alpha models (e.g., Deep Learning, Transformer, Reinforcement Learning) or new strategies can be introduced by subclassing `IAlphaModel` or `IStrategy` without modifying existing backtest harnesses or execution engines.
- New exchanges (Bybit, OKX, Hyperliquid) can be integrated by adding an adapter implementing `IExchangePort`.

### 2.3 Liskov Substitution Principle (LSP)
- Any model adhering to `IAlphaModel` (whether an OLS Ridge, LightGBM, GMM jump tracker, or Quantile Regressor) can be seamlessly interchanged in the `SignalEngine` without altering consumer behavior.

### 2.4 Interface Segregation Principle (ISP)
- Thin, cohesive protocols:
  - `IHistoricalDataProvider` (for historical batch download).
  - `IStreamingDataProvider` (for live WebSocket market feeds).
  - `IOrderExecutor` (for placing and cancelling orders).
  - `IPositionTracker` (for querying open balances and margins).

### 2.5 Dependency Inversion Principle (DIP)
- High-level modules (e.g., `BacktestEngine`, `LiveTradingBot`) depend strictly on abstractions (`IDataProvider`, `IAlphaModel`, `IRiskManager`), never on concrete third-party SDKs (`ccxt.binance`, `pandas`, `lightgbm`).

---

## 3. Package Structure

```
crypto-alpha-engine/
├── docs/
│   ├── system_architecture.md
│   ├── mathematical_formulation.md
│   └── roadmap.md
├── src/
│   ├── domain/               # Pure business & financial entities
│   │   ├── entities.py       # Bar, Tick, Order, Position, Portfolio, Trade
│   │   ├── enums.py          # OrderType, OrderSide, RegimeType, Timeframe
│   │   └── value_objects.py  # Price, Quantity, Volatility, SharpeRatio
│   ├── ports/                # Abstract interfaces / protocols
│   │   ├── data_ports.py     # IDataProvider, IBarStorage
│   │   ├── feature_ports.py  # IFeatureExtractor
│   │   ├── model_ports.py    # IAlphaModel, IRegimeDetector
│   │   ├── risk_ports.py     # IRiskManager, IPositionSizer
│   │   └── execution_ports.py# IExecutionEngine
│   ├── data/                 # Adapters for data acquisition and persistence
│   │   ├── ccxt_provider.py  # CCXT historical and live data fetcher
│   │   ├── parquet_storage.py# Fast DuckDB/PyArrow storage adapter
│   │   └── funding_collector.py # Perpetual funding rate accumulator
│   ├── features/             # Quantitative feature engineering
│   │   ├── volatility.py     # Parkinson, GK, RS, GARCH(1,1), EWMA
│   │   ├── microstructure.py # Corwin-Schultz, Roll, Amihud
│   │   ├── hurst.py          # Fractal persistence & DFA calibrator
│   │   └── cross_sectional.py# Rank normalization & Beta neutralization
│   ├── models/               # Alpha prediction and density models
│   │   ├── ridge_alpha.py    # L2 regularized cross-sectional ranker
│   │   ├── lgbm_alpha.py     # Gradient boosting non-linear ranker
│   │   ├── student_t_alpha.py# Heavy-tail return density forecaster
│   │   └── gmm_regime.py     # 2-state Gaussian Mixture jump detector
│   ├── risk/                 # Risk management & capital allocation
│   │   ├── vol_targeting.py  # Garman-Klass inverse volatility scaler
│   │   ├── kelly_sizing.py   # Fractional Kelly criterion allocator
│   │   └── circuit_breakers.py# Max drawdown, VaR/CVaR, and jump stops
│   ├── strategies/           # High-level trading strategies
│   │   ├── base_strategy.py  # Abstract strategy orchestrator
│   │   ├── cross_sectional_ls.py # Market-Neutral Dollar-Neutral Long/Short
│   │   ├── hurst_regime_trend.py # Fractal trend / mean-reversion switcher
│   │   └── funding_harvest.py    # Perpetual funding carry harvester
│   ├── backtest/             # Simulation & evaluation engine
│   │   ├── engine.py         # Vectorized & event-driven simulator
│   │   ├── cost_model.py     # Fee (maker/taker) and slippage models
│   │   └── metrics.py        # Sharpe, Sortino, Calmar, Drawdown, Win Rate
│   └── execution/            # Live and paper execution adapters
│       ├── paper_trader.py   # Simulated matching engine with latency
│       └── ccxt_executor.py  # Live exchange order manager via CCXT
├── tests/                    # Comprehensive unit and integration test suite
├── scripts/                  # CLI utilities and entry points
│   ├── download_market_data.py
│   ├── run_backtest.py
│   └── start_paper_trading.py
├── requirements.txt
├── README.md
└── pyproject.toml
```
