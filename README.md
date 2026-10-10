# Crypto Alpha Engine 🚀

[![CI Test Suite](https://github.com/sanjmen/crypto-alpha-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjmen/crypto-alpha-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Clean Architecture](https://img.shields.io/badge/Architecture-Clean%20%2F%20Hexagonal-green.svg)](docs/system_architecture.md)

Institutional-grade quantitative alpha and trading engine for cryptocurrency markets, bridging:
1. **Cross-Sectional Machine Learning (from DataCrunch)**: Rank blending, factor neutralization, and market-neutral long/short equity/crypto portfolios.
2. **Continuous Real-Time Volatility Tracking (from Synth)**: Intraday high-frequency volatility estimators (Parkinson, Garman-Klass), fractal Hurst memory ($H$), heavy-tail Student-$t$ distribution, and 2-state Gaussian Mixture jump risk detection.

---

## 🏛️ Architecture & SOLID Design

Built on **Clean Architecture (Hexagonal / Ports & Adapters)**:
* **Domain (`src/domain/`)**: Pure business logic (Bar, Tick, Signal, Order, Position, Portfolio, Regime).
* **Ports (`src/ports/`)**: Abstract protocols for data, features, models, risk, and execution.
* **Data Layer (`src/data/`)**: High-speed, free crypto data ingestion using `CCXT` (Binance / Bybit) stored in partitioned Parquet and queried via `DuckDB`.
* **Feature Layer (`src/features/`)**: High-efficiency estimators (Parkinson, Garman-Klass, Corwin-Schultz, Hurst DFA, Cross-sectional ranking).
* **Models (`src/models/`)**: Ridge regression, LightGBM, Student-$t$ density, and Gaussian Mixture Models.
* **Risk Engine (`src/risk/`)**: Volatility targeting, Fractional Kelly sizing, VaR/CVaR stops, and jump circuit breakers.
* **Strategies (`src/strategies/`)**: Market-Neutral Long/Short, Fractal Regime Trend-Following, and Perpetual Funding Carry.
* **Backtest Engine (`src/backtest/`)**: Event-driven and vectorized backtest with exchange maker/taker fee schedules and slippage.
* **Execution (`src/execution/`)**: Paper trading and live CCXT order execution.

Detailed architectural specifications: [`docs/system_architecture.md`](docs/system_architecture.md).  
Mathematical formulations: [`docs/mathematical_formulation.md`](docs/mathematical_formulation.md).  
Empirical data census & sources report: [`docs/deep_data_census_report.md`](docs/deep_data_census_report.md).  
Development roadmap: [`docs/roadmap.md`](docs/roadmap.md).

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone git@github.com:sanjmen/crypto-alpha-engine.git
cd crypto-alpha-engine

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Free Crypto Market Data
```bash
# Download 1 year of 1h bars for the Top 10 crypto assets from Binance Futures
python scripts/download_market_data.py --preset top10 --timeframe 1h --days 365
```

### 3. Run Unit Tests
```bash
pytest -v tests/
```

---

## 📊 Core Strategies

| Strategy | Engine Foundation | Key Mechanics | Edge |
| :--- | :--- | :--- | :--- |
| **Market-Neutral Long/Short** | DataCrunch Cross-Sectional | Long Top 15% / Short Bottom 15% Altcoins vs BTC | Zero market beta, pure idiosyncratic alpha |
| **Fractal Hurst Regime Trend** | Synth Intraday Vol & Hurst | Trend when $H > 0.55$, Mean-Revert when $H < 0.45$ | Dynamic adaptation to market phase |
| **Funding Carry Harvester** | Synth Jump & Tail Models | Collect 8h funding rate with GMM jump safety brake | Steady cashflow with catastrophic drawdown protection |

---

## 🛰️ Live Shadow Trading Engine (0 Capital Risk)

Run simulated execution against real-time Binance USDS-M Futures Mainnet prices with local SQLite WAL persistence:

```bash
# 1. Run a single synchronous step (poll Binance Mainnet, evaluate orders, record snapshot):
python scripts/run_shadow_trading.py --once

# 2. Run the continuous background daemon:
python scripts/run_shadow_trading.py --daemon

# 3. Launch the real-time Rich terminal telemetry dashboard:
python scripts/run_shadow_trading.py --monitor
```

---

## 📜 License
MIT License. Developed for quantitative investment and algorithmic trading research.

