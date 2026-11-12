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
Master system operations, empirical barrier audit & decision log (Nov 2026): [`docs/SYSTEM_OPERATION_AND_DECISIONS_2026.md`](docs/SYSTEM_OPERATION_AND_DECISIONS_2026.md).  
22-Month & Sub-Hourly quantitative research report: [`reports/high_frequency_and_alpha_research_2026.md`](reports/high_frequency_and_alpha_research_2026.md).  
Milestone M13–M15 specification: [`docs/MILESTONE_M13_YIELD_ACCELERATION.md`](docs/MILESTONE_M13_YIELD_ACCELERATION.md).  
Mathematical formulations: [`docs/mathematical_formulation.md`](docs/mathematical_formulation.md).  
Empirical data census & sources report: [`docs/deep_data_census_report.md`](docs/deep_data_census_report.md).  
Development roadmap (M1–M15): [`docs/roadmap.md`](docs/roadmap.md).

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

## 🛰️ Unified Multi-Mode Trading Engine (`--mainnet`, `--shadow`, `--testnet`)

Run Production (`--mainnet`) and Shadow Lab (`--shadow`) simultaneously in isolated processes and SQLite WAL databases (`data/live_trading.db` vs. `data/shadow_trading.db`), locally or on Google Cloud Tokyo (`asia-northeast1-a`) via Docker + Zero-Trust IAP:

```bash
# 1. Production Live Mainnet (uses data/live_trading.db):
python scripts/run_trading.py --daemon --mainnet
python scripts/run_trading.py --monitor --mainnet

# 2. Local Shadow Lab in parallel (0 capital risk, uses data/shadow_trading.db):
python scripts/run_trading.py --daemon --shadow
python scripts/run_trading.py --monitor --shadow

# 3. Check live status of Tokyo VM containers (crypto-mainnet & crypto-shadow):
./scripts/deploy_to_gcp.sh --status
```

---

## 📜 License
MIT License. Developed for quantitative investment and algorithmic trading research.

