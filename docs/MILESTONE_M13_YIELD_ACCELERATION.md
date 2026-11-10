# Milestone M13: Parallel Multi-Mode Engine, Yield Acceleration (+1.0%–1.3%/day) & Smart Money Integration

## 1. Executive Summary & 20.4-Day Empirical Baseline

Between **2026-10-21 (10:57 UTC)** and **2026-11-10 (20:56 UTC)** (`20.42 days` / `490.0 hours`), the system executed **159,031 telemetry snapshots** and **1,553 trades** (`71.9%` Maker fill rate, `$177,850` total volume) on Binance USDS-M Futures Mainnet market data:

| Metric | Full 20.4-Day Run | Mature Phase (Last 12 Days: Oct 30 – Nov 10) | Peak High-Water Mark (Nov 9) |
| :--- | :--- | :--- | :--- |
| **Starting $\rightarrow$ Ending Equity** | `$1,000.00` $\rightarrow$ `$1,111.72 USDT` | `$1,000.98` $\rightarrow$ `$1,111.72 USDT` | `$1,000.00` $\rightarrow$ **`$1,152.82 USDT`** |
| **Net Cumulative Return** | **`+11.17%`** (`+$111.72 USDT`) | **`+11.06%`** (`+$110.74 USDT`) | **`+15.28%`** (`+$152.82 USDT`) |
| **Realized Cash Balance** | **`$1,120.38 USDT`** (`+$120.38`) | `+$106.39 USDT` | **`$1,124.97 USDT`** (`+$124.97`) |
| **Geometric Daily Return (`%`)** | **`+0.52% / day`** | **`+0.89% / day`** | **`+0.75% / day`** (`+1.22%/day` in Phase 2) |
| **Daily Net PnL (`$ USDT`)** | **`+$5.47 / day`** (`+$5.90/day` Cash) | **`+$9.31 / day`** | **`+$8.00 / day`** (`+$12.76/day` in Phase 2) |
| **Max Drawdown from Initial** | **`-1.96%`** (Low: `$980.44`) | **`-1.47%`** | — |
| **Annualized Sharpe Ratio (`365d`)** | **`5.26`** (Daily Vol: `1.90%`) | **`6.41`** | — |

### Strategy & Factor PnL Attribution (`1,553 Trades`)

| Strategy / Factor | Trades | Maker % | Volume (`$`) | Fees Paid | Gross PnL | **Net PnL (`$`)** | Action for M13 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`F7_HighLiq_Size (Quality)`** | `380` | `92.4%` | `$50,364` | `-$11.14` | `+$63.95` | **`+$52.81`** | Core Pillar #1 — Increase baseline prior weight |
| **`F4_Rev_24_72h (MeanRev)`** | `277` | `92.8%` | `$53,459` | `-$11.24` | `+$50.74` | **`+$39.50`** | Core Pillar #2 — Increase baseline prior weight |
| **`F3_Fast_Swing_PB (12h-48h)`** | `51` | `98.0%` | `$7,384` | `-$1.57` | `+$23.89` | **`+$22.32`** | Highest EV/trade (`+0.30%`) — Boost with Smart Money |
| **`F5_Carry_Neg (72h Fund)`** | `424` | `94.3%` | `$56,033` | `-$12.18` | `+$29.72` | **`+$17.54`** | Core Pillar #4 — Expand via 300+ Perp Harvester |
| **`Early Breakout`** *(Anomaly)* | `108` | `1.9%` | `$1,692` | `-$0.67` | `+$3.32` | **`+$2.65`** | Convert entry to `GTX` Maker + Dynamic `% Equity` size |
| **`Extreme Drop Bounce`** *(Anomaly)* | `256` | `12.1%` | `$3,895` | `-$1.46` | `+$3.65` | **`+$2.19`** | Convert entry to `GTX` Maker + Dynamic `% Equity` size |
| **`F1_Macro_Idio_PB (7d-30d)`** | `7` | `100.0%` | `$614` | `-$0.12` | `+$0.48` | **`+$0.36`** | Retain as low-turnover macro anchor |
| **`Pump & Funding Fade`** *(Anomaly)* | `32` | `6.3%` | `$500` | `-$0.19` | `+$0.30` | **`+$0.10`** | Convert entry to `GTX` Maker |
| **`F6_Liq_Surge (Vol Exp)`** | `18` | `94.4%` | `$3,909` | `-$0.79` | `-$16.86` | **`-$17.65`** | **Gate behind `t-stat >= +0.75` (Issue #52)** |

---

## 2. Milestone M13 Issue Roadmap (`#51` – `#58`)

### Phase 0: Parallel Multi-Mode Execution Architecture
* **Issue #51 (`Issue-0` — COMPLETED) — Unified Multi-Mode Trading Launcher (`scripts/run_trading.py`)**:
  * Enables running **Production (`--mainnet`)** and **Shadow Lab (`--shadow`) / Testnet (`--testnet`)** simultaneously in separate processes without database, lock, or credential interference:
    * `--shadow`: Read-only Mainnet feeds $\rightarrow$ `data/shadow_trading.db`
    * `--testnet`: Binance Futures Testnet execution $\rightarrow$ `data/testnet_trading.db`
    * `--mainnet`: Live Binance USDS-M Futures execution $\rightarrow$ `data/live_trading.db`
  * Exposes M13 experimental flags (`--f6-min-tstat`, `--anomaly-pct-equity`, `--anomaly-maker-chase`, `--enable-smart-money`, `--enable-funding-harvester`) so every new alpha module is validated in `--shadow` / `--testnet` before promoting to `--mainnet`.

### Phase 1: The 4 Direct Levers to Reach `+1.0% – +1.3% / day`
* **Issue #52 (Lever #1) — Factor #6 (`F6_Liq_Surge`) High-Conviction Gating (`t-stat >= +0.75`) & Weight Reallocation**:
  * Gates `F6_Liq_Surge` behind `t-stat >= +0.75` (eliminating `-1.76%` historical drag) and reallocates baseline prior weight to the 4 proven alpha champions (`F7`, `F4`, `F3`, `F5`).
* **Issue #53 (Lever #2) — Post-Only (`GTX`) Maker Limit-Chase & Dynamic `% Equity` Sizing for Anomaly Strategies**:
  * Upgrades `Early Breakout`, `Extreme Drop Bounce`, and `Pump & Funding Fade` from raw `MARKET` Taker entries (`0.05%` fee) to `GTX` Maker Limit-Chase (`0.02%` fee) and scales slot size dynamically as a percentage of current `total_equity` (`--anomaly-pct-equity 0.045`).
* **Issue #54 (Levers #3 & #4) — Market-Neutral `3.0x` Base Leverage Calibration & Multi-Slice Order Chunking (`$5k–$50k`)**:
  * Calibrates `3.0x` base leverage with automatic drawdown step-down (`-0.5x` if drawdown $\ge 6\%$) and adds automatic multi-slice child order execution in `SmartExecutionRouter` when `MarketCapacityCeiling` recommends `recommended_chunks > 1`.

### Phase 2: Binance Institutional Ecosystem Integration (Tier 1 & Tier 2)
* **Issue #55 (Tier 1A) — Binance Institutional Positioning Factor (`F8_Smart_Money_Flow` via `fapiData`)**:
  * Integrates `fapiDataGetTopLongShortPositionRatio` (Top 20% Whales), `fapiDataGetGlobalLongShortAccountRatio` (Retail Crowd), `fapiDataGetTakerlongshortRatio` (Taker Buy/Sell Volume), and `fapiDataGetOpenInterestHist` ($\Delta\text{OI}$) to overweight assets where Whales accumulate against Retail.
* **Issue #56 (Tier 1B) — Extreme Funding Rate Harvester (`300+` USDS-M Perpetuals with Beta-Neutral Hedge)**:
  * Scans `fapiPublicGetPremiumIndex` across all `300+` perpetuals prior to the 8h funding timestamps (`00:00`, `08:00`, `16:00 UTC`) to capture extreme negative funding (`<= -0.20%` per interval) via a beta-neutral `LONG Altcoin / SHORT BTC-USDT` pair.
* **Issue #57 (Tier 1C) — Yield-Bearing Futures Collateral Integration (`BFUSD` & `LDUSDT` Cross-Margin Accounting)**:
  * Supports `BFUSD` and `LDUSDT` in `BinanceLiveBroker` and `ShadowExecutionBroker` (`Multi-Assets Margin` mode + unified `USDT + BFUSD + LDUSDT` equity accounting) to capture **`~4%–8% APY`** passive collateral yield on top of algorithmic PnL.
* **Issue #58 (Tier 2) — Quarterly Delivery Basis Regime Indicator (`fapiDataGetBasis`) & Ecosystem Guardrails**:
  * Uses `BTC` and `ETH` quarterly futures annualized basis (`CURRENT_QUARTER` vs Spot Index) as a macro regime scalar while enforcing explicit exclusions against negative-EV retail instruments (long-only `eapi` Options theta decay, `8–14% APR` Crypto Loans, non-24/7 TradFi perps).
