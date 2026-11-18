# Milestone M16: Multi-Modal Alpha Factory & High-Sharpe Orthogonal Discovery

## 1. Quantitative Motivation & Mathematical Target

Following the deployment of **Phase 1 (`Cushion-Safe CPPI` + `Manga P2 Causal 1m Dollar-Bar TSMOM`)** and **Phase 2 (`Event-Driven Snipers M09, M23 & M04`)** in `100% USDT` Single-Asset Cross-Margin (`commits 510533e`, `ef515a9`, `cfbed64`), the production system achieves on the 4-year `82`-symbol `1m` Grand Backtest (`2023–2026`):
- **Constant `2.5x` Base Leverage:** `CAGR = +125.7%` (`~0.224% / day` geometric), `Sharpe = 2.48`, `MaxDD = 21.2%` (`2026 Holdout DD = 15.3%`), `Calmar = 5.92`.
- **With `Stepped Leverage` (`2.5x -> 5.0x` + Maker Limit-Chase `3.2 bps`):** `~0.33%–0.42% / day` geometric (`$2,700 -> $242,994` over 4Y; `$2,700 -> $5,625` in `2026` Holdout, `Min Equity = $2,539 SAFE`).

### Why `~0.60%–1.00% / day` at `MaxDD <= 20%` Requires Multi-Modal Inputs
By the Fundamental Law of Active Management ($\text{IR} \approx \text{IC} \times \sqrt{\text{Breadth}} - \text{Costs}$) and portfolio combination ($\text{Sharpe}_{\text{total}} = \sqrt{\sum_{k=1}^K \text{Sharpe}_k^2}$ for uncorrelated sleeves):
1. **Volatility Ceiling under `MaxDD <= 20%`:** Portfolio annualized volatility cannot exceed `~35%–45%` without risking `>20%` drawdowns during crypto fat-tail regimes. Achieving `~0.60%–1.00% / day` at `~40%` annualized volatility therefore requires lifting **Portfolio Net Sharpe from `2.48` to `5.0–7.5`**.
2. **The Two Walls of Pure Futures OHLCV (`1m`):**
   - **Positioning Blindness:** Futures `1m` price/volume bars cannot distinguish *new aggressive shorting* (`ΔP < 0, ΔOI >> 0` -> continuation) from *forced long liquidation cascades* (`ΔP < 0, ΔOI << 0` -> violent bounce), nor *real Spot cash buying* (`Spot CVD >> 0, Perp Basis < 0`) from synthetic perpetual leverage.
   - **Taker Fee Wall (`8.4–12 bps` RT):** Fast `1m–15m` exhaustion reversals have positive gross alpha (`+8 to +14 bps`) that is erased by market taker fees unless executed passively via **`Post-Only (GTX)` Wick-Snipers** (`$300–$1,200` agile clips with near-zero market impact).

---

## 2. Milestone M16 Architecture & Issue Roadmap (`#67` – `#72`)

All data storage and compute-heavy mining run **100% on Google Drive (`gdrive:trading/crypto-alpha-engine/data/`) + Google Colab** (`0 MB` local market data storage on Mac), while live production stays **100% in `USDT` (`BFUSD = OFF`)**.

| Order | Issue | Title & Scope | Primary Deliverable |
| :---: | :---: | :--- | :--- |
| **1** | **#67 (`M16-1`)** | **Data/Drive: Zero-Local Streaming Ingestion of `5m` Derivatives Metrics (`OI`, Whale/Retail `L/S`), `1m` Spot Klines & `200+` Perp Universe** | `metrics/5m/`, `spot_bars/1m/`, `universe_catalog.parquet` (`onboard_timestamp` + dynamic `1h/2h/4h/8h` funding intervals) in Google Drive (`0 MB` local). |
| **2** | **#68 (`M16-2`)** | **FeatureLab/MultiModal: Causal Microstructure Feature Library (`ΔOI` Quadrants, Whale-vs-Crowd, `1m` Spot-Perp Basis & Spot CVD Lead-Lag)** | `src/features/multimodal_alpha.py` (`~90–120` strictly causal features) + `2023–2026` Out-of-Sample IC & Conditional Payoff Atlas. |
| **3** | **#69 (`M16-3`)** | **ExecutionLab/Maker: Conservative `1m` Passive `Post-Only (GTX)` Wick-Sniper & Queue-Penetration Simulator** | Pessimistic `1m` `GTX` fill simulator requiring strict price-through (`1–3 bps`), `1–5m` TTL auto-cancel, and worst-case same-bar SL ordering. |
| **4** | **#70 (`M16-4`)** | **Discovery/Engine: Automated Symbolic & Shallow-Tree Alpha Mining on Colab with Deflated Sharpe (`DSR >= 0.95`) & CPCV** | `3–5` surviving interpretable sleeves passing `Train (23-24) -> Val (25) -> Sealed Holdout (26)`, `DSR >= 0.95`, era `Sharpe >= 1.8`, and `corr < 0.20` vs prod. |
| **5** | **#71 (`M16-5`)** | **Portfolio/Synthesis: Orthogonal Multi-Engine Risk Budgeting & Grand 4-Year `1m` Stress-Test (`MaxDD <= 20%`, `$2,500` Floor SAFE)** | Unified `200+` symbol `1m` Grand Portfolio simulation (`100% USDT`) + Forensic Go/No-Go report quantifying exact `% / day`, `Sharpe`, and `MaxDD`. |
| **6** | **#72 (`M16-6`)** | **Prod/Integration: Live Real-Time Multi-Modal Feeds (`5m` OI, `1m` Spot-Perp, `GTX` Snipers), Shadow Burn-In & Mainnet Rollout** | Feature-flagged Clean Architecture adapters, SQLite restart hydration, `inv1–inv6` audit, `--shadow` burn-in, and GCP Tokyo deployment. |

---

## 3. Non-Negotiable Governance Rules

1. **100% USDT Collateral:** Single-Asset `USDT` Cross-Margin (`ONE_WAY` mode); zero `BFUSD` conversion.
2. **Zero Local Market Data:** All raw and intermediate Parquets reside exclusively in `gdrive:trading/crypto-alpha-engine/data/`.
3. **Anti-Overfitting Iron Law:** `2026` remains a **sealed holdout** (never used to select features or thresholds); all multi-trial searches are penalized via **Deflated Sharpe Ratio (`DSR >= 0.95`)**.
4. **Capital Preservation First:** No candidate sleeve may enter `--mainnet` if it pushes 4-year or `2026` `MaxDD > 20.0%` or threatens the `$2,500.00` Hard Equity Floor (`Cushion-Safe CPPI` active at all times).
