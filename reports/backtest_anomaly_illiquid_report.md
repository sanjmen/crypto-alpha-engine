# 6-Month Backtest Report: Extreme Anomaly & Illiquid Tail Strategies

- **Horizon**: 2026-04-14 to 2026-10-14 (Last 183 days up to today 2026-10-14)
- **Starting Capital**: $1,000.00 USDT
- **Universe Size**: 23 assets (Core pairs + Illiquid tail + Anomaly pump/dump movers)
- **Execution Model**: Maker Limit Orders (0.02% fee, 0.0% slippage) + 4.0% Turnover Deadband + 8h Funding Settlements

## 1. Executive Performance Comparison

| KPI Metric | Baseline (Core Meta) | Anomaly Test Mode ($15 Sizing) | Anomaly Prod Mode ($100 Sizing) |
| :--- | :---: | :---: | :---: |
| **Final Equity** | $397.19 | $415.00 | **$475.82** |
| **Total Net Return** | -60.28% | -58.50% | **-52.42%** |
| **Annualized Return (CAGR)** | -83.98% | -82.53% | **-77.08%** |
| **Sharpe Ratio** | -3.01 | -2.90 | **-2.22** |
| **Sortino Ratio** | -3.80 | -3.75 | **-2.88** |
| **Max Drawdown (MDD)** | -61.04% | -59.13% | -55.08% |
| **Daily Win Rate** | 40.2% (74W / 110L) | 40.8% (75W / 109L) | **41.3%** (76W / 108L) |
| **Profit Factor** | 0.62 | 0.63 | **0.67** |
| **Total Maker Fees Paid** | $27.79 | $27.68 | $28.32 |
| **Perpetual Funding PnL** | $+23.71 | $+23.87 | **$+25.47** |

## 2. Event-Driven Anomaly Signals Breakdown

- **❄️ Oversold Bounce Signals**: 67 events captured (violent dumps >15% + RSI < 28)
- **⚡ Early Breakout Signals**: 48 events captured (+8% 4h impulse + 2.5x volume)
- **🔥 Pump Funding Fade Signals**: 18 events captured (+25% 24h pump + funding rate > +0.05%)

## 3. Daily Capital Curve (Last 15 Days)

| Date | Start Equity | End Equity | Daily PnL | Daily Return | Cumulative Return | Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-09-30 | $508.46 | $489.35 | $-19.11 | -3.76% | -51.07% | -51.47% |
| 2026-10-01 | $489.35 | $508.19 | $+18.85 | +3.85% | -49.18% | -49.60% |
| 2026-10-02 | $508.19 | $493.30 | $-14.90 | -2.93% | -50.67% | -51.08% |
| 2026-10-03 | $493.30 | $471.11 | $-22.19 | -4.50% | -52.89% | -53.28% |
| 2026-10-04 | $471.11 | $487.24 | $+16.13 | +3.42% | -51.28% | -51.68% |
| 2026-10-05 | $487.24 | $512.83 | $+25.60 | +5.25% | -48.72% | -49.14% |
| 2026-10-06 | $512.83 | $531.05 | $+18.22 | +3.55% | -46.89% | -47.33% |
| 2026-10-07 | $531.05 | $540.06 | $+9.01 | +1.70% | -45.99% | -46.44% |
| 2026-10-08 | $540.06 | $589.66 | $+49.59 | +9.18% | -41.03% | -41.52% |
| 2026-10-09 | $589.66 | $582.24 | $-7.42 | -1.26% | -41.78% | -42.26% |
| 2026-10-10 | $582.24 | $567.85 | $-14.38 | -2.47% | -43.21% | -43.68% |
| 2026-10-11 | $567.85 | $584.88 | $+17.03 | +3.00% | -41.51% | -41.99% |
| 2026-10-12 | $584.88 | $529.39 | $-55.49 | -9.49% | -47.06% | -47.50% |
| 2026-10-13 | $529.39 | $519.15 | $-10.24 | -1.94% | -48.09% | -48.51% |
| 2026-10-14 | $519.15 | $475.82 | $-43.33 | -8.35% | -52.42% | -52.81% |
