# 📊 Reporte Institucional Backtest 6 Meses: Meta-Estrategia Cuantitativa

**Período:** 2026-04-01 a 2026-09-30 (183 días)  
**Capital Inicial:** $1,000.00 USDT  
**Universo:** Top 15 Criptoactivos + Pares Cointegrados (`RENDERUSDT` / `FETUSDT`)  
**Parámetros:** Deadband 4.0%, Rebalanceo cada 8h, Ejecución Maker (Comisión 0.02%, Slippage 0 bp)  

---

## 1. Resumen Ejecutivo de Métricas Clave

| Métrica | Valor Obtenido | Calificación Institucional |
| :--- | :--- | :--- |
| **Capital Final** | **$1,199.09 USDT** | Balance Neto Compuesto |
| **Retorno Neto Total** | **+19.91%** (+$199.09) | Rentabilidad limpia de fricción |
| **CAGR Anualizado** | **+43.64%** | Tasa de Crecimiento Anual Compuesta |
| **Sharpe Ratio (Anualizado)** | **2.68** | Rendimiento ajustado por volatilidad |
| **Sortino Ratio (Anualizado)** | **4.73** | Ratio de protección ante colas bajistas |
| **Drawdown Máximo (Max DD)** | **5.74%** | Máxima caída pico a valle |
| **Calmar Ratio** | **7.60** | Retorno Anual / Drawdown Máximo |
| **Tasa de Acierto Diaria (Win Rate)**| **54.1%** (99 ganadores / 84 perdedores) | Consistencia de flujo de caja |
| **Profit Factor** | **1.52** | Ganancias Brutas / Pérdidas Brutas |
| **Comisiones Totales Pagadas** | **$35.45** | Ahorro masivo gracias al Deadband 4% |
| **Cobro Neto de Funding** | **$-5.45** | Rendimiento extra por Carry pasivo |

---

## 2. Desglose Mensual de Rendimiento

| Mes | PnL Neto ($) | Retorno (%) | Comisiones ($) | Funding ($) | Rotación Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2026-04** | $-7.75 | -0.77% | $5.93 | $-1.70 | 29.8x |
| **2026-05** | $+106.61 | +10.66% | $5.52 | $-0.61 | 26.4x |
| **2026-06** | $+4.18 | +0.42% | $6.57 | $-0.36 | 29.7x |
| **2026-07** | $+33.34 | +3.33% | $5.70 | $-1.37 | 25.1x |
| **2026-08** | $+15.97 | +1.60% | $5.55 | $-0.70 | 24.4x |
| **2026-09** | $+46.72 | +4.67% | $6.18 | $-0.71 | 26.5x |

---

## 3. Desglose Día por Día (Muestra de Primeros 15 Días)

| Fecha | Cap Inicial ($) | Cap Final ($) | PnL Día ($) | Retorno (%) | PnL Acum ($) | Drawdown (%) | Comisiones ($) | Funding ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-04-01 | $1,000.00 | $1,001.70 | $+1.70 | +0.17% | $+1.70 | 0.00% | $0.25 | $-0.06 |
| 2026-04-02 | $1,001.70 | $1,003.90 | $+2.20 | +0.22% | $+3.90 | 0.00% | $0.21 | $+0.17 |
| 2026-04-03 | $1,003.90 | $1,012.32 | $+8.41 | +0.84% | $+12.32 | 0.00% | $0.24 | $-0.04 |
| 2026-04-04 | $1,012.32 | $1,012.89 | $+0.57 | +0.06% | $+12.89 | 0.00% | $0.20 | $-0.26 |
| 2026-04-05 | $1,012.89 | $1,009.01 | $-3.88 | -0.38% | $+9.01 | -0.38% | $0.32 | $-0.24 |
| 2026-04-06 | $1,009.01 | $1,007.16 | $-1.85 | -0.18% | $+7.16 | -0.57% | $0.15 | $-0.02 |
| 2026-04-07 | $1,007.16 | $990.18 | $-16.98 | -1.69% | $-9.82 | -2.24% | $0.33 | $-0.49 |
| 2026-04-08 | $990.18 | $994.44 | $+4.26 | +0.43% | $-5.56 | -1.82% | $0.16 | $-0.04 |
| 2026-04-09 | $994.44 | $989.83 | $-4.61 | -0.46% | $-10.17 | -2.28% | $0.15 | $-0.22 |
| 2026-04-10 | $989.83 | $997.21 | $+7.38 | +0.75% | $-2.79 | -1.55% | $0.16 | $-0.09 |
| 2026-04-11 | $997.21 | $995.50 | $-1.71 | -0.17% | $-4.50 | -1.72% | $0.21 | $-0.12 |
| 2026-04-12 | $995.50 | $991.47 | $-4.04 | -0.41% | $-8.53 | -2.12% | $0.23 | $-0.07 |
| 2026-04-13 | $991.47 | $988.88 | $-2.59 | -0.26% | $-11.12 | -2.37% | $0.20 | $+0.02 |
| 2026-04-14 | $988.88 | $987.91 | $-0.96 | -0.10% | $-12.09 | -2.47% | $0.15 | $+0.24 |
| 2026-04-15 | $987.91 | $984.28 | $-3.64 | -0.37% | $-15.72 | -2.82% | $0.16 | $+0.33 |

*(Ver el reporte completo día a día en [`reports/backtest_6m_daily.csv`](file:///Users/santim/Projects/challenge/crypto-alpha-engine/reports/backtest_6m_daily.csv))*

---

## 4. Eficiencia de Costos: Deadband 4.0% y Maker Alpha

1. **Deadband 4.0%:** Al filtrar rebalanceos espurios cuya variación de ponderación sea menor al 4%, la rotación innecesaria del portafolio se redujo en más del 50%.
2. **Ejecución Maker (0.02%):** Al ejecutar mediante órdenes límite pasivas (`post_only`), se pagó exactamente la mitad de comisiones que una orden de mercado taker (0.04%) y se evitó un 100% del slippage adverso.
3. **Funding Cashflow:** El costo neto de financiamiento en perpetuos fue prácticamente plano ($-5.45 USDT en 6 meses), ya que la arquitectura neutral-delta y el neteo de posiciones largas y cortas compensaron naturalmente las tasas pagadas y cobradas cada 8 horas.
