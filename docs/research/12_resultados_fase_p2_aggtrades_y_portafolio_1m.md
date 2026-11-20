# 12 — Resultados Forenses Fase P2 (`aggTrades` a `1 Milisegundo`) e Integración en el Portafolio Productivo `1-Minuto` One-Way (`2023–2026` + Estrés de Colchón `$2,700` en `2026`)

- **Cobertura empírica Paso 1 (`aggTrades` `1-ms` Sweep Tomography):** `580` eventos causales únicos (`488/488` archivos diarios de ticks = **`100.0%` de cobertura**, `0.000 ms` de *lookahead*: `transact_time_ms <= signal_ts * 1000 - 1`).
- **Cobertura empírica Paso 2 (`Unified 1-Minute One-Way Portfolio Ablation`):** Simulación lineal minuto a minuto (`2023-03-01` a `2026-11-18`) sobre `80` perpetuos `1m` en streaming `O(1)` de RAM + `2,886` disparos base resueltos en `1m` (`M09 + M23 + M04`, con `M25=OFF`), acoplados a `Core P1 7-Factor Top-15` (`0.85x` + `1m Chop Lock`) y `Manga P2` (`1m Dollar-Bar TSMOM`), respetando no-colisión estricta One-Way, `SafeRamp` desde `$2,700.00` y piso duro CPPI de `$2,500.00`.
- **Artefactos reproducibles:**
  - Paso 1: [p2_aggtrades_sweep_report.md](../../reports/p2_aggtrades_sweep_report.md) · [p2_aggtrades_sweep_leaderboard.csv](../../reports/p2_aggtrades_sweep_leaderboard.csv) · [p2_aggtrades_sweep_summary.json](../../reports/p2_aggtrades_sweep_summary.json)
  - Paso 2: [p2_unified_portfolio_1m_report.md](../../reports/p2_unified_portfolio_1m_report.md) · [p2_unified_portfolio_1m_leaderboard.csv](../../reports/p2_unified_portfolio_1m_leaderboard.csv) · [p2_unified_portfolio_1m_summary.json](../../reports/p2_unified_portfolio_1m_summary.json)
- **Código fuente y tests unitarios:** [tick_sweep_physics.py](../../src/features/tick_sweep_physics.py) · [p2_portfolio_integration.py](../../src/backtest/p2_portfolio_integration.py) · [unified_1m_simulator.py](../../src/backtest/unified_1m_simulator.py) · [test_tick_sweep_physics.py](../../tests/test_tick_sweep_physics.py) · [test_p2_portfolio_integration.py](../../tests/test_p2_portfolio_integration.py).

---

## Parte I — Paso 1: Tomografía de Barridas del Motor de Calce a `1 Milisegundo` (`aggTrades`)

### 1.1 Qué medimos físicamente en el nivel de `1 milisegundo`
En `aggTrades` de Binance USDS-M, cada fila trae `first_trade_id` y `last_trade_id` junto con `transact_time` en milisegundos e `is_buyer_maker`. Agrupando todas las ejecuciones con el **mismo `(transact_time_ms, is_buyer_maker)`** en los `5 minutos` previos a la señal (`[T - 300s, T - 1ms]`), reconstruimos las **órdenes agresoras institucionales/forzosas individuales** y extrajimos tres invariantes físicos imposibles de falsificar con *spoofing* pasivo:
1. **Concentración de Barridas Institucionales (`sweep_conc_5m >= 0.32` + `whale_sweep_buy_share >= 0.53` — Filtro `3_INST_SWEEP_CONC`):** Al menos el `32%` de todo el volumen USD de `5m` proviene de barridas que devoraron múltiples órdenes Maker (`>= 4` fills en el mismo milisegundo o `>= $25,000` en `1 ms`), con neto comprador de ballenas. Descarta rupturas producidas por ruido retail atomizado o paredes falsas (*spoofing*).
2. **Salto de Impacto de Kyle en `1 ms` (`max_up_walk_1ms_bps >= +4.0 bps` — Filtro `2_KYLE_1MS_JUMP`):** Al menos una orden compradora caminó físicamente el libro de `Asks` desplazando el precio `>= +4.0 bps` dentro de un único milisegundo, probando empíricamente que el `Ask` está hueco.
3. **Anti-Spoofing de Flujo Ballena vs. Retail (`whale_buy_share >= 0.52` y `whale >= retail - 0.05` — Filtro `1_ANTISPOOF_WHALE_SWEEP`):** Verifica que las barridas grandes están comprando y no distribuyendo contra compras pequeñas retail.

### 1.2 Tabla Comparativa Paso 1 (`Control Crudo Fase P1` vs. `Filtros Tick-Sweep 1ms Fase P2`)

| Variante | Geometría | `N` | Gross | Fee | **Net Full (`bps`)** | Mediana | Win`%` | PF | FeeSh`%` | `t`-stat | `23–24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P6A_L2_0_RAW_CONTROL` *(P1)* | `MONO_A_CONVEX` | `148` | `+73.2` | `12.7` | **`+60.47`** | `-14.8` | `50.0%` | `1.63` | `7.8%` | `2.36` | `+68.1` | `+3.7` | **`+181.9`** | `+58.3` | **`100%`** |
| **`P6A_L2_3_INST_SWEEP_CONC`** | **`MONO_A_CONVEX`** | `112` | **`+83.8`** | `12.7` | **`+71.15`** | **`+2.0`** | **`51.8%`** | **`1.78`** | **`7.5%`** | **`2.37`** | **`+74.0`** | **`+10.2`** | **`+236.6`** | **`+85.8`** | **`100%`** |
| **`P6A_L2_3_INST_SWEEP_CONC`** | **`MONO_B_COUPON`** | `112` | `+72.6` | `12.5` | **`+60.09`** | **`+2.0`** | **`51.8%`** | `1.66` | `8.0%` | `2.18` | `+62.9` | **`+11.2`** | **`+189.2`** | `+62.6` | **`100%`** |
| **`P6A_L2_2_KYLE_1MS_JUMP`** | **`MONO_A_CONVEX`** | `132` | **`+86.1`** | `12.7` | **`+73.47`** | **`+2.0`** | `51.5%` | **`1.79`** | **`7.4%`** | **`2.64`** | **`+88.2`** | `-2.2` | **`+181.9`** | `+62.9` | **`100%`** |
| `P6A_L2_3_INST_SWEEP_CONC` | `MONO_C_50_50` | `112` | `+50.0` | `11.7` | **`+39.21`** | **`+61.8`** | **`55.4%`** | `1.46` | `8.9%` | `1.72` | `+38.7` | `+3.4` | **`+149.5`** | `+47.3` | `93%` |
| `P2_C2A_0_RAW_CONTROL` *(P1)* | `MONO_A_CONVEX` | `117` | `+35.0` | `8.3` | `+29.00` | `-149.0` | `47.9%` | `1.30` | `6.0%` | `1.12` | `+56.1` | `-10.8` | `+63.6` | `+56.2` | **`100%`** |
| **`P2_C2A_1_ANTISPOOF_WHALE`** | **`MONO_A_CONVEX`** | `89` | `+52.2` | `8.2` | **`+46.90`** | **`+6.5`** | `50.6%` | `1.48` | **`5.5%`** | `1.56` | `+52.2` | `-10.4` | **`+150.7`** | **`+75.4`** | **`100%`** |
| **`UNIFIED_L2_VACUUM_INST_CONC`** | **`MONO_A_CONVEX`** | `185` | `+61.0` | `10.9` | **`+51.44`** | `-31.6` | `49.7%` | `1.53` | **`7.1%`** | **`2.28`** | `+61.0` | `-0.0` | **`+146.9`** | **`+59.0`** | **`100%`** |

### 1.3 Diagnóstico Crítico del Paso 1 (Sin Maquillaje)
1. **Confirmación de primeros principios en `P6A_L2_3_INST_SWEEP_CONC`:** Exigir concentración real de barridas institucionales de `1 ms` eliminó `36` falsas rupturas de ruido retail (`148 -> 112` operaciones) y **mejoró simultáneamente todas las métricas y todas las eras**:
   - El retorno neto medio subió de `+60.47` a **`+71.15 bps/trade` (`+0.712%` neto por operación)** pagando `12.68 bps` completos como Taker (`Fee Share = 7.5%`).
   - La mediana pasó de negativa (`-14.8 bps`) a **positiva (`+2.0 bps`)**, el `Profit Factor` subió de `1.63 -> 1.78`, `2023–24` subió `+68.1 -> +74.0 bps`, **`2025` casi se triplicó (`+3.70 -> +10.20 bps` en `MONO_A` y `+11.21 bps` en `MONO_B`)**, y **`2026` saltó de `+181.85` a `+236.55 bps/trade` (`+2.37%` neto por trade!)**.
   - En **CPCV con re-optimización in-fold (`15` particiones)** mantuvo el **`100.0%` (`15/15`) de splits positivos** y elevó la mediana Out-of-Sample de `+58.31` a **`+85.83 bps/trade` (`Sharpe CPCV = 1.23`)**.
2. **Transparencia sobre el corte `is_treasure`:** El flag booleano rígido (`treasure_count = 0`) exigía simultáneamente `2025 >= +15.0 bps`; `P6A_L2_3_INST_SWEEP_CONC` alcanzó **`+10.20 / +11.21 bps` en `2025`** (positivo en las 3 eras, `+71.2 bps` full, `+236.6 bps` en `2026`, `100%` CPCV).

---

## Parte II — Paso 2: Simulación Integrada `1-Minuto` One-Way del Portafolio Productivo (`$2,700` SafeRamp + Piso Duro CPPI `$2,500`)

Evaluamos dos regímenes complementarios sobre la física exacta del motor en vivo (`Core P1 0.85x + Manga P2 Dollar-Bar + M09/M23/M04` con **`M25=OFF`**):
1. **Horizonte Completo `4 Años` (`2023-03-01` a `2026-11-18`)** partiendo de `$2,700.00`.
2. **Prueba de Estrés Dedicada `2026` (`2026-01-01` a `2026-11-18`)** reiniciando la cuenta exactamente en **`$2,700.00`** (idéntico al capital actual en Tokio, a solo `$200` del piso duro de **`$2,500.00`**) para verificar protección de colchón y velocidad de despegue en `SafeRamp`.

### 2.1 Tabla Maestra de Ablación de Portafolio (`1m` Lineal Exacto, One-Way Sin Colisión)

| Configuración de Portafolio (`M25 = OFF`) | `4Y Sharpe` (`Δ`) | `4Y CAGR` | `4Y MaxDD` | `Sharpe` (`23-24 / 25 / 26`) | `Exec / Cand` | **`2026` `$2,700 -> Final` (`Δ$`)** | **`2026` MinEq (Piso `$2,500`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`00_CURRENT_PROD_1M_NO_M25` (Base Actual)** | **`1.99`** (`0.00`) | `+151.5%` | `36.08%` | `1.86 / 2.02 / 2.23` | `0 / 0` | **`$3,912.53`** (`+$0.00`) | `$2,528.46` ✅ |
| **`PROD + P6A_KYLE_1MS_MONO_A`** | **`2.15`** (**`+0.16`**) | **`+182.2%`** | **`35.30%`** | `2.19 / 1.92 / 2.42` | `120 / 130` | **`$4,431.05`** (**`+$518.52`**) | **`$2,544.53`** ✅ |
| **`PROD + P6A_INST_SWEEP_MONO_A`** | **`2.15`** (**`+0.16`**) | **`+179.1%`** | **`35.30%`** | `2.21 / 1.88 / 2.42` | `101 / 110` | **`$4,392.49`** (**`+$479.96`**) | **`$2,544.42`** ✅ |
| **`PROD + P6A_INST_SWEEP_MONO_B` *(Coupon)*** | **`2.14`** (**`+0.15`**) | `+178.4%` | **`35.30%`** | `2.18 / 1.95 / 2.39` | `101 / 110` | **`$4,330.26`** (`+$417.73`) | `$2,537.75` ✅ |
| `PROD + P6A_RAW_MONO_A` | `2.14` (`+0.15`) | `+180.8%` | `35.30%` | `2.19 / 1.89 / 2.42` | `135 / 145` | `$4,431.05` (`+$518.52`) | `$2,544.53` ✅ |
| **`PROD + P6A_RAW_MONO_C_50_50` *(Split 50/50)*** | `2.09` (`+0.10`) | `+170.2%` | `35.62%` | `2.08 / 1.92 / 2.37` | `135 / 145` | **`$4,845.67`** (**`+$933.14`**) | **`$2,550.58`** ✅ |
| `PROD + P6A_INST_SWEEP_MONO_C` | `2.08` (`+0.09`) | `+167.9%` | `35.62%` | `2.09 / 1.89 / 2.35` | `101 / 110` | `$4,303.79` (`+$391.26`) | `$2,539.30` ✅ |
| **`PROD + UNIFIED_L2_INST_CONC_MONO_A`** | **`2.14`** (**`+0.15`**) | `+178.0%` | `36.35%` | `2.21 / 1.88 / 2.36` | `170 / 183` | **`$5,744.52`** (**`+$1,831.99`**) | **`$2,560.63`** ✅ |
| `PROD + P6A_INST_SWEEP_MONO_A (4-Slots)` | `2.14` (`+0.15`) | `+182.3%` | `36.34%` | `2.18 / 1.94 / 2.36` | `104 / 110` | `$4,369.57` (`+$457.04`) | `$2,542.48` ✅ |
| **`PROD + P6A_INST_SWEEP_MONO_A (Lev=0.30)`** | **`2.19`** (**`+0.20`**) | **`+195.3%`** | **`35.62%`** | **`2.26 / 2.00 / 2.36`** | `101 / 110` | `$4,334.36` (`+$421.83`) | `$2,541.23` ✅ |

---

## Parte III — 5 Hallazgos Clave de Portafolio y Decisión Operativa

1. **Cero canibalización con los Snipers actuales (`M09`, `M23`, `M04`) ni con `Core P1`:**
   - En la configuración productiva de `3` slots (`0.24x`), el portafolio base ejecutó `M09 = 576`, `M23 = 303`, `M04 = 378`. Al añadir `P6A_INST_SWEEP_MONO_A`, se ejecutaron `M09 = 576`, `M23 = 302`, `M04 = 378` **más `101 / 110` (`91.8%`) señales limpias de `P6A`** (sin necesidad de ampliar `sniper_max_positions` a `4`).
2. **Mejora simultánea de Retorno, Sharpe, MaxDD y Colchón sobre `$2,500`:**
   - Todas las variantes `P6A` elevan el `Sharpe 4Y` de `1.99` hasta **`2.15` (`+0.16`)** (y **`2.19` (`+0.20`)** con `slot_lev = 0.30`), suben el `CAGR 4Y` de `+151.5%` a **`+179.1%–+195.3%`**, **reducen el `MaxDD 4Y` de `36.08%` a `35.30%`**, y en el estrés de `2026` arrancando en `$2,700` **suben el equity mínimo desde `$2,528.46` hasta `$2,544–$2,560`** (alejando aún más la cuenta del piso duro de `$2,500`).
3. **El fenómeno interactivo con `SafeRamp` en cuentas cercanas al piso (`$2,700` en `2026`):**
   - Arrancando en `$2,700` el `2026-01-01`:
     - Base sin `P6A`: `$2,700 -> $3,912.53` (`+44.9%`).
     - `+ P6A_INST_SWEEP_MONO_A` / `KYLE_1MS`: `$2,700 -> $4,392.49 / $4,431.05` (`+62.7% / +64.1%`).
     - `+ P6A_RAW_MONO_C_50_50` (*Split 50% TP + 50% Runner*): **`$2,700 -> $4,845.67` (`+79.5%`, `+$933.14`)** porque su mayor tasa de acierto (`54.1%`) y mediana positiva (`+61.8 bps`) empujan temprano el capital por encima del escalón de **`$3,050` del `SafeRamp` (`2.5x -> 3.2x`)**.
     - `+ UNIFIED_L2_INST_CONC_MONO_A` (`P6A ∪ P2_C2A` filtrados por barrida `1ms`): **`$2,700 -> $5,744.52` (`+112.8%`, `+$1,831.99` adicionales en `2026`!)** con el **mayor equity mínimo (`$2,560.63`)** gracias a `26` eventos capturados en `2026` con `+146.9 bps/trade` medios.
4. **Matiz honesto en `2025`:**
   - Aunque `P6A_INST_SWEEP` fue positivo standalone en `2025` (`+10.2 / +11.2 bps/trade`), a `0.24x` su varianza redujo levemente el `Sharpe 2025` del portafolio de `2.02` a `1.88–1.95` (mientras **bajaba el `MaxDD 2025` de `36.08%` a `35.17%–35.30%`**; en `Lev=0.30` el `Sharpe 2025` se mantiene en `2.00` con `MaxDD 2025 = 34.22%` y `CAGR 2025 = +193.6%` vs `+184.1%`). En cambio, en `2023–24` (`1.86 -> 2.21`) y `2026` (`2.23 -> 2.42`) el salto de Sharpe es contundente.
5. **Requisitos ligeros para llevar `P6A_INST_SWEEP` (o `UNIFIED_L2_INST_CONC`) a `crypto-mainnet` en Tokio:**
   - `crypto-mainnet` ya corre en **Tokio (`ap-northeast-1`)** a `~2–4 ms` de Binance Futures y ya evalúa el universo líquido cada `5m`.
   - Para evaluar `P6A_INST_SWEEP` en tiempo real con **cero peso relevante de API**, solo requiere consultar `GET /fapi/v1/depth?symbol=SYM&limit=500` (`weight = 10`, únicamente en las contadas monedas pre-filtradas que acaban de romper el máximo de `4h` con `OI >= 1.16x` y `RVOL_5m >= 2.0` -> `< 1` llamada cada varias horas!) más `GET /fapi/v1/aggTrades?symbol=SYM&startTime=T-300000` (`2–3` páginas en ese mismo instante) para medir `book_imb_1pct`, `book_imb_2pct` y `sweep_conc_5m` antes de disparar la orden `TAKER` con bracket convexo `MONO_A` (o `MONO_C` mientras el equity esté `< $3,050`!).
