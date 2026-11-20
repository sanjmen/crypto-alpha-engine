# 14 — Resultados Forenses Paso 1: Auditoría `1-Minuto` de la Barrera de Rebalanceo, Rebalanceos Intraciclo e Histéresis en `Core P1` (`2023–2026`)

## 1. Motivación Empírica desde Producción Real (`live_trading.db` en Tokio)

En la auditoría en vivo ejecutada con [inspect_production_status.py](../../scripts/inspect_production_status.py) sobre `crypto-mainnet` (`2026-11-20`), observamos una asimetría contundente entre las **3 barreras de salida**:
1. **`TAKE_PROFIT` (`+5.0%` Core / `+3.5%` Chop):** `22` cierres (`100%` ganadores) -> **`+$437.17` netos**.
2. **`STOP_LOSS` / `Profit-Lock` (`+3.5% -> +1.5%`):** `65` cierres (`64.6%` ganadores) -> **`+$117.18` netos**.
3. **Barrera #1 más tocada — Rebalanceo (`LIMIT` / `MARKET`):** `273` cierres (`75.8%` del total) -> **`-$652.22` netos**, con **3 rebalanceos intraciclo fuera de horario en solo 19 horas** (`17:40`, `22:44` y `03:16 UTC` además de `08:00`, `16:00` y `00:00 UTC`) disparados por `enable_intracycle_rebalance=True` ([shadow_daemon.py](../../src/execution/shadow_daemon.py#L3184-L3241)).

### El Mecanismo Físico del Sangrado Intraciclo
Cuando `2` posiciones de un mismo lado (por ejemplo `Shorts`) tocan `Take-Profit` o `Profit-Lock` a mitad de una vela de `8 horas` durante una caída rápida:
* Las posiciones del lado opuesto (`Longs`) están temporalmente en retroceso intrabarra (`-2%` a `-4%`).
* `is_intracycle_rebalance_needed()` ve que la pata *Short* bajó respecto a la pata *Long* (`net_drift >= 25%`) y dispara **`run_rebalance_cycle()` completo en pleno piso intrabarra**:
  1. **Cristaliza pérdidas flotantes** en los `Longs` justo en el extremo intrabarra (`75–120 min` antes del cierre natural de `8h`).
  2. **Abre nuevos `Shorts` en el piso de la caída** (o rota toda la cartera pagando *spread* y comisiones dos veces: en el intraciclo y otra vez a las `:00 / :08 / :16 UTC`).

---

## 2. Diseño del Experimento `1-Minuto` (`2023–2026` + `$2,700` `2026` Cushion + `15`-Split CPCV)

Mediante [core_rebalance_barrier_lab.py](../../src/backtest/core_rebalance_barrier_lab.py), simulamos a resolución exacta de **`1 minuto` lineal** (`80` símbolos `1m`, `zero np.log`, colisión `One-Way` estricta) todo el motor productivo actual (**`Core P1 + Manga P2 + M09/M23/M04 + M26_L2SweepAirPocket`**, `M25=OFF`, `SafeRamp $2,700` y piso duro CPPI `$2,500.00`), comparando `10` configuraciones de barrera de rebalanceo.

Fuente reproducible: [core_rebalance_barrier_report.md](../../reports/core_rebalance_barrier_report.md) y [core_rebalance_barrier_summary.json](../../reports/core_rebalance_barrier_summary.json).

| Configuración | `4Y Sharpe` | `4Y CAGR` | `4Y MaxDD` | `SR 23–24 / 2025 / 2026` | `CPCV SR (Min / Pos%)` | Rebal. Intraciclo | Cierres Rebal. | Micro-Ajustes | `FeeDrag%` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00_LIVE_INTRACYCLE_FULL_RERANK` *(Actual Vivo)* | `2.02` | `+168.8%` | `38.23%` | `2.16 / 1.74 / 2.12` | `2.01` (`0.61` / `100%`) | **`767`** | `6,791` | `19,643` | `256.3%` | **`$3,633.64`** (`$2,502.85` ⚠️ al filo) |
| **`01_STRICT_8H_NO_INTRACYCLE`** *(Reloj 8h Puro)* | **`2.19`** | **`+188.1%`** | **`35.23%`** | **`2.25 / 1.95 / 2.42`** | **`2.17` (`0.86` / `100%`)** | **`0`** | `6,268` | `15,537` | `245.9%` | **`$5,050.17`** (**`$2,550.02`** ✅) |
| `02_COLLAPSED_TRIM_ONLY_INTRACYCLE` | `2.06` | `+157.3%` | `37.52%` | `2.27 / 1.92 / 1.89` | `2.04` (`0.82` / `100%`) | `0` (`197` trims) | `6,247` | `16,143` | `250.8%` | `$4,253.59` (`$2,519.99` ✅) |
| `03_PARTIAL_TRIM_BUFFER_3.5PCT` | `2.14` | `+171.7%` | `37.54%` | `2.40 / 1.93 / 1.94` | `2.11` (`0.81` / `100%`) | `0` | `6,248` | **`11,914`** | `251.6%` | `$4,351.23` (`$2,547.66` ✅) |
| `04_HYSTERESIS_BHS_3_7_DB_1.75` | `2.05` | `+154.1%` | `38.13%` | `2.16 / 1.95 / 2.02` | `2.03` (`0.77` / `100%`) | `0` | `6,093` | `16,123` | `246.4%` | `$4,485.56` (`$2,525.16` ✅) |
| `05_HYSTERESIS_BHS_3_8_DB_2.25` | `2.00` | `+144.8%` | `37.80%` | `1.93 / 1.95 / 2.21` | `1.99` (`0.89` / `100%`) | `0` | `5,946` | `15,643` | `240.5%` | `$4,892.43` (`$2,555.17` ✅) |
| `06_FAST_SWING_TSTAT_GATE_0.50` | `1.97` | `+143.1%` | **`32.10%`** | `2.03 / 2.19 / 1.68` | `1.95` (`0.95` / `100%`) | `0` | `6,167` | `15,830` | `247.2%` | `$4,022.30` (`$2,547.01` ✅) |
| `07_SCORE_SMOOTH_EMA_6H` | `1.96` | `+143.5%` | `34.44%` | `1.70 / 2.34 / 2.00` | `1.96` (`1.11` / `100%`) | `0` | `5,906` | `16,243` | `238.0%` | `$4,325.07` (`$2,522.36` ✅) |
| **`08_UNIFIED_SHIELD_A`** *(BHS 3,7 + DB 1.75 + Buf 3.5%)* | **`2.18`** | `+179.8%` | `37.76%` | `2.39 / 1.96 / 2.09` | **`2.16` (`0.89` / `100%`)** | `0` | `6,079` | **`11,976`** | `252.0%` | **`$5,083.09`** (`$2,549.29` ✅) |
| **`09_UNIFIED_SHIELD_B_FULL`** *(Defensivo MaxDD)* | **`2.15`** | `+173.3%` | **`30.15%`** | `2.08 / 2.47 / 1.92` | **`2.15` (`1.15` / `100%`)** | `0` | **`5,670`** | **`11,638`** | **`236.7%`** | `$4,468.09` (`$2,541.61` ✅) |

---

## 3. Hallazgos Clave y Acción Quirúrgica de Mayor Impacto

1. **Apagar `enable_intracycle_rebalance` (`00 -> 01_STRICT_8H_NO_INTRACYCLE`) es una mejora estricta de Pareto en TODAS las métricas y en TODAS las épocas:**
   * **Sube `4Y Sharpe` de `2.02 -> 2.19` (`+0.17`)** y **`4Y CAGR` de `+168.8% -> +188.1%`**.
   * **Reduce `4Y MaxDD` en `-3.00%`** (`38.23% -> 35.23%`).
   * **Mejora las 3 épocas sin excepción:** `2023–24` (`2.16 -> 2.25`), `2025` (`1.74 -> 1.95`) y **`2026` (`2.12 -> 2.42`)**.
   * **Elimina `767` rebalanceos fuera de horario, `523` cierres prematuros y `4,106` micro-ajustes**, ahorrando `-10.3%` de *fee drag* directo.
   * **En la prueba de estrés real `2026` arrancando con `$2,700`:** el capital final salta de `$3,633.64` a **`$5,050.17` (`+$1,416.53` adicionales!)** y el equity mínimo sube de `$2,502.85` (a solo `$2.85` de tocar el piso) a **`$2,550.02` (`+$47.17` más de margen sobre `$2,500`)**.
2. **Incluso recortar `reduceOnly` a mitad de ciclo (`02_COLLAPSED_TRIM_ONLY`) rinde menos que esperar al reloj de `8h` (`2.06` vs `2.19`):**
   * Cuando una pata toca `Take-Profit` rápido dentro de las `8h`, dejar correr la pata sobreviviente con su propio `Take-Profit / Profit-Lock` hasta el cierre natural de `00:00 / 08:00 / 16:00 UTC` captura reversiones a la media que el rebalanceo intraciclo cortaba en pérdida.
3. **Segundo nivel opcional (`08_UNIFIED_SHIELD_A` vs `09_UNIFIED_SHIELD_B_FULL`):**
   * Sumar el **Buffer Anti-Micro-Recortes de `3.5%` del Equity (`~$95`)** elimina casi el **`40%` (`-7,667`) de las órdenes chicas de ajuste parcial** (las de `$31–$51` que vimos en `live_trading.db`) manteniendo `4Y Sharpe = 2.18` y alcanzando **`$5,083.09`** en `2026`.
   * Y si se prioriza mínima caída histórica sobre retorno bruto en `2026`, `09_UNIFIED_SHIELD_B_FULL` (`FastGate >= 0.50 + EMA 6h`) comprime el `4Y MaxDD` de `38.23%` hasta **`30.15%`** y sube `2025` a `Sharpe = 2.47`.
