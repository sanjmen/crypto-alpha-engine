# 15 — Resultados Forenses Paso 2: Física `1-Milisegundo` (`aggTrades`) de Clímax, Desaceleración y Absorción en `P4_M4` (`2023–2026`)

## 1. Resumen Ejecutivo y Veredicto (`NO-GO` para `P4_M4` Taker; Hallazgo Clave de Microestructura)

Completamos el **Paso 2** evaluando los **`281` eventos únicos de liquidación anclada (`P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG`)** con **`100.0%` de cobertura (`268/268` archivos diarios de `aggTrades`)** a resolución exacta de **`1 milisegundo`** y causalidad estricta (`transact_time_ms <= entry_ts * 1000 - 1`, **`0.000 ms` de lookahead**), mediante [liquidation_absorption_p4_lab.py](../../src/models/liquidation_absorption_p4_lab.py).

Fuente reproducible: [p4_m4_liquidation_aggtrades_report.md](../../reports/p4_m4_liquidation_aggtrades_report.md) y [p4_m4_liquidation_aggtrades_summary.json](../../reports/p4_m4_liquidation_aggtrades_summary.json).

* **Sobrevivientes de las 4 Compuertas Universales (`>= +25 bps` Full, `>= +15 bps` en `23-24` y `2025`, `>= +20 bps` en `2026`, `CPCV >= 80%`):** **`0 / 18` (`NO-GO` estricto — NO se despliega `P4_M4` en producción).**

---

## 2. Tabla de Resultados (`281` Eventos, `100.0%` Cobertura `aggTrades 1-ms`)

| Variante (`1-ms` `aggTrades`) | Geometría | `N` | Gross (`bps`) | Fee (`bps`) | **Net Full (`bps`)** | Win `%` | `23–24` (`bps`) | **`2025` (`bps`)** | **`2026` (`bps`)** | **CPCV OOS (`bps`)** | **CPCV `%`** | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`P4_M4_2_CASCADE_EXHAUSTION_DECEL`** *(Desaceleración de barridas vendedoras en el último `1m` `<= 0.45x`)* | `MONO_C_SPLIT_50_50` | `219` | `+16.4` | `11.9` | **`+4.72`** | `50.2%` | `-25.9` | **`+13.4`** | **`+62.9`** | **`+60.0`** | `73%` | **Mejor variante** (da vuelta `2025` a positivo, pero falla `23–24`) |
| `P4_M4_0_RAW_CONTROL` *(Sin filtro `1-ms`)* | `MONO_C_SPLIT_50_50` | `281` | `+14.4` | `12.0` | **`+2.65`** | `49.8%` | `-17.8` | **`-8.4`** | **`+70.9`** | `+54.2` | `73%` | Falla `23–24` y `2025` |
| `P4_M4_1_TRUE_LIQ_CLIMAX_1MS` *(`makers >= 12` en `1ms`)* | `MONO_C_SPLIT_50_50` | `280` | `+13.8` | `12.0` | **`+2.09`** | `49.6%` | `-19.1` | **`-8.4`** | **`+70.9`** | `+54.2` | `73%` | `280/281` ya tenían clímax real |
| `P4_M4_2_CASCADE_EXHAUSTION_DECEL` | `MONO_A_CONVEX_TRAIL` | `219` | `+12.4` | `13.0` | **`-0.22`** | `43.4%` | `-22.9` | **`-2.8`** | **`+54.0`** | `+52.4` | `67%` | Falla `23–24` |
| `P4_M4_3_WHALE_ABSORPTION_1M` *(Barridas Taker compradoras `>= 56%`)* | `MONO_C_SPLIT_50_50` | `156` | `-2.1` | `12.1` | **`-13.83`** | `45.5%` | `-23.1` | **`-54.1`** | **`+80.9`** | `-11.9` | `40%` | **Trampa física descubierta (ver §3)** |
| `P4_M4_5_STRICT_INST_SNAPBACK` | `MONO_C_SPLIT_50_50` | `84` | `-29.1` | `12.2` | **`-41.11`** | `39.3%` | `-56.1` | **`-13.6`** | `-22.1` | `-50.1` | `13%` | Sobre-retraso de entrada |

---

## 3. Por qué `aggTrades` `1-ms` funcionó distinto en Rupturas (`M26` 🏆) vs. Caídas de Liquidación (`P4_M4` ❌)

Este contraste entre **Fase P2 (`M26`)** y **Paso 2 (`P4_M4`)** revela una ley fundamental de cómo operan las instituciones en el motor de calce de Binance:

1. **La desaceleración vendedora (`P4_M4_2`) SÍ dio vuelta `2025` de `-8.4 bps` a `+13.4 bps`:**
   * Exigir que las barridas vendedoras de `1-ms` en el último minuto hayan caído a `< 45%` de su pico previo eliminó `62` trampas de cuchillo cayendo y llevó la mediana por operación a positiva (`+7.78 bps`, `2025 = +13.4 bps`, `2026 = +62.9 bps`, `CPCV OOS = +60.0 bps`).
2. **Por qué exigir barridas Taker COMPRADORAS en el fondo (`P4_M4_3` / `P4_M4_5`) empeoró `2025` a `-54.1 bps`:**
   * En una **ruptura alcista (`M26`)**, la institución o liquidación forzosa necesita cruzar el *spread* como **Taker Comprador (`1-ms` Buy Sweep)** porque el precio se escapa hacia arriba.
   * En cambio, en el **fondo de una cascada de liquidación (`P4_M4`)**, los verdaderos *market makers* institucionales absorben las liquidaciones **de forma PASIVA sentados en el `Bid` (`Maker` Limit Buy)** — lo cual aparece en `aggTrades` como *agresor vendedor* (`is_buyer_maker = True`) comiéndose el muro pasivo. Cuando en el minuto del fondo ya hubo barridas Taker compradoras masivas (`>= 60%` en `P4_M4_5`), **el rebote elástico de la mecha ya ocurrió dentro de esa vela de `1m`**; entrar `Taker` al cierre de esa vela compra tarde el techo del rebote corto y paga `12–13 bps` de comisión.
3. **Por qué `2023–2024` siguió en `-25.9 bps`:**
   * Comprar después del cierre de una vela `1m` verde tras una liquidación regala los primeros `25–45 bps` de la mecha y paga comisión Taker. Para monetizar rebotes de liquidación (`MEC-03 / MEC-10`) de manera estructural en todas las épocas, la entrada no puede ser `Taker` al cierre de `1m`, sino **órdenes `Maker` pasivas plantadas dentro de la mecha en el segundo exacto del racimo de `!forceOrder@arr`**.

---

## 4. Conclusión Operativa Conjunta (Paso 1 + Paso 2)

1. **Paso 1 (`Core P1 Rebalance Barrier`) -> `GO` CONTUNDENTE:** Apagar el rebalanceo intraciclo fuera de horario (`enable_intracycle_rebalance=false`, reloj estricto de `8h` `01_STRICT_8H_NO_INTRACYCLE`, opcionalmente con *buffer* anti-micro-recortes `08_UNIFIED_SHIELD_A`) mejora **las 3 épocas (`23–24`, `2025`, `2026`)**, sube `4Y Sharpe` de `2.02 -> 2.19`, baja `4Y MaxDD` `-3.00%`, y suma **`+$1,416.53`** en `2026` desde `$2,700` alejando el mínimo del piso de `$2,500` (`$2,550.02` vs `$2,502.85`).
2. **Paso 2 (`P4_M4` + `aggTrades 1-ms`) -> `NO-GO` DISCIPLINADO:** No se agrega `P4_M4` en modo `Taker 1m` a producción porque sigue perdiendo `-25.9 bps` en `2023–2024`; el portafolio vivo de Snipers queda limpio con **`M09 + M23 + M04 + M26`** (`M25=OFF`).
