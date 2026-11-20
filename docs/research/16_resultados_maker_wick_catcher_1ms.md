# 16 — Resultados Forenses Análisis 1: `GTX` Maker Limit Wick-Catcher a Resolución `1-Milisegundo` (`aggTrades`, `2023–2026`)

## 1. Hipótesis Evaluada y Veredicto Ejecutivo (`NO-GO` Definitivo por Selección Adversa)

En el documento anterior ([15_resultados_p4_m4_absorcion_1ms_liquidaciones.md](15_resultados_p4_m4_absorcion_1ms_liquidaciones.md)) demostramos que entrar como **Taker al cierre de `1m`** tras una cascada de liquidación (`P4_M4`) llega `25–45 bps` tarde y paga `12 bps` de comisión (`+4.72 bps` Full, `-25.9 bps` en `2023–2024`). Quedaba abierta una única pregunta física: **¿pueden órdenes límite pasivas `GTX` (`Post-Only` Maker, `2 bps` fee) plantadas dentro de la mecha durante el estallido de `1-ms` capturar el rebote elástico?**

Implementamos el simulador exacto de cola y latencia a **`1 milisegundo`** en [maker_wick_catcher_lab.py](../../src/models/maker_wick_catcher_lab.py) (con tests unitarios en [test_maker_wick_catcher_lab.py](../../tests/test_maker_wick_catcher_lab.py)) y auditamos las **`281` ventanas de cascadas de liquidación ancladas por solvencia cross-venue (`2023–2026`, `83` símbolos, `1,345` ejecuciones `GTX`)**.

Fuente reproducible: [maker_wick_catcher_report.md](../../reports/maker_wick_catcher_report.md) y [maker_wick_catcher_summary.json](../../reports/maker_wick_catcher_summary.json).

* **Sobrevivientes de las 4 Compuertas Universales (`>= +25 bps` Full, `>= +15 bps` en `23–24` y `2025`, `>= +20 bps` en `2026`, `CPCV >= 80%`):** **`0 / 5` (`NO-GO` definitivo — `0%` de splits CPCV positivos en los 5 perfiles).**

---

## 2. Realismo Físico del Simulador de Cola `1-ms` + `1-s`

Para evitar el sesgo clásico de los backtests ingenuos de órdenes límite (asumir que tocar el precio ejecuta la orden en la punta de la cola y rebota), exigimos **cuatro leyes de física real de matching engine**:

1. **Detección causal a `1-ms` (`0.000 ms` lookahead):** Se detecta la primera barrida forzada vendedora (`is_buyer_maker = True`, `>= $35k–$75k` y `>= 8–14` órdenes Maker barridas en el mismo milisegundo `T_trig_ms`).
2. **Latencia de red + API (`+250 ms`):** La orden `GTX` recién llega al libro en `T_place_ms = T_trig_ms + 250 ms`. Si en `T_place_ms` el mercado ya cayó por debajo de `P_limit`, el motor rechaza la orden `GTX` (`Post-Only Reject`) como hace Binance en vivo.
3. **Prioridad de cola conservadora (*Strict Trade-Through*):** La orden `GTX` **solo** se considera ejecutada si después de `T_place_ms` ocurre `aggTrade < P_limit * (1 - 1 bp)` (el precio atraviesa nuestro nivel) o se ejecutan `>= $25,000` de ventas agresivas exactamente a `<= P_limit` vaciando la cola delante nuestro.
4. **Resolución post-fill a `1 segundo` pesimista (`SL` antes que `TP`):** Dentro de cada segundo posterior al fill (`hasta 90 min`), si el mínimo toca el `Stop-Loss` Taker y el máximo toca el `Take-Profit` Maker en el mismo segundo, gana siempre el `Stop-Loss` (`2 bps` Maker entry + `5 bps` Taker SL = `7 bps`; o `2 + 2 = 4 bps` en salida `TP` Maker).

---

## 3. Tabla Completa de Resultados (`1,345` Fills `GTX`, `2023–2026`, `15-Split` True In-Fold CPCV)

| Perfil `GTX` Maker (`1-ms`) | Offset Bid | `SL / TP` (`bps`) | `N` Fills | Gross (`bps`) | Fee (`bps`) | **Net Full (`bps`)** | Win `%` | `SL / TP / TIME` | `23–24` (`bps`) | **`2025` (`bps`)** | **`2026` (`bps`)** | **CPCV OOS (`bps`)** | **CPCV `%`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MW5_GTX_EXTREME_CLIMAX_60BPS` *(Clímax `$75k`, Ladder `-60/-100`)* | `-60 / -100 bps` | `-220 / +115` | `249` | `-66.1` | `6.0` | **`-72.14`** | `35.7%` | `158 / 89 / 2` | `-87.4` | **`-52.5`** | **`-56.8`** | `-80.6` | **`0%`** |
| `MW1_GTX_SHALLOW_25BPS` | `-25 bps` | `-140 / +65` | `276` | `-81.5` | `6.5` | **`-88.01`** | `28.3%` | `196 / 77 / 3` | `-111.3` | **`-69.4`** | **`-57.9`** | `-75.6` | **`0%`** |
| `MW3_GTX_DEEP_70BPS` | `-70 bps` | `-200 / +110` | `270` | `-92.3` | `6.6` | **`-98.90`** | `25.6%` | `201 / 66 / 3` | `-115.2` | **`-80.4`** | **`-84.5`** | `-93.3` | **`0%`** |
| `MW4_GTX_LADDER_35_70BPS` *(`50%` `-35` + `50%` `-70`)* | `-35 / -70 bps` | `-185 / +85` | `275` | `-94.7` | `6.3` | **`-101.06`** | `25.1%` | `204 / 68 / 3` | `-121.0` | **`-82.8`** | **`-79.0`** | `-96.4` | **`0%`** |
| `MW2_GTX_MEDIUM_45BPS` | `-45 bps` | `-165 / +85` | `275` | `-97.4` | `6.7` | **`-104.11`** | `21.8%` | `211 / 59 / 5` | `-125.2` | **`-84.3`** | **`-81.4`** | `-94.1` | **`0%`** |

---

## 4. Diagnóstico Físico ("Bajado a Tierra"): La Maldición del Ganador (*Winner's Curse* de Glosten–Milgrom)

¿Por qué poner órdenes `Maker` pasivas dentro de la mecha rindió **mucho peor** (`-72` a `-104 bps`) que esperar confirmación `1m` Taker (`P4_M4_2 = +4.7 bps` Full, `+62.9 bps` en `2026`), a pesar de ahorrar la mitad de las comisiones (`6.0 bps` vs `12.0 bps`)?

1. **Cuando el rebote en `V` es genuino, tu orden `GTX` NUNCA se llena:**
   * En los verdaderos pisos de liquidación donde una ballena ya absorbió toda la venta forzada en `T_trig_ms`, el precio rebota instantáneamente hacia arriba en los siguientes `250 ms`. Tu orden límite pasiva sentada `-25`, `-45` o `-70 bps` más abajo se queda colgada sin ejecutarse y expira tras `90 s` (`0 bps`, te perdiste el `100%` de los rebotes limpios).
2. **Tu orden `GTX` SOLO se llena cuando viene una segunda o tercera avalancha (*Secondary Margin-Call Cascade*):**
   * Para que el mercado baje otros `-25` a `-100 bps` después del primer estallido de `1-ms` y **atraviese tu cola** (`trade-through`), tiene que desatarse una segunda ola de liquidaciones encadenadas. En ese escenario, el flujo tóxico te llena la orden y sigue de largo directo hasta tu `Stop-Loss` en el **`71%–77%` de las veces (`196–211 SLs` contra apenas `59–77 TPs`)**.
3. **Cierre definitivo de la línea de *Knife-Catching* direccional:**
   * Sin cancelación HFT co-localizada (`< 1 ms` cancel-on-toxicity antes de ser barrido) ni cobertura delta-neutral instantánea contra otro activo/venue, intentar atajar cascadas de liquidación de forma direccional sufre selección adversa estructural severa. Se archiva definitivamente toda la familia `P4_M4 / MW1..MW5` (`0` capital asignado en producción).
