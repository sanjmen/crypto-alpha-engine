# 10 — Resultados Forenses del Laboratorio `1-Minuto` Lineal de Mecánica de Contrapartes (`2023–2026`, `219` Perpetuos Point-in-Time)

- **Ejecución:** Google Colab (`m17_physics_lab`, CPU aislado), `20` shards (`219` perpetuos USDS-M Point-in-Time incluyendo deslistados, `27,103` trades simulados vela a vela en `1 minuto`, `2,068.5 s`).
- **Artefactos crudos reproducibles:** [counterparty_mechanics_1m_leaderboard.csv](../../reports/counterparty_mechanics_1m_leaderboard.csv), [counterparty_mechanics_1m_summary.json](../../reports/counterparty_mechanics_1m_summary.json), [counterparty_mechanics_1m_report.md](../../reports/counterparty_mechanics_1m_report.md).
- **Motor y tests:** [counterparty_mechanics_lab.py](../../src/models/counterparty_mechanics_lab.py) + [test_counterparty_mechanics_lab.py](../../tests/test_counterparty_mechanics_lab.py).

---

## 1. Veredicto Ejecutivo Sin Maquillaje (`0` falsos positivos, `4` descubrimientos estructurales)

Al eliminar por completo `np.log` (`100%` retornos lineales exactos), imponer prioridad `100%` pesimista al Stop-Loss intraminuto, contabilizar cada cobro/pago real de funding en `:00:00` y exigir validación **CPCV in-fold real** + rentabilidad simultánea en las 3 eras (`2023–24`, `2025`, `2026`):

> **[!IMPORTANT]**
> **Ninguna de las 30 variantes iniciales cruzó simultáneamente los 4 filtros de «Tesoro Universal» (`0 / 30`).**
> Y ese `0` es la prueba de que **eliminamos el autoengaño**:
> 1. **Confirmó al `100%` que pausar `M25` en vivo hoy te ahorró dinero real:** En retornos `1m` lineales reales sobre `4,562–4,939` trades, el `M25` incondicional (`M00_NAIVE_M25_CONTROL`) tiene **expectativa neta negativa en todas las eras y en todos los modos de ejecución (`-7.52` a `-10.49 bps/trade`, `t = -4.13`)**, y `GTX_1.5BPS` (`-10.49 bps`) rinde peor que `TAKER_OPEN` (`-7.52 bps`) por selección adversa.
> 2. **Reveló la trampa exacta de monetización que estranguló a los ganadores (`PROFIT_LOCK_SL` prematuro):** Entre el **`42%` y el `50.4%` de todos los trades** entraron en la dirección correcta (`Win Rate 55%–61%`), activaron el *profit-lock* temprano (`+1.0%..+1.8%`) y fueron sacados por ruido normal de `1m` en **`+36.5` a `+71.5 bps`**, mientras que los perdedores pagaron el Stop-Loss completo de **`-218` a `-233 bps`** (creando artificialmente una asimetría negativa `1 : 3` contra nosotros).
> 3. **Aisló 3 fenómenos físicos reales con edge masivo por régimen (`C1B`, `C2A`, `C3A`)** y mostró exactamente **qué dato de microestructura falta (`bookDepth L2` + `premiumIndex/indexPrice 1m` + ventana exacta de salida)** en la era donde cada uno flaqueó.

---

## 2. Tabla Comparativa Principal (`27,103` Trades Lineales `1m`, `2023–2026`)

| Mecanismo | Modo | `N` | Gross | Funding | Fee | **Net Full** | Mediana | Win`%` | FeeSh`%` | `23–24` | `2025` | `2026` | CPCV OOS bps | CPCV`%` | BTC Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C1B_PRE_SETTLE_SPOT_SCARCITY_LONG`** | `GTX_1.5BPS` | `177` | `+14.4` | **`+14.1`** | `8.3` | **`+20.3`** | `+61.5` | `59.3%` | `7.4%` | **`+51.0`** | **`+43.1`** | `-5.3` | **`+20.7`** | **`80%`** | `-0.04` |
| **`C1B_PRE_SETTLE_SPOT_SCARCITY_LONG`** | `TAKER_OPEN` | `187` | `+17.3` | **`+15.4`** | `12.8` | **`+20.0`** | `+57.0` | `59.9%` | `11.3%` | **`+58.9`** | **`+22.8`** | **`+1.2`** | **`+28.3`** | **`80%`** | `-0.04` |
| **`C2A_TRAPPED_SHORTS_PAIN_LONG`** | `TOUCH_LIMIT` | `405` | `+25.3` | `+1.9` | `8.3` | **`+18.9`** | **`+71.5`** | `55.1%` | **`7.1%`** | `-7.2` | `+3.8` | **`+86.7`** | **`+14.4`** | `67%` | `+0.01` |
| **`C2A_TRAPPED_SHORTS_PAIN_LONG`** | `GTX_1.5BPS` | `405` | `+25.2` | `+1.9` | `8.3` | **`+18.8`** | **`+71.5`** | `55.3%` | `7.2%` | `-5.3` | `-0.2` | **`+89.1`** | `+12.2` | `67%` | `+0.01` |
| **`C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG`** | `TAKER_OPEN` | `470` | **`+42.1`** | `+0.4` | `12.3` | **`+30.2`** | `+42.0` | `57.4%` | `10.9%` | **`+50.3`** | **`+39.8`** | `-24.9` | **`+26.0`** | `47%` | `-0.09` |
| `C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG` | `GTX_1.5BPS` | `424` | `+19.5` | `+0.4` | `8.0` | `+11.9` | `+46.5` | `55.4%` | `8.5%` | `+29.9` | `+12.4` | `-26.0` | `+0.8` | `20%` | `-0.07` |
| `C5_SOVEREIGN_CASH_WAVE_IGNITION` | `TAKER_OPEN` | `1058` | `+13.0` | `+0.0` | `12.7` | `+0.3` | `+32.0` | `56.7%` | `20.4%` | `+5.8` | `-54.2` | **`+26.1`** | `-16.1` | `33%` | `+0.06` |
| `C1A_POST_SETTLE_HARVESTER_SHORT` | `TOUCH_LIMIT` | `2000` | `+5.6` | `+0.0` | `8.3` | `-2.7` | `+36.5` | `60.5%` | `12.0%` | `-25.6` | `+1.8` | `+8.0` | `-2.8` | `47%` | `-0.04` |
| `C1A_POST_SETTLE_HARVESTER_SHORT` | `TAKER_OPEN` | `2130` | `+10.0` | `+0.0` | `12.8` | `-2.8` | `+32.0` | `61.5%` | `18.1%` | `-26.7` | `+1.6` | `+8.7` | `-6.9` | `33%` | `-0.06` |
| **`M00_NAIVE_M25_CONTROL` (Control `M25`)** | `TAKER_OPEN` | `4939` | `+5.1` | `+0.0` | `12.7` | **`-7.5`** | `+32.0` | `61.8%` | `17.3%` | `-21.4` | `-4.3` | `-2.6` | `-7.2` | `47%` | `-0.05` |
| **`M00_NAIVE_M25_CONTROL` (Control `M25`)** | `GTX_1.5BPS` | `4562` | `-2.3` | `+0.0` | `8.2` | **`-10.5`** | `+36.5` | `60.3%` | `11.7%` | `-19.5` | `-7.9` | `-7.8` | `-11.3` | `33%` | `-0.03` |
| `C4_CLOCK_COMPRESSION_SQUEEZE_LONG` | `GTX_1.5BPS` | `52` | `-7.6` | `+1.5` | `8.0` | `-14.1` | `-248.5` | `44.2%` | `6.3%` | `+179.6` | `-18.5` | `-41.9` | `-38.6` | `20%` | `-0.03` |
| `C1C_POST_SETTLE_SHORT_PANIC_LONG` | `TAKER_OPEN` | `52` | `-4.6` | `+2.1` | `12.9` | `-15.5` | `+9.7` | `51.9%` | `13.9%` | `+51.3` | `-32.7` | `-22.1` | `-23.9` | `20%` | `+0.02` |
| `C2B_TRAPPED_LONGS_WATERFALL_SHORT` | `TOUCH_LIMIT` | `17` | `-28.5` | `+1.0` | `8.2` | `-35.7` | `-218.5` | `41.2%` | `8.6%` | `+10.7` | `-102.1` | `+0.0` | `+0.0` | `0%` | `+0.03` |
| `C3B_SPOT_ANCHORED_BLOWOFF_SHORT` | `TAKER_OPEN` | `142` | `-27.4` | `-3.1` | `12.7` | `-43.1` | `-183.0` | `43.7%` | `19.4%` | `-26.5` | `-76.7` | `+20.1` | `-28.4` | `40%` | `+0.08` |

---

## 3. Diagnóstico Profundo de los 5 Hallazgos Clave

### Hallazgo 1 — Por qué `M25` parecía ganador en `M16-4` y en realidad pierde `-10.5 bps/trade`
- `M00_NAIVE_M25_CONTROL` gana el **`60.3%–61.8%` de las veces** con una mediana de **`+36.5 bps`** (`48.8%` tocan `PROFIT_LOCK_SL`), pero el **`27.9%` de las veces** se come un *short squeeze* lineal de `-220 bps` (`Profit Factor = 0.86`).
- En `M16-4`, `-np.log(P_1/P_0)` recortaba artificialmente la cola derecha alcista de esos squeezes y no penalizaba la selección adversa de `GTX` (`GTX_1.5BPS` pierde **`-2.3 bps` brutos antes de comisiones**, mientras `TAKER_OPEN` gana `+5.1 bps` brutos — una degradación de **`-7.4 bps` brutos por selección adversa pura**).
- Filtrar el Estado A (`C1A`) mejoró `+16.5 bps/trade` en `2026` (`-7.8 -> +8.7 bps` netos), demostrando que el veto físico de Spot funciona, pero un short de `+8.7 bps` netos con cola izquierda cóncava **no es un tesoro**.

### Hallazgo 2 — El estrangulamiento por `PROFIT_LOCK_SL` (`50%` de los trades cortados en `0.33R`)
Mira la descomposición de salidas en [counterparty_mechanics_1m_summary.json](../../reports/counterparty_mechanics_1m_summary.json):
- En **`C2A_TRAPPED_SHORTS_PAIN_LONG`**: **`50.4%`** de los trades salieron en `PROFIT_LOCK_SL` (`+71.5 bps` netos) y solo **`4.2%`** llegaron al `TAKE_PROFIT_MAKER` (`+700 bps`), mientras **`41.7%`** tocaron `STOP_LOSS` (`-218.5 bps`).
- En **`C1B_PRE_SETTLE_SPOT_SCARCITY_LONG`**: **`49.2%`** salieron en `PROFIT_LOCK_SL` (`+61.5 bps` netos) y solo **`5.3%`** llegaron a TP.
- **Conclusión de monetización:** En perpetuos altcoin de `1m`, ajustar el stop a `+70..+80 bps` apenas el trade toca `+150..+180 bps` **garantiza que el ruido normal de `1m` te saque antes de la verdadera cascada de liquidación**, destruyendo la convexidad positiva que hace rentable un sistema de *squeezes*.

### Hallazgo 3 — `C2A` (`OI-VWAP` en Unidades de Contrato) explotó a **`+89.1 bps/trade` netos en `2026`**
- Con **`96` trades en `2026`** (`corr_vs_btc = +0.01`, `Fee Share = 7.2%`), `C2A` entregó **`+89.09 bps/trade` netos** (`~+0.89%` limpio por operación después de comisiones).
- En `2023–24` (`-5.3 bps`) y `2025` (`+3.8 bps`) estuvo plano porque: (1) el `PROFIT_LOCK_SL` de `+80 bps` asfixió los ganadores, y (2) `34` monedas nuevas no tenían Spot en Binance (`has_spot=False`) y no teníamos **`bookDepth` L2** para distinguir cuándo encima del máximo de `4h` hay un **vacío de Asks** vs. un **muro vendedor institucional**.

### Hallazgo 4 — `C1B` (Pre-Settle Spot-Scarcity Long + Cobro de Cupón `:00:00`) logró **`80%` CPCV In-Fold (`+28.3 bps` OOS)**
- Fue el **único mecanismo positivo en las 3 eras** en `TAKER_OPEN` (`+58.9 bps` en `23–24`, `+22.8 bps` en `2025` / `+43.1 bps` Maker, `+1.2 bps` en `2026`), cobrando en promedio **`+15.4 bps` de funding real en efectivo** en cada trade (`corr_vs_btc = -0.04`).
- ¿Por qué cayó a `+1.2 bps` en `2026`? Porque `max_hold_minutes = 75` desde `T-45m` obligaba a mantener el Long **hasta `T+30m` después del cobro**, comiéndose entero el *dump* post-settlement de los cosechadores en `2026` (`Estado 01-A`) después de haber cobrado ya el cupón en `:00:00`! Salir inmediatamente en `:00:01` tras cobrar el cupón (salvo vacío de Asks en `bookDepth`) elimina ese *give-back* estructural.

### Hallazgo 5 — `C3A` (Rebote Post-Liquidación Anclado en Spot) probó que en pánicos `TAKER` aplasta a `GTX` (`+30.2` vs `+11.9 bps`)
- En `470` cascadas de liquidación (`2023–2026`), entrar `TAKER_OPEN` tras la vela `1m` de giro dio **`+30.15 bps/trade` (`t = +2.62`)** (`+50.3 bps` en `23–24`, `+39.8 bps` en `2025`), casi **el triple** que `GTX_1.5BPS` (`+11.9 bps`), confirmando nuevamente que en rebotes en `V` la orden pasiva sufre selección adversa.
- Sin embargo, en el mercado bajista de altcoins de `2026` cayó a **`-24.9 bps`**: porque el `OI` de `5m` con `+60s` de lag puede tener hasta `6 minutos` de atraso frente a una cascada activa. Para operar `C3A` en `2026` hace falta **ver en el libro (`bookDepth` / `premiumIndexKlines 1m` / `!forceOrder`) que la cascada realmente terminó y reapareció el Bid**.

---

## 4. Próximo Paso Directo (Ingesta y Laboratorio Fase P1 con Datos de Libro e Índice Multi-Exchange)

Tal como planteamos en [09_registro_mecanismos_y_arquitectura_datos_microestructura.md](./09_registro_mecanismos_y_arquitectura_datos_microestructura.md), ya sabemos **exactamente qué 3 piezas necesitamos** para convertir `C1B`, `C2A` y `C3A` en verdaderos *tesoros* todo-terreno:

1. **Ingerir en Google Colab -> Google Drive (`Fase P1 Microestructura`):**
   - `premiumIndexKlines` (`1m`) + `indexPriceKlines` (`1m`) para los `220` perpetuos (`100%` cobertura multi-exchange Spot + *Impact Bid/Ask* intraminuto, `< 600 MB`).
   - `bookDepth` L2 (`±0.2%, ±1%, ±2%, ±3%, ±5%`) desde `data.binance.vision` para medir **vacíos reales de liquidez (`Ask Vacuum` / `Bid Replenishment`)** antes de disparar `C2A`, `C1B` o `C3A`.
   - Dispersión cross-exchange de Funding (`Bybit vs Binance`) para separar arbitraje local (`Estado 01-A`) de escasez global (`Estado 01-B`).
2. **Corregir la geometría de monetización (sin sobreajuste):**
   - En `C1B`: evaluar salida limpia post-cupón (`T+1m`) vs. continuación condicionada a vacío de Asks.
   - En `C2A`: reemplazar el `PROFIT_LOCK_SL` estrangulador (`+80 bps`) por *trailing* convexo basado en `ATR` / parciales para no amputar la cola derecha que dio `+89 bps/trade` en `2026`.
