# 19 — Resultados Forenses Frente B: Veto Causal `L2 bookDepth` sobre Snipers Productivos (`M09/M23/M04`) y Auditoría Anti-Sesgo de Cascadas `SHORT` a `1-ms` (`P8`, `P9`, `P10`)

## 1. Resumen Ejecutivo y Decisión `GO / NO-GO`

En el **Frente B** ([sniper_l2_and_cascade_short_lab.py](../../src/models/sniper_l2_and_cascade_short_lab.py), [frente_b_report.md](../../reports/frente_b_report.md), [frente_b_causal_audit_report.md](../../reports/frente_b_causal_audit_report.md)) cruzamos los `82` perpetuos del universo Sniper (`2023–2026`) contra instantáneas causales reales de **`L2 bookDepth` (`±1% / ±2%` de profundidad en USD, `t_book <= entry_ts`, `0.000 ms` lookahead)** y flujos **`aggTrades` a `1 milisegundo`**, sometiendo además cada hallazgo a una segunda **verificación forense anti-condicionamiento y ablación en portafolio `1m`**.

### Veredicto Consolidado del Frente B:
1. **`GO` Accionable (`M17-4`, para próxima ventana programada) — Veto Quirúrgico No-Destructivo de Muro `L2` (`L2_NO_WALL`) en `EventDrivenSniperEngine` (`M09` y `M23`):**
   - Vetar entradas `LONG` cuando existe un muro vendedor real a `+1%` (`book_imb_1pct < -0.15`) y entradas `SHORT` contra un muro comprador a `-1%` (`book_imb_1pct > +0.15`) mejora simultáneamente:
     - **`M09_LONG` (`TREASURE 🏆`):** sube de `+41.06 -> +50.09 bps/trade` (`+22%`), mediana `+17.45 -> +25.76 bps`, **`2026` `+62.98 -> +78.69 bps`**, `Profit Factor 1.53 -> 1.65` y **`CPCV In-Fold Sharpe 0.65 -> 0.88`** (`86.7%` splits).
     - **`M09_SHORT`:** duplica con creces la estabilidad inter-split **`CPCV%` de `33.3% (5/15) -> 73.3% (11/15)`** (`CPCV Sharpe -0.13 -> +0.23`, `CPCV OOS +39.5 -> +44.9 bps`) al filtrar `287` cortos estrellados contra muros de compra reales.
     - **`M23_BREADTH_LONG`:** sube de `+11.44 -> +13.85 bps`, cuadruplica la mediana (`+3.44 -> +13.69 bps`), eleva `2026` (`+22.01 -> +27.57 bps`) y sube `CPCV%` de `40.0% -> 53.3%`.
2. **Descubrimiento de Interacción de Portafolio (`FB0..FB4`) — NO apagar de cuajo `M09_SHORT` ni `M04`:**
   - Aunque standalone en `2026` `M09_SHORT` rinde `+7.4 bps` y `M04` cae a `-43.0 bps`, la simulación exacta `1m` del portafolio completo demuestra que **eliminar `M09_SHORT` baja el `4Y Sharpe` de `2.19 -> 2.01`** (`$5,050 -> $4,561`) porque `M09_SHORT` es el único Sniper bajista activo durante desplomes (`P90 = +412 bps`), y **eliminar `M04` baja el `4Y Sharpe` de `2.19 -> 2.11`**. Por lo tanto, la intervención correcta es **aplicar el veto `L2_NO_WALL` (`M17-4`) sin mutilar la cobertura bajista**.
3. **`NO-GO` Disciplinado (`0` capital expuesto) — `P8` (`Cascada Waterfall SHORT 1-ms`), `P9` y `P10`:**
   - **Salvada forense crítica en `P8`:** En la primera corrida sobre las `281` ventanas de `P4_M4`, `P8D` pareció un supuesto *Treasure* (`+126.4 bps`, `77.9%` Win, `100%` CPCV, `DSR = 1.00`). Nuestra auditoría interna detectó que buscar el *burst* en `[T_sig - 15m, T_sig)` dentro de ventanas seleccionadas en `T_sig` condicionaba sobre la caída hasta `T_sig`. Al re-correr `P8` **estrictamente post-ancla (`t_ms >= T_anchor`, `0.000 ms` condicionamiento)** tanto sobre las `281` anclas `P4_M4` como sobre **`1,631` cierres causales de pánico `1h`**, el retorno bruto colapsa a **`+1.9 a +3.4 bps`** y el neto a **`-5.4 a -22.2 bps/trade`** (`0%–47%` CPCV). Queda descartado con certeza matemática total.

---

## 2. Parte B1 — Radiografía `L2 bookDepth` de los Snipers Productivos (`82` Símbolos `1m`, `2023–2026`)

Fuente: [frente_b_sniper_l2_leaderboard.csv](../../reports/frente_b_sniper_l2_leaderboard.csv).

| Mecanismo x Filtro `L2` | `N` | Gross | Fee | **Net Full** | Mediana | `PF` | `23–24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV SR (`Pos%`)** | Treasure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M09_LONG__L2_NO_WALL`** | `245` | `+62.5` | `12.4` | **`+50.1`** | **`+25.8`** | **`1.65`** | `+51.0` | **`+17.8`** | **`+78.7`** | **`+51.4`** | **`0.88` (`87%`)** | **`TREASURE 🏆`** |
| `M09_LONG__RAW` *(actual)* | `299` | `+53.4` | `12.4` | **`+41.1`** | `+17.5` | `1.53` | `+41.6` | `+16.3` | `+63.0` | `+43.1` | `0.65` (`87%`) | `TREASURE 🏆` |
| `M09_LONG__L2_STRICT_IMB` | `48` | `+11.5` | `12.6` | **`-1.1`** | `-92.1` | `0.99` | `-8.2` | `-29.5` | `+45.9` | `-36.3` | `-0.45` (`33%`) | `—` |
| **`M09_SHORT__L2_NO_WALL`** | `773` | `+46.5` | `12.7` | **`+33.8`** | `-11.1` | `1.37` | `+67.0` | `+37.3` | `+3.4` | **`+44.9`** | **`0.23` (`73%`)** | `—` |
| `M09_SHORT__RAW` *(actual)* | `1060` | `+47.6` | `12.7` | **`+34.9`** | `-13.8` | `1.37` | `+51.8` | `+52.9` | `+7.4` | `+39.5` | `-0.13` (`33%`) | `—` |
| **`M23_BREADTH_LONG__L2_NO_WALL`** | `729` | `+25.9` | `12.0` | **`+13.9`** | **`+13.7`** | **`1.19`** | `+0.8` | **`+21.8`** | **`+27.6`** | **`+19.0`** | **`0.03` (`53%`)** | `—` |
| `M23_BREADTH_LONG__RAW` *(actual)* | `930` | `+23.4` | `12.0` | **`+11.4`** | `+3.4` | `1.16` | `+1.0` | `+18.6` | `+22.0` | `+16.5` | `-0.09` (`40%`) | `—` |
| `M04_NEGFUND_LONG__L2_NO_WALL` | `510` | `+25.8` | `14.1` | **`+11.6`** | `-23.5` | `1.10` | `+37.7` | `+47.0` | `-46.4` | `+12.4` | `-0.10` (`47%`) | `—` |
| `M04_NEGFUND_LONG__RAW` *(actual)* | `597` | `+24.6` | `14.0` | **`+10.6`** | `+3.3` | `1.09` | `+29.6` | `+45.0` | `-43.0` | `+5.9` | `0.01` (`53%`) | `—` |
| `P9_FAILED_BREAKOUT_TRAP_SHORT__RAW` | `1789` | `+4.4` | `14.1` | **`-9.8`** | `-10.4` | `0.90` | `-10.4` | `+43.2` | `-11.0` | `-9.5` | `0.34` (`80%`) | `—` |
| `P10_BTC_FLUSH_IMMUNE_LONG__RAW` | `23` | `+55.3` | `13.8` | **`+41.5`** | `+63.9` | `1.39` | `+136.9` | `-61.0` | `+1.9` | `+0.0` | `0.00` (`0%`) | `—` |

### Lección Física Clave (`L2_NO_WALL` vs `L2_STRICT_IMB`):
* **Por qué funciona `L2_NO_WALL` (`imb >= -0.15` en Long):** Cuando un perpetuo rompe al alza pero enfrenta `> 1.35x` más dólares en `Ask +1%` que en `Bid -1%`, el comprador Taker se estrella contra inventario institucional pasivo (`-4.9 bps` promedio en los `54` trades vetados de `M09_LONG`).
* **Por qué `L2_STRICT_IMB` (`imb >= +0.12` en Long) es una trampa (`-1.1 bps`, mediana `-92.1 bps`):** Exigir una pila gruesa de Bids justo debajo de una ruptura `1h` selecciona episodios de *spoofing* donde creadores de mercado exhiben Bids falsos para inducir compras Taker minoristas y descargar inventario arriba. **El libro L2 debe usarse como veto de pared opuesta (`L2_NO_WALL`), jamás para perseguir desequilibrio positivo en rupturas lentas.**

---

## 3. Verificación en Portafolio Unificado `1m` (`FB0..FB4`)

Fuente: [frente_b_causal_audit_report.md](../../reports/frente_b_causal_audit_report.md).

| Config ID | **`4Y SR`** | `4Y CAGR` | `4Y DD` | `23–24 / 25 / 26 SR` | `CPCV SR (Pos%)` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`FB0_CURRENT_PROD_ALL_SNIPERS`** | **`2.19`** | **`+188.1%`** | `35.23%` | `2.25 / 1.95 / 2.42` | **`2.17` (`100%`)** | **`$5,050.17`** (**`$2,550.02` ✅**) |
| `FB1_M09_LONG_ONLY_DROP_M09_SHORT` | `2.01` | `+150.7%` | `35.56%` | `1.97 / 1.76 / 2.43` | `1.98` (`100%`) | `$4,560.72` (`$2,526.13` ✅) |
| `FB2_DROP_M04_NEGFUND` | `2.11` | `+173.7%` | `36.89%` | `2.13 / 1.69 / 2.65` | `2.09` (`100%`) | `$4,921.53` (`$2,528.12` ✅) |
| `FB3_DROP_M09_SHORT_AND_M04` | `1.98` | `+146.8%` | `35.17%` | `1.92 / 1.57 / 2.60` | `1.94` (`100%`) | `$5,152.92` (`$2,513.42` ✅) |
| `FB4_M09_LONG_PRIORITY_OVER_SHORT` | `2.09` | `+169.4%` | `36.89%` | `2.13 / 1.65 / 2.61` | `2.07` (`100%`) | `$4,921.53` (`$2,528.12` ✅) |

* **Conclusión de portafolio:** Aunque `FB3` gana `+$102.75` puntuales en `2026`, sacrifica `-0.21` de `4Y Sharpe` y estrecha el colchón sobre `$2,500` de `$50.02` a `$13.42` porque `M09_SHORT` aporta ganancias convexas descorrelacionadas en correcciones cuando `M09_LONG`, `M23` y `M26` (todos `LONG`) están inactivos. Mantener todas las patas y depurarlas con **`L2_NO_WALL` (`M17-4`)** preserva esa convexidad bajista subiendo `CPCV%` de `M09_SHORT` de `33% -> 73%` y `M09_LONG` a `+50.1 bps`.

---

## 4. Auditoría Forense Anti-Sesgo sobre `P8` (`1-ms` Waterfall `SHORT`): De `+126.4 bps` Condicionado a `-7.4 bps` Causal

| Régimen de Ventana en `P8D_WATERFALL_SHORT_HEAVY_CLIMAX_40BPS` | Anclas | Trades | Gross (`bps`) | Fee (`bps`) | **Net Full (`bps`)** | Win`%` | `CPCV%` | Diagnóstico Forense |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1) Pre-Ancla `[T_sig - 15m, T_sig)` sobre `P4_M4`** *(sesgado)* | `281` | `163` | `+136.3` | `9.9` | **`+126.4`** | `77.9%` | `100%` | **Ilusión por condicionamiento de ventana:** `P4_M4` exigía cierre hundido + flush OI en `T_sig` (`hasta 15m después del fill`). |
| **2) Post-Ancla Estricto `[T_sig, T_sig + 30m)` sobre `P4_M4`** (`0.000ms` sesgo) | `281` | `107` | **`-4.6`** | `10.0` | **`-14.5`** | `43.0%` | `47%` | Sin saber el futuro a `15m`, la cascada ya se agotó. |
| **3) Post-Cierre Causal `1h` Panic-Flush `[T_close, T_close + 30m)`** (`0.000ms` sesgo) | `1,631` | `555` | **`+2.6`** | `9.9` | **`-7.4`** | `43.8%` | **`0%`** | Deriva bruta casi nula (`+2.6 bps` < `9.9 bps` comisión Taker). |

> **Cierre científico de cascadas de liquidación (`Docs 15, 16 y 19`):** Una vez impresa una ráfaga de liquidación `1-ms` observable públicamente, el mercado alcanza un **equilibrio martingala de microestructura**: quien pone límites `GTX` para comprar el cuchillo pierde `-72 a -104 bps` por selección adversa (*Doc 16*), y quien vende `Taker` persiguiendo la ruptura post-ancla apenas captura `+2 a +3 bps` brutos que no cubren los `10 bps` de comisión Taker (*Doc 19*). El único punto donde el flujo `1-ms` + `L2` genera alfa real y repetible (`+71.2 bps`, `100%` CPCV) es **antes** del agotamiento, en la ignición alcista hacia un **vacío L2 (`MEC-06 / M26`)**.
