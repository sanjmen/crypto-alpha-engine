# 18 — Resultados Forenses Frente A: Auditoría de Ejecución Real en Binance Mainnet (`$283,024` / `641` Órdenes) y Laboratorio `1-Minuto` de Desfase del Reloj de Funding (`2023–2026`)

## 1. Resumen Ejecutivo y Action Items Aprobados para la Próxima Ventana de Despliegue

Auditamos la base productiva viva de Tokio (`data/live_trading.db`, abierta estrictamente en modo lectura `?mode=ro` sin tocar el contenedor) con [audit_live_execution_microstructure.py](../../scripts/audit_live_execution_microstructure.py) ([live_execution_microstructure_audit.json](../../reports/live_execution_microstructure_audit.json)) y contrastamos los hallazgos en un simulador sub-minuto exacto (`1m`, `2023–2026`, [execution_timing_and_router_lab.py](../../src/backtest/execution_timing_and_router_lab.py), [execution_timing_router_report.md](../../reports/execution_timing_router_report.md)).

> **Regla operativa vigente:** Producción en Tokio (`crypto-mainnet`) permanece **congelada sin cambios** durante varios días para acumular historial limpio del reloj de `8h` + `M26`. Los **3 hallazgos accionables** abajo quedan formalizados en código de investigación y en **GitHub Issues (`M17-1`, `M17-2`, `M17-3`)** listos para aplicar juntos en la próxima ventana programada.

---

## 2. Radiografía Empírica de Ejecución Real en Binance Mainnet (`641` Órdenes Llenas, `$283,024.04`)

### 2.1. Costo Real por Vía de Ejecución (`LIMIT` Chase vs. `Market Fallback`)
| Vía de Ejecución en Vivo | Órdenes | Nocional (`$`) | `% Maker` (`$`) | Slippage Pond. (`bps`) | Comisión Pond. (`bps`) | **Costo Total Real (`bps`)** | Latencia Mediana (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LIMIT` llenada como `MAKER` (`GTX`)** | `400` | `$174,159` | `100.0%` | `+0.51` | `2.00` | **`2.51 bps`** | `4.98 s` |
| **`LIMIT` caída a `TAKER` tras `3x5s` Chase** | `151` | `$80,865` | `0.0%` | `+1.20` | `4.17` | **`5.37 bps`** | `13.73 s` |
| **Total `LIMIT` Chase (`Core P1 + P2`)** | `551` | `$255,024` | `68.3%` | `+0.73` | `2.69` | **`3.42 bps`** | `7.22 s` |

### 2.2. Hallazgo Accionable #1 — El Impuesto de `2.22x` por Chocar contra `:00:00–:02:00 UTC`
| Ventana Horaria de Envío (`LIMIT` Orders) | Órdenes | Nocional (`$`) | `% Maker` (`$`) | Slippage Pond. (`bps`) | Comisión Pond. (`bps`) | **Costo Total Real (`bps`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dentro de Liquidación de Funding (`00/08/16h :00–:02 UTC`)** | `73` | `$21,528` | **`51.7%`** | **`+3.46 bps`** | `3.42 bps` | **`6.88 bps`** ⚠️ |
| **Fuera de la Tormenta de Liquidación (resto del reloj)** | `478` | `$233,496` | **`69.8%`** | **`+0.48 bps`** | `2.62 bps` | **`3.10 bps`** ✅ |

* **Causa física:** En el segundo `:00:00` de las `00:00, 08:00 y 16:00 UTC`, los arbitrajistas de funding desarman inventario a nivel global; el *spread* se ensancha y el libro vibra, rechazando órdenes `GTX` (`Maker %` cae `18.1 puntos`) y multiplicando el *slippage* por **`7.2x`** (`+3.78 bps` extra por pata).

### 2.3. Hallazgo Accionable #2 — Desconexión REST Quote + `OBI` en Mid-Caps (`RENDER`, `FET`, `ADA`, `DOGE`)
| Grupo de Símbolos (`LIMIT` Chase en Vivo) | `% Maker` (`$`) | Costo Total Real (`bps`) | Latencia Mediana (`s`) | Diagnóstico de Código ([live_broker.py](../../src/execution/live_broker.py)) |
| :--- | :---: | :---: | :---: | :--- |
| `XRP` (`83.8%`), `ETH` (`83.1%`), `ARB` (`80.1%`), `SOL` (`78.3%`), `BTC` (`77.0%`) | `77%–84%` | `2.5–6.2 bps` | `3.9–7.2 s` | Libro profundo absorbe la latencia REST. |
| **`RENDER` (`19.9%`), `FET` (`40.9%`), `ADA` (`40.0%`), `DOGE` (`45.4%`)** | **`20%–45%`** | `hasta +20.4 bps` | **`13.73 s`** | `execute_maker_limit_chase` hace un llamado REST `fetch_order_book` (`~100ms`) antes de cada `GTX` e ignora el `OBI` del router: en esos `100ms` el tope se mueve `1 tick`, Binance rechaza el `Post-Only` y tras `13.7s` cae a `Market`. |

* **Solución directa (`M17-1`):** Leer `(best_bid, best_ask)` en **`0 ms`** desde el WebSocket ya conectado en memoria (`ShadowExecutionBroker.market_bids / market_asks`, con fallback REST solo si tiene `> 3s` de antigüedad) e inyectar el `OBI` al chase (si `flow_align >= +0.45`, cruzar directo al toque en `t=0` ahorrando los `13.7s` de deriva adversa).

### 2.4. Hallazgo Accionable #3 — Residuo de Cancelación `P1 + P2` cuando `Gross < $2,000` (`ETH $13.91`)
* En la instantánea viva (`Gross = $1,634.78 < $2,000`), el cruce parcial entre `Core P1` y `Manga P2` dejó una posición residual abierta en `ETH/USDT` Short de **`$13.91`** porque [shadow_daemon.py](../../src/execution/shadow_daemon.py) solo filtra stubs cuando `core_gross_usd >= $2,000`.
* **Lección crítica del stress-test `1m`:** Aplicar un corte fijo de `$155` cuando `gross < $2,000` es peligroso porque cerca de `$2,557` de equity el gobernador CPPI achica las 10 patas a `~$135` y un corte de `$155` las anula todas (*CPPI Cash-Lock*). La fórmula exacta segura (`M17-3`) es: cuando `gross < $2,000`, suprimir **únicamente residuos de cancelación `P1 + P2` (`eff_p2 != 0`) menores a `max($25, 0.05 * gross_usd)`** (`~$80` hoy), eliminando residuos de `$14` sin tocar jamás las patas de `Core P1`.

---

## 3. Simulación `1-Minuto` Descontaminada (`2023–2026` + `$2,700` en `2026`): Rebalancear en `T - 5m` (`23:55 / 07:55 / 15:55 UTC`) vs `:00` vs `T + 3m`

Una vez aislado el filtro seguro de residuos `P1+P2` sin bloqueo CPPI ([execution_timing_router_report.md](../../reports/execution_timing_router_report.md)), la tabla completa sobre `80` símbolos `1m` (`2023–2026` y arranque fresco `$2,700` el `01-Ene-2026` contra el piso duro de `$2,500`) confirma de punta a punta las 3 palancas:

| Exp ID | Offset | Costo (`bps`) | **`4Y SR`** | `4Y CAGR` | `4Y DD` | `23–24 / 25 / 26 SR` | `CPCV SR (Pos%)` | `FeeDrag%` | `StubsSup` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A00_LEGACY_IDEALIZED_COUPON_4.20BPS` | `+0m` | `4.20` | **`2.19`** | `+188.1%` | `35.23%` | `2.25 / 1.95 / 2.42` | `2.17` (`100%`) | `245.9%` | `70` | **`$5,050.17`** (`$2,550.02` ✅) |
| **`A06_EMPIRICAL_LIVE_AT_00M_6.88BPS`** | **`+0m`** | **`6.88`** | **`1.40`** | `+76.6%` | `38.19%` | `1.64 / 1.06 / 1.47` | `1.39` (`100%`) | **`329.5%`** | `87` | **`$3,270.89`** (**`$2,499.46` ⚠️**) |
| `A07_EMPIRICAL_POST_STORM_PLUS_03M_3.10BPS` | `+3m` | `3.10` | **`2.03`** | `+167.7%` | `35.41%` | `2.31 / 1.51 / 2.13` | `1.99` (`100%`) | `199.2%` | `169` | **`$4,007.01`** (`$2,522.14` ✅) |
| **`A08_EMPIRICAL_PRE_SETTLE_MINUS_05M_3.10BPS`** | **`-5m`** | **`3.10`** | **`2.49` 🏆** | **`+239.4%`** | `36.59%` | **`2.05 / 2.84 / 2.87`** | **`2.48` (`100%`)** | **`197.1%`** | `203` | **`$5,141.67`** (**`$2,540.90` ✅**) |
| **`A09_WS_QUOTE_OBI_ROUTER_PLUS_03M_2.55BPS`** | **`+3m`** | **`2.55`** | **`2.18`** | `+196.5%` | **`34.86%`** | `2.40 / 1.72 / 2.33` | `2.16` (`100%`) | **`174.3%`** | `163` | **`$4,124.01`** (`$2,526.43` ✅) |

### Conclusiones clave y trazabilidad de Issues en GitHub:
1. **El costo real de `:00:00` (`6.88 bps`, `A06`) es letal:** hunde el `4Y Sharpe` a **`1.40`**, recorta `$1,779` del capital final `2026` (`$3,270.89`) y llega a rozar/perforar el piso de `$2,500` por sangrado de comisiones (`MinEq = $2,499.46`).
2. **Adelantar el rebalanceo `5 minutos` a `23:55 / 07:55 / 15:55 UTC` (`A08`, `100%` causal sobre la vela cerrada `55m` antes):** dispara el **`4Y Sharpe` a `2.49`** (`+1.09` sobre `:00:00` real y `+0.30` incluso sobre el backtest idealizado `A00`!), con **`2025 SR = 2.84`**, **`2026 SR = 2.87`**, **`100%` splits CPCV (`2.48`)**, recortando `132.4 puntos` de `FeeDrag` y llevando `$2,700` a **`$5,141.67` (`MinEq = $2,540.90` ✅)**.
3. **Issues creados para la próxima ventana programada:**
   - **Issue `#73` (`M17-1`):** Zero-Latency WS Top-of-Book & OBI-Aware `GTX` Limit Chase en `BinanceLiveBroker`.
   - **Issue `#74` (`M17-2`):** Rebalanceo Pre-Settlement `T - 5m` (`23:55 / 07:55 / 15:55 UTC`) desacoplado de la tormenta `:00:00`.
   - **Issue `#75` (`M17-3`):** Supresión proporcional de residuos de cancelación `P1+P2` bajo `$2,000` gross sin *CPPI Cash-Lock*.
