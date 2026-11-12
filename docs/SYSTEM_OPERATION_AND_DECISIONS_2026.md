# 🏛️ Documentación Maestra del Sistema: Funcionamiento, Evidencia Empírica y Registro de Decisiones (Noviembre 2026)

**Última actualización:** 12 de Noviembre de 2026  
**Estado en Producción (`--mainnet`):** Activo en Google Cloud Tokyo (`asia-northeast1-a`, VM `crypto-alpha-engine`, IP estática `35.194.111.133`) sobre Cuenta Master de Binance USDS-M Futures (`$3,000+ USDT`, Cross-Margin, One-Way Mode, `5.0x` Leverage).  
**Estado en Laboratorio (`--shadow`):** Activo en paralelo en el mismo nodo (`data/shadow_trading.db`) evaluando en cuarentena las extensiones experimentales de M13.

---

## 1. Cómo Funciona el Sistema (Arquitectura Operativa End-to-End)

El motor de trading se ejecuta a través del lanzador multi-modo unificado [`scripts/run_trading.py`](../scripts/run_trading.py) y está orquestado por [`ShadowTradingDaemon`](../src/execution/shadow_daemon.py), conectando cuatro ciclos concurrentes con aislamiento estricto de base de datos SQLite WAL (`data/live_trading.db` para `--mainnet`, `data/shadow_trading.db` para `--shadow`, `data/testnet_trading.db` para `--testnet`).

### 1.1. Los 4 Ciclos Operativos de `ShadowTradingDaemon`

1. **Ciclo Rápido (*Fast Cycle* — cada `5s` + Streaming WebSocket L1 en tiempo real):**
   - [`ResilientWebSocketClient`](../src/execution/ws_manager.py) recibe cotizaciones `bookTicker` L1 (`bid`, `ask`, `price`), deduplica ticks sin cambio de precio y cede el GIL (`time.sleep(0.001)`) para mantener el uso de CPU `< 4%`.
   - [`ensure_position_brackets()`](../src/execution/shadow_daemon.py) verifica que toda posición abierta tenga sus órdenes de salida (`TAKE_PROFIT` y/o `STOP_LOSS`) registradas y sincronizadas.
   - [`check_pending_orders()`](../src/execution/shadow_broker.py) evalúa todas las órdenes pendientes (`LIMIT`, `TAKE_PROFIT`, `STOP_LOSS`) contra el *top-of-book* actual; si una orden bracket se ejecuta, cancela automáticamente su par opuesto (lógica **OCO — One-Cancels-the-Other**).
   - [`manage_trailing_and_profit_locks()`](../src/execution/shadow_daemon.py) rastrea el mejor precio alcanzado desde la apertura (`peak_price`) en cada posición y, cuando la ganancia máxima intraciclo alcanza el umbral de **Profit-Lock (`+3.5%` en Core / `+4.5%` en Anomalías)**, coloca o sube automáticamente un `Stop-Profit` garantizado en **`+1.5%`** (`+2.5%` en Anomalías).
   - En modo `--mainnet`, sincroniza cada `60 segundos` el balance real de la billetera en Binance (`fapiPrivateV2GetBalance`) mediante [`sync_from_live_exchange()`](../src/execution/shadow_broker.py) para reflejar depósitos, transferencias o pagos de *funding* sin reiniciar el contenedor.

2. **Escaneo de Anomalías (*Anomaly Scan* — cada `30s`):**
   - Analiza los `300+` contratos perpetuos USDS-M de Binance filtrando activos con volumen 24h `>= $20M USDT` y excluyendo contratos con vencimiento, opciones (`-C`/`-P`), tokens apalancados y activos TradFi con cierre de mercado ([`_is_tradfi_or_gapped_symbol()`](../src/execution/shadow_daemon.py)).
   - Evalúa las estrategias satélite de eventos extremos ([`src/strategies/extreme_anomaly.py`](../src/strategies/extreme_anomaly.py)):
     - **`Extreme Drop Bounce`**: Reversión alcista tras capitulaciones `<= -15%` en 24h con `RSI < 28` (`TP +6%`, `SL -5%`).
     - **`Early Breakout`**: Impulso alcista temprano (`+8%` en 4h con explosión de volumen `>= 2.5x`, `TP +15%`, `SL -4%`).
     - **`Pump & Funding Fade`**: Corto en agotamiento de *pumps* verticales (`>= +25%` en 24h con *funding rate* extremo `>= +0.05%`, `TP +8%`, `SL -8%`).
     - **`Extreme Funding Harvester`** *(solo en `--shadow`, en cuarentena por Issue `#64`)*: Captura de tasas de financiamiento negativas extremas (`<= -0.20%` c/ 8h) con cobertura beta-neutral en `BTC/USDT`.
   - Aplica el escudo **Open Interest Squeeze Guard (Issue `#46`)** para bloquear cortos cuando el Open Interest se expande violentamente (`+8%` en 1h), y cierra cualquier posición satélite estancada tras **`48 horas` (`Time-Stop`)**.

3. **Ciclo de Rebalanceo Cuantitativo (*Rebalance Cycle* — cada `4 horas` o por deriva de neutralidad intraciclo):**
   - Descarga velas históricas de `1h` y `15m`, tasas de financiamiento, ratios de posicionamiento institucional (`fapiData`: Ballenas Top 20% vs. Retail) y base trimestral (`fapiDataGetBasis` `CURRENT_QUARTER`).
   - Ejecuta el **Motor Causal Online de 7+1 Factores** ([`SevenFactorOnlineEngine`](../src/strategies/seven_factor_engine.py)):
     - **`F1_Macro_Idio_PB`**: Momentum idiosincrático (neutral a Beta de BTC) de `7d / 14d / 30d` entrando en retrocesos de `4h–24h`.
     - **`F2_Med_Idio_PB`**: Momentum idiosincrático intermedio de `3d / 7d` entrando en retrocesos de `12h`.
     - **`F3_Fast_Swing_PB`**: Impulso rápido de `12h / 24h / 48h` ponderado por volumen entrando en retrocesos de `4h`.
     - **`F4_Rev_24_72h`**: Reversión a la media transversal de `24h–72h`.
     - **`F5_Carry_Neg`**: Arbitraje contrarian de tasas de financiamiento (`-funding_72h`).
     - **`F6_Liq_Surge`**: Expansión estructural de liquidez en dólares (`168h` vs `720h`), filtrada por la **Puerta de Alta Convicción (`t-stat >= +0.75`, Issue `#61`)** para evitar operar cuando el factor está en racha negativa.
     - **`F7_HighLiq_Size`**: Factor de calidad/liquidez institucional (`log(DollarVol_168h)`).
     - **`F8_Smart_Money_Flow` (Issue `#62`)**: Divergencia de posicionamiento entre Ballenas (`TopLongShortPositionRatio`) y Retail (`GlobalLongShortAccountRatio`) confirmada por flujo Taker.
   - Pondera dinámicamente cada factor según su **t-statistic causal fuera de muestra** en ventanas rodantes de `36h`, `96h` y `240h` (`priors=None`, 100% adaptativo al régimen actual), modula la exposición por el régimen de **Base Trimestral (`Quarterly Delivery Basis`)** y por **Volatility Targeting de 72h**, y fuerza **Neutralidad Estricta en Dólares ($\sum w_{\text{Long}} = -\sum w_{\text{Short}}$)**.
   - Pasa los pesos por el controlador de **Apalancamiento Escalonado con Histéresis (`SteppedLeverageController`)** y aplica un **Deadband de Rotación del `4.0%`**: solo se envían órdenes en las monedas cuyo peso objetivo cambió en más de `4%`, reduciendo las comisiones a menos de la mitad.

4. **Ciclo de Financiamiento y Rendimiento de Colateral (`00:00`, `08:00`, `16:00 UTC`):**
   - Registra los cobros/pagos de *Funding Rate* de cada posición perpetua abierta y contabiliza el rendimiento pasivo (`~5.5% APY`) sobre colateral `BFUSD` / `LDUSDT` en cuentas Cross-Margin.

---

## 2. Mecánica de Posiciones Abiertas, Órdenes Bracket y Ejecución Maker

### 2.1. Por qué las Órdenes se Ejecutan como `Maker` (`0.02%`) y Cómo Funciona el Libro Local vs. Binance
- **Prioridad `100% Maker (Post-Only GTX)`:** Tanto en el rebalanceo Core como en las entradas de Anomalías, [`SmartExecutionRouter`](../src/execution/execution_router.py) y [`BinanceLiveBroker`](../src/execution/live_broker.py) consultan el *top-of-book* L1 en tiempo real (`fetch_best_bid_ask`) y el desequilibrio de profundidad L2 (`Order Book Imbalance — OBI`) para colocar órdenes límite pasivas `Post-Only (GTX)` en la punta del libro. Si la orden no se llena tras la ventana de *Maker Chase*, hace fallback controlado para garantizar que el portafolio nunca quede desbalanceado.
- **Gestión de Brackets (`TAKE_PROFIT` y `Profit-Lock`):**
  - Cada posición abierta del Core tiene asociada en la base de datos SQLite una orden pendiente `TAKE_PROFIT` en **`+5.0%`** desde su precio de entrada (`shd_tp_...`, rol `MAKER`).
  - El demonio monitorea los precios cada `5 segundos` (y en cada tick WebSocket). Cuando el precio toca el nivel de `TAKE_PROFIT` o el nivel de `Profit-Lock` (`+1.5%` una vez superado `+3.5%` de ganancia pico), [`ShadowExecutionBroker.check_pending_orders()`](../src/execution/shadow_broker.py) dispara el cierre de la posición con `reduceOnly=True` hacia Binance Mainnet.
  - Por diseño, los brackets dinámicos son gestionados por el motor de vigilancia continua del demonio en su base de datos local y disparados a Binance en el instante del cruce, evitando chocar con el límite de `10` órdenes algorítmicas condicionales simultáneas de Binance Futures cuando se mantienen 14–18 posiciones abiertas.

### 2.2. Cooldown de 4 Horas y Re-centrado Automático de Neutralidad Intraciclo
- Cuando una posición Core se cierra a mitad del ciclo de 4 horas por `TAKE_PROFIT` (`+5.0%`) o `Profit-Lock` (`+1.5%`):
  1. **Cooldown de 4 Horas ([`_apply_core_bracket_cooldowns()`](../src/execution/shadow_daemon.py)):** El símbolo entra en enfriamiento durante `4.0 horas` para esa misma dirección (`LONG` o `SHORT`), impidiendo que el siguiente rebalanceo vuelva a abrir inmediatamente la misma posición agotada.
  2. **Re-centrado de Neutralidad ([`is_intracycle_rebalance_needed()`](../src/execution/shadow_daemon.py)):** Si el cierre intraciclo deja el portafolio desbalanceado entre Longs y Shorts por encima del umbral de deriva (`25%`), tras `15 minutos` de asentamiento (`intracycle_post_stop_settle_sec = 900s`) el demonio dispara automáticamente un rebalanceo intraciclo que re-centra $\sum \text{Longs} = \sum \text{Shorts}$.

---

## 3. Pruebas Realizadas y Evidencia Empírica

### 3.1. Estadística Forense Real en `shadow_trading.db` (`22 Días`: `2026-10-21` a `2026-11-12`)

Durante los 22 días de operación continua en tiempo real se registraron **`1,673` ejecuciones** (`703` aperturas/escalados y **`970` cierres de posición**). El desglose exacto por mecanismo de salida y por bloque estratégico es el siguiente:

#### A. Motor Core 7-Factor (`764` Salidas Cerradas)
| Barrera / Mecanismo de Salida | Cierres (`N`) | % de Salidas | Win Rate | Retorno Medio (Mediana) | PnL Bruto (`$`) | Comisiones (`$`) | **PnL Neto (`$`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `TAKE_PROFIT` (`+8.0%` Maker)** | `8` | `1.0%` | `100.0%` | **`+8.35%`** (`+8.70%`) | `+$77.22` | `$0.20` | **`+$77.02`** |
| **2. `TRAILING_STOP` (`Pico >= +4.5%, Dist 2%`)** | `44` | `5.8%` | `100.0%` | **`+4.08%`** (`+3.64%`) | `+$353.46` | `$3.64` | **`+$349.82`** |
| **3. `PROFIT_LOCK` (`Pico >= +3.5% -> +1.5%`)** | `34` | `4.5%` | `100.0%` | **`+1.18%`** (`+0.81%`) | `+$78.31` | `$2.55` | **`+$75.76`** |
| **4. `HARD_STOP_LOSS` (`<= 0%`)** | `0` | `0.0%` | — | — | `$0.00` | `$0.00` | **`$0.00`** |
| **5. `REBALANCE_4H_EXIT` (Rotación 4h Maker)** | `678` | `88.7%` | `44.7%` | **`-0.25%`** (`-0.08%`) | `-$307.48` | `$20.00` | **`-$327.48`** |
| **Total Salidas Core (`764` cierres)** | **`764`** | **`100.0%`** | **`50.9%`** | **`+0.16%`** (`+0.01%`) | **`+$201.51`** | **`$26.38`** | **`+$175.13`** *(+$159.44 tras fees de apertura)* |

#### B. Estrategias Satélite de Anomalías (`206` Salidas Cerradas)
| Barrera / Mecanismo de Salida | Cierres (`N`) | % de Salidas | Win Rate | Retorno Medio (Mediana) | PnL Bruto (`$`) | Comisiones (`$`) | **PnL Neto (`$`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `TAKE_PROFIT` (`+6% / +8% / +15%`)** | `35` | `17.0%` | `100.0%` | **`+6.25%`** (`+5.66%`) | `+$35.81` | `$0.11` | **`+$35.70`** |
| **2. `TRAILING_STOP` (`> +2.4%`)** | `49` | `23.8%` | `100.0%` | **`+3.71%`** (`+3.10%`) | `+$29.56` | `$0.32` | **`+$29.25`** |
| **3. `PROFIT_LOCK` (`+0.8%..+2.4%`)** | `24` | `11.7%` | `100.0%` | **`+1.35%`** (`+0.80%`) | `+$4.94` | `$0.15` | **`+$4.79`** |
| **4. `HARD_STOP_LOSS` (`-4% / -5% / -8%`)** | `97` | `47.1%` | `0.0%` | **`-4.87%`** (`-5.25%`) | `-$82.39` | `$0.69` | **`-$83.07`** |
| **5. `TIME_STOP_48H`** | `1` | `0.5%` | `100.0%` | **`+0.09%`** (`+0.09%`) | `+$0.01` | `$0.01` | **`+$0.01`** |
| **Total Salidas Anomalías (`206` cierres)** | **`206`** | **`100.0%`** | **`52.9%`** | **`-0.19%`** (`+0.79%`) | **`-$12.06`** | **`$1.27`** | **`-$13.33`** |

---

### 3.2. Backtest Comparativo de 14 Configuraciones de Barreras (`TP`, `SL`, `Profit-Lock`, `Trailing Stop`)

Al observar en `shadow_trading.db` que solo `8` operaciones Core habían alcanzado el `TP +8.0%`, mientras que `44` operaciones habían superado `+4.5%` pero terminaron cerrando por `Trailing Stop` con un retorno medio de `+4.08%` (mediana `+3.64%`) pagando comisión Taker (`0.04%`), ejecutamos una simulación histórica rigurosa sobre las velas reales de `1h` (`high`/`low` intrabarra) en **`52 días` (`2026-09-20` a `2026-11-11`)** y en la ventana exacta de **`22 días` de Shadow (`2026-10-21` a `2026-11-11`)**:

| Configuración Evaluada en Core 7-Factor (`Reb 4h`, `3.0x Base Vol-Targeted`) | Retorno `52 Días` | Retorno `22 Días` *(Ventana Shadow)* | Sharpe Ratio | Max Drawdown | Salidas (`TP / Trail / Lock / SL / Reb`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Puro Rebalanceo 4h (Sin TP, Sin SL, Sin Trail/Lock)** | `-3.5%` | `-2.9%` | `-0.79` | `-21.33%` | `TP:   0 | Tr:  0 | Lk:  0 | SL:  0 | Reb:505` |
| **1. Shadow Histórico (`TP 8%`, `Lock 3.5->1.5%`, `Trail 4.5/2%`, `Sin SL`)** | `+21.6%` | `+9.5%` | `2.76` | `-15.43%` | `TP:  20 | Tr:118 | Lk: 72 | SL:  0 | Reb:434` |
| **2. ⭐ ÓPTIMA ADOPTADA (`TP 5.0%` + `Lock 3.5->1.5%`, SIN Trailing, Sin SL)** | **`+23.4%`** | **`+12.6%`** | **`2.70`** | **`-14.75%`** | `TP: 128 | Tr:  0 | Lk: 87 | SL:  0 | Reb:434` |
| **3. Variante `TP 4.5%` + `Lock 3.0->1.5%` (SIN Trailing, Sin SL)** | `+15.6%` | **`+13.6%`** | `1.96` | **`-13.02%`** | `TP: 134 | Tr:  0 | Lk:137 | SL:  0 | Reb:419` |
| **4. Solo TP Fijo `+5.0%` (Sin Trailing, Sin Profit-Lock, Sin SL)** | `+10.3%` | `+11.7%` | `1.29` | `-14.89%` | `TP: 145 | Tr:  0 | Lk:  0 | SL:  0 | Reb:450` |
| **5. Solo TP Fijo `+4.0%` (Sin Trailing, Sin Profit-Lock, Sin SL)** | `+9.3%` | `+11.4%` | `1.24` | `-13.96%` | `TP: 204 | Tr:  0 | Lk:  0 | SL:  0 | Reb:430` |
| **6. Solo TP Fijo `+3.0%` (Sin Trailing, Sin Profit-Lock, Sin SL)** | `+16.3%` | `+8.5%` | `2.37` | `-11.72%` | `TP: 306 | Tr:  0 | Lk:  0 | SL:  0 | Reb:405` |
| **7. Solo TP Fijo `+2.0%` (Sin Trailing, Sin Profit-Lock, Sin SL)** | `-3.1%` | `-0.3%` | `0.09` | `-17.06%` | `TP: 419 | Tr:  0 | Lk:  0 | SL:  0 | Reb:338` |
| **8. Solo TP Fijo `+8.0%` (Sin Trailing, Sin Profit-Lock, Sin SL)** | `-1.4%` | `-0.9%` | `-0.18` | `-18.99%` | `TP:  55 | Tr:  0 | Lk:  0 | SL:  0 | Reb:489` |
| **9. Con `SL -2.5%` (`TP 8%`, `SL -2.5%`, `Lock 3.5->1.5%`, `Trail 4.5/2%`)** | **`-1.7%`** | **`-8.7%`** | `0.26` | `-20.48%` | `TP:  29 | Tr:124 | Lk: 82 | SL:408 | Reb:286` |
| **10. Con `SL -3.5%` (`TP 8%`, `SL -3.5%`, `Lock+Trail`)** | `+1.2%` | `-1.6%` | `0.65` | `-14.67%` | `TP:  28 | Tr:130 | Lk: 88 | SL:289 | Reb:327` |
| **11. Con `SL -5.0%` (`TP 8%`, `SL -5.0%`, `Lock+Trail`)** | `+10.6%` | `+4.9%` | `2.04` | `-16.75%` | `TP:  28 | Tr:133 | Lk: 80 | SL:179 | Reb:353` |
| **12. Con `SL Catastrófico -7.0%` (`TP 8%`, `SL -7.0%`, `Lock+Trail`)** | `+9.9%` | `+3.8%` | `1.82` | `-14.58%` | `TP:  20 | Tr:133 | Lk: 89 | SL: 96 | Reb:389` |

---

## 4. Registro de Decisiones Clave de Ingeniería y Estrategia

### Decisión 1: Adopción de `TP Fijo +5.0% (Maker)` + `Profit-Lock (+3.5% -> +1.5%)` SIN `Trailing Stop` ni `SL -2.5%` (`commits cfdb24b` y `4cdbb2a`)
- **Contexto:** Inicialmente probamos colocar un `Stop-Loss` individual de `-2.5%` en el Core tras observar un *short squeeze* aislado en `ARB/USDT` (`-$86.69`) y `RENDER/USDT` (`-$20.31`) el 12 de noviembre.
- **Evidencia:** El backtest de barreras demostró que un `SL -2.5%` permanente en un portafolio neutral de 14 altcoins se dispara `408 veces` en 52 días por ruido intrahora normal (`2%–3%`), pagando `0.04%` Taker y forzando al rebalanceador de neutralidad a recortar las posiciones ganadoras del lado opuesto, reduciendo el retorno de `+21.6%` a `-1.7%`. Asimismo, demostró que bajar el `Take-Profit` fijo de `+8.0%` a **`+5.0%` (ejecución `Maker 0.02%`)** y **desactivar el `Trailing Stop`** (conservando el `Profit-Lock` de `+3.5% -> +1.5%`) captura limpiamente el `+5.0%` en las `128` excursiones ganadoras que antes devolvían `-2.0%` en el Trailing Stop, elevando el retorno a **`+23.4%` (`52d`) / `+12.6%` (`22d`)** y reduciendo el Max Drawdown a **`-14.75%`**.
- **Acción:** Se implementó, probó con tests unitarios (`191 passed`) primero en `--shadow` y luego se unificó en `--mainnet`.

### Decisión 2: Promoción Selectiva a Producción (Milestone M15) vs. Cuarentena en Shadow
- **Promovido a `--mainnet` (`commit c4a7ae5`):**
  1. **Issue `#61` (`priors=None` Dynamic Online IC + `F6 Gate t>=0.75`):** Eliminó el arrastre de factores con t-stat negativo (como `F5_Carry` en `-2.81` o `F6_Liq` en `-0.83`) que antes seguían recibiendo peso por tener *priors* fijos distintos de cero.
  2. **Issue `#62` (`F8_Smart_Money_Flow`, `Quarterly Delivery Basis` y `Yield Collateral`):** Integró el flujo institucional de Ballenas vs. Retail y el régimen de base trimestral con riesgo controlado.
  3. **Issue `#63` (`60s` Live Wallet Balance Sync):** Sincronización continua del capital de billetera real cada 60 segundos.
- **Mantenido en Cuarentena en `--shadow`:**
  1. **Issue `#64` (`Extreme Funding Harvester`):** En cuarentena porque sus 4 ejecuciones en Shadow tocaron el Stop-Loss tras la liquidación del *funding* (`-$10.30` netos).
  2. **Issue `#65` (`4.5% Dynamic Anomaly Sizing`):** En cuarentena (`$45` de tope en `--mainnet` vs. `$135` en `--shadow`) hasta que la tasa de acierto neta de las anomalías supere el `55%` sin depender de stops cortos.

### Decisión 3: Optimización de Almacenamiento SQLite WAL y Event-Loop WebSocket (Milestone M14 — `commits 25592cd` y `5868923`)
- **Contexto:** Tras 20 días, `shadow_trading.db` acumulaba `162,792` filas en `accounts` (`25 MB`) y el hilo WebSocket consumía `~21%` de CPU procesando ticks L1 repetidos.
- **Acción (Issue `#59`):**
  - Se implementó `prune_historical_telemetry(retention_days=7)` en [`ShadowDatabase`](../src/execution/database.py) con `PRAGMA auto_vacuum = INCREMENTAL` y `wal_checkpoint(TRUNCATE)` cada 24h, conservando el **100% del historial de `trades`, `funding_settlements` y `rebalance_snapshots`** y podando solo snapshots de telemetría mayores a 7 días (reduciendo la DB de `25 MB` a `2.8 MB`).
  - Se agregó *throttling* de snapshots de cuenta (`1 cada 60s`), deduplicación de ticks L1 sin cambio de precio y cesión de event-loop (`1ms sleep`), bajando el uso de CPU del contenedor a `< 4%`.
  - **Issue `#60` (Redis 7 Streams):** Quedó documentado como alternativa en reserva en caso de escalar a múltiples consumidores en el futuro.

### Decisión 4: Infraestructura Zero-Trust en Google Cloud Tokyo (`asia-northeast1-a`) y CI/CD (`commits 2ad8526` y `cfdb24b`)
- **Despliegue:** VM `crypto-alpha-engine` (`e2-medium`, Ubuntu 24.04 LTS, 4 GB RAM + 2 GB Swap) en **Tokio (`asia-northeast1-a`)**, a `< 2ms` de los servidores de Binance Futures (`fapi.binance.com`).
- **Seguridad Zero-Trust:** Regla de firewall `default-allow-ssh` eliminada (`0` puertos públicos abiertos a Internet); acceso administrativo y de CI/CD exclusivamente a través de túneles autenticados **Google Cloud IAP (`35.235.240.0/20`)**.
- **Pipeline CI/CD (`.github/workflows/deploy.yml`):** Cada `git push origin main` ejecuta la suite completa de `191` tests unitarios en GitHub Actions y, si pasa al 100%, empaqueta y despliega automáticamente los contenedores `crypto-mainnet` y `crypto-shadow` en Tokio reutilizando una única imagen Docker (`crypto-alpha-engine:latest`).

---

## 5. Comandos de Operación y Monitoreo Rápido

```bash
# 1. Ver el estado en vivo de Producción (--mainnet) y Shadow (--shadow) desde tu Mac:
./scripts/deploy_to_gcp.sh --status

# 2. Abrir el Dashboard Rich en vivo conectado a la VM de Tokio (vía túnel IAP):
gcloud compute ssh santim@crypto-alpha-engine --zone=asia-northeast1-a --project=my-project-alpha-514420 --tunnel-through-iap -- -t "docker exec -it crypto-mainnet python scripts/run_trading.py --monitor --mainnet"

# 3. Abrir el Dashboard Rich de Shadow Lab en vivo:
gcloud compute ssh santim@crypto-alpha-engine --zone=asia-northeast1-a --project=my-project-alpha-514420 --tunnel-through-iap -- -t "docker exec -it crypto-shadow python scripts/run_trading.py --monitor --shadow"

# 4. Ejecutar la suite completa de 191 tests unitarios localmente:
.venv/bin/python -m pytest tests/ -q
```
