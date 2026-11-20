# 09 — Registro Vivo de Mecanismos de Mercado y Arquitectura de Datos de Microestructura (`Order Book`, `Liquidaciones`, `Cross-Exchange` y `On-Chain`)

> **Propósito del documento:** Este archivo es el **registro maestro permanente** del proyecto para:
> 1. Catalogar cada **mecanismo físico de mercado** observado (en backtest `1m` Point-in-Time y en producción real `crypto-mainnet`), identificando **quién es la contraparte forzada**, **cuándo falla** y **las distintas formas de monetizarlo** (no quedarnos con una sola regla rígida cuando el fenómeno subyacente admite mejores estructuras de captura).
> 2. Definir con rigor técnico qué **nuevas fuentes de datos** (`L2 Order Book Depth`, `Impact Premium Index 1m`, `Barridos Tick-a-Tick aggTrades`, `Stream Real de Liquidaciones !forceOrder`, `Dispersión Cross-Exchange` y `Mapa On-Chain de Liquidaciones / Depósitos CEX`) agregan información verdaderamente ortogonal que las velas `OHLCV + OI 5m` son incapaces de ver, listas para ingerir y evaluar en Google Colab.

---

## Parte I — Qué datos nuevos realmente importan (y cuáles son humo)

Con `OHLCV 1m` + `OI 5m` vemos **el resultado pasado** de las órdenes, pero estamos ciegos a **dos variables físicas críticas**:
1. **La elasticidad real del libro de órdenes ($\partial P / \partial Q$)** — cuánto cuesta mover el precio `+1%` o `-1%` *en este instante*.
2. **La micro-identidad del agresor** — si `$1M` de volumen comprador fueron `2,000` órdenes retail dispersas en `60s` o **`1` sola orden de liquidación forzosa** barriendo `140` niveles del libro en `1 milisegundo`.

A continuación clasificamos las **5 fuentes de datos de próxima generación** por orden de **impacto directo en `net_bps/trade` vs. costo de cómputo/almacenamiento**:

| Prioridad | Dataset | Fuente Histórica (`Backtest PiT`) | Fuente Live (`crypto-mainnet` Tokyo) | Qué ve que `OHLCV 1m` NO puede ver | Peso / Viabilidad en Colab + Drive |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P1 (Inmediato)** | **`bookDepth` L2 Asimetría de Libro (`±0.2%` a `±5%`)** | `data.binance.vision/data/futures/um/daily/bookDepth/{SYM}/` | `GET /fapi/v1/depth?limit=100` (`5` weight, `3 ms`) o `@depth20@100ms` | Mide los **dólares reales apoyados en Bid vs Ask** a `±0.2%, ±1%, ±2%, ±3%, ±5%`. Detecta **vacíos de liquidez (*air pockets*)** donde un pequeño flujo forzado dispara `+3%` sin resistencia, o muros reales que frenan un breakout falso. | **Muy liviano (`~2–4 GB` Zstd)** para el universo líquido en Google Drive. Ya tenemos el parser base en `src/data/binance_vision_downloader.py`. |
| **P1 (Inmediato)** | **`premiumIndexKlines` + `indexPriceKlines` (`1m`)** | `data.binance.vision/data/futures/um/monthly/premiumIndexKlines/` y `indexPriceKlines/` | `/fapi/v1/premiumIndex` + WebSocket `@markPrice@1s` (`0` weight) | 1) Cubre las **`34` altcoins volátiles sin Spot en Binance** usando el Índice Spot Multi-Exchange real (Coinbase + OKX + Bybit + Gate).<br>2) `premiumIndexKlines` mide `Impact Bid / Impact Ask` sobre el libro real, capturando **mechas intraminuto de iliquidez y descuento extremo** que el `close` oculta. | **Ultra liviano (`< 600 MB` Zstd para los `220` perps `2023–2026`)**. Se descarga en `< 8 min` en Colab. |
| **P2 (Quirúrgico por Eventos)** | **Huella de Liquidaciones y Barridos Institucionales en `aggTrades` (`1ms` Sweep Size)** | `data.binance.vision/data/futures/um/daily/aggTrades/{SYM}/` *(descarga selectiva solo en ventanas `±30m` alrededor de eventos candidatos, no el archivo masivo entero)* | `wss://fstream.binance.com/ws/!forceOrder@arr` (liquidaciones reales de todo Binance, `0` weight) + `@aggTrade` | *(Nota forense: Binance dejó de actualizar `liquidationSnapshot` histórico en S3 en `2023`).* Sin embargo, en `aggTrades` cada fila trae `last_trade_id - first_trade_id + 1`: **cuántas órdenes Maker descansando en el libro fueron comidas por UN SOLO match en `1 ms`**. Una liquidación forzosa barre `50–300` niveles en el mismo milisegundo; el flujo retail nunca hace eso. | Masivo si se baja bruto (`>500 GB`), pero **quirúrgico (`< 1.5 GB`)** si en Colab bajamos solo los `daily/aggTrades` de días/monedas con eventos extremos + grabamos `!forceOrder@arr` 24/7 en Tokyo. |
| **P3 (Cross-Exchange)** | **Dispersión de Funding, OI y Basis `Binance vs Bybit vs Hyperliquid`** | Archivos públicos `public.bybit.com` + API histórica `Hyperliquid` / `Coinalyze` | REST/WS públicos sin autenticación (`Bybit V5` + `Hyperliquid Info API`) | Distingue **arbitraje local de un exchange** (`FR = -50 bps` en Binance pero `-5 bps` en Bybit -> los arbs están *Long Binance / Short Bybit*, listos para cerrar a las `:00:00` = **Estado A puro**) de **escasez global del activo** (`FR = -50 bps` en los 3 exchanges a la vez = **Estado B squeeze global**). | **Liviano (`< 500 MB`)** para series `5m/1h` de funding + OI cross-exchange. |
| **P4 (On-Chain Real, sin humo)** | **1) Mapa On-Chain de Liquidaciones (`Hyperliquid L1`)<br>2) Depósitos de Tesorerías/VCs a Hot-Wallets de Binance** | `Hyperliquid L1` snapshots / nodos públicos + `DefiLlama` / RPC eventos `Transfer` hacia direcciones etiquetadas de Binance (`Arkham/Etherscan labels`) | `POST https://api.hyperliquid.xyz/info` (`clearinghouseState` de top vaults/whales) + monitor de depósitos `ERC20/SPL -> Binance Hot Wallet` | **Descarta métricas on-chain lentas/inútiles** (como MVRV o *exchange netflow* diario de BTC) y ataca 2 flujos causales directos:<br>• En **Hyperliquid**, el precio exacto de liquidación (`liquidationPx`) de ballenas apalancadas es público on-chain (actúa como imán magnético de precio).<br>• En **altcoins mid-cap**, un depósito grande de `unlock`/VC a la hot-wallet de Binance precede ventas reales en Spot por `10–60 min`. | Requiere ingesta específica (`P4`); ideal para armar un **recolector prospectivo continuo** en `crypto-mainnet` + backtest sobre ventanas disponibles. |

---

## Parte II — Registro Maestro de Mecanismos Observados y Formas de Monetizarlos

Cada mecanismo registrado abajo separa **el fenómeno físico observado** (que existe independientemente de nuestros parámetros) de las **estrategias candidatas para monetizarlo**, para que podamos iterar cómo extraer el máximo `%` neto sin casarnos con una única ejecución.

### `MEC-01` — Bifurcación de Liquidación de Funding Extremo Negativo (`FR <= -20 bps`)
- **Fenómeno físico observado (backtest + mainnet real `19-Nov-2026` en `HUMA/USDT` vs `ZKC/USDT`):**
  Cuando el Funding Rate cae por debajo de `-20 bps` (y especialmente con relojes comprimidos de `1h/2h/4h`), convergen **dos poblaciones con incentivos opuestos** en el segundo exacto `:00:00`:
  - **Sub-estado `01-A` (Sobrehang de Cosechadores / *Harvester Unwind*):** Arbitrajistas que entraron *Long Perp / Short Spot (o Short otro CEX)* antes de las `:00:00` únicamente para cobrar el cupón de funding. En `:00:01` su edge futuro cae a cero y desarman vendiendo el Perp.
  - **Sub-estado `01-B` (Escasez Aguda en Spot / *Spot Cash Squeeze*):** Compradores reales en Spot se llevan toda la oferta al contado (`Spot` cotiza `+100` a `+300 bps` por encima del Perp, superando por múltiplos el *funding*). No hay cosechadores suficientes porque **no hay inventario prestable en Spot para cubrirse**; quienes están en el Perp son **shorts direccionales atrapados** esperando que caiga tras pagar funding. Si en `:00:00 -> :01:00` el precio no cae, esos shorts entran en pánico y compran a mercado.
- **Trampas descubiertas:**
  1. Operar siempre Short en `T+1m` mezcla `01-A` (ganador) con `01-B` (explosión alcista en contra).
  2. Entrar Short con orden pasiva `GTX` sufre **selección adversa severa**: no se llena cuando `01-A` colapsa directo, y se llena el `100%` de las veces cuando `01-B` rebota en contra.
- **Vías de monetización a probar/comparar:**
  1. **Monetización `C1A` (Taker / Stop-Entry en confirmación de caída `:00 -> :01`):** Solo entrar Short si `Basis` NO está desbocado (y/o si `Bybit FR` es mucho menos negativo que `Binance FR`, probando arbitraje local) **y** el primer minuto post-cobro rompe a la baja con venta agresora (`bookDepth` con vacío de Bids debajo).
  2. **Monetización `C1B` (Pre-Settlement Carry + Squeeze Long `T-45m`):** Entrar **Long** `30–60 min` antes del cobro cuando `Spot >> Perp` (`Basis <= -80 bps`) + flujo comprador en Spot: se captura la compresión hacia `:00:00` **más el cobro directo del cupón de funding (`+25 a +200 bps`)** en efectivo.
  3. **Monetización `C1C` (Post-Settlement Failed-Dump Long `T+1m`):** Si tras un pago extremo (`<= -25 bps`) con `Spot >> Perp` el minuto `:00` cierra verde con compras Taker, comprar la capitulación tardía de los shorts atrapados.
  4. **Monetización `Delta-Neutral Sintética Cross-Perp`:** Long el Perp en `Estado 01-B` contra Short `BTCUSDT` o `ETHUSDT` ajustado por beta `60m`, aislando el *short squeeze* idiosincrático + el cobro de funding sin riesgo de caída general del mercado.

---

### `MEC-02` — Tomografía de Dolor de Margen (`OI-VWAP` en Unidades de Contrato)
- **Fenómeno físico observado:**
  Cuando el interés abierto medido en **contratos físicos (`sum_open_interest`)** crece `>= 12–25%` sobre su media de `24h` dentro de un rango estrecho y luego el precio se aleja `>= 1.8–2.5 ATR_1h` del **costo promedio ponderado por adición de contratos (`P_oi_vwap`)**, una masa crítica de cuentas con apalancamiento `10x–25x` entra simultáneamente en zona de *margin call*, mientras el OI aún **no** ha caído (aún no capitularon).
- **Trampas descubiertas:**
  - Usar `OI en USD` (`Contratos × Precio`) contamina el ancla porque la simple subida de precio infla el `OI_USD`. Debe calcularse estrictamente sobre `ΔContratos > 0`.
- **Vías de monetización a probar/comparar:**
  1. **Ignición por Ruptura (`C2A` Long / `C2B` Short):** Entrar a favor de la liquidación inminente cuando rompe el extremo de `4h` con volumen y `bookDepth` muestra poca liquidez defensiva del lado opuesto (`Ask depth` flaco arriba de resistencia para `C2A`).
  2. **Escalonado Convexo (*Pyramiding* con riesgo fijo):** Entrar `50%` en la ruptura inicial y `50%` si en los siguientes `5–10m` el `OI` en contratos empieza a caer (`ΔContratos < -1%`) mientras el precio acelera (confirmación física de que el motor de liquidación ya encendió).

---

### `MEC-03` — Elasticidad y Rebote Post-Liquidación Anclado en Spot (`Spot-Anchored Liquidation Snap-Back`)
- **Fenómeno físico observado:**
  Las órdenes de liquidación forzosa en perpetuos son órdenes `IOC` ciegas al precio que se ejecutan **únicamente en el libro del Perpetuo**, vaciando temporalmente el libro (`bookDepth`) y empujando al Perp `-25` a `-120 bps` por debajo (o por encima) de su propio mercado **Spot**, mientras los contratos abiertos (`sum_open_interest`) caen abruptamente (`<= -2%` en `15m`). Una vez agotada la cola de liquidaciones, los market-makers re-arbitran el Perp de vuelta hacia el precio Spot en `5–45 min`.
- **Trampas descubiertas:**
  - Comprar un desplome donde el `OI` **sube** (`ΔContratos > 0`) es suicida: no es una liquidación agotándose, es dinero nuevo abriendo shorts informados (ej. hackeo o mala noticia real donde Spot también colapsa).
- **Vías de monetización a probar/comparar:**
  1. **Reversión Confirmada `1m` (`C3A` Long / `C3B` Short):** Esperar la primera vela `1m` de giro con absorción en Spot tras el vaciado de contratos.
  2. **Captura Maker Profunda (`Sniper Limit Wick Catcher` con `!forceOrder@arr` en vivo):** En producción en Tokyo, escuchar en tiempo real `!forceOrder@arr`; solo cuando estalla un racimo de liquidaciones (`>= $500k` en `10s`) y el Perp se desancla `>= 35 bps` del `Index Price`, plantar órdenes `GTX` escalonadas (`-35 bps`, `-55 bps` respecto a Spot) para cobrar el rebote + rebate Maker.

---

### `MEC-04` — Transición de Fase del Reloj de Funding de Binance (`8h -> 4h -> 2h -> 1h`)
- **Fenómeno físico observado:**
  Cuando un perpetuo toca el techo/piso de funding, Binance comprime discretamente la frecuencia de cobro (`8h -> 4h -> 2h -> 1h`). Ese cambio de regla del exchange multiplica instantáneamente por `2x` a `8x` la tasa de sangrado horario de los shorts (o longs) atrapados, forzando cierres masivos en las horas siguientes a la transición.
- **Vías de monetización a probar/comparar:**
  1. **Continuación Inmediata en la Transición (`C4`):** Operar a favor del squeeze cuando se activa el acortamiento del reloj con `Spot >= Perp`.
  2. **Agotamiento Tardío de Ciclo (`Post-1h Exhaustion`):** Tras `12–24` cobros consecutivos de `1h` donde el volumen Spot se seca (`spot_qvol` cae `>50%` desde el pico) y el `OI` deja de subir, detectar el agotamiento final de la burbuja.

---

### `MEC-05` — Onda de Propagación en Tiempo-Evento desde Shocks Puros de Efectivo en `BTC` (`Sovereign Cash Wave`)
- **Fenómeno físico observado:**
  En tiempo tranquilo, el *lead-lag* lineal entre `BTC` y altcoins está arbitrado en milisegundos por HFTs. **Pero** en los contados eventos (`~20–25` por mes) donde `BTC` salta `>= 0.75%` en `3 minutos` con `>= 3.5x` volumen relativo impulsado por compras/ventas agresivas en **BTC Spot**, los market-makers retiran liquidez en docenas de altcoins de alta beta y varias quedan temporalmente rezagadas respecto a su beta causal de `60m` durante `2–25 minutos`.
- **Vías de monetización a probar/comparar:**
  1. **Catch-Up Direccional (`C5`):** Entrar Taker en las altcoins líquidas (`ADV >= $25M`, `beta_60m >= 1.25`) con mayor brecha de rezago (`lag_gap >= +50 bps`) apenas aparece flujo confirmatorio en su Spot.
  2. **Pair-Trade Neutral de Rezago (`Long Laggard / Short Already-Overshot Peer`):** Dentro del mismo sector (ej. IA o Memes o L1s), ir Long en la que aún se movió `<25%` de su beta contra Short la que ya sobrerreaccionó `>130%` de su beta.

---

### `MEC-06` — Vacío Asimétrico del Libro de Órdenes + Barrida Institucional `1-ms` (`L2 Ask Air-Pocket + aggTrades Sweep`, **VALIDADO FASES P1 & P2 ✅**)
- **Fenómeno físico validado (`220` perpetuos Point-in-Time `2023–2026`, `100%` cobertura `bookDepth` + `aggTrades`):**
  Cuando un perpetuo líquido (`ADV >= $15M`) acumula `>= 1.16x` contratos abiertos sobre su media de `24h` (`FR <= +3 bps`), rompe el máximo de `4h` con `RVOL_5m >= 2.0`, el libro L2 (`bookDepth`) exhibe `>= 1.63x` más dólares en `Bid -1%/-2%` que en `Ask +1%/+2%` (`book_imb_1pct >= +0.24`, `book_imb_2pct >= +0.20`), **y al menos el `32%` del volumen de `5m` proviene de barridas institucionales multi-nivel en `1 milisegundo` (`P6A_L2_3_INST_SWEEP_CONC`)**:
  - **Edge standalone (`MONO_A_CONVEX_TRAIL`, `112` trades Taker):** **`+71.15 bps/trade` netos (`t = +2.37`, `PF = 1.78`, `Fee Share = 7.5%`)**, positivo en las 3 eras (`+74.0` en `23–24`, **`+10.2 / +11.2 bps` en `2025`**, **`+236.55 bps/trade` en `2026`**) y **`100%` (`15/15`) splits CPCV in-fold positivos (`+85.83 bps` OOS)**.
  - **Impacto en portafolio productivo `1m` One-Way ([12_resultados_fase_p2_aggtrades_y_portafolio_1m.md](./12_resultados_fase_p2_aggtrades_y_portafolio_1m.md)):** Sube `4Y Sharpe` de `1.99 -> 2.15–2.19`, reduce `4Y MaxDD` de `36.08% -> 35.30%`, y desde `$2,700` en `2026` sube el capital final de `$3,912.53` a **`$4,392–$5,744`** mejorando el colchón sobre el piso de `$2,500` (`MinEq = $2,544–$2,560`).

---

### `MEC-07` a `MEC-10` — Dispersión Cross-Exchange (`Binance vs Bybit V5`) y Anclaje On-Chain (`Hyperliquid L1`) (**AUDITADOS EN FASE P3/P4 — VER [13_resultados_fase_p3_p4_cross_venue_y_onchain.md](./13_resultados_fase_p3_p4_cross_venue_y_onchain.md)**)
- **Resultados empíricos sobre `215` perpetuos líquidos (`876,516` filas `Bybit V5` + `641,241` filas `Hyperliquid L1`, `1,539` trades `1m`):**
  1. **`MEC-01A / P3_M1` (*Cross-CEX Harvester Unwind Short*):** `-6.21 bps/trade` Full (`-72.4` en `23–24`, `-10.8` en `2025`, `+23.8` en `2026`, `CPCV = 33%`). **Descartado definitivamente** (confirma que `M25` debe permanecer apagado).
  2. **`MEC-08 / P3_M2` (*Global 3-Venue Short-Trap Squeeze Long*):** **Positivo en las 3 eras** (`+12.0` / `+12.1` / `+13.7 bps` -> **`+12.38 bps/trade`** en `MONO_B`, `60%–67%` CPCV), pero `5.7x` inferior al edge de `MEC-06` / `M26` (`+71.15 bps/trade`) y ya cubierto por `M04 + M26`. **No se agrega standalone para no diluir slots.**
  3. **`MEC-09 / P4_M3` (*On-Chain Hyperliquid Lead Long*):** **`-49.5 a -62.0 bps/trade` (`0%` CPCV)**. **Falsificado**: Binance lidera el descubrimiento de precios; cuando Hyperliquid cotiza con prima aislada es *FOMO* minorista que revierte a la baja.
  4. **`MEC-10 / P4_M4` (*Cross-Venue Solvency-Anchored Liquidation Snap-Back*):** Negativo en `2023–2025` (`-17.8` / `-8.4 bps`), pero **`+70.87 bps/trade` en `2026` (`56` trades, `73.3%` CPCV `+49.6 bps` OOS)** tras la maduración institucional de Hyperliquid/Bybit. **En Watchlist** (requiere prueba futura con `!forceOrder@arr` + absorción `aggTrades` `1-ms` antes de tocar producción).

---

## Parte III — Estado Actual del Ciclo de Descubrimiento y Próximas Fronteras

1. **En Producción Viva (`crypto-mainnet` Tokio):** `M25` apagado (`enable_post_settle_sniper=false`) y **`Sniper:M26_L2SweepAirPocket` (`MEC-06` = `L2 Ask Air-Pocket + 1ms Institutional Sweep + MONO_A Convex Trail`)** desplegado y operando en vivo junto a `Core P1 + Manga P2 + M09/M23/M04` bajo el piso duro de `$2,500.00`.
2. **Próximas fronteras de alto valor (sin diluir lo que ya funciona):**
   - **Colector prospectivo `!forceOrder@arr` + `L2/aggTrades` para `MEC-10` (`P4_M4`):** Grabar eventos reales de liquidación forzosa en Binance cruzados con solvencia en Bybit/Hyperliquid y absorción `1-ms` para evaluar si el edge de `+70.9 bps/trade` observado en `2026` se vuelve estructural con datos de liquidación directa.
   - **Optimización de la Barrera de Rebalanceo del Core P1:** Nuestra auditoría en vivo (`scripts/inspect_production_status.py`) mostró que el `75.8%` de los cierres en producción ocurren en la barrera de rebalanceo de ciclo (`LIMIT/MARKET`) mientras que `Profit-Lock` (`+3.5% -> +1.5%`) y `Take-Profit` (`+5.0%`) son fuertemente ganadores (`+$117` y `+$437` netos). Evaluar amortiguación de rotación (*no-trade buffer* / histéresis de ranking) en `Core P1` es una oportunidad directa para reducir costos de rotación sin agregar riesgo direccional.
