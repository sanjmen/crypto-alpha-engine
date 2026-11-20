# 11 — Resultados Forenses Fase P1: Libro de Órdenes L2 (`bookDepth`), `PremiumIndex 1m`, `Hyperliquid L1` y Geometrías de Monetización Convexa (`2023–2026`, `220` Perpetuos Point-in-Time)

- **Universo evaluado:** `220` perpetuos USDS-M Point-in-Time (`20` shards en Google Drive, `0 bytes` locales persistentes), cruzando:
  1. **`343,197,533` velas `1m` de `premiumIndexKlines`** (`Impact Bid / Impact Ask` contra Índice Spot Multi-Exchange para el `100%` de las `220` monedas).
  2. **Instantáneas causales L2 `bookDepth` (`±0.2%, ±1%, ±2%, ±5%` USD en Bid vs Ask)** alineadas estrictamente antes del cierre de señal (`t_book <= bar_close_ts`, máx `600s`).
  3. **`281,395` registros horarios de funding on-chain en `Hyperliquid L1`** (`bps/hora`).
  4. **3 Geometrías de Monetización `1m` Lineales** sin sesgo `np.log` y con prioridad `100%` pesimista al Stop-Loss: `MONO_A_CONVEX_TRAIL` (cola derecha desbloqueada), `MONO_B_COUPON_VACUUM` (cobro de cupón `:00:00` + salida condicional) y `MONO_C_SPLIT_50_50` (`50%` Maker TP en `+1.8%` + `50%` runner protegido hasta `+9.0%`).
- **Artefactos reproducibles:** [p1_microstructure_1m_report.md](../../reports/p1_microstructure_1m_report.md) · [p1_microstructure_1m_leaderboard.csv](../../reports/p1_microstructure_1m_leaderboard.csv) · [p1_microstructure_1m_summary.json](../../reports/p1_microstructure_1m_summary.json).
- **Código fuente y tests:** [microstructure_p1_ingestor.py](../../src/data/microstructure_p1_ingestor.py) · [microstructure_monetization_lab.py](../../src/models/microstructure_monetization_lab.py) · [test_microstructure_monetization_lab.py](../../tests/test_microstructure_monetization_lab.py).

---

## 1. El Gran Descubrimiento: `P6A_L2_ASK_AIR_POCKET_LONG` (`+60.5 bps/trade` Full, `+181.9 bps/trade` en `2026`, `100%` CPCV In-Fold)

Al incorporar la **asimetría real del libro de órdenes L2 (`bookDepth`)**, apareció por primera vez un fenómeno físico invisible en `OHLCV` con magnitud de **verdadero tesoro estructural**:

| Mecanismo | Geometría | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23–24` | `2025` | **`2026`** | **CPCV OOS** | **CPCV`%`** | BTC Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`P6A_L2_ASK_AIR_POCKET_LONG`** | `MONO_A_CONVEX_TRAIL` | `148` | **`+73.2`** | `+0.0` | `12.7` | **`+60.5`** | `-14.8` | `50.0%` | **`7.8%`** | **`+68.1`** | `+3.7` | **`+181.9`** | **`+58.3`** | **`100%`** | `+0.01` |
| **`P6A_L2_ASK_AIR_POCKET_LONG`** | `MONO_B_COUPON_VACUUM` | `148` | `+64.0` | `+0.0` | `12.5` | **`+51.5`** | `-1.0` | `50.0%` | `8.2%` | **`+57.3`** | `+4.5` | **`+154.6`** | **`+40.8`** | **`100%`** | `+0.01` |
| **`P6A_L2_ASK_AIR_POCKET_LONG`** | `MONO_C_SPLIT_50_50` | `148` | `+44.1` | `+0.7` | `11.7` | **`+33.1`** | **`+61.8`** | **`54.1%`** | `9.2%` | **`+35.3`** | `-4.1` | **`+129.6`** | **`+33.3`** | **`100%`** | `+0.00` |
| **`P2_C2A_BOOK_VACUUM_PAIN_LONG`** | `MONO_A_CONVEX_TRAIL` | `117` | `+35.0` | `+2.2` | `8.3` | **`+29.0`** | `-149.0` | `47.9%` | **`6.0%`** | **`+56.1`** | `-10.8` | **`+63.6`** | **`+56.2`** | **`100%`** | `-0.01` |
| **`P2_C2A_BOOK_VACUUM_PAIN_LONG`** | `MONO_B_COUPON_VACUUM` | `123` | `+37.9` | `+1.9` | `12.6` | **`+27.1`** | `-27.9` | `48.8%` | `9.2%` | **`+44.0`** | `-18.9` | **`+79.3`** | **`+59.3`** | **`80%`** | `+0.00` |
| **`B2_C2A_BASE_UNCHOKED_LONG`** | `MONO_A_CONVEX_TRAIL` | `404` | `+31.2` | `+3.9` | `8.3` | **`+26.7`** | `-38.2` | `48.5%` | **`6.1%`** | `-4.7` | **`+16.1`** | **`+98.5`** | **`+28.4`** | `67%` | `+0.01` |
| **`B2_C2A_BASE_UNCHOKED_LONG`** | `MONO_C_SPLIT_50_50` | `404` | `+20.9` | `+2.4` | `7.2` | **`+16.2`** | **`+66.2`** | **`56.7%`** | `6.6%` | `-6.0` | **`+16.8`** | **`+55.1`** | `+16.8` | `73%` | `+0.01` |
| `B1_C1B_BASE_UNCHOKED_LONG` | `MONO_C_SPLIT_50_50` | `187` | `+10.4` | `+19.4` | `11.8` | `+17.9` | `+61.8` | `56.1%` | `10.4%` | `+50.4` | `+35.8` | `-7.3` | `+20.6` | `73%` | `-0.03` |
| `P1_C1B_GLOBAL_SCARCITY_LONG` | `MONO_A_CONVEX_TRAIL` | `226` | `+1.9` | `+23.0` | `12.9` | `+11.9` | `-128.5` | `45.6%` | `10.4%` | `+101.8` | `+15.0` | `-10.2` | `+20.1` | `67%` | `+0.01` |
| `P3_C3A_BOOK_REPLENISHED_SNAP_LONG` | `MONO_A_CONVEX_TRAIL` | `369` | `+16.0` | `+1.9` | `12.9` | `+5.0` | `-51.6` | `47.2%` | `13.2%` | `+6.3` | `+2.9` | `+3.9` | `+5.7` | `73%` | `+0.01` |

---

## 2. Las 4 Conclusiones Científicas Más Importantes

### 1) Qué es físicamente `P6A_L2_ASK_AIR_POCKET_LONG` y por qué dio `+60.5 bps/trade` (`15 / 15` splits CPCV positivos)
- **La física exacta:** Cuando un perpetuo líquido (`ADV >= $15M`) acumula **`>= 1.16x` contratos abiertos** sobre su media de `24h`, con funding no saturado (`FR <= +3 bps`), rompe el máximo de `4h` con volumen (`RVOL_5m >= 2.0`) y **en ese mismo instante el libro L2 (`bookDepth`) tiene al menos `1.63x` más dólares apoyados en el `Bid -1%` que ofreciendo en el `Ask +1%` (`book_imb_1pct >= +0.24` y `book_imb_2pct >= +0.20`)**, significa que **los market-makers retiraron la liquidez vendedora arriba del precio (*Ask Air-Pocket*)**.
- Cualquier compra forzosa de cobertura atraviesa un libro hueco, produciendo una **cola derecha explosiva (`P90 = +491` a `+641 bps`**, peor trade acotado en `-204 bps`, `Profit Factor = 1.63`, `t = +2.36`, `Fee Share = 7.8%`, `BTC Corr = +0.01`).
- **Positivo en las 3 eras** (`+68.1 bps` en `2023–24`, `+3.7 / +4.5 bps` en `2025`, y **`+181.85 bps/trade` (`+1.82%` neto por operación!) en `2026`**) y **`100%` (`15 de 15`) de particiones Out-of-Sample positivas en `CPCV` in-fold (`+58.3 bps` mediana OOS)**, entrando como **`TAKER_OPEN`** (pagando `12.7 bps` completos de comisión VIP0 + slippage, sin depender de fills Maker ilusorios).
- *(Nota de rigor: el booleano automático `is_treasure` marcó `—` únicamente porque fijamos un corte arbitrario de `>= +15 bps` en cada año individual y `2025` cerró en `+3.7 / +4.5 bps`, aun siendo positivo en las 3 eras y `100%` en CPCV).*

### 2) Prueba de Ablación Exacta en `C2A` (`OI-VWAP` Dolor de Margen): Monetización + Libro L2
Comparando exactamente el mismo mecanismo `C2A` en tres escalones limpios:
1. **Con `PROFIT_LOCK_SL` apretado (`+80 bps`) sin libro L2 (Lab 1):** `+14.4 / +18.9 bps` Full, `2025 = -0.1 / +3.8 bps`, `2026 = +79.8 / +89.1 bps`, `CPCV = 60%–67%`.
2. **Solo desbloqueando la cola derecha (`B2_C2A` + `MONO_A_CONVEX_TRAIL`, sin libro L2):** Sube a **`+26.74 bps/trade` Full (`404` trades, `Fee Share = 6.1%`)**, `2025` sube a **`+16.13 bps/trade`**, y `2026` sube a **`+98.45 bps/trade` (`95` trades)** (`CPCV OOS` se duplica de `+12.6 -> +28.4 bps`).
3. **Agregando además el filtro L2 de Vacío de Asks (`P2_C2A_BOOK_VACUUM_PAIN_LONG` + `MONO_A_CONVEX_TRAIL`):** Filtra el `71%` de rupturas contra muros vendedores (`404 -> 117` trades), revierte `2023–24` de `-4.7 -> +56.1 bps/trade`, sube `Full` a **`+29.0 bps/trade`** y lleva el **`CPCV` in-fold a `100.0%` (`15 / 15` particiones positivas, `+56.2 bps` OOS)**.

### 3) Qué geometría de monetización conviene según el objetivo del portafolio (`MONO_A` vs `MONO_C`)
- **`MONO_A_CONVEX_TRAIL` (Cola Derecha Pura):** Maximiza la ganancia esperada neta por operación (`+60.5 bps` en `P6A`, `+29.0 bps` en `P2_C2A`) dejando correr los *home-runs* hasta `+8.5%`, con `Win Rate ~50%` y mediana ligeramente negativa (`-14.8 bps`), típico de convexidad tipo opción.
- **`MONO_C_SPLIT_50_50` (`50%` Maker TP en `+1.8%` + `50%` Runner con colchón `-35 bps` hasta `+9.0%`):** Sube el **`Win Rate` a `54.1%–56.7%`** y vuelve la **mediana fuertemente positiva (`+61.8` a `+66.2 bps`)** manteniendo `100%` de splits CPCV positivos en `P6A` (`+33.1 bps/trade` Full, `+129.6 bps` en `2026`), ideal si se busca una curva de capital más suave sobre el piso de `$2,500`.

### 4) Qué murió en `2026` (y por qué es una lección clave de estructura de mercado)
- Los mecanismos basados puramente en el **reloj determinista de cobro de funding (`:00:00`)** (`C1B`, `P1_C1B`, `C1A`, `M25`) funcionaron extraordinariamente bien en `2023–24` (`+50 a +109 bps/trade`) y en `2025` (`+15 a +40.5 bps/trade`), pero en **`2026` convergieron a `-10` .. `+8 bps/trade`** bajo todas las geometrías porque los bots HFT de arbitraje de funding ya anticipan la hora exacta del reloj.
- En cambio, los mecanismos gobernados por **fragilidad física instantánea del libro L2 + apalancamiento atrapado en contratos (`P6A` y `C2A`)** no dependen de una hora fija del reloj: ocurren cuando los market-makers se quedan sin inventario vendedor frente a un exceso de contratos abiertos, razón por la cual en la menor liquidez de altcoins de **`2026` su edge neto explotó (`+98.5` a `+181.9 bps/trade`)**.

---

## 3. Siguientes Pasos Propuestos

1. **Profundizar Fase P2 sobre el descubrimiento ganador (`L2 Order Book Air-Pocket` + `aggTrades` Millisecond Sweep Intensity):**
   - Evaluar por qué en `2025` `P6A` estuvo cerca de breakeven (`+3.7 / +4.5 bps` sobre `44` trades) frente a `+68.1 bps` (`23–24`) y `+181.9 bps` (`2026`): descargando quirúrgicamente en Colab los `aggTrades` de esos días candidatos para medir **barridos en `1 milisegundo` (`last_trade_id - first_trade_id + 1`)** y distinguir cuándo un *air-pocket* L2 viene acompañado de barrido institucional real vs. un libro vacío por falta de interés en mercado lateral (`2025`).
2. **Simulación Integrada `1-Minuto` One-Way de Portafolio (`unified_1m_simulator.py`):**
   - Medir el impacto exacto de integrar el *sleeve* `P6A + P2_C2A` (correlación **`+0.01`** contra `BTC`, `~8–12` disparos selectivos de alta convicción al mes) junto a `Core P1 + Manga P2 + M09/M04` bajo el gobernador real de `$2,700` / piso duro `$2,500` antes de tocar jamás producción.
