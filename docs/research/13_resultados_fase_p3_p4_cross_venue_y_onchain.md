# 13 — Resultados Forenses Fase P3/P4: Dispersión Cross-Exchange (`Binance vs Bybit V5`) y Flujos On-Chain (`Hyperliquid L1`) (`2023–2026`)

## 1. Resumen Ejecutivo y Decisión Operativa (`NO-GO` para nuevos Snipers P3/P4; `M26` confirmado como estándar único)

Siguiendo la hoja de ruta del registro maestro ([09_registro_mecanismos_y_arquitectura_datos_microestructura.md](./09_registro_mecanismos_y_arquitectura_datos_microestructura.md)), completamos la ingesta multi-exchange y la auditoría `1-minuto` lineal de los **4 mecanismos Cross-Venue & On-Chain (`Fase P3/P4`)** sobre **`215` perpetuos líquidos Point-in-Time (`ADV_7d >= $15M`)** a lo largo de los `20/20` shards (`2023–2026`, `0 bytes` residuales en disco local):

* **Datos Cross-Venue ingeridos y persistidos en Google Drive (`m16_multimodal/cross_venue/`):**
  * **`Bybit V5` Linear Perpetuals (`cex_peer_funding_1h.parquet`):** **`876,516` liquidaciones históricas** normalizadas a `bps/hora` a través de **`214` perpetuos** (`2023–2026`).
  * **`Hyperliquid L1` Validator-Oracle & Carry (`hl_funding_all_v2.parquet`):** **`641,241` filas horarias on-chain** (`funding_rate_1h` + `hl_premium_bps` del oráculo de validadores) a través de **`87` monedas**.
  * Cruzado con `343.2M` velas `1m` lineales de `Binance USDS-M`, `premiumIndexKlines 1m`, `sumOpenInterest` `5m` en contratos (`+60s` lag causal) y instantáneas reales del libro `bookDepth` L2 (`±1% / ±2%`).
* **Total simulado bajo física exacta `1m` lineal (`zero np.log`):** `1,539` operaciones a través de `12` combinaciones (`4` mecanismos × `3` geometrías de monetización convexa) + validación **`CombinatorialPurgedCV` In-Fold (`N=6, k=2 -> 15` particiones purgadas)**.

### Veredicto Cuantitativo Estricto
* **Sobrevivientes de las 4 Compuertas Universales (`>= +25 bps` Full, `>= +15 bps` en `23-24` y `2025`, `>= +20 bps` en `2026`, `Fee Share <= 30%`, CPCV In-Fold `>= 80%`):** **`0 / 12` (`NO-GO` para promoción directa a Mainnet).**
* **Por qué este resultado negativo/selectivo es altamente valioso para el portafolio real:**
  1. **Sepulta definitivamente a `M25` incluso con filtro Cross-CEX (`P3_M1`):** Ni siquiera exigiendo divergencia genuina `Bybit/HL - Binance >= +3.0 bps/h` el *short* post-settlement en `T+1m` logra sobrevivir `2023–2025` (`-72.4 bps` y `-10.8 bps`, `CPCV = 33%`). Haber apagado `M25` en Tokio fue `100%` correcto.
  2. **Demuestra la superioridad física del Libro L2 + `aggTrades` `1-ms` (`M26`) frente a tasas lentas Cross-CEX (`P3_M2`):** El *short-squeeze* sincronizado en 3 exchanges (`P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG`) **sí es positivo en las 3 épocas** (`+12.0` en `23-24`, `+12.1` en `2025`, `+13.7` en `2026` -> **`+12.38 bps/trade`** en `MONO_B`), pero rinde **`5.7 veces menos`** que `M26` (`+71.15 bps/trade` Full, `+236.55 bps` en `2026`, `100%` CPCV) porque las tasas de *funding* de `1h/8h` entre exchanges son públicas y lentas, mientras que el vacío instantáneo del libro L2 (`±1%/±2%`) + barridas de `1 milisegundo` capturan el desequilibrio en el segundo exacto en que ocurre.
  3. **Falsifica el mito de que Hyperliquid lidera a Binance en rupturas (`P4_M3`):** Cuando Hyperliquid paga `+1.8 bps/h` más que Binance en una ruptura, comprar Binance pierde **`-49.5 a -62.0 bps/trade` (`0%` CPCV)** — es *FOMO* minorista on-chain que los *market makers* arbitran vendiendo.
  4. **Detecta un cambio estructural en `2026` para rebotes de liquidación anclados (`P4_M4`):** Perdió dinero en `2023–2024` (`-17.8 bps`) y `2025` (`-8.4 bps`) cuando Hyperliquid aún tenía poca profundidad, pero en **`2026` (`56` trades) saltó a `+70.87 bps/trade` netos (`73.3%` CPCV OOS `+49.58 bps`)**. Por disciplina anti-*recency bias* **no se despliega hoy**, pero queda en **Watchlist de Investigación** para combinarse en el futuro con el stream en vivo `!forceOrder@arr` + absorción a `1ms`.

---

## 2. Tabla Completa de Resultados (`1,539` Trades Lineales `1m`, `215` Perpetuos Cross-Venue)

Fuente reproducible: [p3_p4_cross_venue_report.md](../../reports/p3_p4_cross_venue_report.md), [p3_p4_cross_venue_summary.json](../../reports/p3_p4_cross_venue_summary.json), motor [cross_venue_p3_p4_lab.py](../../src/models/cross_venue_p3_p4_lab.py) e ingestor [cross_venue_p3_ingestor.py](../../src/data/cross_venue_p3_ingestor.py).

| Mecanismo | Geometría de Salida | `N` | Gross (`bps`) | Fnd (`bps`) | Fee (`bps`) | **Net Full (`bps`)** | Win `%` | `t-stat` | `23–24` (`bps`) | **`2025` (`bps`)** | **`2026` (`bps`)** | **CPCV OOS (`bps`)** | **CPCV `%`** | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG`** | `MONO_B_COUPON_VACUUM` | `156` | `+21.2` | `+3.8` | `12.7` | **`+12.38`** | `44.9%` | `+0.56` | `+12.0` | **`+12.1`** | **`+13.7`** | `+5.7` | `60%` | Positivo en 3 eras, pero débil vs `M26` (`+71.2`) |
| `P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG` | `MONO_A_CONVEX_TRAIL` | `156` | `+17.1` | `+4.0` | `12.9` | **`+8.31`** | `44.2%` | `+0.36` | `+5.9` | **`+7.8`** | **`+13.1`** | `+9.7` | `67%` | Dominado por `M26` + `M04` |
| `P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG` | `MONO_C_SPLIT_50_50` | `156` | `+13.1` | `+3.3` | `11.8` | **`+4.70`** | `54.5%` | `+0.27` | `+1.6` | **`+6.1`** | **`+5.8`** | `+10.8` | `67%` | Diluido por fees |
| **`P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG`** | `MONO_C_SPLIT_50_50` | `281` | `+14.4` | `+0.2` | `12.0` | **`+2.65`** | `49.8%` | `+0.21` | `-17.8` | **`-8.4`** | **`+70.9`** | **`+49.6`** | **`73%`** | **Watchlist 2026** (falla eras tempranas `23–25`) |
| `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG` | `MONO_A_CONVEX_TRAIL` | `281` | `+9.5` | `+0.2` | `12.9` | **`-3.17`** | `43.1%` | `-0.23` | `-20.9` | **`-22.2`** | **`+70.1`** | `+44.6` | `73%` | Régimen-dependiente (`2026` `+70.1 bps`) |
| `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG` | `MONO_B_COUPON_VACUUM` | `281` | `+4.6` | `+0.2` | `12.9` | **`-8.11`** | `43.1%` | `-0.61` | `-22.2` | **`-29.8`** | **`+59.7`** | `+40.6` | `67%` | Régimen-dependiente |
| **`P3_M1_CROSS_CEX_ARB_UNWIND_SHORT`** | `MONO_A_CONVEX_TRAIL` | `50` | `+6.8` | `+0.0` | `13.0` | **`-6.21`** | `44.0%` | `-0.24` | `-72.4` | **`-10.8`** | `+23.8` | `-61.4` | `33%` | **Rechazado (`M25` sigue OFF)** |
| `P3_M1_CROSS_CEX_ARB_UNWIND_SHORT` | `MONO_B_COUPON_VACUUM` | `50` | `+6.8` | `+0.0` | `13.0` | **`-6.21`** | `44.0%` | `-0.24` | `-72.4` | **`-10.8`** | `+23.8` | `-61.4` | `33%` | **Rechazado** |
| `P3_M1_CROSS_CEX_ARB_UNWIND_SHORT` | `MONO_C_SPLIT_50_50` | `50` | `-8.0` | `+0.0` | `12.5` | **`-20.52`** | `44.0%` | `-0.92` | `-73.4` | **`-18.2`** | `+1.3` | `-67.3` | `27%` | **Rechazado** |
| **`P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG`** | `MONO_C_SPLIT_50_50` | `26` | `-37.4` | `-0.1` | `12.1` | **`-49.51`** | `50.0%` | `-1.64` | `-48.2` | **`-56.9`** | `0.0` | `-56.0` | `0%` | **Falsificado (`0%` CPCV)** |
| `P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG` | `MONO_B_COUPON_VACUUM` | `26` | `-45.3` | `-0.2` | `13.0` | **`-58.49`** | `46.2%` | `-1.84` | `-68.4` | **`-4.1`** | `0.0` | `-62.7` | `0%` | **Falsificado (`0%` CPCV)** |
| `P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG` | `MONO_A_CONVEX_TRAIL` | `26` | `-48.8` | `-0.2` | `13.0` | **`-61.99`** | `46.2%` | `-1.98` | `-72.5` | **`-4.1`** | `0.0` | `-67.7` | `0%` | **Falsificado (`0%` CPCV)** |

---

## 3. Diagnóstico Físico Detallado Mecanismo por Mecanismo

### 3.1 `P3_M1` (*Cross-CEX Harvester Unwind Short*): Por qué `M25` queda enterrado definitivamente
* **Hipótesis probada:** Quizás `M25` (`Short` en `T+1m` después de `FR <= -20 bps` en Binance) fallaba porque mezclaba escasez global con arbitraje local, y filtrando solo cuando `Bybit` o `Hyperliquid` estuvieran `>= +3.0 bps/h` más arriba aislaríamos el desarme limpio de los *harvesters*.
* **Qué ocurrió en los datos (`50` eventos puros):** Aunque en `2026` dio `+23.8 bps` (`28` trades), en `2023–2024` perdió **`-72.4 bps`** y en `2025` perdió **`-10.8 bps`**, dejando el neto global en **`-6.21 bps/trade` (`CPCV = 33%`)**. Incluso cuando hay divergencia entre exchanges, los *shorts* post-liquidación quedan expuestos a colas violentas alcistas que destruyen la esperanza matemática.

### 3.2 `P3_M2` (*Global 3-Venue Short-Trap Squeeze Long*): Consistente pero `5.7x` inferior a `M26`
* **Hipótesis probada:** Cuando `Binance` y `Bybit/Hyperliquid` tienen *carry* horario negativo al mismo tiempo (`<= -0.8 bps/h`), hay descuento `1m` (`<= -15 bps`), `OI >= 1.10x` y ruptura de `4h`.
* **Qué ocurrió en los datos (`156` trades):** Fue el **único mecanismo P3/P4 positivo en las 3 épocas** (`+12.0` / `+12.1` / `+13.7 bps`, neto `+12.38 bps`), confirmando que la escasez global multi-exchange es real. Sin embargo, su magnitud neta (`+12.4 bps`, `t = 0.56`) está muy lejos de nuestro umbral de promoción (`>= +25 bps`, `CPCV >= 80%`) y ya está cubierta con mucha mejor relación señal/ruido por **`M04` (`15m NegFundSqueeze`)** + **`M26` (`L2 Ask Air-Pocket + 1ms Sweep`, `+71.15 bps/trade`)**.

### 3.3 `P4_M3` (*Hyperliquid On-Chain Lead Long*): Binance lidera el descubrimiento de precios, no al revés
* **Hipótesis probada:** Que un *spike* alcista de *funding* (`>= +1.8 bps/h`) o prima del oráculo (`>= +10 bps`) en `Hyperliquid L1` anticipara una ruptura alcista en `Binance`.
* **Qué ocurrió en los datos (`26` trades):** Perdió en todas las épocas (`-49.5` a `-62.0 bps/trade`, **`0%` de particiones CPCV positivas**, y `0` ocurrencias en `2026`). En la microestructura real, cuando el DEX perpetuo se recalienta mientras `Binance` (el mercado institucional más profundo) permanece frío, la convergencia ocurre porque **Hyperliquid cae hacia Binance**, no porque Binance suba hacia Hyperliquid.

### 3.4 `P4_M4` (*Cross-Venue Solvency-Anchored Liquidation Snap-Back*): El hallazgo estructural de `2026`
* **Hipótesis probada:** Comprar cascadas de liquidación de `15m` en `Binance` (`ΔContratos <= -1.5%`, mecha de descuento `<= -35 bps`, primer minuto de giro comprador) únicamente cuando `Hyperliquid L1` y/o `Bybit` confirman que el activo sigue solvente afuera.
* **Qué ocurrió en los datos (`281` trades):**
  * En `2023–2024` (`-17.8 bps`) y `2025` (`-8.4 bps`), la liquidez externa era demasiado baja para servir de ancla fiable.
  * En **`2026` (`56` trades)** — con `Hyperliquid` y `Bybit` ya consolidados como mercados profundos — el mecanismo saltó a **`+70.87 bps/trade` netos (`MONO_C`)** y **`+70.11 bps/trade` (`MONO_A`)**, con `73.3%` (`11/15`) splits CPCV positivos.
* **Decisión:** No se habilita en vivo hoy porque viola nuestra regla innegociable de supervivencia en las 3 eras históricas (`23–24`, `2025`, `2026`). Si queremos explotar este fenómeno en una iteración futura, la vía correcta es combinarlo con el stream directo de liquidaciones **`!forceOrder@arr` + absorción `aggTrades` `1-ms`**, en lugar de inferir la liquidación solo desde velas de `1m` y `OI` de `5m`.

---

## 4. Síntesis Global del Ciclo de Microestructura (Fases Lab 1 -> P1 -> P2 -> P3/P4)

| Fase | Qué se probó | Ganador / Resultado Clave | Acción tomada en `crypto-mainnet` (Tokio) |
| :--- | :--- | :--- | :--- |
| **Lab 1** (`27,103` trades `1m`) | Auditoría lineal `1m` de `M25` y estrangulamiento de salidas | `M25` pierde `-10.5 bps/trade`; `Profit-Lock` corto ahoga el `45%` de los ganadores | **`M25` APAGADO (`enable_post_settle_sniper=false`)** |
| **Fase P1** (`343.2M` filas + L2 Book) | Asimetría real del Libro `L2` (`±1%/±2%`) + Salida Convexa | `P6A_L2_ASK_AIR_POCKET` (`+60.5 bps/trade`, `100%` CPCV) | Base del motor L2 en [multimodal_feeds.py](../../src/execution/multimodal_feeds.py) |
| **Fase P2** (`aggTrades 1ms` + Portafolio `1m`) | Anti-spoofing `1-ms` + Ablación One-Way en cartera completa (`$2,700`, piso `$2,500`) | **`P6A_L2_3_INST_SWEEP_CONC` (`+71.15 bps` Full, `+236.6 bps` en `2026`, `4Y Sharpe 1.99 -> 2.15`)** | **`Sniper:M26_L2SweepAirPocket` DESPLEGADO Y ACTIVO EN VIVO** |
| **Fase P3/P4** (`215` perps `Bybit + HL`) | Divergencias Cross-CEX y liderazgo/anclaje On-Chain | `0` sobrevivientes de 4 compuertas (`P3_M1` y `P4_M3` falsificados; `P3_M2` `+12.4 bps` inferior a `M26`; `P4_M4` `+70.9 bps` solo en `2026`) | **Sin cambios al motor vivo — cartera limpia y sin dilución de capital** |
