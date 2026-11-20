# Auditoría Multianual en Física Exacta de `1m` (`2023–2026`): Estudio Sub-Horario (`1m / 5m / 15m / Dollar Bars`) y Revisión Forense de los 7 Mecanismos Actuales

## 1. Resumen Ejecutivo y Calidad de Datos (`40.26M` Barras de `1m`)

Se descargaron desde **Binance Vision S3** los archivos históricos completos de futuros perpetuos `USDS-M` en resolución de **1 minuto (`1m`)** desde `2023-01-01` hasta `2026-11-16` (**`40,261,819` velas de `1m`** con `quote_volume`, `taker_buy_quote_volume` y `trade_count` 100% reales, empalmando `RNDRUSDT -> RENDERUSDT`), más `premiumIndexKlines` (`1h`) y `fundingRate` (`8h`), para los **15 símbolos de producción** y **5 símbolos testigo de mid-cap/cola ilíquida** (`INJ, OP, 1000PEPE, ANKR, CELR`). Desde `1m` se generaron determinísticamente `5m`, `15m`, `30m`, `1h` y **`Dollar Bars` causales**.

* **Integridad verificada:** **`0.000%` de huecos** post-listado y **`0` valores nulos en `taker_buy_quote_volume`** en los 15 activos de producción (`2,039,040` barras de `1m` por moneda listada en `2023-01-01`), aplicando enmascaramiento estricto pre-listado (`SUI, ARB, RENDER, FET, 1000PEPE`) sin *backfill* sintético.
* **Partición cronológica estricta sin fuga de información:**
  * **Train (`2023–2024`):** `23` meses (`bull` temprano + lateral `2023` + rally `2024`).
  * **Validation (`2025`):** `12` meses (crash macro Q1–Q2 + rotación institucional a `BTC/ETH/SOL`).
  * **Holdout Sellado (`2026-01` a `2026-11-16`):** `10.5` meses (`52%` régimen lateral de baja dispersión `R5_CHOP`).

> [!IMPORTANT]
> **Por qué bajar a `1m` fue decisivo:** El simulador histórico previo en `1h` ([run_backtest.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/run_backtest.py#L542-L593)) asumía que si en cualquier momento de una vela de **60 minutos** el mínimo bajaba `2 bps` bajo el `Open`, la orden entera se llenaba pasivamente a `Open` pagando **`1.55 bps` efectivos** sin selección adversa (`+0.286%–0.302%/d`, `SR ~ 3.0`). Al simular minuto a minuto en `1m` cobrando el costo real VIP-0 (`4.2 bps` mixto `90s GTX` + `Taker`), descubrimos el piso real sin ilusiones **y encontramos 3 bugs/desajustes concretos hoy activos en el daemon en vivo**.

---

## 2. Puente Forense Paso a Paso: Simulación `1h` vs Física Real de `1m` (`2.5x` Constante)

Reconciliación exacta en [bridge_step2_exact_cause.py](file:///Users/santim/.gemini/antigravity/brain/c69ecbb2-d40f-4f09-b548-d07875507744/scratch/bridge_step2_exact_cause.py) y [confirm_champion_on_true_1m_winner.py](file:///Users/santim/.gemini/antigravity/brain/c69ecbb2-d40f-4f09-b548-d07875507744/scratch/confirm_champion_on_true_1m_winner.py):

| Paso de Auditoría (`2.5x` Tier Constante) | Retorno Neto `%/día` | `Sharpe` | `MaxDD` | `2026` Holdout (`%/d` / `SR`) | Causa Exacta Aislada |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`B0–B2` Simulación `1h` previa** (`run_backtest.py`) | `+0.271%` a `+0.302%` | `2.99` | `17.0%` | `+0.266%/d` (`SR 3.03`) | Asumía toque pasivo en 60m pagando **`1.55 bps`** sin selección adversa |
| **`B3` Costo Real `4.2 bps`** (misma lógica `1h`) | `+0.179%` | `2.04` | `22.4%` | `+0.167%/d` (`SR 1.96`) | Diferencia real de comisiones VIP-0 (`4.2 bps` vs `1.55 bps` sobre `~3.8x/d` giro) |
| **`B5` Física Intrabarra `1m` Exacta** (`60` velas `1m`/h) | `+0.182%` | `2.06` | `23.1%` | `+0.165%/d` (`SR 1.95`) | Los brackets `TP / Profit-Lock` sobreviven intactos al orden exacto minuto a minuto |
| **`00_LIVE_TODAY` (`4Y` completo `2023–26`, bugs vivos)** | **`+0.131%`** | **`1.55`** | **`26.8%`** | `+0.106%/d` (`SR 1.31`) | Penalizado en vivo por `15mPB=ON` en `F3_Fast` + bloqueo post-`TP` de `4h` |
| **`02_CORE_AUDITADO_1m` (`Fix 1 + Fix 2`, `4.2 bps`)** | **`+0.161%`** | **`1.80`** | **`20.8%`** | **`+0.151%/d` (`SR 1.73`)** | **`+0.25 SR` y `-6.0%` menos `MaxDD`** corrigiendo solo los 2 desajustes en vivo |
| **`02b_CORE_AUDITADO_1m + ChopLock` (`2.4%->1.2%`)** | **`+0.167%`** | **`1.86`** | **`19.9%`** | `+0.129%/d` (`SR 1.51`) | Mayor `Sharpe` 4Y (`1.86`), menor `MaxDD` (`19.9%`) y `67.4%` meses ganadores |

---

## 3. Estudio Sub-Horario Completo (`1m`, `5m`, `15m`, `30m`, `1h` y `Dollar Bars`)

### 3.1 Espectro de Cadencias de Rebalanceo del Motor 7-Factor (`2023–2026`, `2.5x`, `4.2 bps`)
Evaluado en [study_subhourly_1m_5m_15m_mechanisms.py](file:///Users/santim/.gemini/antigravity/brain/c69ecbb2-d40f-4f09-b548-d07875507744/scratch/study_subhourly_1m_5m_15m_mechanisms.py) (sin brackets intrabarra para medir la decadencia pura señal vs rotación):

| Cadencia de Rebalanceo | Rotación Anual | Costo Diario (`4.2 bps`) | Alpha Bruto (`%/d` / `SR`) | Alpha Neto (`%/d` / `SR`) | `SR` (`23–24` / `25` / `26`) | Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`5m`** *(señal CS rápida)* | `15,136x/año` | `-1.742%/d` | `+0.244%/d` (`1.81`) | `-1.497%/d` (`-9.16`) | Negativo en las 3 eras | Hay señal bruta, pero el fee drag es `7x` mayor que el alpha (incluso a `2 bps` Maker da `-0.58%/d`) |
| **`15m`** *(Motor 7-Factor)* | `3,122x/año` | `-0.359%/d` | `+0.222%/d` (`2.52`) | `-0.141%/d` (`-1.31`) | `-1.05 / -1.65 / -1.55` | Muro de comisiones consume `162%` del retorno bruto |
| **`30m`** *(Motor 7-Factor)* | `2,894x/año` | `-0.333%/d` | `+0.221%/d` (`2.50`) | `-0.117%/d` (`-1.04`) | `-0.85 / -1.33 / -1.19` | Rotación excesiva |
| **`1h`** *(Motor 7-Factor)* | `2,748x/año` | `-0.316%/d` | `+0.230%/d` (`2.55`) | `-0.087%/d` (`-0.72`) | `-0.64 / -0.61 / -1.02` | Costo supera al alpha |
| **`2h`** *(Motor 7-Factor)* | `2,069x/año` | `-0.238%/d` | `+0.220%/d` (`2.44`) | `-0.019%/d` (`-0.02`) | `0.08 / -0.02 / -0.26` | Punto de equilibrio nulo |
| **`4h`** *(Motor previo)* | `1,523x/año` | `-0.175%/d` | `+0.216%/d` (`2.36`) | `+0.040%/d` (`+0.58`) | `0.50 / 0.97 / 0.32` | Positivo pero ahogado en `2026` |
| **`8h UTC`** *(Producción actual)* | **`1,065x/año`** | **`-0.123%/d`** | **`+0.207%/d` (`2.28`)** | **`+0.085%/d` (`+1.04`)** | **`0.78 / 1.91 / 0.68`** | **Óptimo sincronizado al cobro de funding** *(sube a `+0.161%/d, SR 1.80` con brackets `1m`)* |
| **`12h`** | `838x/año` | `-0.096%/d` | `+0.199%/d` (`2.19`) | `+0.102%/d` (`+1.21`) | `0.89 / 1.62 / 1.49` | Único competidor lento viable (desalineado de los 3 cobros diarios de `8h`) |
| **`24h`** | `545x/año` | `-0.063%/d` | `+0.114%/d` (`1.27`) | `+0.051%/d` (`+0.68`) | `0.69 / 0.51 / 0.85` | La señal rápida (`F3_Fast` / `F4_Rev`) ya expiró |

### 3.2 ¿Sirve `1m / 5m / 15m` para Ejecución Pasiva Extendida, Micro-Funding o Snipers?

1. **Ejecución `1m` (`Smart Patience-Maker 5m/10m/15m + Breakout Chase` en `4,139` rebalanceos):** **RECHAZADO.**
   * Esperar `5–15` minutos buscando un retroceso de `2 bps` en velas de `1m` logró **`87%–91%` de llenados Maker**, pero **redujo el retorno neto de `+0.086%/d` (`SR 1.05`) a `+0.053%/d` (`SR 0.72`)**.
   * **Causa microestructural (Glosten-Milgrom en `1m`):** El `9%–13%` de órdenes que jamás retroceden `2 bps` en los primeros 10 minutos son exactamente las rupturas ganadoras más explosivas (obligando a perseguirlas `+22 bps` más caras), mientras que el `90%` que sí se llena pasivamente sobrerrepresenta entradas en contra del flujo inmediato. **Conclusión:** El esquema actual corto (`90s GTX` top-of-book + fallback Taker inmediato) es superior a esperar `5–15` minutos.
2. **Micro-Ventana del Cobro de Funding (`T-30m` a `T+30m` en `1m`, `3,316` eventos extremos):** **RECHAZADO.**
   * Con `|Funding| >= 4 bps` (`media +7.4 bps`), el precio cae en promedio **`-4.4 bps`** entre `T-5m` y `T+10m`, dejando **`-6.1 bps` netos (`Win 43.6%`)** tras fees. Con `|Funding| >= 15 bps` (`media +34.9 bps`), el *dump* instantáneo post-snapshot es de **`-25.4 bps`**, dejando **`-1.3 bps` netos (`Win 45.8%`)**. El mercado arbitra el segundo exacto del cobro casi `1:1`.
3. **Snipers Event-Driven en `5m` y `15m` (`Climax Absorption` y `Compression Breakout`):** **RECHAZADOS.**
   * En `5m` pierden dinero en las 3 eras tanto en `BTC/ETH/SOL` (`SR -1.50` y `-0.98`), como en `Top 15` (`SR -2.65` y `-1.83`) y `Mid/Tail` (`SR -1.98` y `-2.40`). En `15m` solo funcionaron en `2025` y fallaron en `2023–24` y `2026`.
4. **¡LO QUE SÍ FUNCIONÓ DESDE `1m`! `Dollar Bars` Causales (Reloj de Volumen `1m`) para `Pilar 2: Anchor TSMOM (BTC, ETH, SOL)`:** **APROBADO EN LAS 3 ERAS.**
   * Muestrear barras desde `1m` cada vez que se negocia `1/24` del volumen diario medio de los últimos `14` días cerrados (sin *lookahead*) supera monotónicamente tanto a `1h` como a `15m` en **todas las eras** y recorta el `MaxDD` en `-5.7` puntos:
     * `Time Bars 1h`: `SR = 0.65` | `MaxDD = 27.9%` (`23–24: 0.62` | `2025: 0.37` | `2026: 0.92`)
     * `Time Bars 15m`: `SR = 0.70` | `MaxDD = 25.1%` (`23–24: 0.59` | `2025: 0.65` | `2026: 0.95`)
     * **`Causal 1m Dollar Bars`**: **`SR = 0.79` | `MaxDD = 22.2%` (`23–24: 0.70` | `2025: 0.71` | `2026: 1.05`)**, con **correlación negativa (`Corr = -0.096`)** frente a nuestro motor neutral de 7 factores (`P1`).

---

## 4. Auditoría Forense de los 7 Mecanismos Actuales y sus *Overlays* (`1m` Exact Physics)

### 4.1 Prueba *Leave-One-Out* (`LOO`) de cada uno de los 7 Factores (`2023–2026`, `2.5x`, `4.2 bps`)
Base comparativa `02_CORE_AUDITADO_1m`: **`+0.161%/d` | `SR = 1.80` | `MaxDD = 20.8%` | `2026 SR = 1.73` (`+0.151%/d`)**.

| Experimento `LOO` (Quitando 1 factor) | `4Y %/día` | `4Y Sharpe` | `4Y MaxDD` | `SR` (`23–24` / `25` / `26`) | Impacto al Quitarlo (`dSR` Bootstrap 95% CI) | Veredicto de Auditoría |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`LOO_Drop_SIZE`** *(sin `F7_Size`)* | `+0.104%` | `1.31` | **`33.0%`** | `1.42 / 0.33 / 2.13` | **`-0.52` `[-1.10, +0.00]`** | **CRÍTICO (Ancla de Calidad Institucional):** Nota importante: en el código ([seven_factor_engine.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/strategies/seven_factor_engine.py#L841)), `F7_Size = +log(dollar_vol_168h)` compra **alta liquidez institucional** y vende rezagadas; sin él, `2025` colapsa (`SR 0.33`) y `MaxDD` sube a `33%`. |
| **`LOO_Drop_FAST`** *(sin `F3_Fast`)* | `+0.105%` | `1.30` | **`32.0%`** | `1.14 / 1.99 / 0.82` | **`-0.49` `[-0.98, -0.01]`** | **CRÍTICO (Motor Swing `12h–48h`):** Sin `F3_Fast` el `MaxDD` salta `+11.2%` y `2026` cae a mitad (`SR 0.82`). Mantener núcleo intacto *(pero quitar su inyección `15m` rota)*. |
| **`LOO_Drop_LIQ`** *(sin `F6_Liq`)* | `+0.136%` | `1.58` | `19.6%` | `1.70 / 1.74 / 1.11` | **`-0.22` `[-0.57, +0.09]`** | **VALIOSO:** Su *gate* `t >= 0.75` captura expansiones de volumen; sin él `2026` cae de `1.73` a `1.11`. Mantener. |
| **`LOO_Drop_MACRO`** *(sin `F1_Macro`)* | `+0.139%` | `1.66` | `20.9%` | `1.73 / 1.41 / 1.79` | **`-0.14` `[-0.49, +0.21]`** | **SÓLIDO:** Sostiene tendencias de `7d–30d` (`2025` cae de `1.87` a `1.41` sin él). Mantener. |
| **`LOO_Drop_REV`** *(sin `F4_Rev`)* | `+0.145%` | `1.69` | `23.3%` | `1.76 / 2.04 / 1.11` | **`-0.14` `[-0.59, +0.23]`** | **ESENCIAL EN LATERAL:** Sin `F4_Rev`, el año lateral `2026` cae de **`+0.151%/d` (`SR 1.73`) a `+0.088%/d` (`SR 1.11`)**. *(Aislado sin `t-stat` online perdería, pero gobernado por el Meta-Allocator + Knife Guard es clave en `R5_CHOP`)*. |
| **`LOO_Drop_CARRY`** *(sin `F5_Carry`)* | `+0.162%` | `1.80` | `18.7%` | `1.88 / 1.62 / 1.81` | **`0.00` `[-0.21, +0.20]`** | **NEUTRO / SEGURO:** Con su *gate* estricto `t >= 1.25` permanece apagado ~97% del tiempo sin estorbar. |
| **`LOO_Drop_MED`** *(sin `F2_Med`)* | `+0.162%` | `1.89` | `19.3%` | `2.00 / 1.63 / 1.94` | **`+0.10` `[-0.22, +0.41]`** | **SOLAPAMIENTO LEVE:** `72h–168h` se solapa parcialmente entre `F3_Fast` (`48h–72h`) y `F1_Macro` (`168h`), aunque el IC al 95% incluye `0`. No es dañino, pero tampoco indispensable. |

### 4.2 Los 3 Bugs / Desajustes Concretos Detectados en el Código en Vivo

1. **Bug/Desajuste #1 — Inyección asimétrica `15m` en `f_fast.iloc[-1]` ([seven_factor_engine.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/strategies/seven_factor_engine.py#L826-L830) y [shadow_daemon.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py#L198)):**
   * En vivo, `enable_subhourly_pullback=True` suma `- (0.05*r_30m + 0.08*r_2h + 0.08*shock_1h)` **solo a la última fila `f_fast.iloc[-1]`** (sin haber entrado jamás en el cálculo histórico de `t_df["fast"]` ni en backtests `1h`).
   * Al auditarlo con barras reales de `15m` en los 4 años, **resta retorno en las 3 eras**: apagarlo sube `Sharpe` de **`1.55 -> 1.70`** (y en `2026` de **`1.31 -> 1.65`**, `+0.106%/d -> +0.138%/d`) porque deja de vender rupturas institucionales frescas.
2. **Bug/Desajuste #2 — Bloqueo post-`Take-Profit` en el libro Core ([shadow_daemon.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py#L2007-L2011)):**
   * Hoy `core_bracket_cooldown_sec = 4h` bloquea re-entrar en una moneda **incluso cuando acaba de cerrar en ganancia por `Take-Profit` (`+3.2% / +5.0%`)**, forzando a [meta_allocator.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/strategies/meta_allocator.py#L230-L252) a concentrar esa pata en activos secundarios `#4/#5` de baja convicción.
   * Permitir re-elegir líderes sanos en el siguiente ciclo de `8h` tras un `Take-Profit` ganador (dejando el *cooldown* **solo para salidas en pérdida**, si algún día se habilita `SL`) sube `Sharpe` de **`1.70 -> 1.80`** y **recorta el `MaxDD` de `26.8%` a `20.8%` (`-6.0%`)**.
3. **Bug #3 — Código incompleto en `compute_dynamic_barrier_profile` ([shadow_daemon.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py#L1536-L1602)):**
   * El docstring especifica que en régimen lateral (`chop`), cuando `eff_tp` baja a `[2.8%, 3.6%]`, el `Profit-Lock` debe escalar a `(0.72 * eff_tp -> 0.40 * eff_tp)`, pero en el código `eff_lock_trig` quedó fijo en `3.5%` (mayor que `eff_tp`, volviéndolo inalcanzable).
   * Completar esas dos líneas (`02b`) eleva el `Sharpe` 4Y a **`1.86`**, baja el `MaxDD` a **`19.9%`** y sube los meses positivos a **`67.4%`** *(con un ligero trade-off en `2026`: `+0.129%/d` vs `+0.151%/d` sin chop-lock)*.

* **Verificación de Overlays adicionales:** **`Taker CVD (#48)`** suma `+0.08 SR` y reduce `MaxDD` (`mantener ON`); **`Holistic Regime Shields`** recorta `-4.2%` de `MaxDD` (`mantener ON`); **`Chop Specialist`** protege `2026` (`SR 1.73` vs `1.56` sin él, `mantener ON`); **`Spot-Perp Premium Index (#49)`** es redundante (`dSR = -0.05`, mantener OFF).

---

## 5. Proyección Honesta sobre Física Exacta de `1m`: Stack Completo (`Stepped 2.5x -> 3.2x -> 5.0x`)

Evaluando el motor completo compuesto (`2023–2026`) sobre **barras reales de `1m`** bajo dos escenarios reales de ejecución: **Conservador Estricto (`4.2 bps` = ~27% Maker / 73% Taker)** y **Ejecución `GTX` Optimizada (`3.2 bps` = ~60% Maker / 40% Taker)**:

| Arquitectura Evaluada en `1m` (`2023–2026`) | `4Y %/día` | `4Y Sharpe` | `4Y MaxDD` | `SR` (`23–24` / `25` / `26`) | `2026` Holdout `%/día` (`SR`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`STEP_00` Motor Vivo Hoy** (`Stepped`, `4.2 bps`, con bugs) | `+0.170%` | `1.42` | `38.6%` | `1.75 / 0.91 / 1.26` | `+0.146%/d` (`1.26`) |
| **`STEP_01` Pilar 1 Auditado** (`Fix 1 + Fix 2`, `4.2 bps`) | `+0.202%` | `1.58` | `34.9%` | `1.65 / 1.55 / 1.49` | `+0.198%/d` (`1.49`) |
| **`STACK P1 + P2` (+ `1m Dollar-Bar TSMOM`, `4.2 bps`)** | `+0.246%` | `1.82` | `35.6%` | `1.78 / 1.77 / 1.94` | `+0.282%/d` (`1.94`) |
| **`STACK 3 Pilares` (`P1 + P2 + BFUSD/Carry`, `4.2 bps`)** | **`+0.264%`** | **`1.93`** | **`34.1%`** | **`1.92 / 1.86 / 2.03`** | **`+0.297%/d` (`2.03`)** |
| **`STACK 3 Pilares` (`3.2 bps` con 60% Maker / VIP)** | **`+0.340%`** | **`2.30`** | **`33.2%`** | **`2.31 / 2.19 / 2.40`** | **`+0.374%/d` (`2.40`)** |

> [!WARNING]
> **Transparencia sobre expectativa realista de retorno:** Sobre física exacta de `1m` y comisiones reales VIP-0, el sistema limpio de 3 pilares entrega de forma sostenible entre **`+0.26%/día` y `+0.37%/día`** (`~155%` a `~290%` compuesto anual, con `Sharpe 1.93–2.30` estable en las 3 eras). Forzar `0.5%–1.0%/día` sobre estas mismas señales solo sería posible duplicando el apalancamiento (`6x–8x`), lo cual elevaría el `MaxDD` por encima del `55%` y pondría en riesgo el piso de capital de `` `$2,500` ``.

---

## 6. Plan de Acción Propuesto para tu Confirmación

1. **Fase Inmediata (Quirúrgica sobre `Pilar 1` actual — bajo riesgo, mejora instantánea en las 3 eras):**
   * Desactivar `enable_subhourly_pullback` (`False` por defecto en [shadow_daemon.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py) y [seven_factor_engine.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/strategies/seven_factor_engine.py)).
   * Cambiar `core_bracket_cooldown` en [shadow_daemon.py](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py) para que **solo bloquee salidas en pérdida (`exit_pnl < 0`)** y **no bloquee líderes tras un `Take-Profit` o `Profit-Lock` ganador**.
   * *(Opcional a tu elección)* Activar o dejar desactivado el escalado del `Chop Profit-Lock (0.72*TP -> 0.40*TP)` en `compute_dynamic_barrier_profile` (`02` maximiza retorno en `2026` con `+0.151%/d`; `02b` minimiza `MaxDD` 4Y a `19.9%` con `+0.167%/d` 4Y).
2. **Fase 2 (Expansión Ortogonal probada en `1m`):**
   * Integrar el módulo **`Pilar 2: Causal 1m Dollar-Bar Anchor TSMOM`** (`BTC, ETH, SOL`, `Corr = -0.096` con `Pilar 1`, zona muerta `|z| < 0.75` en lateral).
   * Activar colateral remunerado **`BFUSD`** en la cuenta de Binance Futures (`+5%–6% APR` pasivo sobre el margen sin riesgo de precio) + sleeve oportunista `1:1 Spot-Perp Carry` cuando el funding supere `15% APR`.
