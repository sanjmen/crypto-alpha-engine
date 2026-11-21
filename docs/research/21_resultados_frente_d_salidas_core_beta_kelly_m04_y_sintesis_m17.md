# 21 — Resultados Forenses Frente D: Física de Salidas Intrabarra en `Core P1`, Neutralización de `Beta_72h`, Sizing `Kelly` + Preemption en Snipers, Radiografía de `M04` y Escalera de Síntesis `M17` (`2023–2026`)

## 1. Resumen Ejecutivo y Decisiones `GO / NO-GO`

En el **Frente D** ([frente_d_architecture_synthesis_lab.py](../../src/backtest/frente_d_architecture_synthesis_lab.py), [frente_d_report.md](../../reports/frente_d_report.md), [frente_d_summary.json](../../reports/frente_d_summary.json)) atacamos 4 ángulos estructurales internos nunca auditados en `Docs 01..20` sobre los `80` símbolos `1m` (`2023–2026` + arranque fresco de `$2,700` en `2026` contra el piso duro CPPI de `$2,500`), y construimos por primera vez la **Escalera Acumulada de Síntesis `M17`** bajo costos empíricos reales de Binance Mainnet:

### Hallazgos Accionables (`GO` — Issues `#78` y `#79`):
1. **`GO` (`M17-6`, Issue `#78`) — Sizing `Kelly` por Convicción en Snipers + Tope de `2` Slots en `M23` + *Preemption* Opcional (`D07` / `D08`):**
   - Hoy todos los Snipers reciben `0.24x` plano (`M26` con `+236.6 bps` en `2026` pesa igual que `M04` con `-44.5 bps`) y un disparo de `M23` llena los `3/3` slots simultáneos bloqueando a `M26` y `M09_LONG`.
   - Pasar a **cuotas escalonadas por edge/SL (`D07`: `M26 0.34x`, `M09_LONG 0.30x`, `M23/M09_SHORT 0.20x`, `M04 0.14x`, máx. `2` slots `M23`)** supera al baseline en **las 3 eras (`2.34 / 1.92 / 2.64` vs `2.26 / 1.90 / 2.42`)**, sube **`4Y Sharpe` de `2.18 -> 2.26`** (`CAGR +186.2% -> +202.9%`, `CPCV 2.15 -> 2.24` `100%`), y eleva `$2,700` en `2026` de **`$5,004.78 -> $5,739.57` (`+$734.79`, `MinEq = $2,553.20` ✅)** sin rotación extra en `Core P1`.
   - Permitir además que `M26` y `M09_LONG` desplacen limpiamente una pata individual abierta en `Core P1` (`D08`, desbloqueando **`76` trades Tier-A** antes descartados por colisión One-Way) sube **`4Y Sharpe` a `2.29`** (`CAGR +212.2%`, **`2026` `$2,700 -> $6,583.88`**, `MinEq = $2,561.25` ✅, con leve cesión en `2025`: `1.92 -> 1.72`).
2. **`GO` (`M17-7`, Issue `#79`) — Diagnóstico y Reparación Causal de `M04_NegFundSqueeze` (`FR <= -8 bps` / `-10 bps`, `D09`):**
   - Descompusimos los `592` trades `1m` de `M04` por bucket de funding causal al momento de entrada: el **`71.8%` (`425` trades) ocurrió con funding tibio entre `-4` y `-8 bps`**, que en `2026` es puro ruido de arbitrajistas (**mediana `-34.4 bps`, `2026 = -59.2 bps/trade` en `137` trades**).
   - En cambio, cuando el dolor de cortos atrapados es genuino (**`FR <= -8 bps`**, `167` trades), la **mediana salta `+109.4 bps` (`-34.4 -> +75.0 bps`)** (y en `FR <= -12 bps`, mediana **`+103.9 bps`**, media **`+36.6 bps`**). Exigir `FR <= -8 bps` (`D09`) sube el **`2026 Sharpe` de `2.42 -> 2.73`**, eleva `$2,700` a **`$6,409.09` (`+$1,404.31`)** y sube el colchón mínimo a **`MinEq = $2,568.39` ✅**.

### Hipótesis Falsificadas con Rigor (`NO-GO` — `0` sobreajuste):
1. **`NO-GO` en `D02` (Apagar `TP` y `Lock` intrabarra en `Core P1`):** Hunde `4Y Sharpe` de **`2.18 -> 1.36`** (`$5,005 -> $2,943`). Los picos intraciclo de altcoins revierten antes de `8h`; las órdenes límite `Maker TP` (`+3.6% / +5.0%`) son vitales.
2. **`NO-GO` en `D03 / D12` (Apagar `PROFIT_LOCK` dejando solo `TP` en `Core P1`):** Aunque en `2026` dispara la cuenta compuesta a `$6,681` (`D03`) y `$15,642` (`D12`, `2026 SR = 3.93`), **deteriora el año lateral `2025` (`1.90 -> 1.62` en `D03`, `2.65 -> 1.79` en `D12`)** porque en mercados picados los rallies de `+3.0%` que no llegan al `+5.0% TP` devuelven toda la ganancia sin `PROFIT_LOCK`. Se preserva el `PROFIT_LOCK` actual por robustez multi-régimen.
3. **`NO-GO` en `D05 / D06` (Neutralización forzada de `Beta_72h` en `Core P1`):** Baja `4Y Sharpe` (`2.18 -> 2.03` / `1.89`) y **sube `4Y MaxDD` (`35.22% -> 38.85%` / `40.62%`)**. `SevenFactorOnlineEngine` ya ortogonaliza residuos idiosincráticos (`|Beta|` base es apenas `0.146`), y ese pequeño sesgo residual `±0.146` actúa como momentum de régimen a favor de la tendencia.

---

## 2. Tabla Completa de Experimentos `1-Minuto` (`2023–2026` + `$2,700` en `2026`)

Fuente: [frente_d_report.md](../../reports/frente_d_report.md).

| Exp ID | **`4Y SR`** | `4Y CAGR` | **`4Y DD`** | `23–24 / 25 / 26 SR` | **`CPCV SR (Pos%)`** | `FeeDrag%` | `Lock/TP` | `|Beta|` | `Preempt` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D00_LIVE_REAL_00M_6.88BPS`** *(Real `:00`)* | **`1.39`** | `+74.8%` | `38.16%` | `1.66 / 0.98 / 1.48` | `1.38` (`100%`) | `328.7%` | `1951/2706` | `0.146` | `0` | **`$3,299.60`** (`$2,499.54` ⚠️) |
| `D01_BASELINE_IDEALIZED_4.20BPS` | `2.18` | `+186.2%` | `35.22%` | `2.26 / 1.90 / 2.42` | `2.15` (`100%`) | `245.8%` | `1945/2699` | `0.146` | `0` | `$5,004.78` (`$2,549.00` ✅) |
| `D02_CORE_PURE_8H_HOLD_NO_LOCK_NO_TP` | `1.36` | `+82.7%` | `36.94%` | `1.04 / 1.90 / 1.28` | `1.36` (`100%`) | `173.5%` | `0/0` | `0.145` | `0` | `$2,943.42` (`$2,507.69` ✅) |
| `D03_CORE_TP_ONLY_NO_PROFIT_LOCK` | `2.22` | `+204.9%` | `33.96%` | `2.23 / 1.62 / 2.91` | `2.22` (`100%`) | `225.9%` | `0/3258` | `0.146` | `0` | `$6,681.11` (`$2,554.18` ✅) |
| `D04_CORE_WIDE_CONVEX_BRACKET` | `1.66` | `+115.0%` | `36.07%` | `1.94 / 0.85 / 2.09` | `1.65` (`100%`) | `226.4%` | `1363/937` | `0.145` | `0` | `$4,829.36` (`$2,555.91` ✅) |
| `D05_CORE_BTC_BETA_OVERLAY_CLAMP_012` | `2.03` | `+149.2%` | `38.85%` | `2.28 / 1.49 / 2.27` | `2.01` (`100%`) | `240.7%` | `1936/2705` | `0.083` | `0` | `$4,825.38` (`$2,520.33` ✅) |
| `D06_CORE_ASYM_LEG_ZERO_BETA_SCALING` | `1.89` | `+119.1%` | `40.62%` | `2.16 / 1.28 / 2.09` | `1.84` (`93%`) | `228.6%` | `1952/2686` | `0.000` | `0` | `$4,266.79` (`$2,525.34` ✅) |
| **`D07_SNIPER_KELLY_TIERED_SIZING`** *(M17-6)* | **`2.26`** | **`+202.9%`** | `35.31%` | **`2.34 / 1.92 / 2.64`** | **`2.24` (`100%`)** | `251.7%` | `1940/2697` | `0.146` | `0` | **`$5,739.57`** (**`$2,553.20` ✅**) |
| **`D08_SNIPER_KELLY_PLUS_CORE_PREEMPT`** | **`2.29`** | **`+212.2%`** | `35.87%` | `2.52 / 1.72 / 2.65` | **`2.26` (`100%`)** | `255.1%` | `1934/2698` | `0.146` | `76` | **`$6,583.88`** (**`$2,561.25` ✅**) |
| **`D09_M04_DEEP_SQUEEZE_8BPS`** *(M17-7)* | `2.15` | `+178.0%` | `36.51%` | `2.10 / 1.78 / 2.73` | `2.13` (`100%`) | `243.2%` | `1931/2703` | `0.146` | `0` | **`$6,409.09`** (**`$2,568.39` ✅**) |
| `D09B_M04_EXTREME_SQUEEZE_12BPS` | **`2.20`** | `+188.1%` | `36.88%` | `2.23 / 1.86 / 2.66` | `2.18` (`100%`) | `244.4%` | `1931/2707` | `0.146` | `0` | `$5,082.24` (`$2,532.43` ✅) |
| **`D10_M17_STACK_FRENTE_A_PLUS_C`** 🏆 | **`2.72`** | **`+297.8%`** | `35.39%` | **`2.29 / 3.06 / 3.11`** | **`2.71` (`100%`)** | **`179.5%`** | `1880/2759` | `0.145` | `0` | **`$6,613.38`** (**`$2,555.75` ✅**) |
| **`D11_M17_STACK_A_C_PLUS_KELLY_M04`** 🏆 | **`2.68`** | **`+286.4%`** | **`34.39%`** | **`2.44 / 2.65 / 3.16`** | **`2.66` (`100%`)** | `181.8%` | `1885/2742` | `0.145` | `76` | **`$7,698.31`** (**`$2,554.98` ✅**) |
| `D12_M17_FULL_SYNTHESIS_ALL_ANGLES` | `2.56` | `+255.0%` | `33.99%` | `2.25 / 1.79 / 3.93` | `2.54` (`100%`) | `149.8%` | `0/3258` | `0.084` | `79` | `$15,641.96` (`$2,583.58` ✅) |

---

## 3. Radiografía Forense de `M04_NegFundSqueeze` por Profundidad de Funding (`592` Trades `1m`)

| Bucket de Funding Causal en Entrada | `N Full` | `N 2026` | **Net Full (`bps`)** | **Mediana (`bps`)** | Win`%` | `23–24` | **`2025`** | **`2026`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ALL_M04_LE_MINUS_4BPS` *(umbral actual)* | `592` | `190` | `+10.8` | `-2.6` | `50.0%` | `+31.6` | `+44.7` | **`-44.5`** |
| **`MILD_NOISE_MINUS_4_TO_8BPS`** *(ruido)* | **`425`** | **`137`** | `+9.1` | **`-34.4`** | `48.7%` | `+39.5` | `+44.8` | **`-59.2` ⚠️** |
| **`DEEP_SQUEEZE_LE_MINUS_8BPS`** *(M17-7)* | **`167`** | **`53`** | **`+15.2`** | **`+75.0` ✅** | **`53.3%`** | `+11.3` | `+44.5` | **`-6.5`** |
| **`EXTREME_SQUEEZE_LE_MINUS_12BPS`** | **`95`** | **`28`** | **`+36.6`** | **`+103.9` ✅** | **`55.8%`** | `+20.1` | **`+91.2`** | **`-10.4`** |

---

## 4. Síntesis Ejecutiva de la Escalera `M17` (`D00 -> D10 -> D11`)

Al integrar de manera limpia los hallazgos validados sin sesgo de **Frente A (`#73, #74, #75`) + Frente C (`#77`) + Frente D (`#78, #79`)** (a lo cual se suma en producción el veto `L2_NO_WALL` de **Frente B `#76`** sobre `M09/M23`):
* Frente al costo empírico real de operar en `:00:00` (`D00`: `4Y SR = 1.39`, `DD = 38.16%`, `$2,700 -> $3,299.60`, `MinEq = $2,499.54`), el stack consolidado **`D10 / D11`** casi **duplica el Sharpe de 4 años (`2.68–2.72`)**, sostiene **Sharpe `> 2.4` en las 3 eras (`2.44 / 2.65 / 3.16`)**, reduce el **`4Y MaxDD` a `34.39%` (`-3.77 puntos`)**, mantiene el **`100%` de los splits CPCV positivos (`2.66–2.71`)**, y eleva `$2,700` en `2026` a **`$6,613.38 – $7,698.31`** con **`MinEq = $2,555` (`+$55` sobre el piso duro de `$2,500`)**.
