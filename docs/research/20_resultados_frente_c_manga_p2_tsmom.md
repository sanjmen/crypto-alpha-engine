# 20 — Resultados Forenses Frente C: Reinvención y Ablación del Pilar 2 (`Manga P2: CausalDollarBarTSMOMEngine`, `2023–2026`)

## 1. Resumen Ejecutivo

En el **Frente C** ([manga_p2_tsmom_lab.py](../../src/models/manga_p2_tsmom_lab.py), [frente_c_manga_p2_report.md](../../reports/frente_c_manga_p2_report.md), [frente_c_manga_p2_summary.json](../../reports/frente_c_manga_p2_summary.json)) auditamos por primera vez el **Pilar 2 (`Manga P2`: `CausalDollarBarTSMOMEngine`)** tanto en forma aislada (`1x` Gross con comisiones y funding reales cada `8h`) como integrado en el portafolio `1m` completo (`Core P1 + Manga P2 + Snipers M09/M23/M04/M26`, `2023–2026` y arranque fresco de `$2,700` en `2026` contra el piso duro CPPI de `$2,500`), evaluando `8` arquitecturas sobre 4 ejes creativos:
1. **Ablación pura (`C00` OFF vs `C01` Live Tokyo)** para cuantificar cuánto aporta realmente `Manga P2`.
2. **Paridad de Riesgo por Volatilidad Inversa Causal (`168h`, `C02`)** vs pesos iguales entre `BTC, ETH, SOL`.
3. **Acelerador Asimétrico de Crash (`C03`)** (`z_24 <= -1.40` + `CVD <= -0.035`).
4. **Barras de Desequilibrio de Flujo (`DIBs`, López de Prado AFML Cap. 2, `C04/C07`)** y **Expansión a `5` Soberanos (`BTC, ETH, SOL, BNB, XRP`, `C05..C07`)**.

### Hallazgos Principales:
1. **`Manga P2` es indispensable en producción (`C00 -> C01`):** Posee correlación diaria **negativa (`Corr = -0.088`)** contra `Core P1`, es positivo standalone en las 3 eras (`0.97 / 0.76 / 0.98`), eleva el **`4Y Sharpe` del portafolio de `1.92 -> 2.19` (`+0.27`)**, sube el `CAGR` de `+139.0% -> +188.1%` sin empeorar el drawdown (`35.08% -> 35.23%`), y en `2026` suma **`+$1,076.90` netos** sobre `$2,700` (`$3,973.27 -> $5,050.17`). Esto confirma además que el `Net $` direccional observado en vivo cuando `Manga P2` está corto `BTC/ETH` es **alfa estructural genuino**.
2. **Mejora opcional de bajo riesgo (`M17-5`, prioridad `P2`) — Paridad de Riesgo `Inv-Vol` en `3` Majors (`C02`):** En `C01`, `SOL` tiene la misma cuota nocional que `BTC` pese a casi duplicar su volatilidad. Ponderar `BTC, ETH, SOL` por volatilidad inversa causal (`168h`, `C02`) empata el `4Y Sharpe` (`2.17` vs `2.19`), pero **sube la estabilidad `P2 CPCV` de `86.7% -> 93.3%`**, mejora la descorrelación con `Core P1` (`-0.088 -> -0.092`), baja el `4Y MaxDD` (`35.23% -> 34.93%`), sube el **`2026 Sharpe` de `2.42 -> 2.50`**, eleva el capital `2026` a **`$5,148.02` (`+$97.85`)** y amplía el colchón sobre el piso de `$2,500` (`MinEq = $2,555.15` vs `$2,550.02`).
3. **Líneas descartadas (`NO-GO`):**
   - **Expansión a `5` Soberanos (`BNB, XRP`, `C05..C07`):** Diluye la cuota de tendencia pura de `BTC/ETH/SOL` y baja `4Y Sharpe` a `2.03–2.05` (`$4,384–$4,878` en `2026`).
   - **Acelerador Asimétrico de Crash (`C03`):** Paradoja clásica standalone vs portafolio — sube el Sharpe aislado de `Manga P2` a **`0.97`**, pero dentro del portafolio con gobernador CPPI sufre *whipsaws* en rebotes en `V` dentro de mercados alcistas (`Port 4Y SR` cae a `2.10`, `$4,366` en `2026`).
   - **Flow-Imbalance Bars (`DIB`, `C04`):** Alcanza el máximo terminal puntual en `2026` (`$5,298.98`, `100%` CPCV), pero decae fuerte en el mercado lateral-picado de `2025` (`P2 Solo 2025 SR = 0.32` vs `0.76` en Dollar Bars estándar).

---

## 2. Tabla Comparativa Completa (`2023–2026` Standalone + Portafolio Integrado `1m`)

Fuente: [frente_c_manga_p2_report.md](../../reports/frente_c_manga_p2_report.md).

| Variant ID | `P2 Solo SR` (`23–24 / 25 / 26`) | `P2 CPCV%` | `Corr vs P1` | **`Port 4Y SR`** | `Port CAGR` | `Port DD` | `Port 23–24 / 25 / 26` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C00_NO_MANGA_P2_ABLATION` *(OFF)* | `0.00` (`0.00/0.00/0.00`) | `0%` | `+0.000` | **`1.92`** | `+139.0%` | `35.08%` | `2.05 / 1.77 / 2.00` | `$3,973.27` (`$2,558.12` ✅) |
| **`C01_PROD_3MAJ_EQUAL_DOLLAR_BAR`** *(Vivo)* | **`0.89`** (`0.97/0.76/0.98`) | `87%` | **`-0.088`** | **`2.19` 🏆** | **`+188.1%`** | `35.23%` | `2.25 / 1.95 / 2.42` | **`$5,050.17`** (`$2,550.02` ✅) |
| **`C02_3MAJ_INV_VOL_RISK_PARITY`** *(M17-5)* | **`0.82`** (`0.76/0.68/1.06`) | **`93%`** | **`-0.092`** | **`2.17`** | `+182.4%` | **`34.93%`** | `2.20 / 1.89 / 2.50` | **`$5,148.02`** (**`$2,555.15` ✅**) |
| `C03_3MAJ_ASYM_CRASH_ACCELERATOR` | **`0.97`** (`1.06/0.91/1.00`) | `87%` | `-0.085` | `2.10` | `+174.8%` | `35.81%` | `2.16 / 1.88 / 2.34` | `$4,366.14` (`$2,538.21` ✅) |
| `C04_3MAJ_FLOW_IMBALANCE_BARS_DIB` | `0.72` (`0.81/0.32/0.98`) | `100%` | `-0.097` | `2.11` | `+173.4%` | `35.71%` | `2.14 / 1.86 / 2.38` | `$5,298.98` (`$2,553.06` ✅) |
| `C05_5SOV_BTC_ETH_SOL_BNB_XRP_INV_VOL` | `0.88` (`0.96/0.71/1.03`) | `100%` | `-0.076` | `2.03` | `+159.9%` | `34.29%` | `1.92 / 1.94 / 2.40` | `$4,383.76` (`$2,545.50` ✅) |
| `C06_5SOV_INV_VOL_ASYM_CRASH_ACCEL` | `0.85` (`0.92/0.79/0.93`) | `100%` | `-0.081` | `2.05` | `+165.2%` | `32.75%` | `2.07 / 1.78 / 2.39` | `$4,877.52` (`$2,545.76` ✅) |
| `C07_5SOV_DIB_INV_VOL_ASYM_CRASH` | `0.67` (`0.68/0.34/0.98`) | `100%` | `-0.084` | `2.05` | `+164.1%` | `33.94%` | `2.02 / 1.87 / 2.37` | `$4,674.41` (`$2,540.66` ✅) |

---

## 3. Conclusión Operativa y Priorización

1. **El motor `C01` actualmente desplegado en Tokio ya captura el `99%+` del óptimo teórico multi-año (`4Y Sharpe = 2.19`)**; no requiere ningún cambio urgente.
2. Queda registrado como mejora opcional de bajo riesgo (**Issue `#77` / `M17-5`, prioridad `P2`**) agregar el modo `sizing_mode="inv_vol"` (`168h` causal) en `CausalDollarBarTSMOMEngine` para igualar presupuesto de volatilidad entre `BTC`, `ETH` y `SOL` (`2026 SR = 2.50`, `4Y DD = 34.93%`, `MinEq = $2,555.15`), siempre por detrás en prioridad respecto a las mejoras de **Frente A (`#73, #74, #75`)** y **Frente B (`#76`)**.
