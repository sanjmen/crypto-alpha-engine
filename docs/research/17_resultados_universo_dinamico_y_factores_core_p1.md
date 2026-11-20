# 17 — Resultados Forenses Análisis 2: Universo Dinámico Point-in-Time (`Top-15 / Top-20`) vs. `PROD_TOP_15` y Ablación Limpia de `F3_Fast` / Micro-Recortes sobre Reloj Estricto de `8h` (`2023–2026`)

## 1. Resumen Ejecutivo y Veredicto (`01_STATIC_TOP15_STRICT_8H` Confirmado como Óptimo Productivo)

Ejecutamos el **Análisis 2** sobre física intrabarra exacta de **`1 minuto` (`80` símbolos `1m`, `2023–2026` + stress-test `2026` desde `$2,700` contra piso duro CPPI de `$2,500` + `15-split` `CombinatorialPurgedCV`)** mediante [core_universe_factor_lab.py](../../src/backtest/core_universe_factor_lab.py) (con blindaje anti-contaminación de activos no listados `compute_dynamic_pit_stitched_matrices`, verificado con paridad exacta `atol=1e-12` en [test_core_universe_factor_lab.py](../../tests/test_core_universe_factor_lab.py)).

Fuente reproducible: [core_universe_factor_report.md](../../reports/core_universe_factor_report.md) y [core_universe_factor_summary.json](../../reports/core_universe_factor_summary.json).

### Conclusiones Principales
1. **La configuración que YA está corriendo hoy en vivo en Tokio (`01_STATIC_TOP15_STRICT_8H`) es el óptimo global de crecimiento y Sharpe (`4Y Sharpe = 2.19`, `2026 Sharpe = 2.42`, `CPCV = 2.17` con `100%` splits positivos, `$2,700 -> $5,066.31` en `2026` con `MinEq = $2,550.42` ✅):** No hace falta tocar nada más en producción.
2. **Reemplazar la cesta institucional curada (`PROD_TOP_15`) por un ranking automático de volumen `30d-ADV` (`07..10`, `Sharpe = 0.46–1.07`) degrada severamente el `Core P1`:** El ranking ciego por volumen mete memecoins parabólicas (`1000PEPE`, `WIF`), tokens con *unlocks* abruptos (`WLD`) y *short-squeezes* legados (`BCH`, `FIL`, `ETC`) adentro de un libro **Market-Neutral sin `Stop-Loss` individual**, rompiendo la neutralidad sectorial en `2025` (`Sharpe -0.97` en `Top-15`). Esas monedas explosivas **solo** deben operarse desde el **Pilar 3 (`Snipers M09/M23/M04/M26` sobre `82` monedas)**, donde cada tiro lleva `Stop-Loss` duro estricto.
3. **Desenredando `F3_Fast_Swing` y el *buffer* de `3.5%` sobre reloj puro de `8h`:**
   * **Eliminar `F3_Fast_Swing` (`04_SIX_FACTOR_NO_F3`) es un error (`-$1,075` en `2026`, `4Y Sharpe 2.19 -> 2.06`):** `F3` aporta el motor de captura en swings rápidos de `2023–2024` y `2026`.
   * **La combinación defensiva `06_STATIC_TOP15_TRIM_3.5_FAST_GATE_0.50`** empata en `4Y Sharpe` (`2.18`) y `CAGR` (`+188.3%`), reduciendo `4Y MaxDD` en **`-4.36%` (`35.23% -> 30.87%`)** y subiendo `2025 Sharpe` de `1.95 -> 2.25`, pero cede `-$491.50` de ganancia en `2026` (`$4,574.81` vs `$5,066.31`). Queda documentada como perfil conservador opcional, manteniendo `01` en vivo.

---

## 2. Tabla Completa de Resultados (`1m` Física Lineal sobre Reloj Estricto `8h`, `2023–2026`)

| Experimento | Universo | `4Y SR` | `4Y CAGR` | `4Y MaxDD` | `23–24 / 2025 / 2026 SR` | `CPCV SR (Pos%)` | `RebalClose / Part` | `Lock / TP` | `FeeDrag%` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`01_STATIC_TOP15_STRICT_8H` *(Vivo en Tokio)* ** | `static_15` | **`2.19`** | `+188.1%` | `35.23%` | **`2.25 / 1.95 / 2.42`** | **`2.17` (`100%`)** | `6268 / 15537` | `1945 / 2699` | `245.9%` | **`$5,066.31`** (`$2,550.42` ✅) |
| `06_STATIC_TOP15_TRIM_3.5_FAST_GATE_0.50` | `static_15` | **`2.18`** | **`+188.3%`** | **`30.87%`** | `2.16 / 2.25 / 2.18` | **`2.17` (`100%`)** | `6183 / 11301` | `1855 / 2642` | `247.5%` | `$4,574.81` (`$2,557.06` ✅) |
| `02_STATIC_TOP15_TRIM_BUF_3.5PCT` | `static_15` | `2.16` | `+186.6%` | `35.55%` | `2.32 / 1.94 / 2.22` | `2.15` (`100%`) | `6269 / 11286` | `1919 / 2711` | `246.7%` | `$4,963.29` (`$2,552.52` ✅) |
| `05_STATIC_TOP15_BHS_3_7_DB_1.60_TRIM_3.5` | `static_15` | `2.09` | `+174.2%` | `35.86%` | `2.20 / 1.77 / 2.34` | `2.08` (`100%`) | `6132 / 11240` | `1921 / 2701` | `244.5%` | `$5,212.89` (`$2,551.22` ✅) |
| `04_STATIC_TOP15_SIX_FACTOR_NO_F3` | `static_15` | `2.06` | `+164.0%` | `32.32%` | `2.07 / 2.26 / 1.90` | `2.07` (`100%`) | `5533 / 15239` | `1769 / 2530` | `226.2%` | `$3,991.20` (`$2,555.46` ✅) |
| `03_STATIC_TOP15_FAST_GATE_0.50` | `static_15` | `2.04` | `+162.0%` | `31.25%` | `1.99 / 2.17 / 2.07` | `2.03` (`100%`) | `6203 / 15261` | `1872 / 2637` | `243.8%` | `$4,426.24` (`$2,556.43` ✅) |
| `08_DYNAMIC_PIT_TOP20_30D_ADV` | `dynamic_pit_20` | `1.07` | `+52.2%` | `42.00%` | `1.57 / 0.90 / 0.18` | `0.97` (`93%`) | `8632 / 16458` | `2325 / 3496` | `227.6%` | `$2,566.59` (`$2,503.44` ✅) |
| `09_DYNAMIC_PIT_TOP15_TRIM_3.5_FAST_GATE` | `dynamic_pit_15` | `0.99` | `+41.8%` | `49.51%` | `1.66 / -0.43 / 0.86` | `0.85` (`80%`) | `6357 / 9857` | `1766 / 2693` | `188.0%` | `$2,558.03` (`$2,479.06` ⚠️) |
| `07_DYNAMIC_PIT_TOP15_30D_ADV` | `dynamic_pit_15` | `0.76` | `+27.3%` | `52.89%` | `1.47 / -0.97 / 0.86` | `0.62` (`73%`) | `6567 / 14568` | `1809 / 2742` | `184.1%` | `$2,576.71` (`$2,516.96` ✅) |
| `10_DYNAMIC_PIT_TOP15_MONTHLY_SEASONED_90D` | `dynamic_pit_15` | `0.46` | `+10.5%` | `48.73%` | `0.80 / 0.15 / 0.45` | `0.42` (`80%`) | `6662 / 7827` | `1753 / 2622` | `140.0%` | `$2,516.05` (`$2,450.15` ⚠️) |

---

## 3. Por qué `PROD_TOP_15` supera por tanto al Ranking Automático por Volumen (`07..10`)

Las instantáneas reales de membresía Point-in-Time en [core_universe_factor_summary.json](../../reports/core_universe_factor_summary.json) revelan exactamente qué activos metió el selector automático de `30d-ADV`:

* **`2023-06`:** Entraron `LTCUSDT, OPUSDT, FILUSDT, ETCUSDT, GALAUSDT`.
* **`2024-06`:** Entraron `BCHUSDT, WLDUSDT, FILUSDT, 1000PEPEUSDT, WIFUSDT` (desplazando a `LINK, APT, FET, RENDER, SUI`).
* **`2025-06`:** Entraron `LTCUSDT, BCHUSDT, AAVEUSDT, 1000PEPEUSDT, WIFUSDT`.
* **`2026-06`:** Entraron `BCHUSDT, TAOUSDT, FILUSDT, 1000PEPEUSDT`.

La razón estructural es **la diferencia física entre el Pilar 1 (`Core P1`) y el Pilar 3 (`Snipers`)**:
1. En **`Core P1`**, las posiciones individuales **no usan `Stop-Loss` duro** porque el riesgo de mercado está cubierto *cruzando patas Long contra patas Short* de infraestructura L1/IA con betas estables y topes por clúster sectorial (`ALT_SECTOR_CLUSTERS`: `FAST_L1`, `ALT_L1`, `AI_COMPUTE`, `PAY_MEME`, `INFRA_L2`).
2. Cuando un ranking puramente volumétrico mete a **`1000PEPE` y `WIF`** (memes puros que se disparan por *hype* de volumen minorista) o a **`WLD` y `FIL`** (sujetos a *squeezes* de baja flotación o desbloqueos de tesorería), el factor `F4_Rev` o el especialista `Range-Arb` se les pone **Short sin `Stop-Loss` duro** durante una manía minorista, sufriendo colas pesadas idiosincráticas que las patas Long en `BTC/ETH/LTC` no compensan (`2025 Sharpe = -0.97`, `MaxDD = 52.9%`).
3. En cambio, en nuestra arquitectura actual de **3 Pilares**, las `82` monedas explosivas (`PEPE, WIF, TAO, WLD, PENDLE, BARD...`) se operan **únicamente a través de los Snipers (`M09, M23, M04, M26`)** con `Stop-Loss` duro individual y *trailing* convexo, mientras `Core P1` opera exclusivamente las `15` redes líquidas estructurales (`PROD_TOP_15`). **Este experimento valida empíricamente por qué esa separación de 3 Pilares es indispensable.**
