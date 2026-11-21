# 22 — Resultados Forenses Frente E1: Basis & Carry Harvester (`Manga P4`) y Resolución Definitiva del Issue `#64` (`M15-4`, `2023–2026`)

## 1. Resumen Ejecutivo y Decisión `NO-GO` Disciplinada

En el **Frente E1** ([basis_and_carry_harvester_lab.py](../../src/models/basis_and_carry_harvester_lab.py), [frente_e1_report.md](../../reports/frente_e1_report.md), [frente_e1_summary.json](../../reports/frente_e1_summary.json)) auditamos sobre los `82` perpetuos líquidos (`1m` klines + `1h` `premiumIndexKlines` `Spot-Perp Basis` + cupones reales `:00/:08/:16`, `2023–2026`, `0.000 ms` lookahead) las dos grandes familias de cosecha de *Carry*:
1. **Cosechadores tácticos de cupón Pre-Settlement (`H1..H5`, resolución directa de [Issue #64](https://github.com/sanjmen/crypto-alpha-engine/issues/64) `M15-4`)**, usando funding predicho estrictamente causal en `T - 60m` (`7/8` horas cerradas del TWAP de `premiumIndex` + cupón previo).
2. **Pods continuos multi-ciclo `Manga P4` (`P4_C1..P4_C4`)** e integración en el portafolio `1m` (`E1_P0..E1_P6` sobre el stack `M17 D11`, `$2,700` en `2026` vs piso duro CPPI de `$2,500`).

### Veredicto Consolidado (`0` sobrevivientes — mantener `100%` del margen en el Stack `M17 D10/D11`):
1. **Resolución definitiva del [Issue #64](https://github.com/sanjmen/crypto-alpha-engine/issues/64) (`M15-4 Extreme Funding Harvester` $\rightarrow$ Archivar / Mantener `funding_harvester_strategy=None` permanente):**
   - El baseline ingenuo (`H1`, `2,905` trades) reproduce la pérdida observada en Shadow en `2026` (`-5.6 bps/trade`, mediana `-25.8 bps`).
   - Aplicando **exactamente las 3 reglas propuestas en el Issue `#64`** (`H2`: `ADV >= $75M`, ventana `T - 30m -> T + 3m`, alineación de tendencia `4h`, `497` trades), el cosechador cobra `+17.0 bps` de cupón pero **pierde `-13.0 bps` en precio** + `7.1 bps` de comisión = **`-3.1 bps/trade` neto (`-4.2 bps` en `2026`, `PF = 0.95`)**.
   - Incluso cubriendo la beta minuto a minuto contra `BTCUSDT` (`H5`: `-2.6 bps` full, `-3.1 bps` en `2026`) o exigiendo descuento Spot-Perp extremo `>= 25 bps` (`H3`: cae a **`+0.4 bps`** en `333` trades de `2026`, mediana `-69.7 bps`, `40%` CPCV), **el descuento "ex-dividendo" intraminuto alrededor de `:00:00:00` se come entre el `60%` y el `80%` del cupón**, dejando menos que las comisiones de ida y vuelta.
2. **Por qué NO agregar una `Manga P4` continua de Carry en una cuenta de `$2,700`:**
   - **`Cash-and-Carry` clásico (`Long Spot 1x + Short Perp`, `P4_C1`):** Es ultra seguro (`4Y Sharpe = 5.62`, `MaxDD = 0.26%`), pero rinde apenas **`+2.8%` CAGR** (`0.0%` en `2026` porque `BTC/ETH/SOL` casi no sostuvieron funding `>= +2.5 bps / 8h`). Partir `20%` del capital hacia `Spot 1x` (`E1_P1`) baja el `MaxDD` (`34.39% -> 28.39%`), pero **recorta `-82.1%/año` de `CAGR` (`+286.4% -> +204.3%`) y cuesta `-$1,354.70` en `2026` (`$7,698.31 -> $6,343.61`)** por costo de oportunidad del margen. *(Solo tendrá sentido como tesorería cuando el capital supere la capacidad de `USD-M`, `> $100k`).*
   - **Carry Cross-Perp dentro de `USD-M` (`Short Alts Euforia vs Long BTC`, `P4_C2..P4_C4`):** Aunque en `2026` suma dólares (`$7,698 -> $8,008–$8,533`), **sufre explosiones idiosincráticas devastadoras en *altseasons* (`2023–2024`: `Sharpe -1.14 a -2.68`, `MaxDD 48.7%–81.5%`)**, degradando el `4Y Sharpe` del portafolio (`2.68 -> 2.47–2.65`). Esto explica por qué `Core P1` ya exige sabiamente `t_stat >= +1.25` sobre `F5_Carry` antes de darle peso.

---

## 2. Tablas Empíricas Completas (`2023–2026`)

Fuente: [frente_e1_report.md](../../reports/frente_e1_report.md).

### 2.1. Parte E1.1 — Cosechadores de Cupón Pre-Settlement (`Issue #64` `M15-4`, `1m` Physics)
| Mech ID | `N` | Precio (`bps`) | Cupón (`bps`) | Fee | **Net Full** | Mediana | Win`%` | `23–24` | **`2025`** | **`2026`** | **CPCV OOS (`Pos%`)** | Treasure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H3_BASIS_ANCHORED_NEG_COUPON_30M` | `399` | `-5.8` | `+18.7` | `7.9` | **`+5.0`** | `-69.7` | `36.8%` | `+39.4` | `+20.2` | **`+0.4`** | `+24.2` (`40%`) | `—` |
| `H1_SHADOW_NAIVE_UNFILTERED_60M` | `2905` | `-4.4` | `+14.8` | `8.5` | **`+1.9`** | `-25.8` | `42.5%` | `+5.7` | `+18.0` | **`-5.6`** | `+1.4` (`53%`) | `—` |
| **`H2_ISSUE64_EXACT_RECALIB_30M`** | `497` | **`-12.9`** | **`+17.0`** | `7.1` | **`-3.1`** | `-30.4` | `42.7%` | `-7.3` | `+10.9` | **`-4.2`** | `+22.8` (`67%`) | `—` |
| `H5_BETA_NEUTRAL_COUPON_PAIR_30M` | `408` | `-5.9` | `+9.7` | `6.4` | **`-2.6`** | `-18.4` | `39.7%` | `-9.1` | `+7.5` | **`-3.1`** | `+5.7` (`33%`) | `—` |
| `H4_FROTH_CONTANGO_SHORT_COUPON_30M` | `9` | `-64.7` | `+7.7` | `8.0` | **`-65.0`** | `-159.7` | `22.2%` | `-65.0` | `+0.0` | `+0.0` | `+0.0` (`0%`) | `—` |

### 2.2. Parte E1.2 — Pods Continuos `Manga P4` (`1x` Standalone)
| Carry Pod ID | Activo`%` | **`4Y SR`** | `4Y CAGR` | `4Y DD` | `23–24 / 25 / 26 SR` | `CPCV SR (Pos%)` | `Corr P1` | `Corr P2` | `FeeDrag%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P4_C1_SPOT_PERP_CASH_AND_CARRY_MAJORS` | `18.4%` | **`5.62`** | `+2.8%` | `0.26%` | `7.99 / 3.10 / 0.00` | `4.72` (`80%`) | `+0.066` | `-0.064` | `0.9%` |
| `P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY` | `3.7%` | **`-0.94`** | `-12.6%` | `48.72%` | `-1.64 / 0.00 / 0.61` | `-0.50` (`27%`) | `-0.065` | `-0.025` | `1.0%` |
| `P4_C3_USDM_SYMMETRIC_DISPERSION_POD` | `21.3%` | **`-0.81`** | `-27.9%` | `81.45%` | `-2.68 / -0.08 / 0.87` | `-0.94` (`27%`) | `-0.023` | `+0.007` | `13.7%` |
| `P4_C4_WIDE_TOP35_CARRY_BHS` | `36.4%` | **`-0.15`** | `-7.2%` | `62.28%` | `-1.14 / 1.09 / 0.86` | `0.03` (`60%`) | `-0.050` | `-0.057` | `13.1%` |

### 2.3. Parte E1.3 — Integración sobre el Stack `M17 D11` (`$2,700` en `2026` vs Piso `$2,500`)
| Port ID | **`4Y SR`** | `4Y CAGR` | `4Y DD` | `23–24 / 25 / 26 SR` | `CPCV SR (Pos%)` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E1_P0_M17_STACK_D11_NO_CARRY`** *(Óptimo)* | **`2.68` 🏆** | **`+286.4%`** | **`34.39%`** | **`2.44 / 2.65 / 3.16`** | **`2.66` (`100%`)** | **`$7,698.31`** (`$2,554.98` ✅) |
| `E1_P1_CAPITAL_SPLIT_20PCT_SPOT_PERP` | `2.69` | `+204.3%` | `28.39%` | `2.48 / 2.66 / 3.16` | `2.67` (`100%`) | `$6,343.61` (`$2,586.72` ✅) |
| `E1_P2_USDM_FROTH_CARRY_OVERLAY_035X` | `2.61` | `+269.6%` | `34.39%` | `2.26 / 2.65 / 3.20` | `2.59` (`100%`) | `$7,806.01` (`$2,579.80` ✅) |
| `E1_P3_USDM_SYMMETRIC_POD_OVERLAY_035X` | `2.47` | `+249.7%` | `35.81%` | `1.93 / 2.58 / 3.30` | `2.43` (`100%`) | `$8,532.77` (`$2,574.43` ✅) |
| `E1_P4_LOW_CONVICTION_CARRY_PARKING_050X` | `2.65` | `+279.3%` | `34.39%` | `2.33 / 2.65 / 3.25` | `2.63` (`100%`) | `$8,007.94` (`$2,579.80` ✅) |
| `E1_P5_WIDE_TOP35_CARRY_OVERLAY_035X` | `2.64` | `+280.0%` | `36.81%` | `2.24 / 2.73 / 3.28` | `2.64` (`100%`) | `$8,128.73` (`$2,579.80` ✅) |
| `E1_P6_D11_PLUS_H3_BASIS_ANCHORE` | `2.69` | `+290.3%` | `34.39%` | `2.47 / 2.67 / 3.14` | `2.67` (`100%`) | `$7,580.94` (`$2,579.80` ✅) |
