# 23 — Resultados Forenses Frente E2: Anatomía Standalone `8h`, Ablación Leave-One-Factor-Out (`LOFO`) y Meta-Asignador Online de `SevenFactorOnlineEngine` (`2023–2026`)

## 1. Resumen Ejecutivo (Cierre del Mapa Completo de Investigación `Docs 01..23`)

En el **Frente E2** ([frente_e2_seven_factor_allocator_lab.py](../../src/backtest/frente_e2_seven_factor_allocator_lab.py), [frente_e2_report.md](../../reports/frente_e2_report.md), [frente_e2_summary.json](../../reports/frente_e2_summary.json)) abrimos el capó del motor principal **`Core P1` (`SevenFactorOnlineEngine`)** sobre física exacta de `1 minuto` (`80` símbolos `1m`, `2023–2026` + arranque fresco de `$2,700` en `2026` contra el piso duro CPPI de `$2,500`, sobre el stack realista `M17 D11`):

### Conclusiones Clave:
1. **`6` de los `7` factores + los escudos de régimen están `100%` justificados (quitar cualquiera de ellos lastima el portafolio):**
   - **`F7_Size` (`Prior 0.22`, `24.2%` share):** Aunque standalone en `4 años` parece plano (`SR = 0.09`), quitarlo (`E2_07`) **derrumba el `4Y Sharpe` de `2.68 -> 1.93`**, hunde `2025` (`2.65 -> 0.67`) y dispara el `MaxDD` (`34.4% -> 42.3%`) porque comprar Majors líquidas contra small-caps es el escudo principal en mercados laterales/bajistas de altcoins (`Solo 2025 SR = 1.86`).
   - **`F1_Macro` (`E2_01`):** Quitarlo baja `4Y Sharpe` `2.68 -> 2.23` (`-$2,577` en `2026`).
   - **`F4_Rev` (`E2_04`):** Solo pierde (`-2.38` SR), pero **dentro del ensemble con guardias anti-cuchillo** aporta `+0.38` de `4Y Sharpe` (`2.68 -> 2.30` si se quita, `-$1,601` en `2026`).
   - **`F3_Fast` (`E2_03`):** Quitarlo recorta **`-$4,093` en `2026` (`$7,698 -> $3,605`)**.
   - **`F5_Carry` (`E2_05`) y `F6_Liq` (`E2_06`):** Quitarlos cuesta `-$1,556` y `-$1,738` en `2026` respectivamente. Además, **`F6_Liq` (`expansión de volumen institucional 168h/720h`) es el factor individual más fuerte de todos (`Solo 4Y SR = 1.82`, `100%` CPCV, `2026 SR = 2.42`)** pese a tener apenas `0.04` de prior (`5.0%` de cuota).
   - **Overlays `chop_range_arb` (`E2_08`) y `holistic_regime_shields` (`E2_09`):** Validados; apagarlos empeora `MaxDD` (`36.1%–36.6%`) o degrada `2025` (`2.65 -> 2.16`). Inyectar `premiumIndex` (`Issue #49`, `E2_10` — ya activo en vivo en `crypto-mainnet`) sube `4Y SR` de **`2.68 -> 2.72`** y baja `4Y DD` a **`33.81%`**.
2. **Único hallazgo accionable de ajuste fino (`M17-8`, Issue `#80`, prioridad `P2` opcional) — Redundancia Colineal de `F2_Med` (`E2_02`):**
   - `F2_Med` (`0.40 * zi[72] + 0.60 * zi[168]`, `12.4%` de cuota viva, `Solo 2025 SR = -0.32`) **duplica exactamente horizontes que ya están en `F3_Fast` (`zi[72]`) y `F1_Macro` (`zi[168]`)**, robándole peso a `F1_Macro` y `F6_Liq`.
   - Al podar únicamente `F2_Med` (`E2_02_DROP_F2_MED`): **`4Y Sharpe` salta de `2.68 -> 2.88` (`+0.20`)**, **`4Y CAGR` sube de `+286.4% -> +355.6%`**, **`CPCV Mean Sharpe` sube de `2.66 -> 2.85` (`100%` splits)**, **`2026 SR` sube de `3.16 -> 3.33`** y `MinEq` mejora a **`$2,558.56` ✅** (configurable por flag de priors sin tocar la lógica del motor).

---

## 2. Tablas Empíricas Completas (`2023–2026`)

Fuente: [frente_e2_report.md](../../reports/frente_e2_report.md).

### 2.1. Parte E2.1 — Anatomía Standalone de los `7` Factores (`1x` Neutral `8h`, `2.55 bps` + Funding Real)
| Factor | Prior | Cuota Viva`%` | **`Solo 4Y SR`** | `Solo CAGR` | `Solo DD` | `23–24 / 25 / 26 SR` | `CPCV SR (Pos%)` | `Turn/Día` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`liq` (`F6`)** | `0.04` | `5.0%` | **`1.82` 🏆** | **`+55.1%`** | **`18.48%`** | **`1.85 / 1.27 / 2.42`** | **`1.83` (`100%`)** | `0.31x` |
| **`macro` (`F1`)** | `0.25` | `18.1%` | **`1.23`** | `+29.4%` | `17.68%` | `1.52 / 0.79 / 1.02` | `1.17` (`93%`) | `0.75x` |
| `med` (`F2`) *(colineal)* | `0.14` | `12.4%` | `0.98` | `+22.0%` | `34.86%` | `1.41 / -0.32 / 1.52` | `0.94` (`80%`) | `0.95x` |
| `fast` (`F3`) | `0.19` | `18.0%` | `0.84` | `+17.8%` | `46.29%` | `1.77 / -1.46 / 1.63` | `0.82` (`73%`) | `2.40x` |
| `carry` (`F5`) | `0.18` | `3.3%` | `0.11` | `-0.1%` | `31.52%` | `0.31 / 0.55 / -0.96` | `0.13` (`67%`) | `0.60x` |
| **`size` (`F7`)** *(escudo 2025)* | `0.22` | `24.2%` | `0.09` | `-0.7%` | `47.34%` | `-0.50 / 1.86 / -0.98` | `0.09` (`60%`) | `0.08x` |
| `rev` (`F4`) *(solo sin guardia)* | `0.16` | `19.0%` | `-2.38` | `-47.3%` | `92.16%` | `-2.89 / -0.96 / -2.92` | `-2.39` (`0%`) | `2.27x` |

### 2.2. Parte E2.2 — Ablación `LOFO` (`F1..F7`), Overlays y Meta-Asignador (`Stack M17 D11`, `$2,700` en `2026`)
| Exp ID | **`4Y SR`** | `4Y CAGR` | `4Y DD` | `23–24 / 25 / 26 SR` | **`CPCV SR (Pos%)`** | `FeeDrag%` | **`2026` `$2,700 -> Final` (`MinEq`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E2_00_BASELINE_D11_7FACTOR`** | **`2.68`** | `+286.4%` | `34.39%` | `2.44 / 2.65 / 3.16` | `2.66` (`100%`) | `181.8%` | **`$7,698.31`** (`$2,554.98` ✅) |
| `E2_01_DROP_F1_MACRO` | `2.23` | `+184.8%` | `37.56%` | `2.00 / 2.52 / 2.35` | `2.19` (`100%`) | `173.0%` | `$5,121.10` (`$2,508.11` ✅) |
| **`E2_02_DROP_F2_MED`** *(M17-8)* | **`2.88` 🏆** | **`+355.6%`** | `34.86%` | **`2.82 / 2.63 / 3.33`** | **`2.85` (`100%`)** | `192.7%` | **`$7,407.10`** (**`$2,558.56` ✅**) |
| `E2_03_DROP_F3_FAST` | `2.43` | `+230.3%` | `31.84%` | `2.45 / 2.92 / 1.83` | `2.41` (`100%`) | `170.9%` | `$3,605.25` (`$2,527.98` ✅) |
| `E2_04_DROP_F4_REV` | `2.30` | `+217.5%` | `34.10%` | `2.20 / 2.23 / 2.59` | `2.26` (`100%`) | `168.8%` | `$6,097.73` (`$2,565.41` ✅) |
| `E2_05_DROP_F5_CARRY` | `2.53` | `+260.8%` | `34.60%` | `2.36 / 2.42 / 3.04` | `2.51` (`100%`) | `175.3%` | `$6,142.45` (`$2,528.18` ✅) |
| `E2_06_DROP_F6_LIQ` | `2.60` | `+265.2%` | `33.90%` | `2.51 / 2.66 / 2.77` | `2.58` (`100%`) | `181.6%` | `$5,960.07` (`$2,558.09` ✅) |
| `E2_07_DROP_F7_SIZE` | `1.93` | `+131.9%` | `42.32%` | `1.86 / 0.67 / 3.46` | `1.89` (`100%`) | `165.3%` | `$8,313.46` (`$2,547.88` ✅) |
| `E2_08_DISABLE_CHOP_RANGE_ARB` | `2.65` | `+297.3%` | `36.11%` | `2.50 / 2.62 / 3.04` | `2.60` (`100%`) | `181.5%` | `$6,751.91` (`$2,529.35` ✅) |
| `E2_09_DISABLE_HOLISTIC_SHIELDS` | `2.69` | `+336.3%` | `36.57%` | `2.35 / 2.16 / 3.91` | `2.63` (`100%`) | `188.7%` | `$10,496.11` (`$2,544.91` ✅) |
| **`E2_10_ENABLE_PREMIUM_BASIS_ALPHA`** *(ya vivo)* | **`2.72`** | `+298.7%` | **`33.81%`** | **`2.52 / 2.67 / 3.18`** | **`2.70` (`100%`)** | `183.2%` | **`$7,587.71`** (**`$2,558.79` ✅**) |
| `E2_11_ENABLE_BTC_LEAD_LAG_6H` | `2.76` | `+311.7%` | `33.68%` | `2.55 / 2.68 / 3.27` | `2.73` (`100%`) | `181.5%` | `$6,005.41` (`$2,530.34` ✅) |
| `E2_12_ZERO_PRIORS_PURE_ONLINE_IC` | `2.64` | `+288.1%` | `30.20%` | `2.58 / 2.17 / 3.36` | `2.61` (`100%`) | `183.8%` | `$7,745.94` (`$2,572.76` ✅) |
| `E2_13_8H_ALIGNED_META_LOOKBACKS` | `2.56` | `+239.8%` | `33.15%` | `2.66 / 2.25 / 2.88` | `2.51` (`100%`) | `146.8%` | `$5,332.65` (`$2,525.47` ✅) |
| `E2_14_TAIL_POWER_165_CONCENTRATED` | `2.66` | `+281.8%` | `35.11%` | `2.33 / 2.66 / 3.29` | `2.63` (`100%`) | `181.5%` | `$10,931.48` (`$2,579.36` ✅) |
