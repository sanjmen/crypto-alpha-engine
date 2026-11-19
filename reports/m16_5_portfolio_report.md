# M16-5 Grand 4-Year Multi-Engine Portfolio Synthesis & `$2,500` Floor Stress-Test (`100% USDT`)

- **Capital & Risk Governance:** `$2,700.00` initial equity (`100% USDT`, `BFUSD=OFF`), `SteppedLeverageController` Two-Phase SafeRamp (`$2,700 -> $3,000` boost) + Cushion-Safe CPPI Governor (`$2,500.00` Hard Floor, `$240.00` smooth buffer).

| Portfolio Variant | `23-26` Sharpe | Lift | `%` Daily Geom | CAGR `%` | MaxDD `%` | Min Eq (`$2.7k`) | `2026` Hold Sharpe | `2026` `%` Daily | `2026` MaxDD | Cons. Sharpe | Go `M16-6` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00_CURRENT_PROD` | **`0.44`** | `+0.00` | **`+0.013%`** | `+4.8%` | **`29.40%`** | **`$2,701.91`** | **`-1.24`** | `-0.018%` | `10.32%` | `-0.27` | **`GO ✅`** |
| `01_PROD_PLUS_A1_FAST_1H_POST_SETTLE_UNWIND_20BPS` | **`2.06`** | `+1.62` | **`+0.096%`** | `+41.8%` | **`23.63%`** | **`$2,693.91`** | **`-0.17`** | `-0.009%` | `16.48%` | `1.24` | **`NO-GO`** |
| `01_PROD_PLUS_A3_MIDCAP_LOW_VOL_QUALITY_8H` | **`0.58`** | `+0.15` | **`+0.023%`** | `+8.6%` | **`23.45%`** | **`$2,654.47`** | **`-0.34`** | `-0.010%` | `11.23%` | `-0.16` | **`NO-GO`** |
| `99_M16_FULL_ORTHOGONAL_ENSEMBLE` | **`2.15`** | `+1.72` | **`+0.100%`** | `+44.0%` | **`15.58%`** | **`$2,633.96`** | **`1.13`** | `+0.044%` | `13.59%` | `1.35` | **`GO ✅`** |
