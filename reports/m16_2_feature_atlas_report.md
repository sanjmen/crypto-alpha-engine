# M16-2 Multi-Modal Microstructure Feature Atlas (2023–2026 Walk-Forward)

- **Universe Evaluated:** `219` Point-in-Time USDS-M Perpetuals (`4,143,581` 30m decision rows, `ADV_7d >= $15M` tradable gate)
- **Causal Features Audited:** `114` across Families A, B, C, and D (`+60s` metrics safety lag, causal settlement clock)
- **Sign-Consistent Orthogonal Features (`|corr| < 0.35` across all 3 eras):** `74` (`4h` horizon) | `79` (`1h` horizon)

## Top 20 Features — 4h Horizon (Ranked by Out-of-Sample Composite Score)

| Rank | Feature | Family | Pol | Train t | Val 2025 t | Holdout 2026 t | 2026 Tail Net Maker (bps) | 2026 Tail Net Taker (bps) | Corr to Existing | 3-Era Consistent |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `pre_settle_neg_squeeze_80bps` | `Family_C_Funding_Clock` | `-1` | `+2.52` | `+6.33` | **`+6.59`** | **`+524.9`** | `+519.1` | `+0.06` | `YES` |
| 2 | `pre_settle_neg_squeeze_40bps` | `Family_C_Funding_Clock` | `-1` | `+2.67` | `+8.93` | **`+6.97`** | **`+277.9`** | `+272.1` | `+0.10` | `YES` |
| 3 | `pre_settle_neg_squeeze_20bps` | `Family_C_Funding_Clock` | `-1` | `+2.83` | `+7.63` | **`+5.06`** | **`+143.5`** | `+137.7` | `+0.14` | `YES` |
| 4 | `compressed_clock_neg_funding_impulse` | `Family_C_Funding_Clock` | `-1` | `+2.96` | `+6.74` | **`+5.52`** | **`+89.0`** | `+83.2` | `+0.12` | `YES` |
| 5 | `parkinson_vol_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+6.45` | `+7.80` | **`+7.92`** | **`+52.5`** | `+46.7` | `+0.25` | `YES` |
| 6 | `corwin_schultz_spread_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+6.83` | `+8.11` | **`+7.60`** | **`+44.5`** | `+38.7` | `+0.25` | `YES` |
| 7 | `funding_ann_pct` | `Family_C_Funding_Clock` | `+1` | `+2.56` | `+9.01` | **`+6.83`** | **`+33.5`** | `+27.7` | `-0.16` | `YES` |
| 8 | `clock_compression_ratio` | `Family_C_Funding_Clock` | `-1` | `+4.53` | `+7.93` | **`+4.99`** | **`+43.0`** | `+37.2` | `+0.04` | `YES` |
| 9 | `funding_rate_bps` | `Family_C_Funding_Clock` | `+1` | `+3.74` | `+7.04` | **`+6.89`** | **`+32.0`** | `+26.2` | `-0.17` | `YES` |
| 10 | `rvol_shock_4h_vs_7d` | `Family_D_CrossSec_MidCap` | `-1` | `+4.57` | `+7.17` | **`+7.23`** | **`+28.9`** | `+23.1` | `+0.49` | `YES` |
| 11 | `funding_interval_hours` | `Family_C_Funding_Clock` | `+1` | `+4.52` | `+8.48` | **`+4.23`** | **`+36.9`** | `+31.1` | `-0.09` | `YES` |
| 12 | `in_post_settlement_window` | `Family_C_Funding_Clock` | `-1` | `+2.17` | `+4.43` | **`+4.77`** | **`+46.1`** | `+40.3` | `-0.03` | `YES` |
| 13 | `in_pre_settlement_window` | `Family_C_Funding_Clock` | `-1` | `+1.90` | `+4.15` | **`+5.26`** | **`+43.9`** | `+38.1` | `+0.01` | `YES` |
| 14 | `mins_to_next_settlement` | `Family_C_Funding_Clock` | `+1` | `+4.68` | `+5.43` | **`+4.96`** | **`+31.3`** | `+25.5` | `+0.02` | `YES` |
| 15 | `mins_since_settlement` | `Family_C_Funding_Clock` | `+1` | `+2.45` | `+5.94` | **`+2.34`** | **`+41.2`** | `+35.4` | `+0.01` | `YES` |
| 16 | `cs_rank_funding_bps` | `Family_D_CrossSec_MidCap` | `+1` | `+2.60` | `+5.14` | **`+4.00`** | **`+19.8`** | `+14.0` | `-0.08` | `YES` |
| 17 | `oi_vel_12h` | `Family_A_Derivatives_OI` | `-1` | `+1.74` | `+6.12` | **`+5.80`** | **`-0.9`** | `-6.7` | `+0.18` | `YES` |
| 18 | `vol_of_vol_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+4.60` | `+5.63` | **`+4.34`** | **`+10.2`** | `+4.4` | `+0.28` | `YES` |
| 19 | `is_compressed_funding_clock` | `Family_C_Funding_Clock` | `-1` | `+4.50` | `+7.38` | **`+4.17`** | **`-287.1`** | `-292.9` | `+0.11` | `YES` |
| 20 | `rvol_shock_1h_vs_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+2.44` | `+5.46` | **`+3.34`** | **`+12.9`** | `+7.1` | `+0.32` | `YES` |

## Top 15 Features — 1h Fast Sniper Horizon

| Rank | Feature | Family | Pol | Train t | Val 2025 t | Holdout 2026 t | 2026 Tail Net Maker (bps) | Corr to Existing | 3-Era Consistent |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `pre_settle_pos_crowding_20bps` | `Family_C_Funding_Clock` | `-1` | `+0.94` | `+2.05` | **`-0.34`** | **`+139.7`** | `-0.05` | `YES` |
| 2 | `pre_settle_neg_squeeze_40bps` | `Family_C_Funding_Clock` | `-1` | `+1.42` | `+5.69` | **`+4.97`** | **`+58.8`** | `+0.10` | `YES` |
| 3 | `pre_settle_neg_squeeze_80bps` | `Family_C_Funding_Clock` | `-1` | `+1.69` | `+5.12` | **`+2.68`** | **`+65.6`** | `+0.06` | `YES` |
| 4 | `pre_settle_neg_squeeze_20bps` | `Family_C_Funding_Clock` | `-1` | `+1.85` | `+4.06` | **`+3.41`** | **`+31.8`** | `+0.18` | `YES` |
| 5 | `compressed_clock_neg_funding_impulse` | `Family_C_Funding_Clock` | `-1` | `+0.76` | `+4.29` | **`+3.40`** | **`+23.6`** | `+0.18` | `YES` |
| 6 | `parkinson_vol_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+3.75` | `+5.09` | **`+3.89`** | **`+11.9`** | `+0.27` | `YES` |
| 7 | `corwin_schultz_spread_24h` | `Family_D_CrossSec_MidCap` | `-1` | `+3.29` | `+5.22` | **`+3.62`** | **`+9.4`** | `+0.29` | `YES` |
| 8 | `funding_rate_bps` | `Family_C_Funding_Clock` | `+1` | `+0.15` | `+5.32` | **`+3.72`** | **`+7.9`** | `-0.18` | `YES` |
| 9 | `funding_ann_pct` | `Family_C_Funding_Clock` | `+1` | `+0.20` | `+5.04` | **`+3.56`** | **`+7.9`** | `-0.17` | `YES` |
| 10 | `rvol_shock_4h_vs_7d` | `Family_D_CrossSec_MidCap` | `-1` | `+2.45` | `+4.44` | **`+3.70`** | **`+6.3`** | `+0.48` | `YES` |
| 11 | `quad_new_longs_1h` | `Family_A_Derivatives_OI` | `-1` | `+6.11` | `+4.05` | **`+4.45`** | **`+0.5`** | `+0.20` | `YES` |
| 12 | `in_pre_settlement_window` | `Family_C_Funding_Clock` | `-1` | `+2.94` | `+2.28` | **`+3.77`** | **`+8.5`** | `+0.06` | `YES` |
| 13 | `ret_z_1b` | `Family_D_CrossSec_MidCap` | `-1` | `+7.15` | `+2.55` | **`+4.71`** | **`-6.5`** | `+0.12` | `YES` |
| 14 | `ret_1b` | `Family_D_CrossSec_MidCap` | `-1` | `+7.73` | `+2.77` | **`+4.47`** | **`+0.2`** | `+0.16` | `YES` |
| 15 | `mins_since_settlement` | `Family_C_Funding_Clock` | `+1` | `+1.90` | `+2.00` | **`+3.08`** | **`+13.6`** | `+0.04` | `YES` |
