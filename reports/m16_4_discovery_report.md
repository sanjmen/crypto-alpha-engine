# M16-4 Automated Symbolic & Shallow-Tree Alpha Discovery Tournament (2023–2026)

- **Total Candidate Mechanisms Audited (`M`):** `59` (penalized via Bailey & López de Prado `DSR >= 0.95`)
- **Mechanisms Passing ALL 4 Hard Anti-Overfitting Gates (`CPCV >= 1.8`, `DSR >= 0.95`, `Sharpe >= 1.8` in ALL 3 eras, `|corr| < 0.20`):** `1`
- **Mutually Diverse Orthogonal Mechanisms Selected for `M16-5` (`|pairwise corr| <= 0.55`):** `A1_FAST_1H_POST_SETTLE_UNWIND_20BPS`

| Rank | Mechanism ID | Archetype | Full Sharpe | `23-24` Tr | `2025` Val | `2026` Hold | CPCV Med | DSR | Fee Drag `%` | Corr vs Prod | All 4 Gates |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `A1_FAST_1H_POST_SETTLE_UNWIND_20BPS` | `Archetype_1_Funding_Settlement_Clock` | **`+2.08`** | `+2.43` | `+1.98` | **`+2.82`** | `+2.31` | `0.951` | `25.9%` | **`+0.07`** | `PASS` |
| 2 | `A1_FAST_1H_POST_SETTLE_UNWIND_35BPS` | `Archetype_1_Funding_Settlement_Clock` | **`+2.01`** | `+3.15` | `+1.61` | **`+2.69`** | `+2.38` | `0.926` | `14.3%` | **`+0.06`** | `no` |
| 3 | `A5_ORTHO_RIDGE_L1000_EMA25` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.60`** | `+0.67` | `+1.13` | **`+3.40`** | `+1.35` | `0.784` | `11.5%` | **`+0.20`** | `no` |
| 4 | `A5_ORTHO_RIDGE_L100_EMA25` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.56`** | `+0.64` | `+1.12` | **`+3.34`** | `+1.31` | `0.764` | `11.5%` | **`+0.20`** | `no` |
| 5 | `A5_SHALLOW_TREE_D3_MIDCAP_EMA30` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.72`** | `+1.29` | `+0.56` | **`+3.48`** | `+1.76` | `0.854` | `8.0%` | **`+0.18`** | `no` |
| 6 | `A3_ALL219_QUALITY_PLUS_FUNDING_DRIFT_EMA12` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.55`** | `+0.55` | `+1.61` | **`+3.24`** | `+1.50` | `0.761` | `5.1%` | **`+0.27`** | `no` |
| 7 | `A5_ORTHO_RIDGE_L1000_EMA15` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.47`** | `+0.51` | `+0.85` | **`+3.51`** | `+1.16` | `0.707` | `7.8%` | **`+0.19`** | `no` |
| 8 | `A5_ORTHO_RIDGE_L100_EMA15` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.48`** | `+0.50` | `+0.87` | **`+3.51`** | `+1.19` | `0.711` | `7.8%` | **`+0.19`** | `no` |
| 9 | `A3_ALL219_LOW_VOL_SPREAD_QUALITY_EMA12` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.34`** | `+0.45` | `+1.09` | **`+3.35`** | `+1.50` | `0.617` | `3.3%` | **`+0.27`** | `no` |
| 10 | `A3_ALL219_LOW_VOL_SPREAD_QUALITY_EMA25` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.38`** | `+0.44` | `+1.10` | **`+3.48`** | `+1.41` | `0.643` | `4.9%` | **`+0.28`** | `no` |
| 11 | `A3_MIDCAP204_QUALITY_PLUS_FUNDING_DRIFT_EMA12` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.49`** | `+0.34` | `+1.20` | **`+3.46`** | `+1.30` | `0.726` | `5.2%` | **`+0.20`** | `no` |
| 12 | `A5_SHALLOW_TREE_D2_MIDCAP_EMA30` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.26`** | `+0.38` | `+0.31` | **`+3.48`** | `+0.81` | `0.553` | `5.4%` | **`+0.20`** | `no` |
| 13 | `A5_SHALLOW_TREE_D3_MIDCAP_EMA15` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.49`** | `+1.02` | `+0.27` | **`+3.33`** | `+1.54` | `0.723` | `6.0%` | **`+0.17`** | `no` |
| 14 | `A5_SHALLOW_TREE_D2_MIDCAP_EMA15` | `Archetype_5_Symbolic_Shallow_Tree` | **`+1.04`** | `+0.19` | `+0.21` | **`+3.10`** | `+0.71` | `0.383` | `4.0%` | **`+0.20`** | `no` |
| 15 | `A3_ALL219_QUALITY_PLUS_FUNDING_DRIFT_EMA25` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.27`** | `+0.17` | `+1.65` | **`+2.72`** | `+1.24` | `0.560` | `8.1%` | **`+0.26`** | `no` |
| 16 | `A3_MIDCAP204_LOW_VOL_SPREAD_QUALITY_EMA25` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.19`** | `+0.16` | `+0.57` | **`+3.42`** | `+0.93` | `0.500` | `5.4%` | **`+0.21`** | `no` |
| 17 | `A3_MIDCAP204_LOW_VOL_SPREAD_QUALITY_EMA12` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.17`** | `+0.11` | `+0.64` | **`+3.33`** | `+1.12` | `0.482` | `3.9%` | **`+0.21`** | `no` |
| 18 | `A3_MIDCAP204_QUALITY_PLUS_FUNDING_DRIFT_EMA25` | `Archetype_3_MidCap_Microstructure_Quality_CS` | **`+1.23`** | `-0.04` | `+1.11` | **`+3.11`** | `+1.11` | `0.529` | `8.1%` | **`+0.20`** | `no` |
| 19 | `A1_POST_SETTLE_NEG_UNWIND_40BPS_K5` | `Archetype_1_Funding_Settlement_Clock` | **`+0.66`** | `+1.67` | `-0.07` | **`+1.56`** | `+0.76` | `0.150` | `3.2%` | **`+0.03`** | `no` |
| 20 | `A1_POST_SETTLE_NEG_UNWIND_40BPS_K3` | `Archetype_1_Funding_Settlement_Clock` | **`+0.65`** | `+1.67` | `-0.08` | **`+1.55`** | `+0.76` | `0.148` | `3.2%` | **`+0.03`** | `no` |
