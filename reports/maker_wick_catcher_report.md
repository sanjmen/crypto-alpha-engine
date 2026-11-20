# Análisis 1: Passive `GTX` Maker Limit Wick-Catcher at `1-Millisecond` `aggTrades` Resolution (`2023–2026`)

- **Queue & Latency Realism:** `+250ms` API latency after `1-ms` forced sell burst, strict `GTX` rejection if crossed, strict trade-through fill (`price < P_limit * 0.9999` or `>= $25k` sell volume), pessimistic `1s` SL-first post-fill resolution (`1345` fills, `384.3s`).
- **Strict 4-Gate Universal Treasure Survivors:** **`0`**

| Profile | `N` | Gross | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** | Treasure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MW5_GTX_EXTREME_CLIMAX_60BPS` | `249` | `-66.1` | `6.0` | **`-72.1`** | `-172.5` | `35.7%` | `16.1%` | `-87.4` | **`-52.5`** | **`-56.8`** | `-80.5` | `0%` | **`—`** |
| `MW1_GTX_SHALLOW_25BPS` | `276` | `-81.5` | `6.5` | **`-88.0`** | `-147.5` | `28.3%` | `35.9%` | `-111.3` | **`-69.4`** | **`-57.9`** | `-75.6` | `0%` | **`—`** |
| `MW3_GTX_DEEP_70BPS` | `270` | `-92.3` | `6.6` | **`-98.9`** | `-167.5` | `25.6%` | `24.2%` | `-115.2` | **`-80.4`** | **`-84.5`** | `-93.3` | `0%` | **`—`** |
| `MW4_GTX_LADDER_35_70BPS` | `275` | `-94.7` | `6.3` | **`-101.1`** | `-162.5` | `25.1%` | `35.2%` | `-121.0` | **`-82.8`** | **`-79.0`** | `-96.4` | `0%` | **`—`** |
| `MW2_GTX_MEDIUM_45BPS` | `275` | `-97.4` | `6.7` | **`-104.1`** | `-157.5` | `21.8%` | `36.5%` | `-125.2` | **`-84.3`** | **`-81.4`** | `-94.1` | `0%` | **`—`** |
