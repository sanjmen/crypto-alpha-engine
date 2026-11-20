# 1-Minute Linear-Return Counterparty-Mechanics Laboratory (`2023–2026`, `220` Perps)

- **Universe Scanned:** `219` Point-in-Time USDS-M Perps (`20` shards, `27,103` total simulated 1m trades, `2068.5s` runtime).
- **Return Accounting:** `100%` Exact Linear Perpetual Return `side * (P_exit - P_entry) / P_entry + Realized_Funding - Fees` (**zero `np.log` convexity bias**).
- **Intrabar Execution Matrix:** `TAKER_OPEN` (`open[t+1]` + `6.5 bps` fee+slip) vs `TOUCH_LIMIT` (`0 bps` offset, `1.5 bps` price-through, `2.0 bps` Maker) vs `GTX_1.5BPS` (`+1.5 bps` offset, `2.0 bps` price-through, `2.0 bps` Maker) with **`100%` pessimistic same-bar Stop-Loss priority**.
- **Strict Treasure Survivors (`>= +25 bps` Full, `>= +15 bps` in `23-24` & `2025`, `>= +20 bps` in `2026`, `Fee Share <= 30%`, True In-Fold CPCV `>= 80%`):** **`0`**

| Mechanism | Mode | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | `2025` | `2026` | In-Fold CPCV bps | CPCV`%` | BTC Corr | Treasure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1B_PRE_SETTLE_SPOT_SCARCITY_LONG` | `TAKER_OPEN` | `187` | `+17.3` | `+15.4` | `12.8` | **`+20.0`** | `+57.0` | `59.9%` | `11.3%` | `+58.9` | **`+22.8`** | **`+1.2`** | `+28.2` | `80%` | `-0.04` | **`—`** |
| `C1B_PRE_SETTLE_SPOT_SCARCITY_LONG` | `TOUCH_LIMIT` | `178` | `+12.9` | `+13.5` | `8.3` | **`+18.1`** | `+61.5` | `59.0%` | `7.5%` | `+48.1` | **`+37.4`** | **`-4.8`** | `+18.0` | `80%` | `-0.04` | **`—`** |
| `C2A_TRAPPED_SHORTS_PAIN_LONG` | `GTX_1.5BPS` | `405` | `+25.2` | `+1.9` | `8.3` | **`+18.8`** | `+71.5` | `55.3%` | `7.2%` | `-5.3` | **`-0.2`** | **`+89.1`** | `+12.2` | `67%` | `+0.01` | **`—`** |
| `C1B_PRE_SETTLE_SPOT_SCARCITY_LONG` | `GTX_1.5BPS` | `177` | `+14.4` | `+14.1` | `8.3` | **`+20.3`** | `+61.5` | `59.3%` | `7.4%` | `+51.0` | **`+43.1`** | **`-5.3`** | `+20.7` | `80%` | `-0.04` | **`—`** |
| `C2A_TRAPPED_SHORTS_PAIN_LONG` | `TOUCH_LIMIT` | `405` | `+25.3` | `+1.9` | `8.3` | **`+18.9`** | `+71.5` | `55.1%` | `7.1%` | `-7.2` | **`+3.8`** | **`+86.7`** | `+14.3` | `67%` | `+0.01` | **`—`** |
| `C2A_TRAPPED_SHORTS_PAIN_LONG` | `TAKER_OPEN` | `413` | `+25.3` | `+1.9` | `12.8` | **`+14.4`** | `+67.0` | `55.2%` | `11.1%` | `-10.7` | **`-0.1`** | **`+79.8`** | `+12.6` | `60%` | `+0.01` | **`—`** |
| `M00_NAIVE_M25_CONTROL` | `TOUCH_LIMIT` | `4624` | `-1.2` | `+0.0` | `8.2` | **`-9.4`** | `+36.5` | `60.4%` | `11.6%` | `-18.5` | **`-6.0`** | **`-7.6`** | `-10.4` | `33%` | `-0.03` | **`—`** |
| `M00_NAIVE_M25_CONTROL` | `GTX_1.5BPS` | `4562` | `-2.3` | `+0.0` | `8.2` | **`-10.5`** | `+36.5` | `60.3%` | `11.7%` | `-19.5` | **`-7.9`** | **`-7.8`** | `-11.3` | `33%` | `-0.03` | **`—`** |
| `M00_NAIVE_M25_CONTROL` | `TAKER_OPEN` | `4939` | `+5.1` | `+0.0` | `12.7` | **`-7.5`** | `+32.0` | `61.8%` | `17.3%` | `-21.4` | **`-4.3`** | **`-2.6`** | `-7.2` | `47%` | `-0.05` | **`—`** |
| `C1A_POST_SETTLE_HARVESTER_SHORT` | `GTX_1.5BPS` | `1973` | `+4.6` | `+0.0` | `8.3` | **`-3.7`** | `+36.5` | `60.3%` | `12.1%` | `-24.4` | **`-0.6`** | **`+7.5`** | `-3.1` | `47%` | `-0.04` | **`—`** |
| `C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG` | `TAKER_OPEN` | `470` | `+42.1` | `+0.3` | `12.3` | **`+30.1`** | `+42.0` | `57.4%` | `10.9%` | `+50.3` | **`+39.8`** | **`-24.9`** | `+26.0` | `47%` | `-0.09` | **`—`** |
| `C1A_POST_SETTLE_HARVESTER_SHORT` | `TOUCH_LIMIT` | `2000` | `+5.6` | `+0.0` | `8.3` | **`-2.7`** | `+36.5` | `60.5%` | `12.0%` | `-25.6` | **`+1.8`** | **`+8.0`** | `-2.8` | `47%` | `-0.04` | **`—`** |
| `C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG` | `GTX_1.5BPS` | `424` | `+19.5` | `+0.4` | `8.0` | **`+11.9`** | `+46.5` | `55.4%` | `8.5%` | `+29.9` | **`+12.4`** | **`-26.0`** | `+0.8` | `20%` | `-0.07` | **`—`** |
| `C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG` | `TOUCH_LIMIT` | `428` | `+17.1` | `+0.4` | `8.0` | **`+9.4`** | `+46.5` | `54.7%` | `8.6%` | `+27.6` | **`+7.2`** | **`-26.1`** | `-3.5` | `20%` | `-0.06` | **`—`** |
| `C1A_POST_SETTLE_HARVESTER_SHORT` | `TAKER_OPEN` | `2130` | `+10.0` | `+0.0` | `12.8` | **`-2.8`** | `+32.0` | `61.5%` | `18.1%` | `-26.7` | **`+1.6`** | **`+8.7`** | `-6.9` | `33%` | `-0.06` | **`—`** |
| `C1C_POST_SETTLE_SHORT_PANIC_LONG` | `TAKER_OPEN` | `52` | `-4.6` | `+2.1` | `12.9` | **`-15.5`** | `+9.7` | `51.9%` | `13.9%` | `+51.3` | **`-32.7`** | **`-22.1`** | `-23.9` | `20%` | `+0.02` | **`—`** |
| `C4_CLOCK_COMPRESSION_SQUEEZE_LONG` | `GTX_1.5BPS` | `52` | `-7.6` | `+1.5` | `8.0` | **`-14.1`** | `-248.5` | `44.2%` | `6.3%` | `+179.6` | **`-18.5`** | **`-41.9`** | `-38.6` | `20%` | `-0.03` | **`—`** |
| `C4_CLOCK_COMPRESSION_SQUEEZE_LONG` | `TOUCH_LIMIT` | `52` | `-7.8` | `+1.5` | `8.0` | **`-14.2`** | `-248.5` | `44.2%` | `6.3%` | `+179.6` | **`-18.7`** | **`-42.0`** | `-38.8` | `20%` | `-0.03` | **`—`** |
| `C4_CLOCK_COMPRESSION_SQUEEZE_LONG` | `TAKER_OPEN` | `53` | `-6.3` | `+1.5` | `12.5` | **`-17.3`** | `-253.0` | `45.3%` | `9.9%` | `+174.6` | **`-23.5`** | **`-42.0`** | `-35.8` | `20%` | `-0.03` | **`—`** |
| `C1C_POST_SETTLE_SHORT_PANIC_LONG` | `GTX_1.5BPS` | `50` | `-16.0` | `+2.1` | `8.4` | **`-22.2`** | `-183.0` | `48.0%` | `9.4%` | `+71.8` | **`-45.2`** | **`-33.5`** | `-29.8` | `13%` | `+0.02` | **`—`** |
| `C1C_POST_SETTLE_SHORT_PANIC_LONG` | `TOUCH_LIMIT` | `50` | `-16.3` | `+2.1` | `8.4` | **`-22.5`** | `-183.0` | `48.0%` | `9.5%` | `+71.4` | **`-45.6`** | **`-33.6`** | `-29.9` | `13%` | `+0.02` | **`—`** |
| `C5_SOVEREIGN_CASH_WAVE_IGNITION` | `TAKER_OPEN` | `1058` | `+13.0` | `+0.0` | `12.7` | **`+0.3`** | `+32.0` | `56.7%` | `20.4%` | `+5.8` | **`-54.2`** | **`+26.1`** | `-16.1` | `33%` | `+0.06` | **`—`** |
| `C5_SOVEREIGN_CASH_WAVE_IGNITION` | `TOUCH_LIMIT` | `987` | `+7.8` | `+0.0` | `8.2` | **`-0.4`** | `+36.5` | `54.9%` | `13.8%` | `+5.9` | **`-56.2`** | **`+17.6`** | `-5.5` | `33%` | `+0.08` | **`—`** |
| `C5_SOVEREIGN_CASH_WAVE_IGNITION` | `GTX_1.5BPS` | `964` | `+7.1` | `+0.0` | `8.2` | **`-1.1`** | `+36.5` | `54.4%` | `13.7%` | `+5.8` | **`-62.3`** | **`+14.7`** | `-6.5` | `40%` | `+0.07` | **`—`** |
| `C3B_SPOT_ANCHORED_BLOWOFF_SHORT` | `TAKER_OPEN` | `142` | `-27.4` | `-3.1` | `12.7` | **`-43.1`** | `-183.0` | `43.7%` | `19.4%` | `-26.5` | **`-76.7`** | **`+20.1`** | `-28.4` | `40%` | `+0.08` | **`—`** |
| `C3B_SPOT_ANCHORED_BLOWOFF_SHORT` | `GTX_1.5BPS` | `139` | `-32.7` | `-3.1` | `8.2` | **`-44.0`** | `-178.5` | `42.4%` | `13.2%` | `-21.3` | **`-86.1`** | **`+32.0`** | `-32.6` | `33%` | `+0.08` | **`—`** |
| `C3B_SPOT_ANCHORED_BLOWOFF_SHORT` | `TOUCH_LIMIT` | `139` | `-32.8` | `-3.1` | `8.2` | **`-44.2`** | `-178.5` | `42.4%` | `13.3%` | `-21.5` | **`-86.2`** | **`+31.9`** | `-32.6` | `33%` | `+0.08` | **`—`** |
| `C2B_TRAPPED_LONGS_WATERFALL_SHORT` | `TOUCH_LIMIT` | `17` | `-28.5` | `+1.0` | `8.2` | **`-35.7`** | `-218.5` | `41.2%` | `8.6%` | `+10.7` | **`-102.1`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |
| `C2B_TRAPPED_LONGS_WATERFALL_SHORT` | `GTX_1.5BPS` | `16` | `-36.0` | `+0.7` | `8.2` | **`-43.5`** | `-218.5` | `37.5%` | `8.6%` | `+2.1` | **`-102.1`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |
| `C2B_TRAPPED_LONGS_WATERFALL_SHORT` | `TAKER_OPEN` | `17` | `-28.6` | `+0.9` | `12.7` | **`-40.4`** | `-223.0` | `41.2%` | `13.3%` | `+6.0` | **`-106.6`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |

## Mechanism Specifications
- **`M00_NAIVE_M25_CONTROL`** (`Control_M16_6_Baseline`): Unfiltered M25 post-settle SHORT (T+1m, FR <= -20 bps, hold<=54m, :56 guard)
- **`C1A_POST_SETTLE_HARVESTER_SHORT`** (`Pillar_1_PostFunding_Bifurcation_StateA`): State-A Pure Harvester Overhang SHORT (vetoes deep Spot premium, requires Perp basis markup + :00 dump confirm)
- **`C1B_PRE_SETTLE_SPOT_SCARCITY_LONG`** (`Pillar_1_PostFunding_Bifurcation_StateB`): State-B Pre-Settle (T-45m) LONG when Spot >> Perp (Basis <= -45 bps) & FR <= -30 bps (collects +funding + squeeze)
- **`C1C_POST_SETTLE_SHORT_PANIC_LONG`** (`Pillar_1_PostFunding_Bifurcation_StateB`): State-B Post-Settle (T+1m) LONG when Spot >> Perp (Basis <= -45 bps, FR <= -25 bps) & :00 candle refuses to dump
- **`C2A_TRAPPED_SHORTS_PAIN_LONG`** (`Pillar_2_Trapped_OI_Pain_Tomography`): Recursive OI-VWAP Trapped Shorts Detonation LONG (OI >= 1.18x 24h, Pain >= +2.2 ATR, FR <= -8 bps, 4h High Break + Spot Flow)
- **`C2B_TRAPPED_LONGS_WATERFALL_SHORT`** (`Pillar_2_Trapped_OI_Pain_Tomography`): Recursive OI-VWAP Trapped Retail Longs Waterfall SHORT (OI >= 1.18x 24h, Pain >= +2.2 ATR below OI-VWAP, FR >= +8 bps, 4h Low Break)
- **`C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG`** (`Pillar_3_Spot_Perp_Liquidation_Elasticity`): 1m Post-Liquidation Snap-Back LONG (10m Perp drop <= -2.5%, 15m OI flush <= -2.5%, Perp dislocated <= -25 bps below Spot + 1m reversal)
- **`C3B_SPOT_ANCHORED_BLOWOFF_SHORT`** (`Pillar_3_Spot_Perp_Liquidation_Elasticity`): 1m Post-Short-Liquidation Blow-Off SHORT (10m Perp spike >= +3.0%, 15m OI flush <= -2.5%, Perp dislocated >= +30 bps above Spot + 1m turn)
- **`C4_CLOCK_COMPRESSION_SQUEEZE_LONG`** (`Pillar_4_Exchange_Clock_Phase_Transition`): Discrete Binance Funding-Clock Compression Transition (8h -> 4h/2h/1h) LONG when FR <= -20 bps & Spot >= Perp
- **`C5_SOVEREIGN_CASH_WAVE_IGNITION`** (`Pillar_5_Event_Time_Propagation_Wave`): Sovereign 3m BTC Spot+Perp Impulse (>=0.75% on >=3.5x RVOL & >=62% Spot Taker) into lagging high-beta perps
