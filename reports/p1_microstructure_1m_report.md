# Phase-P1 Next-Gen Microstructure (`L2 bookDepth` + `1m PremiumIndex` + `Hyperliquid L1`) & Convex Monetization Lab

- **Universe Scanned:** `220` Point-in-Time Perps (`4,495` simulated 1m trades, `1514.5s` runtime).
- **Monetization Geometries Compared:** `MONO_A_CONVEX_TRAIL` (unchoked right-tail) vs `MONO_B_COUPON_VACUUM` (exact `:00:00` coupon bank + conditional L2 continuation) vs `MONO_C_SPLIT_50_50` (`50%` Maker TP @ `+1.8%` + `50%` risk-free runner @ `+9.0%`).
- **Strict Treasure Survivors (`>= +25 bps` Full, `>= +15 bps` in `23-24` & `2025`, `>= +20 bps` in `2026`, `Fee Share <= 30%`, True In-Fold CPCV `>= 80%`):** **`0`**

| Mechanism | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | `2025` | `2026` | In-Fold CPCV bps | CPCV`%` | BTC Corr | Treasure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P6A_L2_ASK_AIR_POCKET_LONG` | `MONO_B_COUPON_VACUUM` | `148` | `+64.0` | `+0.0` | `12.5` | **`+51.5`** | `-1.0` | `50.0%` | `8.2%` | `+57.3` | **`+4.5`** | **`+154.6`** | `+40.8` | `100%` | `+0.01` | **`—`** |
| `P6A_L2_ASK_AIR_POCKET_LONG` | `MONO_A_CONVEX_TRAIL` | `148` | `+73.2` | `+0.0` | `12.7` | **`+60.5`** | `-14.8` | `50.0%` | `7.8%` | `+68.1` | **`+3.7`** | **`+181.8`** | `+58.3` | `100%` | `+0.01` | **`—`** |
| `P3_C3A_BOOK_REPLENISHED_SNAP_LONG` | `MONO_A_CONVEX_TRAIL` | `369` | `+16.0` | `+1.9` | `12.9` | **`+5.0`** | `-51.6` | `47.2%` | `13.2%` | `+6.3` | **`+2.9`** | **`+3.9`** | `+5.7` | `73%` | `+0.01` | **`—`** |
| `P3_C3A_BOOK_REPLENISHED_SNAP_LONG` | `MONO_B_COUPON_VACUUM` | `369` | `+14.0` | `+1.8` | `12.8` | **`+3.0`** | `-41.2` | `46.9%` | `13.5%` | `+4.1` | **`+3.6`** | **`-1.6`** | `+26.5` | `67%` | `+0.01` | **`—`** |
| `B1_C1B_BASE_UNCHOKED_LONG` | `MONO_B_COUPON_VACUUM` | `187` | `-3.9` | `+28.1` | `12.8` | **`+11.5`** | `-0.5` | `49.7%` | `11.2%` | `+36.6` | **`+17.1`** | **`-3.0`** | `+21.5` | `73%` | `-0.04` | **`—`** |
| `P6A_L2_ASK_AIR_POCKET_LONG` | `MONO_C_SPLIT_50_50` | `148` | `+44.1` | `+0.7` | `11.7` | **`+33.1`** | `+61.8` | `54.1%` | `9.2%` | `+35.3` | **`-4.1`** | **`+129.6`** | `+33.3` | `100%` | `+0.00` | **`—`** |
| `B2_C2A_BASE_UNCHOKED_LONG` | `MONO_A_CONVEX_TRAIL` | `404` | `+31.2` | `+3.9` | `8.3` | **`+26.7`** | `-38.2` | `48.5%` | `6.1%` | `-4.7` | **`+16.1`** | **`+98.5`** | `+28.4` | `67%` | `+0.01` | **`—`** |
| `P3_C3A_BOOK_REPLENISHED_SNAP_LONG` | `MONO_C_SPLIT_50_50` | `369` | `+8.1` | `+1.6` | `12.1` | **`-2.4`** | `-4.0` | `49.6%` | `14.3%` | `-2.8` | **`-5.4`** | **`+3.9`** | `-2.7` | `67%` | `+0.00` | **`—`** |
| `B2_C2A_BASE_UNCHOKED_LONG` | `MONO_C_SPLIT_50_50` | `404` | `+20.9` | `+2.4` | `7.2` | **`+16.2`** | `+66.2` | `56.7%` | `6.6%` | `-6.0` | **`+16.8`** | **`+55.1`** | `+16.8` | `73%` | `+0.01` | **`—`** |
| `P1_C1B_GLOBAL_SCARCITY_LONG` | `MONO_B_COUPON_VACUUM` | `226` | `+0.4` | `+22.9` | `12.7` | **`+10.6`** | `-58.5` | `45.1%` | `10.9%` | `+109.1` | **`+5.5`** | **`-6.1`** | `+23.3` | `67%` | `+0.01` | **`—`** |
| `B1_C1B_BASE_UNCHOKED_LONG` | `MONO_C_SPLIT_50_50` | `187` | `+10.4` | `+19.4` | `11.8` | **`+17.9`** | `+61.8` | `56.1%` | `10.4%` | `+50.4` | **`+35.8`** | **`-7.3`** | `+20.6` | `73%` | `-0.03` | **`—`** |
| `P1_C1B_GLOBAL_SCARCITY_LONG` | `MONO_A_CONVEX_TRAIL` | `226` | `+1.9` | `+23.0` | `12.9` | **`+11.9`** | `-128.5` | `45.6%` | `10.4%` | `+101.8` | **`+15.0`** | **`-10.2`** | `+20.1` | `67%` | `+0.01` | **`—`** |
| `B1_C1B_BASE_UNCHOKED_LONG` | `MONO_A_CONVEX_TRAIL` | `187` | `+1.5` | `+28.2` | `12.9` | **`+16.8`** | `+0.5` | `50.3%` | `10.3%` | `+45.4` | **`+40.5`** | **`-10.3`** | `+21.4` | `67%` | `-0.03` | **`—`** |
| `B2_C2A_BASE_UNCHOKED_LONG` | `MONO_B_COUPON_VACUUM` | `412` | `+32.1` | `+3.2` | `12.7` | **`+22.7`** | `-16.4` | `49.3%` | `9.5%` | `-10.6` | **`+12.5`** | **`+97.4`** | `+35.5` | `67%` | `+0.01` | **`—`** |
| `P2_C2A_BOOK_VACUUM_PAIN_LONG` | `MONO_A_CONVEX_TRAIL` | `117` | `+35.0` | `+2.2` | `8.3` | **`+29.0`** | `-149.0` | `47.9%` | `6.0%` | `+56.1` | **`-10.8`** | **`+63.6`** | `+56.2` | `100%` | `-0.01` | **`—`** |
| `P2_C2A_BOOK_VACUUM_PAIN_LONG` | `MONO_C_SPLIT_50_50` | `117` | `+22.2` | `+2.5` | `7.2` | **`+17.5`** | `+66.2` | `53.8%` | `6.2%` | `+34.9` | **`-11.7`** | **`+45.5`** | `+31.6` | `80%` | `-0.00` | **`—`** |
| `P2_C2A_BOOK_VACUUM_PAIN_LONG` | `MONO_B_COUPON_VACUUM` | `123` | `+37.9` | `+1.9` | `12.6` | **`+27.1`** | `-27.9` | `48.8%` | `9.2%` | `+44.0` | **`-18.9`** | **`+79.3`** | `+59.3` | `80%` | `+0.00` | **`—`** |
| `P1_C1B_GLOBAL_SCARCITY_LONG` | `MONO_C_SPLIT_50_50` | `225` | `-1.5` | `+14.8` | `11.9` | **`+1.4`** | `-21.9` | `49.8%` | `11.3%` | `+74.8` | **`+10.5`** | **`-22.5`** | `-7.4` | `47%` | `+0.01` | **`—`** |
| `P1_C1D_LOCAL_ARB_DISPERSION_SHORT` | `MONO_A_CONVEX_TRAIL` | `2` | `-180.0` | `+0.0` | `13.0` | **`-193.0`** | `-193.0` | `0.0%` | `100.0%` | `-193.0` | **`-193.0`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |
| `P1_C1D_LOCAL_ARB_DISPERSION_SHORT` | `MONO_B_COUPON_VACUUM` | `2` | `-180.0` | `+0.0` | `13.0` | **`-193.0`** | `-193.0` | `0.0%` | `100.0%` | `-193.0` | **`-193.0`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |
| `P1_C1D_LOCAL_ARB_DISPERSION_SHORT` | `MONO_C_SPLIT_50_50` | `2` | `-180.0` | `+0.0` | `13.0` | **`-193.0`** | `-193.0` | `0.0%` | `100.0%` | `-193.0` | **`-193.0`** | **`+0.0`** | `+0.0` | `0%` | `+0.03` | **`—`** |
| `P6B_L2_BID_VACUUM_WATERFALL_SHORT` | `MONO_A_CONVEX_TRAIL` | `8` | `+168.3` | `+0.0` | `11.9` | **`+156.4`** | `-51.7` | `50.0%` | `4.7%` | `-203.0` | **`+331.0`** | **`-100.5`** | `+0.0` | `0%` | `+0.00` | **`—`** |
| `P6B_L2_BID_VACUUM_WATERFALL_SHORT` | `MONO_C_SPLIT_50_50` | `8` | `+137.1` | `+0.0` | `11.3` | **`+125.8`** | `-41.5` | `50.0%` | `4.9%` | `-203.0` | **`+258.5`** | **`-41.5`** | `+0.0` | `0%` | `+0.01` | **`—`** |
| `P6B_L2_BID_VACUUM_WATERFALL_SHORT` | `MONO_B_COUPON_VACUUM` | `8` | `+121.5` | `+0.0` | `11.9` | **`+109.6`** | `-51.7` | `50.0%` | `5.8%` | `-203.0` | **`+256.2`** | **`-100.5`** | `+0.0` | `0%` | `+0.00` | **`—`** |
| `P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG` | `MONO_C_SPLIT_50_50` | `32` | `+0.3` | `+7.4` | `7.4` | **`+0.3`** | `+4.7` | `50.0%` | `7.8%` | `-208.5` | **`+7.0`** | **`+0.0`** | `+0.0` | `33%` | `-0.02` | **`—`** |
| `P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG` | `MONO_A_CONVEX_TRAIL` | `32` | `-7.4` | `+8.1` | `8.5` | **`-7.8`** | `-172.4` | `37.5%` | `7.7%` | `-208.5` | **`-1.3`** | **`+0.0`** | `-1.3` | `0%` | `-0.01` | **`—`** |
| `P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG` | `MONO_B_COUPON_VACUUM` | `35` | `-32.2` | `+7.3` | `12.9` | **`-37.8`** | `-54.9` | `37.1%` | `21.9%` | `-213.0` | **`-32.7`** | **`+0.0`** | `-32.7` | `0%` | `-0.02` | **`—`** |

## Mechanism Profiles
- **`B1_C1B_BASE_UNCHOKED_LONG`** (`Ablation_Control_C1B`): Baseline C1B (T-45m Pre-Settle Spot>>Perp Long, no Book/HL gate) under unchoked monetization
- **`P1_C1B_GLOBAL_SCARCITY_LONG`** (`MEC01B_MultiEx_Carry_Squeeze`): T-45m Pre-Settle Long using 220-sym 1m PremiumIndex <= -75 bps + HL Cross-Venue Sync + L2 Ask-Wall Veto
- **`P1_C1D_LOCAL_ARB_DISPERSION_SHORT`** (`MEC01A_CrossVenue_Harvester_Unwind`): T+1m Post-Settle SHORT ONLY when Binance FR << Hyperliquid FR (local arb overhang) + :00 dump + L2 Ask pressure
- **`B2_C2A_BASE_UNCHOKED_LONG`** (`Ablation_Control_C2A`): Baseline C2A (Contract OI-VWAP Trapped Shorts Pain >= 1.8 ATR, no L2 Book gate) under unchoked monetization
- **`P2_C2A_BOOK_VACUUM_PAIN_LONG`** (`MEC02_MEC06_OIPain_L2AskVacuum`): Contract OI-VWAP Pain >= 1.8 ATR + 220-sym Premium <= -10 bps + L2 Book Ask-Vacuum Gate (vetoes Ask walls)
- **`P3_C3A_BOOK_REPLENISHED_SNAP_LONG`** (`MEC03_LiqFlush_L2BidReplenish`): 15m Contract-OI Flush <= -2% + 1m Impact Premium Low <= -40 bps + L2 Book Bid-Replenishment + Crash Guard
- **`P6A_L2_ASK_AIR_POCKET_LONG`** (`MEC06_Pure_L2_OrderBook_Vacuum`): Pure L2 Book Air-Pocket Breakout LONG (OI >= 1.16x, FR <= +3 bps, 4h High Break + L2 Bid/Ask Imb >= +0.24)
- **`P6B_L2_BID_VACUUM_WATERFALL_SHORT`** (`MEC06_Pure_L2_OrderBook_Vacuum`): Symmetric L2 Bid-Vacuum Waterfall SHORT (OI >= 1.16x, Underwater Longs <= -1.4 ATR, 4h Low Break + L2 Imb <= -0.24)
- **`P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG`** (`MEC07_OnChain_Hyperliquid_CoSqueeze`): Synchronized On-Chain Hyperliquid (<= -3.5 bps/h) + Binance Impact Discount (<= -35 bps) 4h Breakout Squeeze LONG
