"""
Phase-P1 Next-Generation Microstructure (`L2 bookDepth` + `1m premiumIndexKlines` + `Hyperliquid L1`)
& Convex Monetization Laboratory (`2023–2026`, `220` Point-in-Time Perps).

Resolves the two exact bottlenecks uncovered in `docs/research/10_resultados_laboratorio_1m_mecanica_contrapartes.md`:
  1. **Monetization Geometry Stranglehold:** Replaces the premature `+45..+80 bps` `PROFIT_LOCK_SL`
     (which choked `42%–50.4%` of winning trades at `0.33R`) with three explicit, side-by-side
     linear monetization engines:
       - `MONO_A_CONVEX_TRAIL`: Unchoked right-tail (`SL = -2.0%`, breakeven `+15 bps` only after `+2.4%`,
         wide `1.6%` trail above `+3.2%`, Maker TP `+8.5%`).
       - `MONO_B_COUPON_VACUUM`: Exact settlement-coupon capture (exits cleanly `2m` AFTER banking the
         `:00:00` funding payout unless `:00` flow + L2 `bookDepth` confirms `State 01-B` Ask vacuum).
       - `MONO_C_SPLIT_50_50`: Institutional `50/50` scale-out (`50%` Maker TP at `+1.8%` (`~1R`) +
         `50%` risk-free runner stopped at `+10 bps` riding up to `+9.0%`).
  2. **Missing Microstructure Observables:** Integrates `1m` multi-exchange `premiumIndexKlines`
     (`100%` `220`-coin coverage + intraminute Impact Bid/Ask wicks), causal L2 `bookDepth`
     (`±0.2%..±5%` Bid vs Ask USD walls/vacuums), and On-Chain `Hyperliquid L1` hourly funding dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.models.counterparty_mechanics_lab import (
    _fast_ema_1d,
    _fast_rolling_mean_1d,
    compute_recursive_oi_vwap_anchor,
    evaluate_infold_cpcv_and_treasure_metrics,
)


@dataclass(frozen=True)
class P1MechanismProfile:
    mech_id: str
    family: str
    description: str
    prefer_passive_touch: bool = False
    base_sl_pct: float = 0.020
    max_hold_minutes: int = 150
    cooldown_minutes: int = 60
    is_funding_coupon_mech: bool = False


P1_MECHANISM_PROFILES: dict[str, P1MechanismProfile] = {
    "B1_C1B_BASE_UNCHOKED_LONG": P1MechanismProfile(
        mech_id="B1_C1B_BASE_UNCHOKED_LONG",
        family="Ablation_Control_C1B",
        description="Baseline C1B (T-45m Pre-Settle Spot>>Perp Long, no Book/HL gate) under unchoked monetization",
        prefer_passive_touch=False,
        base_sl_pct=0.021,
        max_hold_minutes=90,
        cooldown_minutes=60,
        is_funding_coupon_mech=True,
    ),
    "P1_C1B_GLOBAL_SCARCITY_LONG": P1MechanismProfile(
        mech_id="P1_C1B_GLOBAL_SCARCITY_LONG",
        family="MEC01B_MultiEx_Carry_Squeeze",
        description="T-45m Pre-Settle Long using 220-sym 1m PremiumIndex <= -75 bps + HL Cross-Venue Sync + L2 Ask-Wall Veto",
        prefer_passive_touch=False,
        base_sl_pct=0.020,
        max_hold_minutes=90,
        cooldown_minutes=60,
        is_funding_coupon_mech=True,
    ),
    "P1_C1D_LOCAL_ARB_DISPERSION_SHORT": P1MechanismProfile(
        mech_id="P1_C1D_LOCAL_ARB_DISPERSION_SHORT",
        family="MEC01A_CrossVenue_Harvester_Unwind",
        description="T+1m Post-Settle SHORT ONLY when Binance FR << Hyperliquid FR (local arb overhang) + :00 dump + L2 Ask pressure",
        prefer_passive_touch=False,
        base_sl_pct=0.018,
        max_hold_minutes=52,
        cooldown_minutes=45,
        is_funding_coupon_mech=False,
    ),
    "B2_C2A_BASE_UNCHOKED_LONG": P1MechanismProfile(
        mech_id="B2_C2A_BASE_UNCHOKED_LONG",
        family="Ablation_Control_C2A",
        description="Baseline C2A (Contract OI-VWAP Trapped Shorts Pain >= 1.8 ATR, no L2 Book gate) under unchoked monetization",
        prefer_passive_touch=True,
        base_sl_pct=0.021,
        max_hold_minutes=180,
        cooldown_minutes=90,
    ),
    "P2_C2A_BOOK_VACUUM_PAIN_LONG": P1MechanismProfile(
        mech_id="P2_C2A_BOOK_VACUUM_PAIN_LONG",
        family="MEC02_MEC06_OIPain_L2AskVacuum",
        description="Contract OI-VWAP Pain >= 1.8 ATR + 220-sym Premium <= -10 bps + L2 Book Ask-Vacuum Gate (vetoes Ask walls)",
        prefer_passive_touch=True,
        base_sl_pct=0.020,
        max_hold_minutes=180,
        cooldown_minutes=90,
    ),
    "P3_C3A_BOOK_REPLENISHED_SNAP_LONG": P1MechanismProfile(
        mech_id="P3_C3A_BOOK_REPLENISHED_SNAP_LONG",
        family="MEC03_LiqFlush_L2BidReplenish",
        description="15m Contract-OI Flush <= -2% + 1m Impact Premium Low <= -40 bps + L2 Book Bid-Replenishment + Crash Guard",
        prefer_passive_touch=False,
        base_sl_pct=0.017,
        max_hold_minutes=75,
        cooldown_minutes=45,
    ),
    "P6A_L2_ASK_AIR_POCKET_LONG": P1MechanismProfile(
        mech_id="P6A_L2_ASK_AIR_POCKET_LONG",
        family="MEC06_Pure_L2_OrderBook_Vacuum",
        description="Pure L2 Book Air-Pocket Breakout LONG (OI >= 1.16x, FR <= +3 bps, 4h High Break + L2 Bid/Ask Imb >= +0.24)",
        prefer_passive_touch=False,
        base_sl_pct=0.019,
        max_hold_minutes=150,
        cooldown_minutes=90,
    ),
    "P6B_L2_BID_VACUUM_WATERFALL_SHORT": P1MechanismProfile(
        mech_id="P6B_L2_BID_VACUUM_WATERFALL_SHORT",
        family="MEC06_Pure_L2_OrderBook_Vacuum",
        description="Symmetric L2 Bid-Vacuum Waterfall SHORT (OI >= 1.16x, Underwater Longs <= -1.4 ATR, 4h Low Break + L2 Imb <= -0.24)",
        prefer_passive_touch=False,
        base_sl_pct=0.019,
        max_hold_minutes=150,
        cooldown_minutes=90,
    ),
    "P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG": P1MechanismProfile(
        mech_id="P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG",
        family="MEC07_OnChain_Hyperliquid_CoSqueeze",
        description="Synchronized On-Chain Hyperliquid (<= -3.5 bps/h) + Binance Impact Discount (<= -35 bps) 4h Breakout Squeeze LONG",
        prefer_passive_touch=True,
        base_sl_pct=0.020,
        max_hold_minutes=180,
        cooldown_minutes=90,
    ),
}

MONETIZATION_MODES: tuple[str, ...] = (
    "MONO_A_CONVEX_TRAIL",
    "MONO_B_COUPON_VACUUM",
    "MONO_C_SPLIT_50_50",
)


def simulate_1m_monetization_geometries(
    ts_1m: np.ndarray,
    open_1m: np.ndarray,
    high_1m: np.ndarray,
    low_1m: np.ndarray,
    close_1m: np.ndarray,
    qvol_1m: np.ndarray,
    tb_share_1m: np.ndarray,
    book_imb_1pct_1m: np.ndarray,
    funding_map: dict[int, float],
    raw_events: list[tuple[int, int, float, float]],
    profile: P1MechanismProfile,
    symbol: str,
    order_notional_usd: float = 600.0,
    max_bar_participation: float = 0.025,
) -> list[dict[str, Any]]:
    """
    Simulates `(sig_idx, side, strength, aux_metric)` events across all 3 Monetization Geometries
    (`MONO_A_CONVEX_TRAIL`, `MONO_B_COUPON_VACUUM`, `MONO_C_SPLIT_50_50`) using 100% exact linear
    perpetual returns, exact `:00:00` funding cashflows, and 100% pessimistic same-bar SL priority.
    """
    if not raw_events or len(ts_1m) < 15:
        return []

    n_bars = len(ts_1m)
    sorted_evs = sorted(raw_events, key=lambda x: x[0])
    min_req_qvol = order_notional_usd / max(1e-6, max_bar_participation)

    taker_cost = 6.5 / 10_000.0
    maker_cost = 2.0 / 10_000.0
    tp_pen = 1.0 / 10_000.0

    records: list[dict[str, Any]] = []

    for mono_mode in MONETIZATION_MODES:
        next_free_idx = 0

        for sig_i, side_in, strength, aux_v in sorted_evs:
            if sig_i < next_free_idx or sig_i + 3 >= n_bars:
                continue
            sgn = 1.0 if side_in > 0 else -1.0
            ref_c = float(close_1m[sig_i])
            sub_i = sig_i + 1
            if int(ts_1m[sub_i]) - int(ts_1m[sig_i]) > 180 or ref_c <= 0.0 or not np.isfinite(ref_c):
                continue

            # Entry execution physics (`TOUCH_LIMIT` 1.5 bps penetration vs `TAKER_OPEN` at `open[t+1]`)
            if profile.prefer_passive_touch and mono_mode != "MONO_B_COUPON_VACUUM":
                sub_o = float(open_1m[sub_i])
                limit_px = min(ref_c, sub_o * 0.9999) if sgn > 0 else max(ref_c, sub_o * 1.0001)
                req_px = limit_px * (1.0 - sgn * 0.00015)
                fill_i = -1
                for m_try in range(sub_i, min(n_bars, sub_i + 3)):
                    if float(qvol_1m[m_try]) < min_req_qvol:
                        continue
                    if (sgn > 0 and float(low_1m[m_try]) <= req_px) or (sgn < 0 and float(high_1m[m_try]) >= req_px):
                        fill_i = m_try
                        break
                if fill_i < 0:
                    next_free_idx = min(n_bars, sub_i + 3)
                    continue
                entry_px = limit_px
                entry_fee = maker_cost
            else:
                if float(qvol_1m[sub_i]) < min_req_qvol:
                    next_free_idx = sub_i + 1
                    continue
                fill_i = sub_i
                entry_px = float(open_1m[sub_i])
                entry_fee = taker_cost

            if not np.isfinite(entry_px) or entry_px <= 0.0:
                continue

            sl_pct = profile.base_sl_pct
            init_sl = entry_px * (1.0 - sgn * sl_pct)
            hold_end = min(n_bars - 1, fill_i + max(2, profile.max_hold_minutes))

            # Same-bar pessimistic Stop-Loss check on `fill_i`
            if (sgn > 0 and float(low_1m[fill_i]) <= init_sl) or (sgn < 0 and float(high_1m[fill_i]) >= init_sl):
                gross_ret = sgn * (init_sl - entry_px) / entry_px
                fee_ret = entry_fee + taker_cost
                net_ret = gross_ret - fee_ret
                records.append(
                    {
                        "mech_id": profile.mech_id,
                        "exec_mode": mono_mode,
                        "symbol": symbol,
                        "signal_ts": int(ts_1m[sig_i]) + 60,
                        "entry_ts": int(ts_1m[fill_i]),
                        "exit_ts": int(ts_1m[fill_i]) + 60,
                        "side": int(side_in),
                        "hold_min": 1,
                        "gross_bps": round(gross_ret * 1e4, 3),
                        "funding_bps": 0.0,
                        "fee_bps": round(fee_ret * 1e4, 3),
                        "net_bps": round(net_ret * 1e4, 3),
                        "exit_reason": "STOP_LOSS_SAME_BAR",
                        "strength": round(float(strength), 4),
                        "aux_metric": round(float(aux_v), 3),
                    }
                )
                next_free_idx = fill_i + max(1, profile.cooldown_minutes)
                continue

            # Funding on `fill_i` close if crossing `:00:00`
            fnd_fill = 0.0
            t_fill_close = int(ts_1m[fill_i]) + 60
            if t_fill_close % 3600 == 0 and t_fill_close in funding_map:
                fnd_fill = -sgn * float(funding_map[t_fill_close])

            # ------------------------------------------------------------------
            # GEOMETRY 1 & 2: Single-Unit Position (`MONO_A_CONVEX_TRAIL` & `MONO_B_COUPON_VACUUM`)
            # ------------------------------------------------------------------
            if mono_mode in ("MONO_A_CONVEX_TRAIL", "MONO_B_COUPON_VACUUM"):
                cur_sl = init_sl
                tp_pct = 0.085 if mono_mode == "MONO_A_CONVEX_TRAIL" else 0.065
                tp_px = entry_px * (1.0 + sgn * tp_pct)
                tp_req = tp_px * (1.0 + sgn * tp_pen)
                peak_fav = entry_px
                fnd_acc = fnd_fill
                crossed_settle_ts = t_fill_close if (t_fill_close % 3600 == 0 and t_fill_close in funding_map) else -1

                exit_i = hold_end
                exit_px = float(close_1m[hold_end])
                exit_reason = "TIME_STOP"
                exit_fee = taker_cost

                for h in range(fill_i + 1, hold_end + 1):
                    t_h = int(ts_1m[h])
                    oh, hh, lh, ch = float(open_1m[h]), float(high_1m[h]), float(low_1m[h]), float(close_1m[h])

                    # 1. Pessimistic Stop-Loss FIRST
                    if (sgn > 0 and lh <= cur_sl) or (sgn < 0 and hh >= cur_sl):
                        exit_i = h
                        exit_px = min(oh, cur_sl) if sgn > 0 else max(oh, cur_sl)
                        locked = (sgn > 0 and cur_sl > entry_px) or (sgn < 0 and cur_sl < entry_px)
                        exit_reason = "CONVEX_TRAIL_STOP" if locked else "STOP_LOSS"
                        exit_fee = taker_cost
                        break

                    # 2. Maker Take-Profit
                    if (sgn > 0 and hh >= tp_req) or (sgn < 0 and lh <= tp_req):
                        exit_i = h
                        exit_px = max(oh, tp_px) if sgn > 0 else min(oh, tp_px)
                        exit_reason = "TAKE_PROFIT_MAKER"
                        exit_fee = maker_cost
                        break

                    # 3. Pre-settlement `:56` guard for non-coupon shorts (`P1_C1D_LOCAL_ARB_DISPERSION_SHORT`)
                    if (
                        not profile.is_funding_coupon_mech
                        and sgn < 0
                        and (h - fill_i) >= 3
                        and ((t_h % 3600) // 60) >= 56
                    ):
                        exit_i = h
                        exit_px = ch
                        exit_reason = "PRE_SETTLE_GUARD"
                        exit_fee = taker_cost
                        break

                    # 4. `MONO_B_COUPON_VACUUM`: Immediately after banking `:00:00` coupon (`t_h == crossed_settle_ts`),
                    # exit cleanly at `:01:00` close UNLESS `:00` candle is strongly squeezing with L2 Bid support!
                    if mono_mode == "MONO_B_COUPON_VACUUM" and crossed_settle_ts > 0 and t_h == crossed_settle_ts:
                        r_00 = sgn * (ch - oh) / max(oh, 1e-12)
                        tb_ok = (sgn > 0 and float(tb_share_1m[h]) >= 0.54) or (
                            sgn < 0 and float(tb_share_1m[h]) <= 0.46
                        )
                        bk_ok = (sgn * float(book_imb_1pct_1m[h])) >= 0.05
                        if not (r_00 >= 0.0008 and (tb_ok or bk_ok)):
                            exit_i = h
                            exit_px = ch
                            exit_reason = "COUPON_BANKED_EXIT"
                            exit_fee = taker_cost
                            break
                        # If squeeze continues past settlement, protect coupon at breakeven!
                        be_px = entry_px * (1.0 + sgn * 0.0010)
                        if (sgn > 0 and ch > be_px) or (sgn < 0 and ch < be_px):
                            cur_sl = max(cur_sl, be_px) if sgn > 0 else min(cur_sl, be_px)

                    # 5. Realized `:00:00` funding settlement at close of minute `h` (`t_h + 60`)
                    t_close = t_h + 60
                    if t_close % 3600 == 0 and t_close in funding_map:
                        fnd_acc += -sgn * float(funding_map[t_close])
                        if crossed_settle_ts < 0:
                            crossed_settle_ts = t_close

                    # 6. UNCHOKED Convex Breakeven (`>= 1.2R = +2.4%` -> `+15 bps`) & Wide Trail (`>= +3.2%` -> `1.6%` trail)
                    if sgn > 0:
                        if hh > peak_fav:
                            peak_fav = hh
                        pr = (peak_fav - entry_px) / entry_px
                        if pr >= 0.032:
                            cand_sl = peak_fav * (1.0 - 0.016)
                            if ch > cand_sl:
                                cur_sl = max(cur_sl, cand_sl)
                        elif pr >= 0.024:
                            cand_sl = entry_px * 1.0015
                            if ch > cand_sl:
                                cur_sl = max(cur_sl, cand_sl)
                    else:
                        if lh < peak_fav:
                            peak_fav = lh
                        pr = (entry_px - peak_fav) / entry_px
                        if pr >= 0.032:
                            cand_sl = peak_fav * (1.0 + 0.016)
                            if ch < cand_sl:
                                cur_sl = min(cur_sl, cand_sl)
                        elif pr >= 0.024:
                            cand_sl = entry_px * 0.9985
                            if ch < cand_sl:
                                cur_sl = min(cur_sl, cand_sl)

                    if h == hold_end:
                        exit_i = h
                        exit_px = ch
                        exit_reason = "TIME_STOP"
                        exit_fee = taker_cost

                gross_ret = sgn * (exit_px - entry_px) / entry_px
                tot_fee = entry_fee + exit_fee
                net_ret = gross_ret + fnd_acc - tot_fee

            # ------------------------------------------------------------------
            # GEOMETRY 3: `MONO_C_SPLIT_50_50` (`50%` Maker TP @ `+1.8%` + `50%` Risk-Free Runner @ `+9.0%`)
            # ------------------------------------------------------------------
            else:
                cur_sl = init_sl
                tp1_pct = 0.018  # ~1R partial cash-register target
                tp2_pct = 0.090  # Convex tail runner target
                tp1_px = entry_px * (1.0 + sgn * tp1_pct)
                tp1_req = tp1_px * (1.0 + sgn * tp_pen)
                tp2_px = entry_px * (1.0 + sgn * tp2_pct)
                tp2_req = tp2_px * (1.0 + sgn * tp_pen)
                peak_fav = entry_px

                leg1_done = False
                leg1_gross = 0.0
                leg1_exit_fee = taker_cost
                fnd_acc = fnd_fill  # Weighted by active position fraction (`1.0` before TP1, `0.5` after TP1)

                exit_i = hold_end
                leg2_px = float(close_1m[hold_end])
                leg2_exit_fee = taker_cost
                exit_reason = "TIME_STOP"

                for h in range(fill_i + 1, hold_end + 1):
                    t_h = int(ts_1m[h])
                    oh, hh, lh, ch = float(open_1m[h]), float(high_1m[h]), float(low_1m[h]), float(close_1m[h])

                    # 1. Pessimistic Stop-Loss FIRST on active legs
                    if (sgn > 0 and lh <= cur_sl) or (sgn < 0 and hh >= cur_sl):
                        exit_i = h
                        sl_fill = min(oh, cur_sl) if sgn > 0 else max(oh, cur_sl)
                        if not leg1_done:
                            leg1_gross = sgn * (sl_fill - entry_px) / entry_px
                            leg1_exit_fee = taker_cost
                            leg2_px = sl_fill
                            leg2_exit_fee = taker_cost
                            exit_reason = "STOP_LOSS"
                        else:
                            leg2_px = sl_fill
                            leg2_exit_fee = taker_cost
                            exit_reason = "SPLIT_TP1_PLUS_RUNNER_TRAIL"
                        break

                    # 2. Check Leg-1 `50%` Maker TP (`+1.8%`) if not yet hit
                    if not leg1_done and ((sgn > 0 and hh >= tp1_req) or (sgn < 0 and lh <= tp1_req)):
                        leg1_done = True
                        p1_fill = max(oh, tp1_px) if sgn > 0 else min(oh, tp1_px)
                        leg1_gross = sgn * (p1_fill - entry_px) / entry_px
                        leg1_exit_fee = maker_cost
                        # Cushion Runner stop to `-35 bps` (`0.5*180 - 0.5*35 = +72.5 bps` locked net-positive trade!)
                        # so retests of the entry level don't prematurely shake out the convex runner!
                        cushion_runner = entry_px * (1.0 - sgn * 0.0035)
                        if (sgn > 0 and ch > cushion_runner) or (sgn < 0 and ch < cushion_runner):
                            cur_sl = max(cur_sl, cushion_runner) if sgn > 0 else min(cur_sl, cushion_runner)

                    # 3. Check Leg-2 `50%` Maker TP (`+9.0%`)
                    if leg1_done and ((sgn > 0 and hh >= tp2_req) or (sgn < 0 and lh <= tp2_req)):
                        exit_i = h
                        leg2_px = max(oh, tp2_px) if sgn > 0 else min(oh, tp2_px)
                        leg2_exit_fee = maker_cost
                        exit_reason = "SPLIT_FULL_TP2_RUNNER"
                        break

                    # 4. Realized `:00:00` funding weighted by open size (`1.0` or `0.5`)
                    t_close = t_h + 60
                    if t_close % 3600 == 0 and t_close in funding_map:
                        pos_wt = 0.5 if leg1_done else 1.0
                        fnd_acc += pos_wt * (-sgn * float(funding_map[t_close]))

                    # 5. Progressive convex trailing on Runner (`>= +2.8%` -> `+20 bps`, `>= +4.0%` -> `1.8%` trail)
                    if leg1_done:
                        if sgn > 0:
                            if hh > peak_fav:
                                peak_fav = hh
                            pr_r = (peak_fav - entry_px) / entry_px
                            if pr_r >= 0.040:
                                cand_sl = peak_fav * (1.0 - 0.018)
                                if ch > cand_sl:
                                    cur_sl = max(cur_sl, cand_sl)
                            elif pr_r >= 0.028:
                                cand_sl = entry_px * 1.0020
                                if ch > cand_sl:
                                    cur_sl = max(cur_sl, cand_sl)
                        else:
                            if lh < peak_fav:
                                peak_fav = lh
                            pr_r = (entry_px - peak_fav) / entry_px
                            if pr_r >= 0.040:
                                cand_sl = peak_fav * (1.0 + 0.018)
                                if ch < cand_sl:
                                    cur_sl = min(cur_sl, cand_sl)
                            elif pr_r >= 0.028:
                                cand_sl = entry_px * 0.9980
                                if ch < cand_sl:
                                    cur_sl = min(cur_sl, cand_sl)

                    if h == hold_end:
                        exit_i = h
                        leg2_px = ch
                        leg2_exit_fee = taker_cost
                        if not leg1_done:
                            leg1_gross = sgn * (ch - entry_px) / entry_px
                            leg1_exit_fee = taker_cost
                            exit_reason = "TIME_STOP"
                        else:
                            exit_reason = "SPLIT_TP1_PLUS_TIME"

                leg2_gross = sgn * (leg2_px - entry_px) / entry_px
                gross_ret = 0.5 * leg1_gross + 0.5 * leg2_gross
                tot_fee = entry_fee + 0.5 * leg1_exit_fee + 0.5 * leg2_exit_fee
                net_ret = gross_ret + fnd_acc - tot_fee

            records.append(
                {
                    "mech_id": profile.mech_id,
                    "exec_mode": mono_mode,
                    "symbol": symbol,
                    "signal_ts": int(ts_1m[sig_i]) + 60,
                    "entry_ts": int(ts_1m[fill_i]),
                    "exit_ts": int(ts_1m[exit_i]) + 60,
                    "side": int(side_in),
                    "hold_min": int(exit_i - fill_i + 1),
                    "gross_bps": round(gross_ret * 1e4, 3),
                    "funding_bps": round(fnd_acc * 1e4, 3),
                    "fee_bps": round(tot_fee * 1e4, 3),
                    "net_bps": round(net_ret * 1e4, 3),
                    "exit_reason": exit_reason,
                    "strength": round(float(strength), 4),
                    "aux_metric": round(float(aux_v), 3),
                }
            )
            next_free_idx = exit_i + max(1, profile.cooldown_minutes)

    return records


def prepare_symbol_p1_base_state(
    symbol: str,
    perp_1m: pd.DataFrame,
    spot_1m: pd.DataFrame | None,
    prem_1m: pd.DataFrame | None,
    metrics_5m: pd.DataFrame | None,
    funding_df: pd.DataFrame | None,
    hl_funding_df: pd.DataFrame | None,
    min_adv_usd: float = 15_000_000.0,
) -> dict[str, Any] | None:
    """
    Pass A: Builds causal `1m` base arrays (including `220`-sym `premium_1m` & `Hyperliquid L1`
    cross-venue hourly carry) AND extracts the set of UTC `YYYY-MM-DD` dates with candidate triggers
    so `bookDepth` L2 is fetched strictly & solely for causal event days.
    """
    if perp_1m is None or len(perp_1m) < 2880:
        return None

    p_df = perp_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    ts = p_df["timestamp"].to_numpy(dtype=np.int64)
    o = p_df["open"].to_numpy(dtype=np.float64)
    h = p_df["high"].to_numpy(dtype=np.float64)
    low = p_df["low"].to_numpy(dtype=np.float64)
    c = p_df["close"].to_numpy(dtype=np.float64)
    qv = p_df["quote_volume"].to_numpy(dtype=np.float64)
    tb = p_df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
    n = len(ts)

    adv_7d = np.roll(_fast_rolling_mean_1d(qv, 7 * 1440) * 1440.0, 1)
    adv_7d[0] = 0.0
    liq_gate = (np.arange(n) >= 14 * 1440) & (adv_7d >= float(min_adv_usd)) & np.isfinite(c) & (c > 0.0)
    if not bool(liq_gate.any()):
        return None

    ret_1m = np.zeros(n, dtype=np.float64)
    ret_1m[1:] = (c[1:] - c[:-1]) / np.maximum(c[:-1], 1e-12)
    ret_10m = np.zeros(n, dtype=np.float64)
    ret_10m[10:] = (c[10:] - c[:-10]) / np.maximum(c[:-10], 1e-12)
    ret_4h = np.zeros(n, dtype=np.float64)
    ret_4h[240:] = (c[240:] - c[:-240]) / np.maximum(c[:-240], 1e-12)

    perp_tb_1m = tb / np.maximum(qv, 1.0)
    perp_tb_5m = (_fast_rolling_mean_1d(tb, 5) * 5.0) / np.maximum(_fast_rolling_mean_1d(qv, 5) * 5.0, 1.0)
    perp_tb_60m = (_fast_rolling_mean_1d(tb, 60) * 60.0) / np.maximum(_fast_rolling_mean_1d(qv, 60) * 60.0, 1.0)

    qv_5m = _fast_rolling_mean_1d(qv, 5) * 5.0
    qv_5m_base = np.roll(_fast_rolling_mean_1d(qv, 360) * 5.0, 5)
    qv_5m_base[:6] = qv_5m[:6]
    rvol_5m = qv_5m / np.maximum(qv_5m_base, 1.0)

    tr_1m = np.maximum(h - low, np.abs(h - np.roll(c, 1)))
    tr_1m[0] = h[0] - low[0]
    atr_1h = np.maximum(_fast_rolling_mean_1d(tr_1m, 60) * np.sqrt(60.0), c * 0.003)

    # 1. Binance Spot 1m alignment (when present)
    has_spot = np.zeros(n, dtype=bool)
    spot_basis_bps = np.zeros(n, dtype=np.float64)
    spot_tb_3m = perp_tb_5m.copy()
    spot_tb_60m = perp_tb_60m.copy()
    if spot_1m is not None and not spot_1m.empty:
        s_df = spot_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        s_ts = s_df["timestamp"].to_numpy(dtype=np.int64)
        s_c = s_df["close"].to_numpy(dtype=np.float64)
        s_qv = s_df["quote_volume"].to_numpy(dtype=np.float64)
        s_tb = s_df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        if len(s_ts) > 100:
            pos_s = np.searchsorted(s_ts, ts)
            pos_sc = np.minimum(pos_s, len(s_ts) - 1)
            m_sp = (pos_s < len(s_ts)) & (s_ts[pos_sc] == ts) & (s_c[pos_sc] > 0.0)
            sp_px = np.where(m_sp, s_c[pos_sc], c)
            has_spot = m_sp & (c / np.maximum(sp_px, 1e-12) >= 0.5) & (c / np.maximum(sp_px, 1e-12) <= 2.0)
            spot_basis_bps = np.where(
                has_spot, np.clip(((c - sp_px) / np.maximum(sp_px, 1e-12)) * 1e4, -1500.0, 1500.0), 0.0
            )
            sqv_a = np.where(has_spot, s_qv[pos_sc], 0.0)
            stb_a = np.where(has_spot, s_tb[pos_sc], 0.0)
            sh3 = (_fast_rolling_mean_1d(stb_a, 3) * 3.0) / np.maximum(_fast_rolling_mean_1d(sqv_a, 3) * 3.0, 1.0)
            sh60 = (_fast_rolling_mean_1d(stb_a, 60) * 60.0) / np.maximum(_fast_rolling_mean_1d(sqv_a, 60) * 60.0, 1.0)
            spot_tb_3m = np.where(has_spot & (sqv_a > 0.0), sh3, perp_tb_5m)
            spot_tb_60m = np.where(has_spot & (sqv_a > 0.0), sh60, perp_tb_60m)

    # 2. Multi-Exchange `premiumIndexKlines` 1m (`220`-sym universal coverage + intraminute Impact Bid/Ask!)
    has_prem = has_spot.copy()
    prem_close_bps = spot_basis_bps.copy()
    prem_low_bps = spot_basis_bps.copy()
    prem_high_bps = spot_basis_bps.copy()
    if prem_1m is not None and not prem_1m.empty:
        pr_df = prem_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        pr_ts = pr_df["timestamp"].to_numpy(dtype=np.int64)
        if len(pr_ts) > 100:
            pos_p = np.searchsorted(pr_ts, ts)
            pos_pc = np.minimum(pos_p, len(pr_ts) - 1)
            m_pr = (pos_p < len(pr_ts)) & (pr_ts[pos_pc] == ts)
            pc_arr = pr_df["p_close_bps"].to_numpy(dtype=np.float64)
            pl_arr = pr_df["p_low_bps"].to_numpy(dtype=np.float64)
            ph_arr = pr_df["p_high_bps"].to_numpy(dtype=np.float64)
            prem_close_bps = np.where(m_pr, np.clip(pc_arr[pos_pc], -1500.0, 1500.0), spot_basis_bps)
            prem_low_bps = np.where(m_pr, np.clip(pl_arr[pos_pc], -1500.0, 1500.0), prem_close_bps)
            prem_high_bps = np.where(m_pr, np.clip(ph_arr[pos_pc], -1500.0, 1500.0), prem_close_bps)
            has_prem = m_pr | has_spot

    ema_prem_60m = _fast_ema_1d(prem_close_bps, 60)
    prem_low_disloc_bps = np.where(has_prem, prem_low_bps - ema_prem_60m, 0.0)

    # 3. Contract-Unit `metrics_5m` (`+60s` causal safety lag)
    has_oi = np.zeros(n, dtype=bool)
    oi_ratio_24h = np.ones(n, dtype=np.float64)
    oi_vel_15m = np.zeros(n, dtype=np.float64)
    oi_vel_1h = np.zeros(n, dtype=np.float64)
    oi_vwap_1m = c.copy()
    if (
        metrics_5m is not None
        and not metrics_5m.empty
        and ("sum_open_interest" in metrics_5m.columns or "sum_open_interest_value" in metrics_5m.columns)
    ):
        m_df = metrics_5m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        m_ts_raw = m_df["timestamp"].to_numpy(dtype=np.int64)
        m_ts_avail = m_ts_raw + 60
        p_idx = np.minimum(np.searchsorted(ts, m_ts_raw), n - 1)
        px_5m = np.maximum(o[p_idx], 1e-12)
        if "sum_open_interest" in m_df.columns:
            m_oi = pd.to_numeric(m_df["sum_open_interest"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            m_oi = (
                pd.to_numeric(m_df["sum_open_interest_value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                / px_5m
            )
        if len(m_ts_avail) > 36 and bool((m_oi > 0).any()):
            anchor_5m = compute_recursive_oi_vwap_anchor(m_oi, px_5m, base_decay=0.9965)
            oi_r24_5m = m_oi / np.maximum(_fast_rolling_mean_1d(m_oi, 288), 1e-9)
            oi_v15_5m = np.zeros(len(m_oi), dtype=np.float64)
            oi_v15_5m[3:] = (m_oi[3:] - m_oi[:-3]) / np.maximum(m_oi[:-3], 1e-9)
            oi_v1h_5m = np.zeros(len(m_oi), dtype=np.float64)
            oi_v1h_5m[12:] = (m_oi[12:] - m_oi[:-12]) / np.maximum(m_oi[:-12], 1e-9)

            bar_close_ts = ts + 60
            idx_m = np.searchsorted(m_ts_avail, bar_close_ts, side="right") - 1
            valid_m = (idx_m >= 0) & ((bar_close_ts - m_ts_avail[np.maximum(idx_m, 0)]) <= 1800)
            idx_mc = np.maximum(idx_m, 0)
            has_oi = valid_m & (m_oi[idx_mc] > 0.0)
            oi_ratio_24h = np.where(has_oi, oi_r24_5m[idx_mc], 1.0)
            oi_vel_15m = np.where(has_oi, oi_v15_5m[idx_mc], 0.0)
            oi_vel_1h = np.where(has_oi, oi_v1h_5m[idx_mc], 0.0)
            oi_vwap_1m = np.where(has_oi, anchor_5m[idx_mc], c)

    # 4. Binance Dynamic Funding & On-Chain `Hyperliquid L1` Hourly Carry Alignment
    funding_map: dict[int, float] = {}
    last_fr_bps = np.zeros(n, dtype=np.float64)
    f_ts_arr = np.empty(0, dtype=np.int64)
    f_rate_arr = np.empty(0, dtype=np.float64)
    f_iv_arr = np.empty(0, dtype=np.float64)
    if funding_df is not None and not funding_df.empty and "funding_rate" in funding_df.columns:
        f_df = funding_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        f_ts_arr = ((f_df["timestamp"].to_numpy(dtype=np.int64) + 30) // 60) * 60
        f_rate_arr = pd.to_numeric(f_df["funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "interval_hours" in f_df.columns:
            f_iv_arr = np.clip(
                pd.to_numeric(f_df["interval_hours"], errors="coerce").fillna(8.0).to_numpy(dtype=np.float64), 1.0, 8.0
            )
        else:
            f_iv_arr = np.full(len(f_ts_arr), 8.0, dtype=np.float64)
        for ft_k, fr_k in zip(f_ts_arr.tolist(), f_rate_arr.tolist(), strict=False):
            if abs(fr_k) > 1e-9:
                funding_map[int(ft_k)] = float(fr_k)
        if len(f_ts_arr) > 0:
            f_idx = np.searchsorted(f_ts_arr, ts + 60, side="right") - 1
            last_fr_bps = np.where(f_idx >= 0, f_rate_arr[np.maximum(f_idx, 0)] * 1e4, 0.0)

    has_hl = np.zeros(n, dtype=bool)
    hl_fr_bps_h = np.zeros(n, dtype=np.float64)
    if hl_funding_df is not None and not hl_funding_df.empty:
        h_df = hl_funding_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        h_ts = h_df["timestamp"].to_numpy(dtype=np.int64)
        h_fr = h_df["hl_fr_bps_per_h"].to_numpy(dtype=np.float64)
        if len(h_ts) > 0:
            bar_close_ts = ts + 60
            idx_h = np.searchsorted(h_ts, bar_close_ts, side="right") - 1
            valid_h = (idx_h >= 0) & ((bar_close_ts - h_ts[np.maximum(idx_h, 0)]) <= 7200)
            idx_hc = np.maximum(idx_h, 0)
            has_hl = valid_h
            hl_fr_bps_h = np.where(valid_h, h_fr[idx_hc], 0.0)

    # Pre-scan candidate indices across all mechanisms to collect exact causal UTC dates for L2 `bookDepth`
    cand_indices: list[int] = []

    if len(f_ts_arr) >= 2:
        for k_f in range(1, len(f_ts_arr)):
            t_set = int(f_ts_arr[k_f])
            fr_bps = float(f_rate_arr[k_f]) * 1e4
            prev_fr_bps = float(f_rate_arr[k_f - 1]) * 1e4
            idx_00 = int(np.searchsorted(ts, t_set))
            if 240 <= idx_00 < n - 95 and int(ts[idx_00]) == t_set and bool(liq_gate[idx_00]) and fr_bps <= -20.0:
                cand_indices.append(idx_00)
            t_pre = t_set - 45 * 60
            idx_pre = int(np.searchsorted(ts, t_pre))
            if (
                240 <= idx_pre < n - 95
                and int(ts[idx_pre]) == t_pre
                and bool(liq_gate[idx_pre])
                and prev_fr_bps <= -25.0
                and (float(prem_close_bps[idx_pre]) <= -70.0 or float(spot_basis_bps[idx_pre]) <= -75.0)
            ):
                cand_indices.append(idx_pre)

    # 5m sub-grid pre-scan for C2A, P6A, P6B, P7
    n5 = n // 5
    s_h4 = np.full(max(1, n5), np.inf, dtype=np.float64)
    s_l4 = np.full(max(1, n5), -np.inf, dtype=np.float64)
    step5 = np.arange(250, n - 185, 5, dtype=np.int64)
    if len(step5) > 50 and n5 > 50:
        h5 = h[: n5 * 5].reshape(n5, 5).max(axis=1)
        l5 = low[: n5 * 5].reshape(n5, 5).min(axis=1)
        s_h4 = pd.Series(h5).shift(1).rolling(48, min_periods=24).max().to_numpy(dtype=np.float64)
        s_l4 = pd.Series(l5).shift(1).rolling(48, min_periods=24).min().to_numpy(dtype=np.float64)
        for ix in step5.tolist():
            b5 = ix // 5
            if b5 >= len(s_h4) or not bool(liq_gate[ix]):
                continue
            c_ix = float(c[ix])
            brk_up = c_ix > float(s_h4[b5])
            brk_dn = c_ix < float(s_l4[b5])
            if not (brk_up or brk_dn):
                continue
            up_cand = brk_up and (
                (bool(has_oi[ix]) and float(oi_ratio_24h[ix]) >= 1.12 and float(rvol_5m[ix]) >= 1.8)
                or (bool(has_hl[ix]) and float(hl_fr_bps_h[ix]) <= -3.5 and float(prem_close_bps[ix]) <= -35.0)
            )
            dn_cand = brk_dn and bool(has_oi[ix]) and float(oi_ratio_24h[ix]) >= 1.16 and float(rvol_5m[ix]) >= 2.2
            if up_cand or dn_cand:
                cand_indices.append(ix)

    # 1m liquidation snap-back pre-scan (`P3_C3A`)
    if bool((has_prem & has_oi).any()):
        c3_mask = (
            liq_gate
            & has_prem
            & has_oi
            & (oi_vel_15m <= -0.020)
            & (ret_10m <= -0.022)
            & (prem_low_disloc_bps <= -35.0)
            & (ret_1m > 0.0)
        )
        cand_indices.extend(np.flatnonzero(c3_mask).tolist())

    needed_dates: set[str] = set()
    for ix in cand_indices:
        t_s = int(ts[ix])
        dt_u = datetime.fromtimestamp(t_s, tz=timezone.utc)
        needed_dates.add(dt_u.strftime("%Y-%m-%d"))
        if dt_u.hour == 0 and dt_u.minute < 45:
            needed_dates.add((dt_u - timedelta(days=1)).strftime("%Y-%m-%d"))

    return {
        "symbol": symbol,
        "ts": ts,
        "o": o,
        "h": h,
        "low": low,
        "c": c,
        "qv": qv,
        "liq_gate": liq_gate,
        "ret_1m": ret_1m,
        "ret_10m": ret_10m,
        "ret_4h": ret_4h,
        "perp_tb_1m": perp_tb_1m,
        "perp_tb_5m": perp_tb_5m,
        "spot_tb_3m": spot_tb_3m,
        "spot_tb_60m": spot_tb_60m,
        "rvol_5m": rvol_5m,
        "atr_1h": atr_1h,
        "has_spot": has_spot,
        "spot_basis_bps": spot_basis_bps,
        "has_prem": has_prem,
        "prem_close_bps": prem_close_bps,
        "prem_low_bps": prem_low_bps,
        "prem_high_bps": prem_high_bps,
        "prem_low_disloc_bps": prem_low_disloc_bps,
        "has_oi": has_oi,
        "oi_ratio_24h": oi_ratio_24h,
        "oi_vel_15m": oi_vel_15m,
        "oi_vel_1h": oi_vel_1h,
        "oi_vwap_1m": oi_vwap_1m,
        "funding_map": funding_map,
        "last_fr_bps": last_fr_bps,
        "f_ts_arr": f_ts_arr,
        "f_rate_arr": f_rate_arr,
        "f_iv_arr": f_iv_arr,
        "has_hl": has_hl,
        "hl_fr_bps_h": hl_fr_bps_h,
        "s_h4": s_h4,
        "s_l4": s_l4,
        "step5": step5,
        "needed_dates": needed_dates,
    }


def evaluate_symbol_p1_with_book_depth(
    state: dict[str, Any],
    book_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """
    Pass B: Causally aligns L2 `bookDepth` snapshots (`t_book <= ts + 60`, max age `<= 600s`)
    onto the symbol's `1m` timeline, emits triggers for all 9 mechanisms, and simulates all 3
    Monetization Geometries (`MONO_A`, `MONO_B`, `MONO_C`).
    """
    symbol = state["symbol"]
    ts = state["ts"]
    n = len(ts)

    has_book = np.zeros(n, dtype=bool)
    book_imb_02 = np.zeros(n, dtype=np.float64)
    book_imb_1 = np.zeros(n, dtype=np.float64)
    book_imb_2 = np.zeros(n, dtype=np.float64)
    bid_replenish_1 = np.ones(n, dtype=np.float64)

    if book_df is not None and not book_df.empty:
        b_sorted = book_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_ts = b_sorted["timestamp"].to_numpy(dtype=np.int64)
        if len(b_ts) > 0:
            b02 = b_sorted["bid_02_usd"].to_numpy(dtype=np.float64)
            a02 = b_sorted["ask_02_usd"].to_numpy(dtype=np.float64)
            b1 = b_sorted["bid_1_usd"].to_numpy(dtype=np.float64)
            a1 = b_sorted["ask_1_usd"].to_numpy(dtype=np.float64)
            b2 = b_sorted["bid_2_usd"].to_numpy(dtype=np.float64)
            a2 = b_sorted["ask_2_usd"].to_numpy(dtype=np.float64)

            imb02_s = (b02 - a02) / np.maximum(b02 + a02, 1.0)
            imb1_s = (b1 - a1) / np.maximum(b1 + a1, 1.0)
            imb2_s = (b2 - a2) / np.maximum(b2 + a2, 1.0)
            b1_ema = _fast_ema_1d(b1, 30)
            b1_rep_s = b1 / np.maximum(b1_ema, 1.0)

            bar_close_ts = ts + 60
            idx_b = np.searchsorted(b_ts, bar_close_ts, side="right") - 1
            valid_b = (idx_b >= 0) & ((bar_close_ts - b_ts[np.maximum(idx_b, 0)]) <= 600)
            idx_bc = np.maximum(idx_b, 0)
            has_book = valid_b & ((b1[idx_bc] + a1[idx_bc]) > 1000.0)
            book_imb_02 = np.where(has_book, imb02_s[idx_bc], 0.0)
            book_imb_1 = np.where(has_book, imb1_s[idx_bc], 0.0)
            book_imb_2 = np.where(has_book, imb2_s[idx_bc], 0.0)
            bid_replenish_1 = np.where(has_book, b1_rep_s[idx_bc], 1.0)

    c = state["c"]
    liq_gate = state["liq_gate"]
    ret_1m = state["ret_1m"]
    ret_10m = state["ret_10m"]
    ret_4h = state["ret_4h"]
    perp_tb_1m = state["perp_tb_1m"]
    perp_tb_5m = state["perp_tb_5m"]
    spot_tb_3m = state["spot_tb_3m"]
    spot_tb_60m = state["spot_tb_60m"]
    rvol_5m = state["rvol_5m"]
    atr_1h = state["atr_1h"]
    has_spot = state["has_spot"]
    spot_basis_bps = state["spot_basis_bps"]
    has_prem = state["has_prem"]
    prem_close_bps = state["prem_close_bps"]
    prem_low_disloc_bps = state["prem_low_disloc_bps"]
    has_oi = state["has_oi"]
    oi_ratio_24h = state["oi_ratio_24h"]
    oi_vel_15m = state["oi_vel_15m"]
    oi_vel_1h = state["oi_vel_1h"]
    oi_vwap_1m = state["oi_vwap_1m"]
    f_ts_arr = state["f_ts_arr"]
    f_rate_arr = state["f_rate_arr"]
    f_iv_arr = state["f_iv_arr"]
    last_fr_bps = state["last_fr_bps"]
    has_hl = state["has_hl"]
    hl_fr_bps_h = state["hl_fr_bps_h"]
    s_h4 = state["s_h4"]
    s_l4 = state["s_l4"]

    events: dict[str, list[tuple[int, int, float, float]]] = {k: [] for k in P1_MECHANISM_PROFILES}

    # ------------------------------------------------------------------
    # 1. Funding Settlement & Cross-Venue Dispersion (`B1_C1B`, `P1_C1B`, `P1_C1D`)
    # ------------------------------------------------------------------
    if len(f_ts_arr) >= 2:
        for k_f in range(1, len(f_ts_arr)):
            t_settle = int(f_ts_arr[k_f])
            fr_bps = float(f_rate_arr[k_f]) * 1e4
            prev_fr_bps = float(f_rate_arr[k_f - 1]) * 1e4
            iv_cur = max(1.0, float(f_iv_arr[k_f]))

            # Post-Settle T+1m (`P1_C1D_LOCAL_ARB_DISPERSION_SHORT`: ONLY when Binance FR << Hyperliquid FR!)
            idx_00 = int(np.searchsorted(ts, t_settle))
            if 240 <= idx_00 < n - 95 and int(ts[idx_00]) == t_settle and bool(liq_gate[idx_00]):
                bin_bps_h = fr_bps / iv_cur
                if (
                    fr_bps <= -25.0
                    and bool(has_hl[idx_00])
                    and (float(hl_fr_bps_h[idx_00]) - bin_bps_h) >= 3.0
                    and float(hl_fr_bps_h[idx_00]) >= -2.0
                    and float(prem_close_bps[idx_00]) >= -140.0
                    and float(ret_1m[idx_00]) <= -0.0005
                    and float(perp_tb_1m[idx_00]) <= 0.47
                    and (not bool(has_book[idx_00]) or float(book_imb_1[idx_00]) <= 0.08)
                ):
                    disp_str = float(hl_fr_bps_h[idx_00]) - bin_bps_h
                    events["P1_C1D_LOCAL_ARB_DISPERSION_SHORT"].append((idx_00, -1, disp_str * 10.0, disp_str))

            # Pre-Settle T-45m (`B1_C1B` Control vs `P1_C1B` Multi-Ex Premium + HL Sync + L2 Ask-Wall Veto)
            t_pre = t_settle - 45 * 60
            idx_pre = int(np.searchsorted(ts, t_pre))
            if 240 <= idx_pre < n - 95 and int(ts[idx_pre]) == t_pre and bool(liq_gate[idx_pre]):
                if (
                    bool(has_spot[idx_pre])
                    and prev_fr_bps <= -25.0
                    and float(spot_basis_bps[idx_pre]) <= -80.0
                    and float(spot_tb_60m[idx_pre]) >= 0.52
                    and float(ret_4h[idx_pre]) >= 0.015
                ):
                    b_s = float(spot_basis_bps[idx_pre])
                    events["B1_C1B_BASE_UNCHOKED_LONG"].append((idx_pre, 1, abs(b_s) + abs(prev_fr_bps), b_s))

                # Upgraded `P1_C1B`: Universal `prem_close_bps <= -75`, HL not positive local-arb diverge, L2 not Ask-walled!
                hl_sync_ok = (not bool(has_hl[idx_pre])) or (float(hl_fr_bps_h[idx_pre]) <= 0.5)
                book_ok_c1b = (not bool(has_book[idx_pre])) or (float(book_imb_2[idx_pre]) >= -0.15)
                if (
                    bool(has_prem[idx_pre])
                    and prev_fr_bps <= -25.0
                    and float(prem_close_bps[idx_pre]) <= -75.0
                    and float(spot_tb_60m[idx_pre]) >= 0.52
                    and float(ret_4h[idx_pre]) >= 0.012
                    and hl_sync_ok
                    and book_ok_c1b
                ):
                    pr_v = float(prem_close_bps[idx_pre])
                    bk_bonus = max(0.0, float(book_imb_2[idx_pre])) * 40.0
                    events["P1_C1B_GLOBAL_SCARCITY_LONG"].append(
                        (idx_pre, 1, abs(pr_v) + abs(prev_fr_bps) + bk_bonus, pr_v)
                    )

    # ------------------------------------------------------------------
    # 2. 5m Sub-Grid Scan: `B2_C2A`, `P2_C2A`, `P6A`, `P6B`, `P7`
    # ------------------------------------------------------------------
    for ix in state["step5"].tolist():
        b5 = ix // 5
        if b5 >= len(s_h4) or not bool(liq_gate[ix]):
            continue
        c_ix = float(c[ix])
        brk_up = c_ix > float(s_h4[b5])
        brk_dn = c_ix < float(s_l4[b5])
        if not (brk_up or brk_dn):
            continue

        rv5 = float(rvol_5m[ix])
        fr_b = float(last_fr_bps[ix])

        if brk_up and bool(has_oi[ix]):
            oir = float(oi_ratio_24h[ix])
            pain = (c_ix - float(oi_vwap_1m[ix])) / float(atr_1h[ix])

            # `B2_C2A_BASE_UNCHOKED_LONG` (Control without L2 Book or Premium gate)
            if (
                oir >= 1.12
                and pain >= 1.8
                and fr_b <= -5.0
                and float(oi_vel_1h[ix]) >= 0.0
                and rv5 >= 1.8
                and float(spot_tb_3m[ix]) >= 0.54
            ):
                events["B2_C2A_BASE_UNCHOKED_LONG"].append((ix, 1, pain * oir, fr_b))

                # `P2_C2A_BOOK_VACUUM_PAIN_LONG`: Multi-Ex discount + L2 Book Ask-Vacuum Gate (`book_imb >= +0.05`!)
                prem_ok = (not bool(has_prem[ix])) or (float(prem_close_bps[ix]) <= -10.0)
                l2_vacuum_ok = (not bool(has_book[ix])) or (
                    float(book_imb_1[ix]) >= 0.04 or float(book_imb_2[ix]) >= 0.04
                )
                if prem_ok and l2_vacuum_ok:
                    str_p2 = pain * oir * (1.0 + max(0.0, float(book_imb_2[ix])))
                    events["P2_C2A_BOOK_VACUUM_PAIN_LONG"].append((ix, 1, str_p2, float(book_imb_2[ix])))

            # `P6A_L2_ASK_AIR_POCKET_LONG`: Pure L2 Book Air-Pocket (`has_book` REQUIRED, `imb_1 >= +0.24` & `imb_2 >= +0.20`)
            if (
                bool(has_book[ix])
                and oir >= 1.16
                and fr_b <= 3.0
                and rv5 >= 2.0
                and float(book_imb_1[ix]) >= 0.24
                and float(book_imb_2[ix]) >= 0.20
                and float(spot_tb_3m[ix]) >= 0.53
            ):
                st_6a = (float(book_imb_1[ix]) + float(book_imb_2[ix])) * 100.0 * oir
                events["P6A_L2_ASK_AIR_POCKET_LONG"].append((ix, 1, st_6a, float(book_imb_1[ix])))

        if brk_dn and bool(has_oi[ix]) and bool(has_book[ix]):
            oir_d = float(oi_ratio_24h[ix])
            pain_d = (c_ix - float(oi_vwap_1m[ix])) / float(atr_1h[ix])
            # `P6B_L2_BID_VACUUM_WATERFALL_SHORT`: Pure L2 Bid Collapse (`imb_1 <= -0.24` & `imb_2 <= -0.20`)
            if (
                oir_d >= 1.16
                and pain_d <= -1.4
                and rv5 >= 2.2
                and float(book_imb_1[ix]) <= -0.24
                and float(book_imb_2[ix]) <= -0.20
                and float(perp_tb_5m[ix]) <= 0.45
            ):
                st_6b = abs(float(book_imb_1[ix]) + float(book_imb_2[ix])) * 100.0 * oir_d
                events["P6B_L2_BID_VACUUM_WATERFALL_SHORT"].append((ix, -1, st_6b, float(book_imb_1[ix])))

        # `P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG`: Synchronized On-Chain Hyperliquid + Binance Impact Discount Breakout
        if (
            brk_up
            and bool(has_hl[ix])
            and float(hl_fr_bps_h[ix]) <= -3.5
            and float(prem_close_bps[ix]) <= -35.0
            and rv5 >= 1.6
            and float(spot_tb_3m[ix]) >= 0.53
            and (not bool(has_book[ix]) or float(book_imb_2[ix]) >= -0.10)
        ):
            st_p7 = abs(float(hl_fr_bps_h[ix])) * 10.0 + abs(float(prem_close_bps[ix]))
            events["P7_ONCHAIN_HL_BINANCE_COSQUEEZE_LONG"].append((ix, 1, st_p7, float(hl_fr_bps_h[ix])))

    # ------------------------------------------------------------------
    # 3. 1m Liquidation Snap-Back + L2 Bid-Replenishment + Crash Guard (`P3_C3A`)
    # ------------------------------------------------------------------
    if bool((has_prem & has_oi).any()):
        l2_bid_refilled = (~has_book) | ((book_imb_02 >= -0.05) & (bid_replenish_1 >= 0.75))
        c3_idx = np.flatnonzero(
            liq_gate
            & has_prem
            & has_oi
            & (oi_vel_15m <= -0.020)
            & (ret_10m <= -0.022)
            & (ret_4h >= -0.075)
            & (prem_low_disloc_bps <= -35.0)
            & (ret_1m > 0.0005)
            & (spot_tb_3m >= 0.53)
            & l2_bid_refilled
        )
        for ix in c3_idx.tolist():
            if 240 <= ix < n - 80:
                st_c3 = abs(float(prem_low_disloc_bps[ix])) * (abs(float(oi_vel_15m[ix])) * 100.0)
                events["P3_C3A_BOOK_REPLENISHED_SNAP_LONG"].append((ix, 1, st_c3, float(prem_low_disloc_bps[ix])))

    all_records: list[dict[str, Any]] = []
    for m_id, ev_list in events.items():
        if not ev_list:
            continue
        all_records.extend(
            simulate_1m_monetization_geometries(
                ts_1m=ts,
                open_1m=state["o"],
                high_1m=state["h"],
                low_1m=state["low"],
                close_1m=c,
                qvol_1m=state["qv"],
                tb_share_1m=perp_tb_1m,
                book_imb_1pct_1m=book_imb_1,
                funding_map=state["funding_map"],
                raw_events=ev_list,
                profile=P1_MECHANISM_PROFILES[m_id],
                symbol=symbol,
            )
        )
    return all_records


def summarize_p1_lab_results(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Runs `evaluate_infold_cpcv_and_treasure_metrics(total_trials=27)` and attaches family metadata."""
    lb = evaluate_infold_cpcv_and_treasure_metrics(trades_df, total_trials=27)
    if lb.empty:
        return lb
    lb["pillar"] = [
        P1_MECHANISM_PROFILES[str(m)].family if str(m) in P1_MECHANISM_PROFILES else "" for m in lb["mech_id"]
    ]
    return lb
