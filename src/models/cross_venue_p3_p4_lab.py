"""
Phase P3/P4 Cross-Exchange (`Bybit V5` / `OKX Swap`) & Full-Universe On-Chain (`Hyperliquid L1`)
Microstructure & Monetization Laboratory (`2023–2026`, Exact `1-Minute` Linear Exchange Physics).

Tests 4 genuine cross-venue & on-chain state-space mechanisms impossible to observe from single-exchange
`OHLCV + OI` data:
  1. `P3_M1_CROSS_CEX_ARB_UNWIND_SHORT` (`MEC-01A` True Cross-CEX Harvester Unwind):
     Enters `T+1m` post-settlement SHORT ONLY when `Binance FR <= -20 bps` (`<= -3.5 bps/h`) WHILE
     `Bybit/OKX` or `Hyperliquid L1` carry is `>= +3.0 bps/h` higher (`>= -1.5 bps/h`), proving
     local cross-exchange arbitrage overhang rather than global spot scarcity, exiting `<= :56` (`0` funding paid).
  2. `P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG` (`MEC-08` Synchronized Multi-Venue Short-Trap Ignition):
     `4h` High `5m` breakout when BOTH `Binance` AND `Bybit/OKX/Hyperliquid` exhibit synchronized negative
     hourly carry (`<= -0.8 bps/h`) + `1m premiumIndex <= -15 bps` + `OI >= 1.10x` + L2 Ask Vacuum.
  3. `P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG` (`MEC-09` On-Chain Hyperliquid Validator-Oracle Lead):
     `4h` High `5m` breakout when On-Chain `Hyperliquid L1` (`1h` funding `>= +1.8 bps/h` or oracle
     `hl_premium_bps >= +10 bps`) leads a still-unpriced/discounted `Binance` perp (`hl - bin >= +1.4 bps/h`).
  4. `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG` (`MEC-10` Cross-Venue Anchored Liquidation Snap-Back):
     Buys `1m` Binance contract-OI liquidation flushes (`oi_vel_15m <= -1.5%`, `prem_low_disloc <= -35 bps`)
     ONLY when On-Chain `Hyperliquid L1` (`hl_premium_bps >= -12 bps`) or `Peer CEX` confirms asset solvency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.models.counterparty_mechanics_lab import (
    _fast_ema_1d,
    _fast_rolling_mean_1d,
    evaluate_infold_cpcv_and_treasure_metrics,
)
from src.models.microstructure_monetization_lab import (
    P1MechanismProfile,
    simulate_1m_monetization_geometries,
)

P3_P4_MECHANISM_PROFILES: dict[str, P1MechanismProfile] = {
    "P3_M1_CROSS_CEX_ARB_UNWIND_SHORT": P1MechanismProfile(
        mech_id="P3_M1_CROSS_CEX_ARB_UNWIND_SHORT",
        family="MEC01A_CrossCEX_Harvester_Unwind",
        description="T+1m Post-Settle SHORT strictly when Binance FR <= -20bps AND PeerCEX/HL >= +3bps/h higher",
        prefer_passive_touch=False,
        base_sl_pct=0.018,
        max_hold_minutes=52,
        cooldown_minutes=45,
        is_funding_coupon_mech=False,
    ),
    "P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG": P1MechanismProfile(
        mech_id="P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG",
        family="MEC08_Global_MultiVenue_ShortTrap",
        description="Synchronized Binance + PeerCEX/HL Negative Carry + 1m Discount + OI>=1.10x 4h Breakout LONG",
        prefer_passive_touch=False,
        base_sl_pct=0.020,
        max_hold_minutes=180,
        cooldown_minutes=90,
        is_funding_coupon_mech=True,
    ),
    "P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG": P1MechanismProfile(
        mech_id="P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG",
        family="MEC09_OnChain_Hyperliquid_OracleLead",
        description="On-Chain Hyperliquid 1h Carry/Premium Lead (>=+1.8bps/h) over Lagging Binance 4h Breakout LONG",
        prefer_passive_touch=False,
        base_sl_pct=0.019,
        max_hold_minutes=150,
        cooldown_minutes=90,
        is_funding_coupon_mech=False,
    ),
    "P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG": P1MechanismProfile(
        mech_id="P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG",
        family="MEC10_CrossVenue_SolvencyAnchored_SnapBack",
        description="Binance 15m OI Liquidation Flush (-1.5%) + Impact Wick (-35bps) Anchored by Healthy HL/PeerCEX",
        prefer_passive_touch=False,
        base_sl_pct=0.017,
        max_hold_minutes=75,
        cooldown_minutes=45,
        is_funding_coupon_mech=False,
    ),
}


def prepare_symbol_p3_p4_state(
    symbol: str,
    perp_1m: pd.DataFrame,
    prem_1m: pd.DataFrame | None,
    metrics_5m: pd.DataFrame | None,
    funding_df: pd.DataFrame | None,
    hl_funding_df: pd.DataFrame | None,
    peer_cex_df: pd.DataFrame | None,
    min_adv_usd: float = 15_000_000.0,
) -> dict[str, Any] | None:
    """
    Pass A: Causally aligns `perp_1m`, `prem_1m`, contract-unit `metrics_5m` (`+60s` lag),
    `Binance` dynamic funding, `Hyperliquid L1` on-chain `1h` carry & oracle premium, and
    `Bybit/OKX` peer-CEX hourly carry (`bps/h`), returning candidate UTC dates for L2 `bookDepth`.
    """
    if perp_1m is None or len(perp_1m) < 2880:
        return None
    if (hl_funding_df is None or hl_funding_df.empty) and (peer_cex_df is None or peer_cex_df.empty):
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

    qv_5m = _fast_rolling_mean_1d(qv, 5) * 5.0
    qv_5m_base = np.roll(_fast_rolling_mean_1d(qv, 360) * 5.0, 5)
    qv_5m_base[:6] = qv_5m[:6]
    rvol_5m = qv_5m / np.maximum(qv_5m_base, 1.0)

    # 1. 1m Multi-Exchange Premium Index
    has_prem = np.zeros(n, dtype=bool)
    prem_close_bps = np.zeros(n, dtype=np.float64)
    prem_low_bps = np.zeros(n, dtype=np.float64)
    if prem_1m is not None and not prem_1m.empty:
        pr_df = prem_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        pr_ts = pr_df["timestamp"].to_numpy(dtype=np.int64)
        if len(pr_ts) > 100:
            pos_p = np.searchsorted(pr_ts, ts)
            pos_pc = np.minimum(pos_p, len(pr_ts) - 1)
            m_pr = (pos_p < len(pr_ts)) & (pr_ts[pos_pc] == ts)
            pc_a = pr_df["p_close_bps"].to_numpy(dtype=np.float64)
            pl_a = pr_df["p_low_bps"].to_numpy(dtype=np.float64)
            prem_close_bps = np.where(m_pr, np.clip(pc_a[pos_pc], -1500.0, 1500.0), 0.0)
            prem_low_bps = np.where(m_pr, np.clip(pl_a[pos_pc], -1500.0, 1500.0), prem_close_bps)
            has_prem = m_pr
    prem_low_disloc_bps = np.where(has_prem, prem_low_bps - _fast_ema_1d(prem_close_bps, 60), 0.0)

    # 2. Contract-Unit 5m Open Interest (`+60s` causal safety lag)
    has_oi = np.zeros(n, dtype=bool)
    oi_ratio_24h = np.ones(n, dtype=np.float64)
    oi_vel_15m = np.zeros(n, dtype=np.float64)
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
            oi_r24_5m = m_oi / np.maximum(_fast_rolling_mean_1d(m_oi, 288), 1e-9)
            oi_v15_5m = np.zeros(len(m_oi), dtype=np.float64)
            oi_v15_5m[3:] = (m_oi[3:] - m_oi[:-3]) / np.maximum(m_oi[:-3], 1e-9)
            bar_close_ts = ts + 60
            idx_m = np.searchsorted(m_ts_avail, bar_close_ts, side="right") - 1
            valid_m = (idx_m >= 0) & ((bar_close_ts - m_ts_avail[np.maximum(idx_m, 0)]) <= 1800)
            idx_mc = np.maximum(idx_m, 0)
            has_oi = valid_m & (m_oi[idx_mc] > 0.0)
            oi_ratio_24h = np.where(has_oi, oi_r24_5m[idx_mc], 1.0)
            oi_vel_15m = np.where(has_oi, oi_v15_5m[idx_mc], 0.0)

    # 3. Binance Dynamic Funding
    funding_map: dict[int, float] = {}
    last_fr_bps = np.zeros(n, dtype=np.float64)
    bin_fr_bps_h = np.zeros(n, dtype=np.float64)
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
            dt_fh = np.diff(f_ts_arr, prepend=f_ts_arr[0] - 8 * 3600) / 3600.0
            f_iv_arr = np.clip(np.round(dt_fh), 1.0, 8.0)
        for ft_k, fr_k in zip(f_ts_arr.tolist(), f_rate_arr.tolist(), strict=False):
            if abs(fr_k) > 1e-9:
                funding_map[int(ft_k)] = float(fr_k)
        if len(f_ts_arr) > 0:
            f_idx = np.searchsorted(f_ts_arr, ts + 60, side="right") - 1
            f_ic = np.maximum(f_idx, 0)
            last_fr_bps = np.where(f_idx >= 0, f_rate_arr[f_ic] * 1e4, 0.0)
            bin_fr_bps_h = np.where(f_idx >= 0, (f_rate_arr[f_ic] * 1e4) / f_iv_arr[f_ic], 0.0)

    # 4. On-Chain Hyperliquid L1 (`hl_fr_bps_per_h`, `hl_premium_bps`)
    has_hl = np.zeros(n, dtype=bool)
    hl_fr_bps_h = np.zeros(n, dtype=np.float64)
    hl_prem_bps = np.zeros(n, dtype=np.float64)
    if hl_funding_df is not None and not hl_funding_df.empty:
        h_df = hl_funding_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        h_ts = h_df["timestamp"].to_numpy(dtype=np.int64)
        h_fr = h_df["hl_fr_bps_per_h"].to_numpy(dtype=np.float64)
        h_pr = (
            h_df["hl_premium_bps"].to_numpy(dtype=np.float64)
            if "hl_premium_bps" in h_df.columns
            else np.zeros(len(h_ts), dtype=np.float64)
        )
        if len(h_ts) > 0:
            bar_close_ts = ts + 60
            idx_h = np.searchsorted(h_ts, bar_close_ts, side="right") - 1
            valid_h = (idx_h >= 0) & ((bar_close_ts - h_ts[np.maximum(idx_h, 0)]) <= 7200)
            idx_hc = np.maximum(idx_h, 0)
            has_hl = valid_h
            hl_fr_bps_h = np.where(valid_h, h_fr[idx_hc], 0.0)
            hl_prem_bps = np.where(valid_h, h_pr[idx_hc], 0.0)

    # 5. Peer CEX (`Bybit V5` / `OKX Swap`) Hourly Carry (`peer_fr_bps_per_h`)
    has_peer = np.zeros(n, dtype=bool)
    peer_fr_bps_h = np.zeros(n, dtype=np.float64)
    if peer_cex_df is not None and not peer_cex_df.empty:
        p_c = peer_cex_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        c_ts = p_c["timestamp"].to_numpy(dtype=np.int64)
        c_fr = p_c["peer_fr_bps_per_h"].to_numpy(dtype=np.float64)
        if len(c_ts) > 0:
            bar_close_ts = ts + 60
            idx_p = np.searchsorted(c_ts, bar_close_ts, side="right") - 1
            valid_p = (idx_p >= 0) & ((bar_close_ts - c_ts[np.maximum(idx_p, 0)]) <= 32400)
            idx_pc = np.maximum(idx_p, 0)
            has_peer = valid_p
            peer_fr_bps_h = np.where(valid_p, c_fr[idx_pc], 0.0)

    # Pre-scan candidate indices using exact non-book gates so `needed_dates` only fetches true event days
    cand_indices: list[int] = []
    if len(f_ts_arr) >= 2:
        for k_f in range(1, len(f_ts_arr)):
            t_set = int(f_ts_arr[k_f])
            fr_bps = float(f_rate_arr[k_f]) * 1e4
            iv_cur = max(1.0, float(f_iv_arr[k_f]))
            idx_00 = int(np.searchsorted(ts, t_set))
            if not (240 <= idx_00 < n - 95 and int(ts[idx_00]) == t_set and bool(liq_gate[idx_00]) and fr_bps <= -20.0):
                continue
            b_h = fr_bps / iv_cur
            peer_div = (float(peer_fr_bps_h[idx_00]) - b_h) if bool(has_peer[idx_00]) else -999.0
            hl_div = (float(hl_fr_bps_h[idx_00]) - b_h) if bool(has_hl[idx_00]) else -999.0
            peer_arb_ok = bool(has_peer[idx_00]) and peer_div >= 3.0 and float(peer_fr_bps_h[idx_00]) >= -1.5
            hl_arb_ok = bool(has_hl[idx_00]) and hl_div >= 3.0 and float(hl_fr_bps_h[idx_00]) >= -1.5
            if (
                b_h <= -3.5
                and (peer_arb_ok or hl_arb_ok)
                and (not bool(has_prem[idx_00]) or float(prem_close_bps[idx_00]) >= -150.0)
                and float(ret_1m[idx_00]) <= -0.0004
                and float(perp_tb_1m[idx_00]) <= 0.47
            ):
                cand_indices.append(idx_00)

    n5 = n // 5
    s_h4 = np.full(max(1, n5), np.inf, dtype=np.float64)
    step5 = np.arange(250, n - 185, 5, dtype=np.int64)
    if len(step5) > 50 and n5 > 50:
        h5 = h[: n5 * 5].reshape(n5, 5).max(axis=1)
        s_h4 = pd.Series(h5).shift(1).rolling(48, min_periods=24).max().to_numpy(dtype=np.float64)
        for ix in step5.tolist():
            b5 = ix // 5
            if b5 >= len(s_h4) or not bool(liq_gate[ix]) or float(c[ix]) <= float(s_h4[b5]):
                continue
            if float(rvol_5m[ix]) < 1.8 or float(perp_tb_5m[ix]) < 0.53:
                continue
            oir = float(oi_ratio_24h[ix]) if bool(has_oi[ix]) else 1.0
            b_h = float(bin_fr_bps_h[ix])
            fr_raw = float(last_fr_bps[ix])

            peer_neg = bool(has_peer[ix]) and float(peer_fr_bps_h[ix]) <= -0.8
            hl_neg = bool(has_hl[ix]) and float(hl_fr_bps_h[ix]) <= -1.2
            no_pos_div = (not bool(has_peer[ix]) or float(peer_fr_bps_h[ix]) <= 0.3) and (
                not bool(has_hl[ix]) or float(hl_fr_bps_h[ix]) <= 0.3
            )
            m2_pre = (
                fr_raw <= -4.0
                and b_h <= -0.8
                and (peer_neg or hl_neg)
                and no_pos_div
                and oir >= 1.10
                and (not bool(has_prem[ix]) or float(prem_close_bps[ix]) <= -15.0)
            )
            m3_pre = False
            if bool(has_hl[ix]) and fr_raw <= 3.0 and b_h <= 0.4 and oir >= 1.08:
                hl_f = float(hl_fr_bps_h[ix])
                hl_p = float(hl_prem_bps[ix])
                if (hl_f >= 1.8 or hl_p >= 10.0) and (hl_f - b_h) >= 1.4:
                    m3_pre = True
            if m2_pre or m3_pre:
                cand_indices.append(ix)

    if bool((has_prem & has_oi & (has_hl | has_peer)).any()):
        hl_solv = has_hl & (hl_prem_bps >= -12.0) & (hl_fr_bps_h >= -2.0)
        peer_solv = has_peer & ((peer_fr_bps_h - bin_fr_bps_h) >= 1.2) & (peer_fr_bps_h >= -1.5)
        m4_mask = (
            liq_gate
            & has_prem
            & has_oi
            & (hl_solv | peer_solv)
            & (oi_vel_15m <= -0.015)
            & (ret_10m <= -0.020)
            & (ret_4h > -0.090)
            & (prem_low_disloc_bps <= -35.0)
            & (ret_1m > 0.0)
            & (perp_tb_1m >= 0.52)
        )
        cand_indices.extend(np.flatnonzero(m4_mask).tolist())

    needed_dates: set[str] = set()
    for ix in cand_indices:
        dt_u = datetime.fromtimestamp(int(ts[ix]), tz=timezone.utc)
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
        "rvol_5m": rvol_5m,
        "has_prem": has_prem,
        "prem_close_bps": prem_close_bps,
        "prem_low_disloc_bps": prem_low_disloc_bps,
        "has_oi": has_oi,
        "oi_ratio_24h": oi_ratio_24h,
        "oi_vel_15m": oi_vel_15m,
        "funding_map": funding_map,
        "last_fr_bps": last_fr_bps,
        "bin_fr_bps_h": bin_fr_bps_h,
        "f_ts_arr": f_ts_arr,
        "f_rate_arr": f_rate_arr,
        "f_iv_arr": f_iv_arr,
        "has_hl": has_hl,
        "hl_fr_bps_h": hl_fr_bps_h,
        "hl_prem_bps": hl_prem_bps,
        "has_peer": has_peer,
        "peer_fr_bps_h": peer_fr_bps_h,
        "s_h4": s_h4,
        "step5": step5,
        "needed_dates": needed_dates,
    }


def evaluate_symbol_p3_p4_mechanisms(
    state: dict[str, Any],
    book_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """
    Pass B: Aligns causal L2 `bookDepth` snapshots, generates triggers for `P3_M1`, `P3_M2`,
    `P4_M3`, and `P4_M4`, and simulates exact `1-minute` linear monetization geometries.
    """
    symbol = state["symbol"]
    ts = state["ts"]
    n = len(ts)

    has_book = np.zeros(n, dtype=bool)
    book_imb_1 = np.zeros(n, dtype=np.float64)
    book_imb_2 = np.zeros(n, dtype=np.float64)
    if book_df is not None and not book_df.empty:
        b_sorted = book_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_ts = b_sorted["timestamp"].to_numpy(dtype=np.int64)
        if len(b_ts) > 0:
            b1 = b_sorted["bid_1_usd"].to_numpy(dtype=np.float64)
            a1 = b_sorted["ask_1_usd"].to_numpy(dtype=np.float64)
            b2 = b_sorted["bid_2_usd"].to_numpy(dtype=np.float64)
            a2 = b_sorted["ask_2_usd"].to_numpy(dtype=np.float64)
            imb1_s = (b1 - a1) / np.maximum(b1 + a1, 1.0)
            imb2_s = (b2 - a2) / np.maximum(b2 + a2, 1.0)
            bar_close_ts = ts + 60
            idx_b = np.searchsorted(b_ts, bar_close_ts, side="right") - 1
            valid_b = (idx_b >= 0) & ((bar_close_ts - b_ts[np.maximum(idx_b, 0)]) <= 600)
            idx_bc = np.maximum(idx_b, 0)
            has_book = valid_b & ((b1[idx_bc] + a1[idx_bc]) > 1000.0)
            book_imb_1 = np.where(has_book, imb1_s[idx_bc], 0.0)
            book_imb_2 = np.where(has_book, imb2_s[idx_bc], 0.0)

    c = state["c"]
    liq_gate = state["liq_gate"]
    ret_1m = state["ret_1m"]
    ret_10m = state["ret_10m"]
    ret_4h = state["ret_4h"]
    perp_tb_1m = state["perp_tb_1m"]
    perp_tb_5m = state["perp_tb_5m"]
    rvol_5m = state["rvol_5m"]
    has_prem = state["has_prem"]
    prem_close_bps = state["prem_close_bps"]
    prem_low_disloc_bps = state["prem_low_disloc_bps"]
    has_oi = state["has_oi"]
    oi_ratio_24h = state["oi_ratio_24h"]
    oi_vel_15m = state["oi_vel_15m"]
    last_fr_bps = state["last_fr_bps"]
    bin_fr_bps_h = state["bin_fr_bps_h"]
    f_ts_arr = state["f_ts_arr"]
    f_rate_arr = state["f_rate_arr"]
    f_iv_arr = state["f_iv_arr"]
    has_hl = state["has_hl"]
    hl_fr_bps_h = state["hl_fr_bps_h"]
    hl_prem_bps = state["hl_prem_bps"]
    has_peer = state["has_peer"]
    peer_fr_bps_h = state["peer_fr_bps_h"]
    s_h4 = state["s_h4"]

    events: dict[str, list[tuple[int, int, float, float]]] = {k: [] for k in P3_P4_MECHANISM_PROFILES}

    # 1. `P3_M1_CROSS_CEX_ARB_UNWIND_SHORT`
    if len(f_ts_arr) >= 2:
        for k_f in range(1, len(f_ts_arr)):
            t_settle = int(f_ts_arr[k_f])
            fr_bps = float(f_rate_arr[k_f]) * 1e4
            iv_cur = max(1.0, float(f_iv_arr[k_f]))
            idx_00 = int(np.searchsorted(ts, t_settle))
            if not (240 <= idx_00 < n - 95 and int(ts[idx_00]) == t_settle and bool(liq_gate[idx_00])):
                continue
            if fr_bps > -20.0:
                continue
            b_h = fr_bps / iv_cur
            peer_div = (float(peer_fr_bps_h[idx_00]) - b_h) if bool(has_peer[idx_00]) else -999.0
            hl_div = (float(hl_fr_bps_h[idx_00]) - b_h) if bool(has_hl[idx_00]) else -999.0
            peer_arb_ok = bool(has_peer[idx_00]) and peer_div >= 3.0 and float(peer_fr_bps_h[idx_00]) >= -1.5
            hl_arb_ok = bool(has_hl[idx_00]) and hl_div >= 3.0 and float(hl_fr_bps_h[idx_00]) >= -1.5
            if (
                b_h <= -3.5
                and (peer_arb_ok or hl_arb_ok)
                and (not bool(has_prem[idx_00]) or float(prem_close_bps[idx_00]) >= -150.0)
                and float(ret_1m[idx_00]) <= -0.0004
                and float(perp_tb_1m[idx_00]) <= 0.47
                and (not bool(has_book[idx_00]) or float(book_imb_1[idx_00]) <= 0.10)
            ):
                best_div = max(peer_div, hl_div)
                events["P3_M1_CROSS_CEX_ARB_UNWIND_SHORT"].append((idx_00, -1, best_div * 12.0, best_div))

    # 2. 5m Sub-Grid: `P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG` & `P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG`
    for ix in state["step5"].tolist():
        b5 = ix // 5
        if b5 >= len(s_h4) or not bool(liq_gate[ix]) or float(c[ix]) <= float(s_h4[b5]):
            continue
        rv5 = float(rvol_5m[ix])
        if rv5 < 1.8 or float(perp_tb_5m[ix]) < 0.53:
            continue
        oir = float(oi_ratio_24h[ix]) if bool(has_oi[ix]) else 1.0
        b_h = float(bin_fr_bps_h[ix])
        fr_raw = float(last_fr_bps[ix])

        # `P3_M2`: Synchronized global negative carry across Binance + PeerCEX/HL + 1m discount + L2 Ask Vacuum
        peer_neg = bool(has_peer[ix]) and float(peer_fr_bps_h[ix]) <= -0.8
        hl_neg = bool(has_hl[ix]) and float(hl_fr_bps_h[ix]) <= -1.2
        no_pos_div = (not bool(has_peer[ix]) or float(peer_fr_bps_h[ix]) <= 0.3) and (
            not bool(has_hl[ix]) or float(hl_fr_bps_h[ix]) <= 0.3
        )
        book_vac_ok = (not bool(has_book[ix])) or (float(book_imb_1[ix]) >= 0.05 or float(book_imb_2[ix]) >= 0.05)
        if (
            fr_raw <= -4.0
            and b_h <= -0.8
            and (peer_neg or hl_neg)
            and no_pos_div
            and oir >= 1.10
            and (not bool(has_prem[ix]) or float(prem_close_bps[ix]) <= -15.0)
            and book_vac_ok
        ):
            ext_c = min(
                float(peer_fr_bps_h[ix]) if peer_neg else 0.0,
                float(hl_fr_bps_h[ix]) if hl_neg else 0.0,
            )
            st_m2 = (abs(b_h) + abs(ext_c)) * 15.0 * oir
            events["P3_M2_GLOBAL_3VENUE_SQUEEZE_LONG"].append((ix, 1, st_m2, ext_c))

        # `P4_M3`: On-Chain Hyperliquid 1h Carry / Validator-Oracle Premium Leading Lagging Binance Perp
        if bool(has_hl[ix]) and fr_raw <= 3.0 and b_h <= 0.4 and oir >= 1.08:
            hl_f = float(hl_fr_bps_h[ix])
            hl_p = float(hl_prem_bps[ix])
            lead_spread = hl_f - b_h
            book_lead_ok = (not bool(has_book[ix])) or (float(book_imb_1[ix]) >= 0.06)
            if (hl_f >= 1.8 or hl_p >= 10.0) and lead_spread >= 1.4 and book_lead_ok:
                st_m3 = (lead_spread * 15.0 + max(0.0, hl_p)) * oir
                events["P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG"].append((ix, 1, st_m3, lead_spread))

    # 3. `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG`
    if bool((has_prem & has_oi & (has_hl | has_peer)).any()):
        hl_solv = has_hl & (hl_prem_bps >= -12.0) & (hl_fr_bps_h >= -2.0)
        peer_solv = has_peer & ((peer_fr_bps_h - bin_fr_bps_h) >= 1.2) & (peer_fr_bps_h >= -1.5)
        m4_idxs = np.flatnonzero(
            liq_gate
            & has_prem
            & has_oi
            & (hl_solv | peer_solv)
            & (oi_vel_15m <= -0.015)
            & (ret_10m <= -0.020)
            & (ret_4h > -0.090)
            & (prem_low_disloc_bps <= -35.0)
            & (ret_1m > 0.0)
            & (perp_tb_1m >= 0.52)
            & ((~has_book) | (book_imb_1 >= -0.05))
        )
        for ix_4 in m4_idxs.tolist():
            if ix_4 < 250 or ix_4 >= n - 95:
                continue
            dis_v = abs(float(prem_low_disloc_bps[ix_4]))
            events["P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG"].append(
                (ix_4, 1, dis_v * (1.0 + abs(float(oi_vel_15m[ix_4])) * 10.0), float(hl_prem_bps[ix_4]))
            )

    all_trades: list[dict[str, Any]] = []
    for m_id, ev_list in events.items():
        if not ev_list:
            continue
        prof = P3_P4_MECHANISM_PROFILES[m_id]
        all_trades.extend(
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
                profile=prof,
                symbol=symbol,
            )
        )
    return all_trades


def evaluate_p3_p4_leaderboard(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates all `(mech_id, exec_mode)` pairs under True In-Fold Refit CPCV (`15` splits)."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    n_trials = int(trades_df[["mech_id", "exec_mode"]].drop_duplicates().shape[0])
    return evaluate_infold_cpcv_and_treasure_metrics(trades_df, total_trials=max(1, n_trials))
