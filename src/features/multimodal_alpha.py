"""
Causal Multi-Modal Microstructure Feature Library & Out-of-Sample IC Atlas (Issue #68 / M16-2).

Implements ~104 strictly causal features across 4 orthogonal families over the M16-1 datasets
(`bars_1m` / `bars_1h`, `spot_bars_1m`, `metrics_5m`, `funding`):
  - Family A: Derivatives Positioning, Open Interest Velocity & 4-Quadrant Liquidation Physics
  - Family B: Spot-vs-Perp Cash-Flow Lead-Lag, 1m Perpetual Basis & Spot-Perp CVD Divergence
  - Family C: Extreme Funding & Dynamic Settlement Clock (`1h / 2h / 4h / 8h`) Microstructure
  - Family D: Expanded 200+ Mid-Cap Dispersion, Causal ADV/Spread Gates & Post-Listing Decay

Strict Causality Invariants:
  1. Base decision timestamps represent **bar close** (`bar_close_ts = bar_open_ts + bar_sec`).
  2. `metrics_5m` snapshots (`M_ts`) are shifted by `+metrics_safety_lag_sec` (default `+60s`)
     before backward `merge_asof(..., direction='backward')` onto `bar_close_ts`.
  3. `funding` settlements (`F_ts`) join via backward `merge_asof`, and `minutes_to_next_settlement`
     is projected strictly from `last_settlement_ts + last_interval_hours * 3600` (zero future peek).
  4. Every rolling mean, z-score, volatility, ADV filter, and cross-sectional rank uses strictly
     trailing data (`t <= bar_close_ts`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erfinv

from src.features.derivatives import DerivativesFeatures
from src.features.microstructure import rolling_amihud_illiquidity, rolling_corwin_schultz_spread


@dataclass(frozen=True)
class MultiModalFeatureConfig:
    """Configuration for causal multi-modal feature synthesis."""

    decision_freq: str = "15min"  # '5min', '15min', or '1h'
    metrics_safety_lag_sec: int = 60
    min_adv_usd: float = 15_000_000.0
    zscore_clip: float = 6.0


def _freq_to_seconds(freq: str) -> int:
    clean = freq.strip().lower()
    if clean in ("1m", "1min"):
        return 60
    if clean in ("5m", "5min"):
        return 300
    if clean in ("15m", "15min"):
        return 900
    if clean in ("1h", "60m", "60min"):
        return 3600
    if clean in ("4h", "240m", "240min"):
        return 14400
    raise ValueError(f"Unsupported decision_freq: {freq}")


def _causal_zscore(series: pd.Series, window: int, min_periods: int | None = None, clip: float = 6.0) -> pd.Series:
    """Strictly causal trailing rolling z-score with zero lookahead."""
    w = max(2, int(window))
    mp = max(2, min_periods if min_periods is not None else min(w, max(4, w // 3)))
    mean = series.rolling(window=w, min_periods=mp).mean()
    std = series.rolling(window=w, min_periods=mp).std(ddof=0)
    z = (series - mean) / std.where(std > 1e-12, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip)


def _relu(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0)


def resample_bars_to_decision_grid(bars_1m: pd.DataFrame, decision_freq: str = "15min") -> pd.DataFrame:
    """
    Deterministically resamples `1m` (or `1h`) OHLCV + Taker Flow bars onto `decision_freq`
    using C-vectorized integer bucket `np.reduceat` (zero DatetimeIndex allocation),
    producing `open_ts` (`timestamp`) and `decision_ts` (`open_ts + bar_sec`, exact bar close).
    """
    if bars_1m is None or bars_1m.empty:
        return pd.DataFrame()
    bar_sec = _freq_to_seconds(decision_freq)

    ts_raw = bars_1m["timestamp"].to_numpy(dtype=np.int64, copy=False)
    n_raw = len(ts_raw)
    if n_raw > 1 and not bool(np.all(ts_raw[:-1] < ts_raw[1:])):
        order = np.argsort(ts_raw, kind="mergesort")
        ts_sorted = ts_raw[order]
        keep_dedup = np.empty(n_raw, dtype=bool)
        keep_dedup[:-1] = ts_sorted[:-1] != ts_sorted[1:]
        keep_dedup[-1] = True
        idx_sel = order[keep_dedup]
        df = bars_1m.iloc[idx_sel]
        ts = ts_sorted[keep_dedup]
    else:
        df = bars_1m
        ts = ts_raw

    close_a = df["close"].to_numpy(dtype=np.float64, copy=False)
    open_a = df["open"].to_numpy(dtype=np.float64, copy=False) if "open" in df.columns else close_a
    valid_px = np.isfinite(close_a) & np.isfinite(open_a)
    if not bool(valid_px.all()):
        if not bool(valid_px.any()):
            return pd.DataFrame()
        df = df.iloc[valid_px]
        ts = ts[valid_px]
        close_a = close_a[valid_px]
        open_a = open_a[valid_px]

    if len(ts) >= 2:
        med_step = int(np.median(np.diff(ts)))
    else:
        med_step = bar_sec

    sym_val = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else None

    if med_step == bar_sec:
        out = df.reset_index(drop=True).copy()
        out["open_ts"] = ts
        out["decision_ts"] = ts + bar_sec
        for opt_c in ("open", "high", "low", "volume", "trade_count", "taker_buy_vol", "taker_buy_quote_volume"):
            if opt_c not in out.columns:
                out[opt_c] = out["close"] if opt_c in ("open", "high", "low") else 0.0
        return out

    buckets = (ts // bar_sec) * bar_sec
    is_start = np.empty(len(buckets), dtype=bool)
    is_start[0] = True
    is_start[1:] = buckets[1:] != buckets[:-1]
    starts = np.flatnonzero(is_start)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = len(buckets) - 1

    open_ts = buckets[starts]
    high_a = df["high"].to_numpy(dtype=np.float64, copy=False) if "high" in df.columns else close_a
    low_a = df["low"].to_numpy(dtype=np.float64, copy=False) if "low" in df.columns else close_a
    vol_a = df["volume"].to_numpy(dtype=np.float64, copy=False) if "volume" in df.columns else None
    qvol_a = df["quote_volume"].to_numpy(dtype=np.float64, copy=False)

    res_cols: dict[str, Any] = {
        "open": open_a[starts],
        "high": np.maximum.reduceat(high_a, starts),
        "low": np.minimum.reduceat(low_a, starts),
        "close": close_a[ends],
        "volume": np.add.reduceat(vol_a, starts) if vol_a is not None else np.zeros(len(starts), dtype=np.float64),
        "quote_volume": np.add.reduceat(qvol_a, starts),
        "trade_count": (
            np.add.reduceat(df["trade_count"].to_numpy(dtype=np.float64, copy=False), starts)
            if "trade_count" in df.columns
            else np.zeros(len(starts), dtype=np.float64)
        ),
        "taker_buy_vol": (
            np.add.reduceat(df["taker_buy_vol"].to_numpy(dtype=np.float64, copy=False), starts)
            if "taker_buy_vol" in df.columns
            else np.zeros(len(starts), dtype=np.float64)
        ),
        "taker_buy_quote_volume": (
            np.add.reduceat(df["taker_buy_quote_volume"].to_numpy(dtype=np.float64, copy=False), starts)
            if "taker_buy_quote_volume" in df.columns
            else np.zeros(len(starts), dtype=np.float64)
        ),
        "timestamp": open_ts,
        "open_ts": open_ts,
        "decision_ts": open_ts + bar_sec,
    }
    if sym_val is not None:
        res_cols["symbol"] = sym_val
    return pd.DataFrame(res_cols)


def compute_symbol_multimodal_features(
    perp_bars: pd.DataFrame,
    spot_bars: pd.DataFrame | None = None,
    metrics_5m: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    symbol: str | None = None,
    onboard_timestamp: int | None = None,
    config: MultiModalFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Computes ~94 single-asset causal multi-modal features across Families A, B, C, and D
    aligned on `decision_ts` (bar close). Cross-sectional panel features (+10 -> 104 total)
    are added across the universe via `compute_cross_sectional_panel_features()`.
    """
    cfg = config or MultiModalFeatureConfig()
    bar_sec = _freq_to_seconds(cfg.decision_freq)
    bph = max(1, 3600 // bar_sec)  # bars per hour (e.g., 4 for 15min, 12 for 5min, 1 for 1h)

    base = resample_bars_to_decision_grid(perp_bars, cfg.decision_freq)
    if base.empty:
        return pd.DataFrame()

    sym = symbol or (str(base["symbol"].iloc[0]) if "symbol" in base.columns else "UNKNOWN")
    base = base.sort_values("decision_ts").reset_index(drop=True)
    n = len(base)

    grid_df = pd.DataFrame({"decision_ts": base["decision_ts"].to_numpy(dtype=np.int64)})
    cols: dict[str, Any] = {
        "timestamp": base["open_ts"].to_numpy(dtype=np.int64),
        "decision_ts": base["decision_ts"].to_numpy(dtype=np.int64),
        "symbol": sym,
        "close": base["close"].astype(np.float64),
        "quote_volume": base["quote_volume"].astype(np.float64),
    }

    close_s = base["close"].astype(np.float64)
    open_s = base["open"].astype(np.float64)
    high_s = base["high"].astype(np.float64)
    low_s = base["low"].astype(np.float64)
    qvol_s = base["quote_volume"].astype(np.float64)
    tb_qvol_s = base["taker_buy_quote_volume"].astype(np.float64)

    # ------------------------------------------------------------------
    # Multi-Horizon Price Returns & Z-Scores
    # ------------------------------------------------------------------
    horizons = {
        "1b": 1,
        "1h": bph,
        "4h": 4 * bph,
        "12h": 12 * bph,
        "24h": 24 * bph,
    }
    z_win_24h = max(8, 24 * bph)
    z_win_72h = max(16, 72 * bph)
    z_win_7d = max(24, 168 * bph)

    log_p = np.log(close_s.clip(lower=1e-12))
    for lbl, steps in horizons.items():
        r = log_p.diff(steps).fillna(0.0)
        cols[f"ret_{lbl}"] = r
        cols[f"ret_z_{lbl}"] = _causal_zscore(r, window=max(z_win_24h, steps * 4), clip=cfg.zscore_clip)

    # ------------------------------------------------------------------
    # FAMILY A: Derivatives Positioning, OI Velocity & Quadrant Physics
    # ------------------------------------------------------------------
    if metrics_5m is not None and not metrics_5m.empty:
        m_df = metrics_5m.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").copy()
        m_df["available_ts"] = m_df["timestamp"].astype(np.int64) + int(cfg.metrics_safety_lag_sec)
        m_cols = [
            "available_ts",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ]
        present_m = [c for c in m_cols if c in m_df.columns]
        joined_m = pd.merge_asof(
            grid_df,
            m_df[present_m].sort_values("available_ts"),
            left_on="decision_ts",
            right_on="available_ts",
            direction="backward",
        )
        oi_units = pd.to_numeric(joined_m.get("sum_open_interest"), errors="coerce").ffill()
        oi_usd = pd.to_numeric(joined_m.get("sum_open_interest_value"), errors="coerce").ffill()
        top_cnt_ls = pd.to_numeric(joined_m.get("count_toptrader_long_short_ratio"), errors="coerce").ffill()
        top_pos_ls = pd.to_numeric(joined_m.get("sum_toptrader_long_short_ratio"), errors="coerce").ffill()
        retail_ls = pd.to_numeric(joined_m.get("count_long_short_ratio"), errors="coerce").ffill()
        taker_ls = pd.to_numeric(joined_m.get("sum_taker_long_short_vol_ratio"), errors="coerce").ffill()
    else:
        oi_units = pd.Series(np.nan, index=base.index)
        oi_usd = pd.Series(np.nan, index=base.index)
        top_cnt_ls = pd.Series(np.nan, index=base.index)
        top_pos_ls = pd.Series(np.nan, index=base.index)
        retail_ls = pd.Series(np.nan, index=base.index)
        taker_ls = pd.Series(np.nan, index=base.index)

    for lbl, steps in horizons.items():
        vel = DerivativesFeatures.open_interest_velocity(oi_units, window=steps).fillna(0.0).clip(-2.0, 2.0)
        cols[f"oi_vel_{lbl}"] = vel
        cols[f"oi_vel_z_{lbl}"] = _causal_zscore(vel, window=max(z_win_24h, steps * 4), clip=cfg.zscore_clip)

    for lbl in ("1h", "4h", "12h"):
        steps = horizons[lbl]
        cols[f"oi_accel_{lbl}"] = (
            DerivativesFeatures.open_interest_acceleration(oi_units, window=steps).fillna(0.0).clip(-2.0, 2.0)
        )
        rz = cols[f"ret_z_{lbl}"]
        oiz = cols[f"oi_vel_z_{lbl}"]
        # 4 Structural Derivatives Quadrants
        cols[f"quad_new_longs_{lbl}"] = (_relu(rz) * _relu(oiz)).clip(0.0, 25.0)
        cols[f"quad_new_shorts_{lbl}"] = (_relu(-rz) * _relu(oiz)).clip(0.0, 25.0)
        cols[f"quad_long_liq_flush_{lbl}"] = (_relu(-rz) * _relu(-oiz)).clip(0.0, 25.0)
        cols[f"quad_short_cover_cap_{lbl}"] = (_relu(rz) * _relu(-oiz)).clip(0.0, 25.0)

    whale_pos_z = _causal_zscore(np.log(top_pos_ls.clip(lower=1e-4)), window=z_win_72h, clip=cfg.zscore_clip)
    whale_cnt_z = _causal_zscore(np.log(top_cnt_ls.clip(lower=1e-4)), window=z_win_72h, clip=cfg.zscore_clip)
    retail_z = _causal_zscore(np.log(retail_ls.clip(lower=1e-4)), window=z_win_72h, clip=cfg.zscore_clip)

    cols["whale_pos_ls_z"] = whale_pos_z
    cols["whale_acc_ls_z"] = whale_cnt_z
    cols["retail_ls_z"] = retail_z
    cols["smart_money_div_raw"] = DerivativesFeatures.smart_money_divergence(top_pos_ls, retail_ls).fillna(0.0)
    cols["smart_money_div_z_24h"] = (whale_pos_z - retail_z).clip(-cfg.zscore_clip, cfg.zscore_clip)
    cols["smart_money_div_vel_4h"] = cols["smart_money_div_z_24h"].diff(horizons["4h"]).fillna(0.0)

    qvol_24h = qvol_s.rolling(window=z_win_24h, min_periods=max(1, bph)).sum()
    oi_crowd = (oi_usd / qvol_24h.where(qvol_24h > 1.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols["oi_to_qvol_24h"] = oi_crowd.clip(0.0, 20.0)
    cols["oi_to_qvol_z_7d"] = _causal_zscore(oi_crowd, window=z_win_7d, clip=cfg.zscore_clip)
    cols["metrics_taker_imb"] = DerivativesFeatures.taker_flow_imbalance(taker_ls.fillna(1.0)).fillna(0.0)

    # ------------------------------------------------------------------
    # FAMILY B: Spot-vs-Perp Cash-Flow Lead-Lag & 1m Perpetual Basis
    # ------------------------------------------------------------------
    perp_signed_flow = (2.0 * tb_qvol_s - qvol_s).fillna(0.0)

    if spot_bars is not None and not spot_bars.empty:
        sp_grid = resample_bars_to_decision_grid(spot_bars, cfg.decision_freq)
        sp_merged = pd.merge_asof(
            grid_df,
            sp_grid[["decision_ts", "close", "quote_volume", "taker_buy_quote_volume"]].sort_values("decision_ts"),
            on="decision_ts",
            direction="backward",
            tolerance=bar_sec * 4,
        )
        sp_close = pd.to_numeric(sp_merged["close"], errors="coerce")
        sp_qvol = pd.to_numeric(sp_merged["quote_volume"], errors="coerce").fillna(0.0)
        sp_tb_qvol = pd.to_numeric(sp_merged["taker_buy_quote_volume"], errors="coerce").fillna(0.0)
        has_spot_bar = sp_close.notna() & (sp_close > 0.0)
    else:
        sp_close = pd.Series(np.nan, index=base.index)
        sp_qvol = pd.Series(0.0, index=base.index)
        sp_tb_qvol = pd.Series(0.0, index=base.index)
        has_spot_bar = pd.Series(False, index=base.index)

    raw_basis_bps = (((close_s - sp_close) / sp_close.where(has_spot_bar, np.nan)) * 1e4).clip(-500.0, 500.0)
    basis_bps = raw_basis_bps.fillna(0.0)
    cols["basis_bps"] = basis_bps
    cols["basis_z_4h"] = _causal_zscore(raw_basis_bps, window=max(6, 4 * bph), clip=cfg.zscore_clip)
    cols["basis_z_24h"] = _causal_zscore(raw_basis_bps, window=z_win_24h, clip=cfg.zscore_clip)
    cols["basis_z_72h"] = _causal_zscore(raw_basis_bps, window=z_win_72h, clip=cfg.zscore_clip)
    cols["basis_vel_1b"] = basis_bps.diff(1).fillna(0.0)
    cols["basis_vel_1h"] = basis_bps.diff(horizons["1h"]).fillna(0.0)
    cols["basis_vel_4h"] = basis_bps.diff(horizons["4h"]).fillna(0.0)
    cols["basis_reversion_pressure"] = (-cols["basis_z_24h"] * (1.0 + _relu(cols["oi_to_qvol_z_7d"]))).clip(-15.0, 15.0)
    cols["basis_trend_confirm"] = (np.sign(cols["ret_4h"]) * cols["basis_z_24h"]).clip(
        -cfg.zscore_clip, cfg.zscore_clip
    )

    spot_signed_flow = (2.0 * sp_tb_qvol - sp_qvol).fillna(0.0)
    for lbl in ("1b", "1h", "4h"):
        w_b = max(1, horizons[lbl])
        p_sum_q = qvol_s.rolling(w_b, min_periods=1).sum()
        p_sum_tb = tb_qvol_s.rolling(w_b, min_periods=1).sum()
        p_share = (p_sum_tb / p_sum_q.where(p_sum_q > 0.0, np.nan)).fillna(0.5).clip(0.0, 1.0)

        s_sum_q = sp_qvol.rolling(w_b, min_periods=1).sum()
        s_sum_tb = sp_tb_qvol.rolling(w_b, min_periods=1).sum()
        s_share = (s_sum_tb / s_sum_q.where(s_sum_q > 0.0, np.nan)).fillna(0.5).clip(0.0, 1.0)

        cols[f"perp_taker_share_{lbl}"] = p_share - 0.5
        cols[f"spot_taker_share_{lbl}"] = (s_share - 0.5).where(s_sum_q > 0.0, 0.0)
        cols[f"spot_vs_perp_share_div_{lbl}"] = (s_share - p_share).where(s_sum_q > 0.0, 0.0)

    for lbl in ("1h", "4h"):
        w_b = max(2, horizons[lbl])
        p_cvd = perp_signed_flow.rolling(w_b, min_periods=1).sum()
        s_cvd = spot_signed_flow.rolling(w_b, min_periods=1).sum()
        p_cvd_z = _causal_zscore(p_cvd, window=z_win_72h, clip=cfg.zscore_clip)
        s_cvd_z = _causal_zscore(s_cvd.where(sp_qvol > 0.0, np.nan), window=z_win_72h, clip=cfg.zscore_clip)
        cols[f"perp_cvd_z_{lbl}"] = p_cvd_z
        cols[f"spot_cvd_z_{lbl}"] = s_cvd_z

    for lbl in ("1b", "1h", "4h", "12h"):
        w_b = max(1, horizons[lbl])
        p_cvd = _causal_zscore(
            perp_signed_flow.rolling(w_b, min_periods=1).sum(), window=z_win_72h, clip=cfg.zscore_clip
        )
        s_cvd = _causal_zscore(
            spot_signed_flow.rolling(w_b, min_periods=1)
            .sum()
            .where(sp_qvol.rolling(w_b, min_periods=1).sum() > 0.0, np.nan),
            window=z_win_72h,
            clip=cfg.zscore_clip,
        )
        cols[f"spot_vs_perp_cvd_div_{lbl}"] = (s_cvd - p_cvd).where(has_spot_bar, 0.0).clip(-10.0, 10.0)

    tot_1h = sp_qvol.rolling(horizons["1h"], min_periods=1).sum() + qvol_s.rolling(horizons["1h"], min_periods=1).sum()
    tot_24h = sp_qvol.rolling(z_win_24h, min_periods=1).sum() + qvol_24h
    dom_1h = (sp_qvol.rolling(horizons["1h"], min_periods=1).sum() / tot_1h.where(tot_1h > 0.0, np.nan)).fillna(0.0)
    dom_24h = (sp_qvol.rolling(z_win_24h, min_periods=1).sum() / tot_24h.where(tot_24h > 0.0, np.nan)).fillna(0.0)
    cols["spot_vol_dominance_1h"] = dom_1h
    cols["spot_vol_dominance_24h"] = dom_24h
    cols["spot_vol_dom_shock_z"] = _causal_zscore(
        dom_1h.where(has_spot_bar, np.nan), window=z_win_72h, clip=cfg.zscore_clip
    )
    sp_ret_1b = np.log(sp_close.clip(lower=1e-12)).diff(1).fillna(0.0)
    cols["spot_lead_ret_div_1b"] = (sp_ret_1b - cols["ret_1b"]).where(has_spot_bar, 0.0)
    cols["spot_buy_perp_liq_combo"] = (_relu(cols["spot_vs_perp_cvd_div_1h"]) * cols["quad_long_liq_flush_1h"]).clip(
        0.0, 50.0
    )

    # ------------------------------------------------------------------
    # FAMILY C: Extreme Funding & Dynamic Settlement Clock (1h/2h/4h/8h)
    # ------------------------------------------------------------------
    if funding is not None and not funding.empty:
        f_df = funding.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").copy()
        f_df["f_ts"] = f_df["timestamp"].astype(np.int64)
        if "interval_hours" not in f_df.columns:
            f_df["interval_hours"] = 8
        joined_f = pd.merge_asof(
            grid_df,
            f_df[["f_ts", "funding_rate", "interval_hours"]].sort_values("f_ts"),
            left_on="decision_ts",
            right_on="f_ts",
            direction="backward",
        )
        f_rate = pd.to_numeric(joined_f["funding_rate"], errors="coerce").fillna(0.0)
        f_iv_h = pd.to_numeric(joined_f["interval_hours"], errors="coerce").fillna(8.0).clip(1.0, 8.0)
        last_f_ts = pd.to_numeric(joined_f["f_ts"], errors="coerce")
    else:
        f_rate = pd.Series(0.0, index=base.index)
        f_iv_h = pd.Series(8.0, index=base.index)
        last_f_ts = pd.Series(np.nan, index=base.index)

    f_bps = (f_rate * 1e4).clip(-300.0, 300.0)
    settlements_per_day = 24.0 / f_iv_h
    cols["funding_rate_bps"] = f_bps
    cols["funding_ann_pct"] = f_rate * settlements_per_day * 365.0 * 100.0
    cols["funding_z_7d"] = _causal_zscore(f_bps, window=z_win_7d, clip=cfg.zscore_clip)
    cols["funding_vel_24h"] = f_bps.diff(z_win_24h).fillna(0.0)
    cols["funding_interval_hours"] = f_iv_h
    cols["is_compressed_funding_clock"] = (f_iv_h <= 4.0).astype(np.float64)
    cols["clock_compression_ratio"] = 8.0 / f_iv_h

    # Strictly causal settlement clock projection from last observed settlement + interval_hours
    dec_ts_s = grid_df["decision_ts"].astype(np.float64)
    mins_since = ((dec_ts_s - last_f_ts) / 60.0).where(last_f_ts.notna(), np.nan)
    next_proj_ts = last_f_ts + f_iv_h * 3600.0
    mins_to_next = ((next_proj_ts - dec_ts_s) / 60.0).where(last_f_ts.notna(), np.nan)

    cols["mins_since_settlement"] = mins_since.fillna(240.0).clip(0.0, 960.0)
    cols["mins_to_next_settlement"] = mins_to_next.fillna(240.0).clip(-60.0, 960.0)

    pre_win = ((mins_to_next >= 5.0) & (mins_to_next <= 60.0)).astype(np.float64)
    post_win = ((mins_since >= 0.0) & (mins_since <= 30.0)).astype(np.float64)
    cols["in_pre_settlement_window"] = pre_win
    cols["in_post_settlement_window"] = post_win
    cols["pre_settle_neg_squeeze_20bps"] = pre_win * (f_bps <= -20.0).astype(np.float64) * (-f_bps / 20.0)
    cols["pre_settle_neg_squeeze_40bps"] = pre_win * (f_bps <= -40.0).astype(np.float64) * (-f_bps / 40.0)
    cols["pre_settle_neg_squeeze_80bps"] = pre_win * (f_bps <= -80.0).astype(np.float64) * (-f_bps / 80.0)
    cols["post_settle_neg_snapback"] = post_win * (f_bps <= -20.0).astype(np.float64) * (-cols["ret_1b"])
    cols["pre_settle_pos_crowding_20bps"] = pre_win * (f_bps >= 20.0).astype(np.float64) * (f_bps / 20.0)
    cols["compressed_clock_neg_funding_impulse"] = (
        cols["clock_compression_ratio"] * _relu(-f_bps - 15.0) * (1.0 + _relu(cols["spot_vs_perp_cvd_div_1h"]))
    ).clip(0.0, 100.0)
    cols["carry_plus_basis_bps"] = f_bps + cols["basis_bps"]

    # ------------------------------------------------------------------
    # FAMILY D (Single-Asset Part): Liquidity Gates, Spread, Vol & Listing Age
    # ------------------------------------------------------------------
    bars_7d = max(1, 7 * 24 * bph)
    bars_30d = max(1, 30 * 24 * bph)
    adv_7d_lag1 = (qvol_s.shift(1).rolling(bars_7d, min_periods=max(1, z_win_24h)).sum() / 7.0).fillna(0.0)
    adv_30d_lag1 = (qvol_s.shift(1).rolling(bars_30d, min_periods=max(1, z_win_24h)).sum() / 30.0).fillna(0.0)
    cols["adv_7d_usd_lag1"] = adv_7d_lag1
    cols["adv_30d_usd_lag1"] = adv_30d_lag1
    cols["tradable_adv_15m_gate"] = (adv_7d_lag1 >= cfg.min_adv_usd).astype(np.float64)

    cols["corwin_schultz_spread_24h"] = pd.Series(
        rolling_corwin_schultz_spread(high_s, low_s, window=z_win_24h), index=base.index
    ).fillna(0.0)
    cols["amihud_illiq_24h"] = pd.Series(
        rolling_amihud_illiquidity(cols["ret_1b"], qvol_s, window=z_win_24h), index=base.index
    ).fillna(0.0)

    log_hl_sq = (np.log(high_s.clip(lower=1e-12) / low_s.clip(lower=1e-12))) ** 2
    park_24h = np.sqrt((log_hl_sq.rolling(z_win_24h, min_periods=2).mean() / (4.0 * np.log(2.0))).clip(lower=0.0))
    cols["parkinson_vol_24h"] = park_24h.fillna(0.0)
    cols["vol_of_vol_24h"] = _causal_zscore(park_24h, window=z_win_7d, clip=cfg.zscore_clip)

    qvol_1h = qvol_s.rolling(horizons["1h"], min_periods=1).sum()
    qvol_4h = qvol_s.rolling(horizons["4h"], min_periods=1).sum()
    avg_1h_in_24h = qvol_24h / 24.0
    avg_4h_in_7d = (qvol_s.rolling(bars_7d, min_periods=max(1, z_win_24h)).sum()) / 42.0
    cols["rvol_shock_1h_vs_24h"] = (
        (qvol_1h / avg_1h_in_24h.where(avg_1h_in_24h > 0.0, np.nan)).fillna(1.0).clip(0.0, 25.0)
    )
    cols["rvol_shock_4h_vs_7d"] = (qvol_4h / avg_4h_in_7d.where(avg_4h_in_7d > 0.0, np.nan)).fillna(1.0).clip(0.0, 25.0)

    listing_ts = int(onboard_timestamp) if onboard_timestamp and onboard_timestamp > 0 else int(base["open_ts"].iloc[0])
    days_listed = ((grid_df["decision_ts"] - listing_ts) / 86400.0).clip(lower=0.0)
    cols["days_since_listing"] = days_listed
    is_new_3_30 = ((days_listed >= 3.0) & (days_listed <= 30.0)).astype(np.float64)
    cols["is_newly_listed_3_30d"] = is_new_3_30
    cols["post_listing_vol_decay_score"] = is_new_3_30 * np.exp(-days_listed / 14.0) * cols["vol_of_vol_24h"]
    cols["post_listing_fade_signal"] = is_new_3_30 * _relu(cols["ret_z_24h"]) * _relu(cols["funding_rate_bps"] / 10.0)

    bar_range = (high_s - low_s).clip(lower=1e-12)
    body_top = np.maximum(open_s, close_s)
    body_bot = np.minimum(open_s, close_s)
    wick_asym = ((body_bot - low_s) - (high_s - body_top)) / bar_range
    cols["wick_rejection_asymmetry_1h"] = wick_asym.rolling(horizons["1h"], min_periods=1).mean().fillna(0.0)

    out = pd.DataFrame(cols)
    assert len(out) == n
    return out


def _vectorized_cs_gaussian_rank(df: pd.DataFrame, group_col: str, val_col: str) -> np.ndarray:
    """C-vectorized cross-sectional Gaussian rank (`u = (rank - 0.5)/N -> sqrt(2)*erfinv(2u - 1)`)."""
    s = pd.to_numeric(df[val_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    tmp = pd.DataFrame({"g": df[group_col].to_numpy(), "v": s.to_numpy(dtype=np.float64)})
    cnt = tmp.groupby("g")["v"].transform("count")
    rnk = tmp.groupby("g")["v"].rank(method="average", na_option="keep")
    valid = rnk.notna() & (cnt > 1)
    u = ((rnk - 0.5) / cnt.where(cnt > 1, 1.0)).clip(1e-4, 1.0 - 1e-4)
    out = np.zeros(len(df), dtype=np.float64)
    if bool(valid.any()):
        out[valid.to_numpy()] = np.sqrt(2.0) * erfinv(2.0 * u[valid].to_numpy(dtype=np.float64) - 1.0)
    return out


def _fast_cs_zscore_preindexed(vals: np.ndarray, ts_codes: np.ndarray, n_ts: int) -> np.ndarray:
    """Zero-allocation `np.bincount` cross-sectional z-score (`ddof=0`) over pre-factorized `ts_codes`."""
    v = vals.astype(np.float64, copy=False)
    finite = np.isfinite(v)
    if bool(finite.all()):
        cnt = np.bincount(ts_codes, minlength=n_ts).astype(np.float64)
        mu_g = np.bincount(ts_codes, weights=v, minlength=n_ts) / np.maximum(cnt, 1.0)
        dv = v - mu_g[ts_codes]
        var_g = np.bincount(ts_codes, weights=dv * dv, minlength=n_ts) / np.maximum(cnt, 1.0)
        std_row = np.sqrt(np.maximum(var_g, 0.0))[ts_codes]
        return np.where(std_row > 1e-8, dv / np.maximum(std_row, 1e-8), 0.0)

    m_f = finite.astype(np.float64)
    v0 = np.where(finite, v, 0.0)
    cnt = np.bincount(ts_codes, weights=m_f, minlength=n_ts)
    mu_g = np.bincount(ts_codes, weights=v0, minlength=n_ts) / np.maximum(cnt, 1.0)
    dv = np.where(finite, v0 - mu_g[ts_codes], 0.0)
    var_g = np.bincount(ts_codes, weights=dv * dv, minlength=n_ts) / np.maximum(cnt, 1.0)
    std_row = np.sqrt(np.maximum(var_g, 0.0))[ts_codes]
    return np.where(finite & (std_row > 1e-8), dv / np.maximum(std_row, 1e-8), 0.0)


def _vectorized_cs_zscore(df: pd.DataFrame, group_col: str, val_col: str) -> np.ndarray:
    """C-vectorized cross-sectional z-score (`(x - mean) / std` with `ddof=0` matching `cross_sectional_zscore`)."""
    ts_codes, uniq_ts = pd.factorize(df[group_col], sort=False)
    v = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return _fast_cs_zscore_preindexed(v, ts_codes, len(uniq_ts))


def _sliding_beta_1d(r_sym: np.ndarray, r_btc: np.ndarray, window: int = 96, min_periods: int = 16) -> np.ndarray:
    """Exact `O(N)` cumulative-sum causal rolling beta (`cov(r_sym, r_btc) / var(r_btc)`)."""
    n = len(r_sym)
    out = np.ones(n, dtype=np.float64)
    if n < min_periods:
        return out
    x = np.nan_to_num(r_btc, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(r_sym, nan=0.0, posinf=0.0, neginf=0.0)
    cx = np.r_[0.0, np.cumsum(x)]
    cy = np.r_[0.0, np.cumsum(y)]
    cxy = np.r_[0.0, np.cumsum(x * y)]
    cxx = np.r_[0.0, np.cumsum(x * x)]

    idx_end = np.arange(1, n + 1)
    idx_start = np.maximum(0, idx_end - window)
    cnt = (idx_end - idx_start).astype(np.float64)

    sx = cx[idx_end] - cx[idx_start]
    sy = cy[idx_end] - cy[idx_start]
    sxy = cxy[idx_end] - cxy[idx_start]
    sxx = cxx[idx_end] - cxx[idx_start]

    denom = cnt - 1.0
    valid_cnt = cnt >= min_periods
    cov = np.where(valid_cnt, (sxy - (sx * sy) / cnt) / np.maximum(denom, 1.0), 0.0)
    var = np.where(valid_cnt, (sxx - (sx * sx) / cnt) / np.maximum(denom, 1.0), 0.0)
    valid = valid_cnt & (var > 1e-12)
    out[valid] = np.clip(cov[valid] / np.maximum(var[valid], 1e-12), -0.5, 3.5)
    return out


def compute_cross_sectional_panel_features(
    symbol_frames: dict[str, pd.DataFrame] | pd.DataFrame,
    benchmark_symbol: str = "BTCUSDT",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Computes Family D cross-sectional panel features (+10 features -> ~104 total) across all
    symbols active at each `decision_ts` without any temporal lookahead (`groupby('decision_ts')` only):
      - Gaussian cross-sectional ranks (`cs_rank_*`)
      - Beta-neutralized 24h residual momentum vs BTC (`cs_resid_mom_24h`)
      - Cross-sectional return dispersion (`cs_dispersion_24h`)
      - Relative OI shock z-score (`cs_rel_oi_shock_z`)
    """
    if isinstance(symbol_frames, dict):
        valid = [df for df in symbol_frames.values() if df is not None and not df.empty]
        if not valid:
            return pd.DataFrame()
        panel = pd.concat(valid, ignore_index=True)
    else:
        panel = symbol_frames if inplace else symbol_frames.copy()
    if panel.empty:
        return panel

    # Sort by ["symbol", "decision_ts"] first to compute exact O(N) causal rolling beta per symbol
    panel.sort_values(["symbol", "decision_ts"], inplace=True, ignore_index=True)
    dec_ts_sym = panel["decision_ts"].to_numpy(dtype=np.int64, copy=False)
    sym_codes, _ = pd.factorize(panel["symbol"], sort=False)
    btc_mask = (panel["symbol"] == benchmark_symbol).to_numpy()

    if bool(btc_mask.any()):
        btc_ts = dec_ts_sym[btc_mask]
        btc_r1h = panel["ret_1h"].to_numpy(dtype=np.float64, copy=False)[btc_mask]
        pos = np.searchsorted(btc_ts, dec_ts_sym)
        pos_clip = np.minimum(pos, len(btc_ts) - 1)
        matched = (pos < len(btc_ts)) & (btc_ts[pos_clip] == dec_ts_sym)
        aligned_btc_1h = np.where(matched, btc_r1h[pos_clip], 0.0)

        r1h_all = panel["ret_1h"].to_numpy(dtype=np.float64, copy=False)
        rolling_beta_sym = np.ones(len(panel), dtype=np.float64)
        s_starts = np.flatnonzero(np.r_[True, sym_codes[1:] != sym_codes[:-1]])
        s_ends = np.r_[s_starts[1:], len(panel)]
        for s0, s1 in zip(s_starts, s_ends, strict=False):
            rolling_beta_sym[s0:s1] = _sliding_beta_1d(r1h_all[s0:s1], aligned_btc_1h[s0:s1], window=96, min_periods=16)
    else:
        rolling_beta_sym = np.ones(len(panel), dtype=np.float64)

    # Sort permutation to ["decision_ts", "symbol"] applied directly to `rolling_beta_sym` (zero column insert!)
    sym_for_lex = pd.factorize(panel["symbol"], sort=True)[0]
    ts_order = np.lexsort((sym_for_lex, dec_ts_sym))
    panel = panel.iloc[ts_order].reset_index(drop=True)
    b_arr = rolling_beta_sym[ts_order]

    ts_codes, uniq_ts = pd.factorize(panel["decision_ts"], sort=False)
    n_ts = len(uniq_ts)
    use_f32 = panel["ret_4h"].dtype == np.float32 if "ret_4h" in panel.columns else False
    out_dtype = np.float32 if use_f32 else np.float64

    cs_cols: dict[str, np.ndarray] = {}
    rank_targets = [
        ("ret_4h", "cs_rank_ret_4h"),
        ("ret_24h", "cs_rank_ret_24h"),
        ("oi_vel_4h", "cs_rank_oi_vel_4h"),
        ("smart_money_div_z_24h", "cs_rank_smart_money_div"),
        ("basis_bps", "cs_rank_basis_bps"),
        ("spot_vs_perp_cvd_div_4h", "cs_rank_spot_cvd_div_4h"),
        ("funding_rate_bps", "cs_rank_funding_bps"),
    ]
    for src_col, dst_col in rank_targets:
        if src_col in panel.columns:
            cs_cols[dst_col] = _vectorized_cs_gaussian_rank(panel, "decision_ts", src_col).astype(out_dtype)
        else:
            cs_cols[dst_col] = np.zeros(len(panel), dtype=out_dtype)

    oi_v4 = panel["oi_vel_4h"].to_numpy(dtype=np.float64, copy=False)
    cs_cols["cs_rel_oi_shock_z"] = _fast_cs_zscore_preindexed(oi_v4, ts_codes, n_ts).astype(out_dtype)

    r_arr = np.nan_to_num(panel["ret_24h"].to_numpy(dtype=np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    cnt_ts = np.bincount(ts_codes, minlength=n_ts).astype(np.float64)
    mu_r = np.bincount(ts_codes, weights=r_arr, minlength=n_ts) / np.maximum(cnt_ts, 1.0)
    dr = r_arr - mu_r[ts_codes]
    var_r_s = np.bincount(ts_codes, weights=dr * dr, minlength=n_ts) / np.maximum(cnt_ts - 1.0, 1.0)
    disp_ts = np.where(cnt_ts > 1.0, np.sqrt(np.maximum(var_r_s, 0.0)), 0.0)
    cs_cols["cs_dispersion_24h"] = disp_ts[ts_codes].astype(out_dtype)

    # C-vectorized Gram-Schmidt beta neutralization per decision_ts: `r_orth = r - (sum(r*b)/sum(b*b))*b`
    sum_rb = np.bincount(ts_codes, weights=r_arr * b_arr, minlength=n_ts)[ts_codes]
    sum_bb = np.bincount(ts_codes, weights=b_arr * b_arr, minlength=n_ts)[ts_codes]
    proj = np.where(sum_bb > 1e-12, (sum_rb / np.maximum(sum_bb, 1e-12)) * b_arr, 0.0)
    tmp_orth = pd.DataFrame({"decision_ts": panel["decision_ts"].to_numpy(copy=False), "orth": r_arr - proj})
    cs_cols["cs_resid_mom_24h"] = _vectorized_cs_gaussian_rank(tmp_orth, "decision_ts", "orth").astype(out_dtype)

    return pd.concat([panel, pd.DataFrame(cs_cols, index=panel.index)], axis=1)


NON_FEATURE_METADATA_COLS: frozenset[str] = frozenset(
    {
        "timestamp",
        "decision_ts",
        "symbol",
        "close",
        "quote_volume",
        "adv_7d_usd_lag1",
        "adv_30d_usd_lag1",
        "days_since_listing",
    }
)


def list_multimodal_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns sorted list of pure alpha feature columns (excluding price/timestamp/metadata columns)."""
    return sorted(c for c in df.columns if c not in NON_FEATURE_METADATA_COLS and not c.startswith("fwd_ret_"))


def attach_forward_returns_for_audit(
    panel: pd.DataFrame,
    forward_bars_map: dict[str, int],
) -> pd.DataFrame:
    """
    Attaches strictly post-`decision_ts` linear forward returns (`(close_{t+h} - close_t) / close_t`)
    per symbol ONLY for offline IC & Conditional Payoff evaluation (never passed into live inference).
    Uses exact linear simple returns (never `np.log`) so Short payoffs do not suffer `+0.5 * sigma^2`
    Jensen convexity inflation.
    """
    out = panel.sort_values(["symbol", "decision_ts"]).copy()
    denom = out["close"].clip(lower=1e-12)
    for lbl, h_bars in forward_bars_map.items():
        h = max(1, int(h_bars))
        fwd_close = out.groupby("symbol")["close"].shift(-h)
        out[f"fwd_ret_{lbl}"] = (fwd_close - denom) / denom
    return out


def evaluate_feature_atlas(
    panel_with_fwd: pd.DataFrame,
    fwd_col: str = "fwd_ret_4h",
    feature_cols: list[str] | None = None,
    tail_z_threshold: float = 2.0,
    maker_roundtrip_bps: float = 3.2,
    taker_roundtrip_bps: float = 9.0,
    periods_per_year: float = 365.25 * 6.0,
    benchmark_daily_pnl: pd.Series | None = None,
    tradable_gate_col: str | None = "tradable_adv_15m_gate",
) -> pd.DataFrame:
    """
    Evaluates standalone Rank IC, t-statistic, L1 turnover, conditional extreme-tail (`|z| >= 2.0`)
    net Maker/Taker payoffs, and daily PnL correlation against existing sleeves across any split
    (`2023–2024` Train, `2025` Validation, `2026` Holdout) using zero-allocation `np.bincount` pre-indexing.
    """
    if panel_with_fwd.empty or fwd_col not in panel_with_fwd.columns:
        return pd.DataFrame()

    fwd_all = panel_with_fwd[fwd_col].to_numpy(dtype=np.float64, copy=False)
    valid_mask = np.isfinite(fwd_all)
    if tradable_gate_col and tradable_gate_col in panel_with_fwd.columns:
        gated_mask = valid_mask & (panel_with_fwd[tradable_gate_col].to_numpy(dtype=np.float64, copy=False) > 0.5)
        if int(gated_mask.sum()) >= 100:
            valid_mask = gated_mask
    if not bool(valid_mask.any()):
        return pd.DataFrame()

    df = panel_with_fwd.loc[valid_mask]
    feats = feature_cols or list_multimodal_feature_columns(df)

    dec_ts = df["decision_ts"].to_numpy(dtype=np.int64, copy=False)
    ts_codes, uniq_ts = pd.factorize(dec_ts, sort=False)
    n_timestamps = len(uniq_ts)
    ts_cnt = np.bincount(ts_codes, minlength=n_timestamps).astype(np.float64)

    fwd_arr = df[fwd_col].to_numpy(dtype=np.float64, copy=False)
    fwd_rank = _fast_cs_zscore_preindexed(fwd_arr, ts_codes, n_timestamps)

    # Pre-sort indices by ["symbol", "decision_ts"] ONCE for fast O(N) lag-1 autocorrelation
    sym_codes, _ = pd.factorize(df["symbol"], sort=False)
    sym_order = np.lexsort((dec_ts, sym_codes))
    sym_sorted = sym_codes[sym_order]
    same_sym_prev = sym_sorted[1:] == sym_sorted[:-1]
    n_same_sym = int(same_sym_prev.sum())

    # Pre-index daily epoch buckets (`decision_ts // 86400 * 86400`) ONCE for daily benchmark correlation
    has_bench = benchmark_daily_pnl is not None and not benchmark_daily_pnl.empty
    if has_bench:
        assert benchmark_daily_pnl is not None
        day_epochs = (dec_ts // 86400) * 86400
        day_codes, uniq_day_epochs = pd.factorize(day_epochs, sort=True)
        n_days = len(uniq_day_epochs)
        day_cnt = np.bincount(day_codes, minlength=n_days).astype(np.float64)
        uniq_dt = pd.to_datetime(uniq_day_epochs, unit="s", utc=True)
        bench_reindexed = benchmark_daily_pnl.reindex(uniq_dt).to_numpy(dtype=np.float64)
        bench_valid = np.isfinite(bench_reindexed)
        n_bench_days = int(bench_valid.sum())
        b_sub = bench_reindexed[bench_valid]
        b_std = float(b_sub.std(ddof=1)) if n_bench_days >= 15 else 0.0
    else:
        day_codes = np.empty(0, dtype=np.int64)
        n_days = 0
        day_cnt = np.empty(0, dtype=np.float64)
        bench_valid = np.empty(0, dtype=bool)
        n_bench_days = 0
        b_sub = np.empty(0, dtype=np.float64)
        b_std = 0.0

    records: list[dict[str, Any]] = []
    for f_col in feats:
        s_raw = np.nan_to_num(df[f_col].to_numpy(dtype=np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        if float(s_raw.std()) < 1e-12:
            continue

        f_cs_z = _fast_cs_zscore_preindexed(s_raw, ts_codes, n_timestamps)
        ic_by_ts = np.bincount(ts_codes, weights=f_cs_z * fwd_rank, minlength=n_timestamps) / np.maximum(ts_cnt, 1.0)
        ic_valid = ic_by_ts[np.isfinite(ic_by_ts) & (ic_by_ts != 0.0)]
        n_ic = len(ic_valid)
        ic_mean = float(ic_valid.mean()) if n_ic > 0 else 0.0
        ic_std = float(ic_valid.std(ddof=1)) if n_ic > 2 else 1.0
        ic_tstat = (ic_mean / (ic_std / np.sqrt(n_ic))) if (n_ic > 2 and ic_std > 1e-12) else 0.0
        ic_ir = (ic_mean / ic_std * np.sqrt(periods_per_year)) if ic_std > 1e-12 else 0.0

        # Feature serial persistence / L1 turnover proxy per symbol
        if n_same_sym > 10:
            s_sym = s_raw[sym_order]
            x_cur = s_sym[1:][same_sym_prev]
            x_lag = s_sym[:-1][same_sym_prev]
            sx = float(x_cur.std())
            sy = float(x_lag.std())
            autocorr_1 = float(np.corrcoef(x_cur, x_lag)[0, 1]) if (sx > 1e-12 and sy > 1e-12) else 0.0
        else:
            autocorr_1 = 0.0
        turnover_proxy = float(max(0.0, 1.0 - max(-1.0, min(1.0, autocorr_1 if np.isfinite(autocorr_1) else 0.0))))

        # Conditional Tail Event Analysis (`|z| >= tail_z_threshold`)
        z_evt = s_raw if ("_z" in f_col or f_col.startswith("cs_")) else f_cs_z
        tail_mask = np.abs(z_evt) >= tail_z_threshold
        n_events = int(tail_mask.sum())
        if n_events >= 5:
            signed_tail_bps = (np.sign(z_evt[tail_mask]) * fwd_arr[tail_mask]) * 1e4
            gross_tail_bps = float(signed_tail_bps.mean())
            abs_gross_bps = abs(gross_tail_bps)
            dir_sign = 1.0 if gross_tail_bps >= 0.0 else -1.0
            dir_rets = dir_sign * signed_tail_bps
            hit_rate = float((dir_rets > 0.0).mean())
            net_maker_bps = abs_gross_bps - maker_roundtrip_bps
            net_taker_bps = abs_gross_bps - taker_roundtrip_bps
            tail_std = float(dir_rets.std(ddof=1)) if n_events > 2 else 1.0
            tail_tstat = (
                float((abs_gross_bps - maker_roundtrip_bps) / (tail_std / np.sqrt(n_events)))
                if tail_std > 1e-9
                else 0.0
            )
        else:
            gross_tail_bps = 0.0
            net_maker_bps = -maker_roundtrip_bps
            net_taker_bps = -taker_roundtrip_bps
            hit_rate = 0.0
            tail_tstat = 0.0

        # Daily factor-mimicking return correlation against benchmark sleeves (`7-Factor + P2 + Snipers`)
        corr_bench = 0.0
        if has_bench and n_bench_days >= 15 and b_std > 1e-12:
            daily_f = np.bincount(day_codes, weights=f_cs_z * fwd_arr, minlength=n_days) / np.maximum(day_cnt, 1.0)
            f_sub = daily_f[bench_valid]
            if float(f_sub.std(ddof=1)) > 1e-12:
                c_val = float(np.corrcoef(f_sub, b_sub)[0, 1])
                corr_bench = c_val if np.isfinite(c_val) else 0.0

        family = "Family_D_CrossSec_MidCap"
        if f_col.startswith(("oi_", "quad_", "whale_", "retail_", "smart_money_", "metrics_taker")):
            family = "Family_A_Derivatives_OI"
        elif f_col.startswith(("basis_", "perp_taker", "spot_", "perp_cvd")):
            family = "Family_B_Spot_Perp_LeadLag"
        elif f_col.startswith(
            (
                "funding_",
                "is_compressed",
                "clock_",
                "mins_",
                "in_pre_",
                "in_post_",
                "pre_settle",
                "post_settle",
                "compressed_clock",
                "carry_plus",
            )
        ):
            family = "Family_C_Funding_Clock"

        records.append(
            {
                "feature": f_col,
                "family": family,
                "n_timestamps": n_timestamps,
                "ic_mean": round(ic_mean, 6),
                "ic_tstat": round(ic_tstat, 3),
                "ic_ir": round(ic_ir, 3),
                "turnover_proxy": round(turnover_proxy, 3),
                "tail_events": n_events,
                "tail_signed_gross_bps": round(gross_tail_bps, 2),
                "tail_net_maker_bps": round(net_maker_bps, 2),
                "tail_net_taker_bps": round(net_taker_bps, 2),
                "tail_maker_tstat": round(tail_tstat, 3),
                "tail_hit_rate": round(hit_rate, 3),
                "corr_to_existing_sleeves": round(corr_bench, 3),
            }
        )

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return res_df
    return res_df.sort_values("ic_tstat", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def evaluate_walk_forward_feature_atlas(
    panel_with_fwd: pd.DataFrame,
    fwd_col: str = "fwd_ret_4h",
    tail_z_threshold: float = 2.0,
    maker_roundtrip_bps: float = 3.2,
    taker_roundtrip_bps: float = 9.0,
    periods_per_year: float = 365.25 * 6.0,
    benchmark_daily_pnl: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Evaluates all ~100 multi-modal features across 3 chronological eras:
      - Train (`2023–2024`): learns polarity (`train_sign`)
      - Validation (`2025`): evaluates out-of-sample IC & tail payoff under `train_sign`
      - Holdout (`2026`): evaluates pure holdout IC, net Maker bps, & correlation to existing sleeves
    """
    if panel_with_fwd.empty:
        return pd.DataFrame()
    years = pd.to_datetime(panel_with_fwd["decision_ts"], unit="s", utc=True).dt.year
    df_train = panel_with_fwd[years <= 2024]
    df_val = panel_with_fwd[years == 2025]
    df_hold = panel_with_fwd[years >= 2026]

    feat_cols = list_multimodal_feature_columns(panel_with_fwd)
    tr_atlas = evaluate_feature_atlas(
        df_train,
        fwd_col=fwd_col,
        feature_cols=feat_cols,
        tail_z_threshold=tail_z_threshold,
        maker_roundtrip_bps=maker_roundtrip_bps,
        taker_roundtrip_bps=taker_roundtrip_bps,
        periods_per_year=periods_per_year,
        benchmark_daily_pnl=benchmark_daily_pnl,
    )
    if tr_atlas.empty:
        return pd.DataFrame()

    va_atlas = (
        evaluate_feature_atlas(
            df_val,
            fwd_col=fwd_col,
            feature_cols=feat_cols,
            tail_z_threshold=tail_z_threshold,
            maker_roundtrip_bps=maker_roundtrip_bps,
            taker_roundtrip_bps=taker_roundtrip_bps,
            periods_per_year=periods_per_year,
            benchmark_daily_pnl=benchmark_daily_pnl,
        ).set_index("feature")
        if not df_val.empty
        else pd.DataFrame()
    )
    ho_atlas = (
        evaluate_feature_atlas(
            df_hold,
            fwd_col=fwd_col,
            feature_cols=feat_cols,
            tail_z_threshold=tail_z_threshold,
            maker_roundtrip_bps=maker_roundtrip_bps,
            taker_roundtrip_bps=taker_roundtrip_bps,
            periods_per_year=periods_per_year,
            benchmark_daily_pnl=benchmark_daily_pnl,
        ).set_index("feature")
        if not df_hold.empty
        else pd.DataFrame()
    )

    rows: list[dict[str, Any]] = []
    for _, tr_r in tr_atlas.iterrows():
        f = str(tr_r["feature"])
        tr_ic = float(tr_r["ic_mean"])
        tr_tail = float(tr_r["tail_signed_gross_bps"])
        pol = 1.0 if (tr_ic if abs(tr_ic) > 1e-6 else tr_tail) >= 0.0 else -1.0

        va_ic_t = pol * float(va_atlas.loc[f, "ic_tstat"]) if (not va_atlas.empty and f in va_atlas.index) else 0.0
        va_tail_gross = (
            pol * float(va_atlas.loc[f, "tail_signed_gross_bps"])
            if (not va_atlas.empty and f in va_atlas.index)
            else 0.0
        )

        ho_ic_t = pol * float(ho_atlas.loc[f, "ic_tstat"]) if (not ho_atlas.empty and f in ho_atlas.index) else 0.0
        ho_ic_ir = pol * float(ho_atlas.loc[f, "ic_ir"]) if (not ho_atlas.empty and f in ho_atlas.index) else 0.0
        ho_tail_gross = (
            pol * float(ho_atlas.loc[f, "tail_signed_gross_bps"])
            if (not ho_atlas.empty and f in ho_atlas.index)
            else 0.0
        )
        ho_events = int(ho_atlas.loc[f, "tail_events"]) if (not ho_atlas.empty and f in ho_atlas.index) else 0
        ho_corr = (
            float(ho_atlas.loc[f, "corr_to_existing_sleeves"])
            if (not ho_atlas.empty and f in ho_atlas.index)
            else float(tr_r["corr_to_existing_sleeves"])
        )

        tr_ic_t_dir = abs(float(tr_r["ic_tstat"])) if tr_ic * pol >= 0 else -abs(float(tr_r["ic_tstat"]))
        sign_consistent = bool(
            (tr_ic_t_dir > 0 or pol * tr_tail > 0)
            and (va_ic_t > 0 or va_tail_gross > 0)
            and (ho_ic_t > 0 or ho_tail_gross > maker_roundtrip_bps)
        )

        rows.append(
            {
                "feature": f,
                "family": tr_r["family"],
                "train_polarity": int(pol),
                "train_ic_tstat": round(tr_ic_t_dir, 3),
                "train_tail_net_maker_bps": round(pol * tr_tail - maker_roundtrip_bps, 2),
                "val_2025_ic_tstat": round(va_ic_t, 3),
                "val_2025_tail_net_maker_bps": round(va_tail_gross - maker_roundtrip_bps, 2),
                "holdout_2026_ic_tstat": round(ho_ic_t, 3),
                "holdout_2026_ic_ir": round(ho_ic_ir, 3),
                "holdout_2026_tail_events": ho_events,
                "holdout_2026_tail_net_maker_bps": round(ho_tail_gross - maker_roundtrip_bps, 2),
                "holdout_2026_tail_net_taker_bps": round(ho_tail_gross - taker_roundtrip_bps, 2),
                "turnover_proxy": float(tr_r["turnover_proxy"]),
                "corr_to_existing_sleeves": round(ho_corr, 3),
                "sign_consistent_3_eras": sign_consistent,
                "composite_oos_score": round(
                    0.45 * va_ic_t + 0.55 * ho_ic_t + 0.08 * max(0.0, ho_tail_gross - maker_roundtrip_bps), 3
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    return out_df.sort_values("composite_oos_score", ascending=False).reset_index(drop=True)
