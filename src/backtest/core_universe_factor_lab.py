"""
Análisis 2 Laboratory: Dynamic Point-in-Time Universe (`Rolling 30d ADV Top-15 / Top-20`) vs
Static `PROD_TOP_15` + Clean `F3_Fast_Swing` Gating & Micro-Trim Buffering on Strict `8h` Clock.

Builds directly on Paso 1's breakthrough (`01_STRICT_8H_NO_INTRACYCLE`, `enable_intracycle_rebalance=False`,
now live in Tokyo) to answer two clean, unconfounded questions on exact `1-minute` linear physics
(`2023–2026` + `$2,700` `2026` Cushion Stress-Test + `15`-split `CombinatorialPurgedCV`):

  1. **Clean Factor & Micro-Trim Refinements on Pure `8h` Clock (`intracycle_mode="off"`):**
     In Paso 1, `03..09` had `collapsed_trim_only` active (`197` mid-cycle trims). Here we isolate
     on pure `8h` clock:
       - `01_STATIC_TOP15_STRICT_8H` (current live Tokyo baseline)
       - `02_STRICT_8H_TRIM_BUF_3.5PCT` (suppress `< 3.5%` equity partial micro-trims)
       - `03_STRICT_8H_FAST_GATE_0.50` (gate `F3_Fast_Swing` unless trailing IC `t >= +0.50`)
       - `04_STRICT_8H_SIX_FACTOR_NO_F3` (remove `F3_Fast_Swing` completely: 6-Factor Core)
       - `05_STRICT_8H_BHS_3_7_DB_1.60_TRIM_3.5` (clean rank retention + micro-trim buffer)
       - `06_STRICT_8H_TRIM_3.5_FAST_GATE_0.50` (combined micro-trim buffer + `F3` IC gate)

  2. **Causal Rolling Point-in-Time Liquidity Universe (`CANDIDATE_POOL_35` -> `Top-15` / `Top-20`):**
     Tests replacing the static 15-coin list (`PROD_TOP_15`) with a strictly causal (`t - 1` lagged
     `30d` dollar volume) weekly-updated `Top-15` or `Top-20` universe with `+4` rank hysteresis
     so emerging institutional leaders (`1000PEPE, WIF, TAO, WLD, PENDLE, INJ, LTC, UNI, AAVE, OP`)
     enter organically when liquid and exit when volume fades, with zero lookahead bias.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    CoreRebalanceBarrierSpec,
    compute_cpcv_portfolio_sharpe_stability,
    simulate_core_barrier_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    compute_portfolio_era_metrics,
    convert_trade_df_to_hourly_sniper_events,
)
from src.backtest.unified_1m_simulator import resolve_sniper_1m_path
from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine
from src.strategies.event_snipers import (
    SNIPER_BRACKET_PROFILES,
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_M26_STRAT,
    SNIPER_UNIVERSE_82,
    EventDrivenSniperEngine,
)
from src.strategies.seven_factor_engine import (
    DEFAULT_PRIORS,
    SevenFactorOnlineEngine,
    cap_side_weights_waterfill,
)

CANDIDATE_POOL_35: tuple[str, ...] = (
    # Current PROD_TOP_15
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "RENDERUSDT",
    "FETUSDT",
    # 20 Additional Major Liquid Perps across L1, DeFi, AI, PoW & Memes
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "INJUSDT",
    "OPUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "WLDUSDT",
    "TAOUSDT",
    "PENDLEUSDT",
    "FILUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "RUNEUSDT",
    "GALAUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
)


@dataclass(frozen=True)
class UniverseFactorExperimentSpec:
    exp_id: str
    description: str
    universe_mode: str  # "static_15", "dynamic_pit_15", "dynamic_pit_20"
    bhs_enter_top_k: int = 3
    bhs_hold_top_k: int = 6
    deadband_mult: float = 1.25
    min_partial_trim_equity_pct: float = 0.009
    fast_min_tstat: float | None = None
    drop_f3_completely: bool = False
    pit_hold_buffer: int = 4
    pit_reselect_hours: int = 168
    pit_min_age_hours: int = 720


UNIVERSE_FACTOR_SPECS: tuple[UniverseFactorExperimentSpec, ...] = (
    UniverseFactorExperimentSpec(
        exp_id="01_STATIC_TOP15_STRICT_8H",
        description="Current Live Tokyo Baseline (Static Top-15, Strict 8h Clock, BHS(3,6), DBx1.25)",
        universe_mode="static_15",
    ),
    UniverseFactorExperimentSpec(
        exp_id="02_STATIC_TOP15_TRIM_BUF_3.5PCT",
        description="Static Top-15 + Strict 8h + Suppress <3.5% Equity Partial Micro-Trims",
        universe_mode="static_15",
        min_partial_trim_equity_pct=0.035,
    ),
    UniverseFactorExperimentSpec(
        exp_id="03_STATIC_TOP15_FAST_GATE_0.50",
        description="Static Top-15 + Strict 8h + Gate F3_Fast_Swing unless Trailing IC t >= +0.50",
        universe_mode="static_15",
        fast_min_tstat=0.50,
    ),
    UniverseFactorExperimentSpec(
        exp_id="04_STATIC_TOP15_SIX_FACTOR_NO_F3",
        description="Static Top-15 + Strict 8h + Drop F3_Fast_Swing Completely (6-Factor Core)",
        universe_mode="static_15",
        drop_f3_completely=True,
    ),
    UniverseFactorExperimentSpec(
        exp_id="05_STATIC_TOP15_BHS_3_7_DB_1.60_TRIM_3.5",
        description="Static Top-15 + Strict 8h + BHS(3,7) + DBx1.60 + 3.5% Trim Buffer",
        universe_mode="static_15",
        bhs_enter_top_k=3,
        bhs_hold_top_k=7,
        deadband_mult=1.60,
        min_partial_trim_equity_pct=0.035,
    ),
    UniverseFactorExperimentSpec(
        exp_id="06_STATIC_TOP15_TRIM_3.5_FAST_GATE_0.50",
        description="Static Top-15 + Strict 8h + 3.5% Trim Buffer + F3_Fast Gate (t >= +0.50)",
        universe_mode="static_15",
        min_partial_trim_equity_pct=0.035,
        fast_min_tstat=0.50,
    ),
    UniverseFactorExperimentSpec(
        exp_id="07_DYNAMIC_PIT_TOP15_30D_ADV",
        description="Dynamic Causal 30d-ADV Top-15 (from 35 Pool, +4 Rank Hysteresis) + Strict 8h",
        universe_mode="dynamic_pit_15",
    ),
    UniverseFactorExperimentSpec(
        exp_id="08_DYNAMIC_PIT_TOP20_30D_ADV",
        description="Dynamic Causal 30d-ADV Top-20 (from 35 Pool, +4 Rank Hysteresis, BHS(4,8)) + Strict 8h",
        universe_mode="dynamic_pit_20",
        bhs_enter_top_k=4,
        bhs_hold_top_k=8,
    ),
    UniverseFactorExperimentSpec(
        exp_id="09_DYNAMIC_PIT_TOP15_TRIM_3.5_FAST_GATE",
        description="Dynamic Causal 30d-ADV Top-15 + Strict 8h + 3.5% Trim Buffer + F3_Fast Gate",
        universe_mode="dynamic_pit_15",
        min_partial_trim_equity_pct=0.035,
        fast_min_tstat=0.50,
    ),
    UniverseFactorExperimentSpec(
        exp_id="10_DYNAMIC_PIT_TOP15_MONTHLY_SEASONED_90D",
        description="Dynamic Monthly 30d-ADV Top-15 (90d Listing Seasoning, +6 Rank Hysteresis) + 3.5% TrimBuf",
        universe_mode="dynamic_pit_15",
        min_partial_trim_equity_pct=0.035,
        pit_hold_buffer=6,
        pit_reselect_hours=720,
        pit_min_age_hours=2160,
    ),
)


def compute_causal_pit_universe_mask(
    dollar_vol_1h: pd.DataFrame,
    raw_close_1h: pd.DataFrame,
    top_k: int = 15,
    hold_buffer: int = 4,
    reselect_hours: int = 168,
    min_age_hours: int = 720,
) -> pd.DataFrame:
    """
    Computes a strictly causal (`shift(1)` lagged `30d` rolling dollar volume) Point-in-Time
    active universe boolean mask with `reselect_hours` cadence and `+hold_buffer` hysteresis:
      - Asset must have `>= min_age_hours` valid 1h bars post-listing (`age_ok`).
      - Currently active members stay active while `adv_rank < top_k + hold_buffer`.
      - Remaining slots up to `top_k` are filled by highest-ADV eligible assets.
    """
    eff_min_age = max(168, int(min_age_hours))
    age_ok = raw_close_1h.notna().cumsum() >= eff_min_age
    adv_30d = dollar_vol_1h.rolling(24 * 30, min_periods=168).mean().shift(1).where(age_ok, -1.0).fillna(-1.0)
    arr_adv = adv_30d.to_numpy(dtype=np.float64)
    n_rows, n_cols = arr_adv.shape
    mask_out = np.zeros((n_rows, n_cols), dtype=bool)

    active_set: set[int] = set()
    for t in range(n_rows):
        if t == 0 or (t % reselect_hours == 0) or not active_set:
            row_v = arr_adv[t]
            valid_idxs = [j for j in range(n_cols) if row_v[j] > 0.0]
            valid_idxs.sort(key=lambda j: -row_v[j])
            rank_of = {j: rk for rk, j in enumerate(valid_idxs)}

            kept = {j for j in active_set if j in rank_of and rank_of[j] < (top_k + hold_buffer)}
            if len(kept) > top_k:
                kept = set(sorted(kept, key=lambda j: rank_of[j])[:top_k])
            for j in valid_idxs:
                if len(kept) >= top_k:
                    break
                kept.add(j)
            active_set = kept

        for j in active_set:
            if arr_adv[t, j] > 0.0:
                mask_out[t, j] = True

    return pd.DataFrame(mask_out, index=dollar_vol_1h.index, columns=dollar_vol_1h.columns)


def compute_dynamic_pit_stitched_matrices(
    bars_1h_pool: dict[str, pd.DataFrame],
    fnd_pool: dict[str, pd.Series],
    pit_mask: pd.DataFrame,
    eng: SevenFactorOnlineEngine,
    warmup_hours: int = 1440,
) -> dict[str, Any]:
    """
    Executes `SevenFactorOnlineEngine.compute_matrices` strictly on the active listed
    Point-in-Time member set `U_seg` of each contiguous reconstitution regime `[s_i, e_i)`
    (with `warmup_hours` causal lookback), completely preventing unlisted `NaN` / zero-volume
    coins from contaminating cross-sectional standardization, factor ICs (`t_df`), or waterfill caps.
    """
    full_idx = pit_mask.index
    pool_cols = list(pit_mask.columns)
    n_rows = len(full_idx)

    close_full = pd.DataFrame(index=full_idx, columns=pool_cols, dtype=np.float64)
    fnd_full = pd.DataFrame(0.0, index=full_idx, columns=pool_cols, dtype=np.float64)
    for sym in pool_cols:
        close_full[sym] = bars_1h_pool[sym]["close"].reindex(full_idx).ffill().bfill()
        if sym in fnd_pool:
            fnd_full[sym] = fnd_pool[sym].reindex(full_idx).fillna(0.0)

    w_masked = pd.DataFrame(0.0, index=full_idx, columns=pool_cols, dtype=np.float64)
    vol_scalar = pd.Series(1.0, index=full_idx, dtype=np.float64)
    alpha_chop = pd.Series(0.0, index=full_idx, dtype=np.float64)
    vol_24h = pd.DataFrame(0.02, index=full_idx, columns=pool_cols, dtype=np.float64)
    t_df = pd.DataFrame(
        0.0,
        index=full_idx,
        columns=["macro", "med", "fast", "rev", "carry", "liq", "size"],
        dtype=np.float64,
    )

    mask_arr = pit_mask.to_numpy(dtype=bool)
    seg_start = 0
    cur_tuple = tuple(pool_cols[j] for j in np.where(mask_arr[0])[0])

    segments: list[tuple[int, int, tuple[str, ...]]] = []
    for t in range(1, n_rows):
        nxt_tuple = tuple(pool_cols[j] for j in np.where(mask_arr[t])[0])
        if nxt_tuple != cur_tuple:
            segments.append((seg_start, t, cur_tuple))
            seg_start = t
            cur_tuple = nxt_tuple
    segments.append((seg_start, n_rows, cur_tuple))

    for s_i, e_i, u_syms in segments:
        if len(u_syms) < 4:
            continue
        w_start_ts = full_idx[max(0, s_i - int(warmup_hours))]
        seg_end_ts = full_idx[e_i - 1]
        b_seg = {s: bars_1h_pool[s].loc[w_start_ts:seg_end_ts] for s in u_syms}
        f_seg = {s: fnd_pool[s].loc[w_start_ts:seg_end_ts] for s in u_syms if s in fnd_pool}
        m_seg = eng.compute_matrices(b_seg, funding_rates=f_seg)

        seg_idx = full_idx[s_i:e_i].intersection(m_seg["close"].index)
        if seg_idx.empty:
            continue
        u_list = [s for s in u_syms if s in m_seg["w_base_1x"].columns]
        w_masked.loc[seg_idx, u_list] = m_seg["w_base_1x"].loc[seg_idx, u_list].to_numpy(dtype=np.float64)
        vol_scalar.loc[seg_idx] = m_seg["vol_scalar"].loc[seg_idx].to_numpy(dtype=np.float64)
        alpha_chop.loc[seg_idx] = m_seg["alpha_chop_overlay"].loc[seg_idx].to_numpy(dtype=np.float64)
        vol_24h.loc[seg_idx, u_list] = m_seg["vol_24h"].loc[seg_idx, u_list].to_numpy(dtype=np.float64)
        for tc in t_df.columns:
            if tc in m_seg["t_df"].columns:
                t_df.loc[seg_idx, tc] = m_seg["t_df"].loc[seg_idx, tc].to_numpy(dtype=np.float64)

    return {
        "close": close_full,
        "funding": fnd_full,
        "w_base_1x": w_masked,
        "w_masked": w_masked,
        "vol_scalar": vol_scalar,
        "alpha_chop_overlay": alpha_chop,
        "vol_24h": vol_24h,
        "t_df": t_df,
    }


def renorm_neutral_masked_weights(w_base: pd.DataFrame, active_mask: pd.DataFrame) -> pd.DataFrame:
    """
    Masks `w_base` to `active_mask`, cross-sectionally demeans inside the active universe,
    preserves `w_base`'s exact regime-shielded `conviction` (`orig_leg`), and re-applies
    the `30%` single-asset and `48%` sector waterfill concentration caps.
    """
    orig_leg = w_base.abs().sum(axis=1) * 0.5
    w_m = w_base.where(active_mask, np.nan)
    demeaned = w_m.sub(w_m.mean(axis=1), axis=0).fillna(0.0)
    pos = demeaned.clip(lower=0.0)
    neg = (-demeaned).clip(lower=0.0)
    ps = pos.sum(axis=1)
    ns = neg.sum(axis=1)
    valid = (ps > 1e-10) & (ns > 1e-10) & (orig_leg > 1e-10)
    wl = pos.div(ps.replace(0.0, 1.0), axis=0).mul(orig_leg, axis=0)
    ws = neg.div(ns.replace(0.0, 1.0), axis=0).mul(orig_leg, axis=0)
    w_out = (wl - ws).mul(valid.astype(float), axis=0)
    return cap_side_weights_waterfill(w_out, max_side_share=0.30, max_cluster_share=0.48)


def build_pool35_1m_bundle(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    symbol_1m_loader: Callable[[str], pd.DataFrame | None],
    symbols_to_stream: list[str],
    eval_start: str = "2023-03-01",
    loader_workers: int = 2,
) -> dict[str, Any]:
    """
    Streams `1m` klines in `O(1)` RAM (`float32` `o/h/l` tensors = `< 0.9 GB` for all `35` candidate
    Core symbols!) and resolves `Manga P2` + `M09/M23/M04` base snipers simultaneously.
    """
    pool_syms = [s for s in CANDIDATE_POOL_35 if s in bars_1h_by_sym]
    bars_1h_pool: dict[str, pd.DataFrame] = {}
    fnd_pool: dict[str, pd.Series] = {}
    for s in pool_syms:
        b = bars_1h_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        if "volume" not in b.columns:
            b["volume"] = b["quote_volume"] / np.maximum(b["close"], 1e-12)
        b.index = pd.DatetimeIndex(pd.to_datetime(b["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        b.drop(columns=["timestamp"], inplace=True)
        bars_1h_pool[s] = b
        if s in funding_by_sym and not funding_by_sym[s].empty:
            f = funding_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_idx = pd.DatetimeIndex(
                pd.to_datetime(((f["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600, unit="s", utc=True)
            )
            f_s = pd.Series(f["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            fnd_pool[s] = f_s[~f_s.index.duplicated(keep="last")]

    btc_idx = bars_1h_pool["BTCUSDT"].index if "BTCUSDT" in bars_1h_pool else next(iter(bars_1h_pool.values())).index
    all_idx = btc_idx.sort_values()
    eval_idx = all_idx[all_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, n_p = len(eval_idx), len(pool_syms)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()
    hr_to_pos = {int(t_h): i for i, t_h in enumerate(ts_eval.tolist())}
    min_eval_s = int(ts_eval[0])
    max_eval_s = int(ts_eval[-1]) + 3600

    open_pool = np.zeros((n_b, n_p), dtype=np.float64)
    o_1m_pool = np.full((n_b, 60, n_p), np.nan, dtype=np.float32)
    h_1m_pool = np.full((n_b, 60, n_p), np.nan, dtype=np.float32)
    l_1m_pool = np.full((n_b, 60, n_p), np.nan, dtype=np.float32)
    for j_c, sym in enumerate(pool_syms):
        op_s = bars_1h_pool[sym]["open"].reindex(eval_idx).ffill().bfill().to_numpy(dtype=np.float64)
        open_pool[:, j_c] = op_s
        o_1m_pool[:, 0, j_c] = op_s.astype(np.float32)
        h_1m_pool[:, 0, j_c] = bars_1h_pool[sym]["high"].reindex(eval_idx).to_numpy(dtype=np.float32)
        l_1m_pool[:, 0, j_c] = bars_1h_pool[sym]["low"].reindex(eval_idx).to_numpy(dtype=np.float32)

    p2_eng = CausalDollarBarTSMOMEngine(max_gross_leverage=0.95)
    p2_w_pool = np.zeros((n_b, n_p), dtype=np.float64)

    sniper_syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    sn_eng = EventDrivenSniperEngine()
    raw_1h_by_sym: dict[str, list[tuple[int, int, str, float, int]]] = {}
    f_maps: dict[str, dict[int, float]] = {}
    f_series_map: dict[str, pd.Series] = {}
    rk_df_1h: pd.DataFrame | None = None
    idx_15m: pd.DatetimeIndex | None = None
    ts_15m: np.ndarray | None = None

    if sniper_syms and n_b > 220:
        ns = len(sniper_syms)
        c_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
        h_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
        l_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
        qv_1h = np.zeros((n_b, ns), dtype=np.float64)
        tb_1h = np.zeros((n_b, ns), dtype=np.float64)
        for j_s, s_sym in enumerate(sniper_syms):
            bs = bars_1h_by_sym[s_sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
            c_1h[:, j_s] = pd.Series(bs["close"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
            h_1h[:, j_s] = pd.Series(bs["high"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
            l_1h[:, j_s] = pd.Series(bs["low"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
            qv_1h[:, j_s] = (
                pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
                .reindex(eval_idx)
                .fillna(0.0)
                .to_numpy()
            )
            tb_1h[:, j_s] = (
                pd.Series(bs["taker_buy_quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
                .reindex(eval_idx)
                .fillna(0.0)
                .to_numpy()
            )
            if s_sym in funding_by_sym and not funding_by_sym[s_sym].empty:
                fd = funding_by_sym[s_sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                fts = ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 60) * 60
                frs = fd["funding_rate"].to_numpy(dtype=np.float64)
                f_maps[s_sym] = {int(k): float(v) for k, v in zip(fts.tolist(), frs.tolist(), strict=False)}
                f_idx_s = pd.DatetimeIndex(pd.to_datetime(fts, unit="s", utc=True))
                f_ser = pd.Series(frs, index=f_idx_s)
                f_series_map[s_sym] = f_ser[~f_ser.index.duplicated(keep="last")]

        adv30 = (
            pd.DataFrame(qv_1h, index=eval_idx, columns=sniper_syms)
            .rolling(24 * 30, min_periods=48)
            .mean()
            .shift(1)
            .fillna(0.0)
        )
        rk_df_1h = adv30.rank(axis=1, ascending=False, method="first").fillna(999.0) - 1.0
        rank_1h = rk_df_1h.to_numpy(dtype=np.int64)
        for en_ts, s_idx, side, st_name, sc_v, rk_v in sn_eng.scan_historical_1h_events(
            ts_eval, sniper_syms, c_1h, h_1h, l_1h, qv_1h, tb_1h, rank_1h
        ):
            raw_1h_by_sym.setdefault(sniper_syms[s_idx], []).append((en_ts, side, st_name, sc_v, rk_v))
        idx_15m = pd.date_range(start=eval_idx[0], end=eval_idx[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC")
        ts_15m = idx_15m.as_unit("s").astype(np.int64).to_numpy()
        del c_1h, h_1h, l_1h, qv_1h, tb_1h
        gc.collect()

    pool_col_idx = {s: j for j, s in enumerate(pool_syms)}
    p2_majors_set = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

    def _consume_1m(sym: str, df_1m_raw: pd.DataFrame | None) -> list[dict[str, Any]]:
        if df_1m_raw is None or df_1m_raw.empty:
            return []
        d1 = df_1m_raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        ts_a = d1["timestamp"].to_numpy(dtype=np.int64)
        o_a = d1["open"].to_numpy(dtype=np.float64)
        h_a = d1["high"].to_numpy(dtype=np.float64)
        l_a = d1["low"].to_numpy(dtype=np.float64)
        c_a = d1["close"].to_numpy(dtype=np.float64)

        if sym in pool_col_idx:
            j_c = pool_col_idx[sym]
            o_1m_pool[:, 0, j_c] = np.nan
            h_1m_pool[:, 0, j_c] = np.nan
            l_1m_pool[:, 0, j_c] = np.nan
            m_ev = (ts_a >= min_eval_s) & (ts_a < max_eval_s)
            ts_sub = ts_a[m_ev]
            hr_b = (ts_sub // 3600) * 3600
            mn_b = (ts_sub % 3600) // 60
            p_arr = np.array([hr_to_pos.get(int(h_v), -1) for h_v in hr_b], dtype=np.int64)
            ok = (p_arr >= 0) & (mn_b >= 0) & (mn_b < 60)
            if np.any(ok):
                r_idx, m_idx = p_arr[ok], mn_b[ok]
                o_1m_pool[r_idx, m_idx, j_c] = o_a[m_ev][ok].astype(np.float32)
                h_1m_pool[r_idx, m_idx, j_c] = h_a[m_ev][ok].astype(np.float32)
                l_1m_pool[r_idx, m_idx, j_c] = l_a[m_ev][ok].astype(np.float32)

        if sym in p2_majors_set and sym in pool_col_idx:
            j_m = pool_col_idx[sym]
            s_p2 = p2_eng.compute_historical_weights({sym: d1}, eval_idx)[sym].shift(1).fillna(0.0)
            p2_w_pool[:, j_m] = s_p2.to_numpy(dtype=np.float64)

        if sym not in sniper_syms:
            return []
        cand_evs: list[tuple[int, int, str, float, int]] = list(raw_1h_by_sym.get(sym, []))
        if idx_15m is not None and ts_15m is not None and rk_df_1h is not None and sym in f_series_map:
            qv_a = d1["quote_volume"].to_numpy(dtype=np.float64) if "quote_volume" in d1.columns else np.zeros_like(c_a)
            tb_a = (
                d1["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
                if "taker_buy_quote_volume" in d1.columns
                else 0.5 * qv_a
            )
            m_15 = (ts_a >= min_eval_s) & (ts_a < max_eval_s)
            if np.any(m_15):
                sub_dt = pd.DatetimeIndex(pd.to_datetime(ts_a[m_15], unit="s", utc=True))
                g15 = (
                    pd.DataFrame(
                        {"high": h_a[m_15], "close": c_a[m_15], "qv": qv_a[m_15], "tb": tb_a[m_15]},
                        index=sub_dt,
                    )
                    .resample("15min")
                    .agg({"high": "max", "close": "last", "qv": "sum", "tb": "sum"})
                    .reindex(idx_15m)
                )
                c15 = g15["close"].to_numpy(dtype=np.float64)[:, None]
                h15 = g15["high"].to_numpy(dtype=np.float64)[:, None]
                qv15 = g15["qv"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                tb15 = g15["tb"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                fr15 = (
                    f_series_map[sym].reindex(idx_15m, method="ffill").fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                )
                rk15 = rk_df_1h[sym].reindex(idx_15m, method="ffill").fillna(999.0).to_numpy(dtype=np.int64)[:, None]
                for en_ts, _z, side, st_name, sc_v, rk_v in sn_eng.scan_historical_15m_events(
                    ts_15m, c15, h15, qv15, tb15, fr15, rk15
                ):
                    cand_evs.append((en_ts, side, st_name, sc_v, rk_v))

        out_evs: list[dict[str, Any]] = []
        fr_sym = f_maps.get(sym, {})
        for en_ts, side, st_name, sc_v, rk_v in cand_evs:
            res = resolve_sniper_1m_path(
                entry_ts=en_ts,
                side=side,
                spec=SNIPER_BRACKET_PROFILES[st_name],
                adv_rank=rk_v,
                ts_1m=ts_a,
                open_1m=o_a,
                high_1m=h_a,
                low_1m=l_a,
                close_1m=c_a,
                fr_map_sym=fr_sym,
            )
            if res is not None:
                ex_ts, hr_ret, is_loss, _ = res
                out_evs.append(
                    {
                        "hr_ts": (en_ts // 3600) * 3600,
                        "strat": st_name,
                        "sym": sym,
                        "entry_ts": en_ts,
                        "exit_ts": ex_ts,
                        "score": sc_v,
                        "hourly_ret": hr_ret,
                        "was_loss": is_loss,
                    }
                )
        return out_evs

    base_events: list[dict[str, Any]] = []
    t_s0 = time.time()
    done_cnt = 0
    batch_sz = max(2, int(loader_workers) * 2)
    with ThreadPoolExecutor(max_workers=max(1, int(loader_workers))) as pool:
        for b_i in range(0, len(symbols_to_stream), batch_sz):
            b_syms = symbols_to_stream[b_i : b_i + batch_sz]
            futs = {pool.submit(symbol_1m_loader, s_k): s_k for s_k in b_syms}
            for fut in as_completed(futs):
                s_k = futs[fut]
                df_l = fut.result()
                base_events.extend(_consume_1m(s_k, df_l))
                del df_l
                done_cnt += 1
                if done_cnt % 12 == 0 or done_cnt == len(symbols_to_stream):
                    print(
                        f"  • Streamed 1m klines {done_cnt}/{len(symbols_to_stream)} symbols ({time.time() - t_s0:.1f}s, float32 O(1) RAM)...",
                        flush=True,
                    )
            gc.collect()

    return {
        "pool_syms": pool_syms,
        "bars_1h_pool": bars_1h_pool,
        "fnd_pool": fnd_pool,
        "eval_idx": eval_idx,
        "open_pool": open_pool,
        "o_1m_pool": o_1m_pool,
        "h_1m_pool": h_1m_pool,
        "l_1m_pool": l_1m_pool,
        "p2_w_pool": p2_w_pool,
        "base_sniper_events": base_events,
    }


def evaluate_universe_and_factor_experiments(
    pool_bundle: dict[str, Any],
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[UniverseFactorExperimentSpec, ...] = UNIVERSE_FACTOR_SPECS,
) -> pd.DataFrame:
    """Runs all 9 `UniverseFactorExperimentSpec` ablations on exact `1m` physics."""
    pool_syms: list[str] = pool_bundle["pool_syms"]
    bars_1h_pool: dict[str, pd.DataFrame] = pool_bundle["bars_1h_pool"]
    fnd_pool: dict[str, pd.Series] = pool_bundle["fnd_pool"]
    eval_idx: pd.DatetimeIndex = pool_bundle["eval_idx"]
    open_pool: np.ndarray = pool_bundle["open_pool"]
    o_1m_pool: np.ndarray = pool_bundle["o_1m_pool"]
    h_1m_pool: np.ndarray = pool_bundle["h_1m_pool"]
    l_1m_pool: np.ndarray = pool_bundle["l_1m_pool"]
    p2_w_pool: np.ndarray = pool_bundle["p2_w_pool"]

    events_by_hr: dict[int, list[dict[str, Any]]] = {}
    for ev_b in pool_bundle["base_sniper_events"]:
        events_by_hr.setdefault(int(ev_b["hr_ts"]), []).append(ev_b)
    allowed_strats = {SNIPER_M09_STRAT, SNIPER_M23_STRAT, SNIPER_M04_STRAT}
    if m26_trades_df is not None and not m26_trades_df.empty:
        sub_26 = m26_trades_df[
            (m26_trades_df["mech_id"] == "P6A_L2_3_INST_SWEEP_CONC")
            & (m26_trades_df["exec_mode"] == "MONO_A_CONVEX_TRAIL")
        ]
        if not sub_26.empty:
            for ev in convert_trade_df_to_hourly_sniper_events(sub_26, SNIPER_M26_STRAT):
                events_by_hr.setdefault(int(ev["hr_ts"]), []).append(ev)
            allowed_strats.add(SNIPER_M26_STRAT)

    static_cols = [s for s in PROD_TOP_15 if s in pool_syms]
    static_idx_in_pool = [pool_syms.index(s) for s in static_cols]

    # Pre-compute SevenFactor matrices only once per unique engine + universe configuration
    mats_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    full_idx_pool = pd.DatetimeIndex(
        sorted({ts for df_b in bars_1h_pool.values() for ts in df_b.index}),
        tz="UTC",
    )
    dvol_pool_df = (
        pd.DataFrame({s: bars_1h_pool[s]["quote_volume"] for s in pool_syms}).reindex(full_idx_pool).fillna(0.0)
    )
    raw_cl_pool_df = pd.DataFrame({s: bars_1h_pool[s]["close"] for s in pool_syms}).reindex(full_idx_pool)

    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}

    rows: list[dict[str, Any]] = []
    for sp in specs:
        min_t_map: dict[str, float] = {"liq": 0.75}
        priors_use = dict(DEFAULT_PRIORS)
        if sp.drop_f3_completely:
            priors_use["fast"] = 0.0
            min_t_map["fast"] = 99.0
        elif sp.fast_min_tstat is not None:
            min_t_map["fast"] = float(sp.fast_min_tstat)

        eng = SevenFactorOnlineEngine(
            priors=priors_use,
            f6_min_tstat=0.75,
            min_tstat_by_factor=min_t_map,
            enable_chop_range_arb=True,
            chop_max_alpha=0.75,
            enable_holistic_regime_shields=True,
        )

        if sp.universe_mode == "static_15":
            c_key = ("static_15", sp.fast_min_tstat, sp.drop_f3_completely)
            if c_key not in mats_cache:
                b_sub = {s: bars_1h_pool[s] for s in static_cols}
                f_sub = {s: fnd_pool[s] for s in b_sub if s in fnd_pool}
                mats_cache[c_key] = eng.compute_matrices(b_sub, funding_rates=f_sub)
            base_mats = dict(mats_cache[c_key])
            cols_here = list(base_mats["close"].columns)
            raw_cl = pd.DataFrame({s: bars_1h_pool[s]["close"] for s in cols_here}).reindex(base_mats["close"].index)
            age_ok = raw_cl.notna().cumsum() >= 168
            base_mats["w_masked"] = base_mats["w_base_1x"].where(age_ok, 0.0)
            o_1m_use = o_1m_pool[:, :, static_idx_in_pool]
            h_1m_use = h_1m_pool[:, :, static_idx_in_pool]
            l_1m_use = l_1m_pool[:, :, static_idx_in_pool]
            open_use = open_pool[:, static_idx_in_pool]
            p2_use = p2_w_pool[:, static_idx_in_pool]
        else:
            top_k = 15 if sp.universe_mode == "dynamic_pit_15" else 20
            c_key = (
                sp.universe_mode,
                sp.fast_min_tstat,
                sp.drop_f3_completely,
                sp.pit_hold_buffer,
                sp.pit_reselect_hours,
                sp.pit_min_age_hours,
            )
            if c_key not in mats_cache:
                eff_age = min(int(sp.pit_min_age_hours), max(168, len(full_idx_pool) // 3))
                pit_mask = compute_causal_pit_universe_mask(
                    dvol_pool_df,
                    raw_cl_pool_df,
                    top_k=top_k,
                    hold_buffer=sp.pit_hold_buffer,
                    reselect_hours=sp.pit_reselect_hours,
                    min_age_hours=eff_age,
                )
                mats_cache[c_key] = compute_dynamic_pit_stitched_matrices(
                    bars_1h_pool=bars_1h_pool,
                    fnd_pool=fnd_pool,
                    pit_mask=pit_mask,
                    eng=eng,
                    warmup_hours=1440,
                )
            base_mats = dict(mats_cache[c_key])
            o_1m_use = o_1m_pool
            h_1m_use = h_1m_pool
            l_1m_use = l_1m_pool
            open_use = open_pool
            p2_use = p2_w_pool

        cb_spec = CoreRebalanceBarrierSpec(
            config_id=sp.exp_id,
            description=sp.description,
            intracycle_mode="off",
            bhs_enter_top_k=sp.bhs_enter_top_k,
            bhs_hold_top_k=sp.bhs_hold_top_k,
            deadband_mult=sp.deadband_mult,
            min_partial_trim_equity_pct=sp.min_partial_trim_equity_pct,
        )

        sim_4y = simulate_core_barrier_1m_portfolio(
            mats=base_mats,
            o_1m_core=o_1m_use,
            h_1m_core=h_1m_use,
            l_1m_core=l_1m_use,
            eval_idx=eval_idx,
            open_core=open_use,
            p2_w_mat=p2_use,
            spec=cb_spec,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])
        bs = sim_4y["barrier_stats"]

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            sim_26 = simulate_core_barrier_1m_portfolio(
                mats=base_mats,
                o_1m_core=o_1m_use[mask_2026],
                h_1m_core=h_1m_use[mask_2026],
                l_1m_core=l_1m_use[mask_2026],
                eval_idx=idx_2026,
                open_core=open_use[mask_2026],
                p2_w_mat=p2_use[mask_2026],
                spec=cb_spec,
                resolved_sniper_events_by_hour=evs_2026,
                allowed_sniper_strats=allowed_strats,
            )
            fresh_final = round(float(sim_26["final_eq"]), 2)
            fresh_min = round(float(sim_26["min_eq"]), 2)
            fresh_ok = bool(fresh_min >= 2500.0)

        rows.append(
            {
                "exp_id": sp.exp_id,
                "description": sp.description,
                "universe_mode": sp.universe_mode,
                "sharpe_4y": m_4y["full_sharpe"],
                "cagr_4y_pct": m_4y["full_cagr_pct"],
                "max_dd_4y_pct": m_4y["full_max_dd_pct"],
                "sharpe_23_24": m_4y["sharpe_23_24"],
                "sharpe_2025": m_4y["sharpe_2025"],
                "sharpe_2026": m_4y["sharpe_2026"],
                "cpcv_mean_sharpe": cpcv_4y["cpcv_mean_sharpe"],
                "cpcv_min_sharpe": cpcv_4y["cpcv_min_sharpe"],
                "cpcv_pos_splits_pct": cpcv_4y["cpcv_pos_splits_pct"],
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_ok,
                "rebal_full_closes": bs["rebal_full_closes"],
                "rebal_partial_adjusts": bs["rebal_partial_adjusts"],
                "profit_lock_exits": bs["profit_lock_exits"],
                "tp_exits": bs["tp_exits"],
                "core_fee_drag_pct": bs["core_fee_drag_pct"],
            }
        )

    return pd.DataFrame(rows)
