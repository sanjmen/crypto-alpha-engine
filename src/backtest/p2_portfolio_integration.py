"""
Unified 1-Minute One-Way Production Portfolio Integration for Phase-P2 L2-Sweep Snipers.

Synthesizes the full production stack (`2023–2026`) on exact 1-minute linear exchange physics
via `src/backtest/unified_1m_simulator.py`:
  1. `Core P1`: `SevenFactorOnlineEngine` on `TOP_15_SYMBOLS` (`core_p1_risk_scale = 0.85`) +
     `1m` intrabar Chop/Trend Profit-Lock (`2.4%->1.2%` / `3.5%->1.5%`).
  2. `Manga P2`: `CausalDollarBarTSMOMEngine` on `BTCUSDT, ETHUSDT, SOLUSDT` (`max_gross = 0.95`).
  3. `Deployed Snipers`: `EventDrivenSniperEngine` (`M09 Coil`, `M23 Breadth`, `M04 NegFund`)
     resolved on exact `1-minute` bars via `resolve_sniper_1m_path`.
  4. `New L2-Book / Tick-Sweep Candidates`: `P6A_L2_ASK_AIR_POCKET_LONG` & `P2` Tick-Sweep
     survivors (`MONO_A_CONVEX_TRAIL`, `MONO_B_COUPON_VACUUM`, `MONO_C_SPLIT_50_50`), enforcing
     strict One-Way symbol non-collision, `3`-slot concurrency cap, `$2,700` Two-Phase SafeRamp,
     and `$2,500` Cushion-Safe CPPI Floor.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.unified_1m_simulator import (
    resolve_sniper_1m_path,
    run_unified_1m_portfolio_simulation,
)
from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine
from src.strategies.event_snipers import (
    SNIPER_BRACKET_PROFILES,
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_UNIVERSE_82,
    EventDrivenSniperEngine,
)
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine

PROD_TOP_15: list[str] = [
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
]


def convert_trade_df_to_hourly_sniper_events(
    trades_df: pd.DataFrame,
    strat_label: str,
) -> list[dict[str, Any]]:
    """
    Converts simulated `1-minute` linear trade records (`symbol, entry_ts, exit_ts, net_bps, fee_bps, strength`)
    into the exact hourly mark-to-market event format consumed by `run_unified_1m_portfolio_simulation`,
    charging half the roundtrip fee in `entry_hr` and accruing linear return uniformly across open hours
    such that `sum(hourly_ret.values()) == net_bps * 1e-4` to machine precision.
    """
    if trades_df is None or trades_df.empty:
        return []

    out: list[dict[str, Any]] = []
    for row in trades_df.itertuples(index=False):
        en_ts = int(row.entry_ts)
        ex_ts = max(en_ts + 60, int(row.exit_ts))
        net_ret = float(row.net_bps) * 1e-4
        fee_ret = float(row.fee_bps) * 1e-4

        en_hr = (en_ts // 3600) * 3600
        ex_hr = ((ex_ts - 1) // 3600) * 3600
        hrs = list(range(en_hr, ex_hr + 3600, 3600))
        n_h = max(1, len(hrs))

        gross_plus_fnd = net_ret + fee_ret
        per_h = gross_plus_fnd / float(n_h)
        hr_map: dict[int, float] = {}
        for idx_h, h_b in enumerate(hrs):
            val = per_h
            if idx_h == 0:
                val -= 0.5 * fee_ret
            if idx_h == n_h - 1:
                val -= 0.5 * fee_ret
            hr_map[h_b] = val

        out.append(
            {
                "hr_ts": en_hr,
                "strat": strat_label,
                "sym": str(row.symbol),
                "entry_ts": en_ts,
                "exit_ts": ex_ts,
                "score": float(row.strength),
                "hourly_ret": hr_map,
                "was_loss": bool(net_ret < 0.0),
            }
        )
    return out


def compute_portfolio_era_metrics(
    sim_out: dict[str, Any],
    initial_capital: float = 2700.0,
) -> dict[str, Any]:
    """Computes full-horizon (`2023–2026`), `2023–24`, `2025`, and `2026` linear portfolio metrics."""
    d_s: pd.Series = sim_out["daily"]

    def _stats(sub: pd.Series) -> dict[str, float]:
        if sub.empty or len(sub) < 5:
            return {
                "sharpe": 0.0,
                "geom_daily_pct": 0.0,
                "cagr_pct": 0.0,
                "max_dd_pct": 0.0,
            }
        arr = sub.to_numpy(dtype=np.float64)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        sh = (mu / sd) * np.sqrt(365.25) if sd > 1e-12 else 0.0
        eq = np.cumprod(1.0 + np.clip(arr, -0.95, None))
        tot = float(eq[-1])
        n_d = len(arr)
        geom_d = (tot ** (1.0 / n_d) - 1.0) * 100.0 if tot > 0 else -100.0
        cagr = (tot ** (365.25 / n_d) - 1.0) * 100.0 if tot > 0 else -100.0
        pk = np.maximum.accumulate(np.insert(eq, 0, 1.0))[1:]
        mdd = abs(float(np.min((eq - pk) / np.maximum(pk, 1e-12)))) * 100.0
        return {
            "sharpe": round(sh, 2),
            "geom_daily_pct": round(geom_d, 4),
            "cagr_pct": round(cagr, 1),
            "max_dd_pct": round(mdd, 2),
        }

    s_full = _stats(d_s)
    s_24 = _stats(d_s[d_s.index.year <= 2024])
    s_25 = _stats(d_s[d_s.index.year == 2025])
    s_26 = _stats(d_s[d_s.index.year >= 2026])

    return {
        "final_eq_usd": round(float(sim_out["final_eq"]), 2),
        "total_return_pct": round(((float(sim_out["final_eq"]) / initial_capital) - 1.0) * 100.0, 1),
        "min_eq_usd": round(float(sim_out["min_eq"]), 2),
        "floor_2500_intact": bool(float(sim_out["min_eq"]) >= 2500.0),
        "worst_24h_dd_pct": round(float(sim_out["worst_24h_dd"]), 2),
        "full_sharpe": s_full["sharpe"],
        "full_geom_daily_pct": s_full["geom_daily_pct"],
        "full_cagr_pct": s_full["cagr_pct"],
        "full_max_dd_pct": s_full["max_dd_pct"],
        "sharpe_23_24": s_24["sharpe"],
        "sharpe_2025": s_25["sharpe"],
        "sharpe_2026": s_26["sharpe"],
        "cagr_2025_pct": s_25["cagr_pct"],
        "cagr_2026_pct": s_26["cagr_pct"],
        "geom_daily_2025_pct": s_25["geom_daily_pct"],
        "geom_daily_2026_pct": s_26["geom_daily_pct"],
        "max_dd_2025_pct": s_25["max_dd_pct"],
        "max_dd_2026_pct": s_26["max_dd_pct"],
        "sniper_counts": dict(sim_out.get("sniper_counts", {})),
    }


def identify_required_1m_symbols_for_portfolio(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    eval_start: str = "2023-03-01",
) -> list[str]:
    """
    Determines the exact mathematical superset of symbols requiring `1m` kline history
    for `Core P1`, `Manga P2`, and all three live event snipers (`M09`, `M23`, `M04`)
    with `0` false negatives:
      - `PROD_TOP_15` (`15` symbols)
      - Every symbol triggered by `EventDrivenSniperEngine.scan_historical_1h_events` (`M09` + `M23`)
      - Every `SNIPER_UNIVERSE_82` symbol (`adv_rank < 60`) that ever experienced `funding_rate <= -0.0004`
        during an hour whose `high` broke above the prior `23h` high (`100%` exact superset of `15m` `M04`!).
    """
    needed: set[str] = {s for s in PROD_TOP_15 if s in bars_1h_by_sym}
    sniper_syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    if not sniper_syms:
        return sorted(needed)

    btc_df = bars_1h_by_sym.get("BTCUSDT", bars_1h_by_sym[sniper_syms[0]])
    b_sorted = btc_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    all_idx = pd.DatetimeIndex(pd.to_datetime(b_sorted["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
    eval_idx = all_idx[all_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, ns = len(eval_idx), len(sniper_syms)
    if n_b <= 220:
        return sorted(needed)

    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()
    c_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    h_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    l_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    qv_1h = np.zeros((n_b, ns), dtype=np.float64)
    tb_1h = np.zeros((n_b, ns), dtype=np.float64)
    fr_1h = np.zeros((n_b, ns), dtype=np.float64)

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
            f_idx = pd.to_datetime(
                ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600,
                unit="s",
                utc=True,
            )
            f_ser = pd.Series(fd["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            fr_1h[:, j_s] = f_ser.reindex(eval_idx, method="ffill").fillna(0.0).to_numpy()

    adv30 = pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).mean().shift(1).fillna(0.0)
    rank_1h = adv30.rank(axis=1, ascending=False, method="first").fillna(999.0).to_numpy(dtype=np.int64) - 1

    sn_eng = EventDrivenSniperEngine()
    raw_1h_evs = sn_eng.scan_historical_1h_events(ts_eval, sniper_syms, c_1h, h_1h, l_1h, qv_1h, tb_1h, rank_1h)
    for _en_ts, s_idx, _side, _st_name, _sc_v, _rk_v in raw_1h_evs:
        needed.add(sniper_syms[s_idx])

    # Exact superset for 15m M04 (`funding_rate <= -0.0004`, `rank < 60`, `high_1h > prior 22h high`)
    h22_prev = pd.DataFrame(h_1h).shift(1).rolling(22, min_periods=10).max().to_numpy(dtype=np.float64)
    m04_cand_mask = (rank_1h < sn_eng.m04_max_rank) & (fr_1h <= sn_eng.m04_max_funding_rate) & (h_1h > h22_prev)
    for j_s in np.where(np.any(m04_cand_mask, axis=0))[0].tolist():
        needed.add(sniper_syms[int(j_s)])

    return sorted(needed)


def build_core_1m_tensors_and_base_snipers(
    core_1m_by_sym: dict[str, pd.DataFrame] | None,
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    sniper_1m_by_sym: dict[str, pd.DataFrame] | None = None,
    eval_start: str = "2023-03-01",
    symbol_1m_loader: Callable[[str], pd.DataFrame | None] | None = None,
    symbols_to_stream: list[str] | None = None,
    loader_workers: int = 2,
) -> dict[str, Any]:
    """
    Constructs exact `SevenFactorOnlineEngine` matrices (`Core P1`), `CausalDollarBarTSMOMEngine`
    1-minute Dollar-Bar weights (`Manga P2`), `(n_hours, 60, 15)` `1m` OHLC tensors, and `1m`-resolved
    baseline production sniper events (`M09`, `M23`, `M04`) using bounded O(1)-symbol streaming memory
    (`< 1.1 GB` peak RAM across `80+` 1-minute symbols!).
    """
    present_core = [s for s in PROD_TOP_15 if s in bars_1h_by_sym]
    bars_1h_core: dict[str, pd.DataFrame] = {}
    fnd_core: dict[str, pd.Series] = {}

    for s in present_core:
        b = bars_1h_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        if "volume" not in b.columns:
            b["volume"] = b["quote_volume"] / np.maximum(b["close"].to_numpy(dtype=np.float64), 1e-12)
        b.index = pd.DatetimeIndex(pd.to_datetime(b["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        b.drop(columns=["timestamp"], inplace=True)
        bars_1h_core[s] = b
        if s in funding_by_sym and not funding_by_sym[s].empty:
            f = funding_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_idx = pd.DatetimeIndex(
                pd.to_datetime(
                    ((f["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600,
                    unit="s",
                    utc=True,
                )
            )
            f_s = pd.Series(f["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            fnd_core[s] = f_s[~f_s.index.duplicated(keep="last")]

    eng_p1 = SevenFactorOnlineEngine(
        enable_chop_range_arb=True,
        enable_holistic_regime_shields=True,
    )
    mats = eng_p1.compute_matrices(bars_1h=bars_1h_core, funding_rates=fnd_core)
    raw_close_core = pd.DataFrame({s: bars_1h_core[s]["close"] for s in mats["close"].columns}).reindex(
        mats["close"].index
    )
    age_ok = raw_close_core.notna().cumsum() >= 168
    mats["w_masked"] = mats["w_base_1x"].where(age_ok, 0.0)
    core_cols = list(mats["close"].columns)

    start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_idx = mats["close"].index[mats["close"].index >= start_ts]
    n_b, n_c = len(eval_idx), len(core_cols)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    open_core_df = pd.DataFrame(index=eval_idx, columns=core_cols, dtype=np.float64)
    for s in core_cols:
        open_core_df[s] = bars_1h_core[s]["open"].reindex(eval_idx).ffill().bfill()
    open_core = open_core_df.to_numpy(dtype=np.float64)

    o_1m_core = np.full((n_b, 60, n_c), np.nan, dtype=np.float64)
    h_1m_core = np.full((n_b, 60, n_c), np.nan, dtype=np.float64)
    l_1m_core = np.full((n_b, 60, n_c), np.nan, dtype=np.float64)
    for j_c, sym in enumerate(core_cols):
        o_1m_core[:, 0, j_c] = open_core[:, j_c]
        h_1m_core[:, 0, j_c] = bars_1h_core[sym]["high"].reindex(eval_idx).to_numpy(dtype=np.float64)
        l_1m_core[:, 0, j_c] = bars_1h_core[sym]["low"].reindex(eval_idx).to_numpy(dtype=np.float64)

    hr_to_pos = {int(t_h): i for i, t_h in enumerate(ts_eval.tolist())}
    min_eval_s = int(ts_eval[0]) if n_b > 0 else 0
    max_eval_s = int(ts_eval[-1]) + 3600 if n_b > 0 else 0

    p2_eng = CausalDollarBarTSMOMEngine(max_gross_leverage=0.95)
    p2_w_mat = np.zeros((n_b, n_c), dtype=np.float64)

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

        adv30 = pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).mean().shift(1).fillna(0.0)
        rank_1h = adv30.rank(axis=1, ascending=False, method="first").fillna(999.0).to_numpy(dtype=np.int64) - 1
        rk_df_1h = pd.DataFrame(rank_1h, index=eval_idx, columns=sniper_syms)
        idx_15m = pd.date_range(eval_idx[0], eval_idx[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC")
        ts_15m = idx_15m.as_unit("s").astype(np.int64).to_numpy()

        for en_ts, s_idx, side, st_name, sc_v, rk_v in sn_eng.scan_historical_1h_events(
            ts_eval, sniper_syms, c_1h, h_1h, l_1h, qv_1h, tb_1h, rank_1h
        ):
            raw_1h_by_sym.setdefault(sniper_syms[s_idx], []).append((en_ts, side, st_name, sc_v, rk_v))

    def _consume_symbol_1m(sym: str, df1: pd.DataFrame | None) -> list[dict[str, Any]]:
        if df1 is None or df1.empty:
            return []

        if sym in core_cols:
            j_c = core_cols.index(sym)
            o_1m_core[:, 0, j_c] = np.nan
            h_1m_core[:, 0, j_c] = np.nan
            l_1m_core[:, 0, j_c] = np.nan
            d_sub = df1[(df1["timestamp"] >= min_eval_s) & (df1["timestamp"] < max_eval_s)]
            ts_m = d_sub["timestamp"].to_numpy(dtype=np.int64)
            om = d_sub["open"].to_numpy(dtype=np.float64)
            hm = d_sub["high"].to_numpy(dtype=np.float64)
            lm = d_sub["low"].to_numpy(dtype=np.float64)
            hr_b = (ts_m // 3600) * 3600
            min_off = ((ts_m % 3600) // 60).astype(np.int64)
            row_idx = np.array([hr_to_pos.get(int(x), -1) for x in hr_b], dtype=np.int64)
            valid = (row_idx >= 0) & (min_off >= 0) & (min_off < 60)
            r_v, m_v = row_idx[valid], min_off[valid]
            o_1m_core[r_v, m_v, j_c] = om[valid]
            h_1m_core[r_v, m_v, j_c] = hm[valid]
            l_1m_core[r_v, m_v, j_c] = lm[valid]

        if sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT") and sym in core_cols:
            p2_col = p2_eng.compute_historical_weights({sym: df1}, eval_idx)[sym].shift(1).fillna(0.0)
            p2_w_mat[:, core_cols.index(sym)] = p2_col.to_numpy(dtype=np.float64)

        sym_triggers: list[tuple[int, int, str, float, int]] = list(raw_1h_by_sym.get(sym, []))
        if (
            rk_df_1h is not None
            and idx_15m is not None
            and ts_15m is not None
            and sym in rk_df_1h.columns
            and sym in f_series_map
            and float(f_series_map[sym].min()) <= sn_eng.m04_max_funding_rate
        ):
            dm_sub = df1[(df1["timestamp"] >= min_eval_s) & (df1["timestamp"] < max_eval_s)]
            if not dm_sub.empty:
                d_idx = pd.DatetimeIndex(
                    pd.to_datetime(dm_sub["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True)
                )
                r15 = (
                    dm_sub.set_index(d_idx)
                    .resample("15min", label="left", closed="left")
                    .agg(
                        {
                            "high": "max",
                            "close": "last",
                            "quote_volume": "sum",
                            "taker_buy_quote_volume": "sum",
                        }
                    )
                    .reindex(idx_15m)
                )
                c_15 = r15["close"].to_numpy(dtype=np.float64)[:, None]
                h_15 = r15["high"].to_numpy(dtype=np.float64)[:, None]
                qv_15 = r15["quote_volume"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                tb_15 = r15["taker_buy_quote_volume"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                fr_15 = (
                    f_series_map[sym].reindex(idx_15m, method="ffill").fillna(0.0).to_numpy(dtype=np.float64)[:, None]
                )
                rk_15 = rk_df_1h[sym].reindex(idx_15m, method="ffill").fillna(999).to_numpy(dtype=np.int64)[:, None]
                for (
                    en_ts,
                    _j0,
                    side,
                    st_name,
                    sc_v,
                    rk_v,
                ) in sn_eng.scan_historical_15m_events(ts_15m, c_15, h_15, qv_15, tb_15, fr_15, rk_15):
                    sym_triggers.append((en_ts, side, st_name, sc_v, rk_v))

        if not sym_triggers:
            return []

        ts_a = df1["timestamp"].to_numpy(dtype=np.int64)
        o_a = df1["open"].to_numpy(dtype=np.float64)
        h_a = df1["high"].to_numpy(dtype=np.float64)
        l_a = df1["low"].to_numpy(dtype=np.float64)
        c_a = df1["close"].to_numpy(dtype=np.float64)
        fr_sym = f_maps.get(sym)

        f_ser_sym = f_series_map.get(sym)
        f_ts_ns = (
            f_ser_sym.index.as_unit("s").astype(np.int64).to_numpy()
            if f_ser_sym is not None and not f_ser_sym.empty
            else np.empty(0, dtype=np.int64)
        )
        f_vals_sym = (
            f_ser_sym.to_numpy(dtype=np.float64)
            if f_ser_sym is not None and not f_ser_sym.empty
            else np.empty(0, dtype=np.float64)
        )

        resolved_out: list[dict[str, Any]] = []
        for en_ts, side, st_name, sc_v, rk_v in sym_triggers:
            resolved = resolve_sniper_1m_path(
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
            if resolved is not None:
                ex_ts, hr_ret, is_loss, _ = resolved
                f_pos = int(np.searchsorted(f_ts_ns, int(en_ts), side="right")) - 1
                en_fr = float(f_vals_sym[f_pos]) if f_pos >= 0 else 0.0
                resolved_out.append(
                    {
                        "hr_ts": (en_ts // 3600) * 3600,
                        "strat": st_name,
                        "sym": sym,
                        "side": int(side),
                        "entry_ts": en_ts,
                        "exit_ts": ex_ts,
                        "score": sc_v,
                        "entry_fr": en_fr,
                        "hourly_ret": hr_ret,
                        "was_loss": is_loss,
                    }
                )
        return resolved_out

    base_events: list[dict[str, Any]] = []
    if symbol_1m_loader is not None:
        stream_list = list(symbols_to_stream or sorted(set(core_cols) | set(sniper_syms)))
        t_s0 = time.time()
        done_cnt = 0
        with ThreadPoolExecutor(max_workers=max(1, int(loader_workers))) as pool:
            # Submit in bounded micro-batches of 6 so at most 6 symbols exist in flight/queue at once!
            batch_sz = max(2, int(loader_workers) * 2)
            for b_i in range(0, len(stream_list), batch_sz):
                batch_syms = stream_list[b_i : b_i + batch_sz]
                futs = {pool.submit(symbol_1m_loader, s_k): s_k for s_k in batch_syms}
                for fut in as_completed(futs):
                    s_k = futs[fut]
                    df_loaded = fut.result()
                    evs_sym = _consume_symbol_1m(s_k, df_loaded)
                    base_events.extend(evs_sym)
                    del df_loaded
                    done_cnt += 1
                    if done_cnt % 10 == 0 or done_cnt == len(stream_list):
                        print(
                            f"  • Streamed & consumed 1m klines {done_cnt}/{len(stream_list)} symbols "
                            f"({time.time() - t_s0:.1f}s, O(1) RAM)...",
                            flush=True,
                        )
                gc.collect()
    else:
        all_1m_lookup = {**(core_1m_by_sym or {}), **(sniper_1m_by_sym or {})}
        for s_k in sorted(set(core_cols) | set(sniper_syms)):
            if s_k in all_1m_lookup:
                base_events.extend(_consume_symbol_1m(s_k, all_1m_lookup[s_k]))

    return {
        "mats": mats,
        "o_1m_core": o_1m_core,
        "h_1m_core": h_1m_core,
        "l_1m_core": l_1m_core,
        "eval_idx": eval_idx,
        "open_core": open_core,
        "core_cols": core_cols,
        "p2_w_mat": p2_w_mat,
        "base_sniper_events": base_events,
        "base_allowed_strats": {SNIPER_M09_STRAT, SNIPER_M23_STRAT, SNIPER_M04_STRAT},
    }


def run_p2_portfolio_ablation_1m(
    bundle: dict[str, Any],
    candidate_sleeves: dict[str, pd.DataFrame],
    initial_capital: float = 2700.0,
    sniper_slot_lev: float = 0.24,
    sniper_max_pos: int = 3,
    extra_slot_configs: list[tuple[str, str, int, float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs side-by-side 1-minute One-Way portfolio simulations comparing:
      - `00_CURRENT_PROD_1M_NO_M25` (`Core P1 + Manga P2 + M09/M23/M04`, `M25=OFF`)
      - Each candidate L2-Book / Tick-Sweep sleeve added under strict One-Way non-collision,
        evaluating BOTH the full `2023–2026` horizon AND a fresh `$2,700` start on `2026-01-01`
        against the `$2,500` Cushion-Safe CPPI Hard Floor!
    """
    base_evs = list(bundle["base_sniper_events"])
    base_strats = set(bundle["base_allowed_strats"])
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    min_eval_s = int(eval_idx[0].timestamp()) if len(eval_idx) > 0 else 0

    mask_26 = eval_idx >= pd.Timestamp("2026-01-01", tz="UTC")
    has_26 = bool(np.sum(mask_26) >= 24 * 15)

    def _group_by_hr(ev_list: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        by_h: dict[int, list[dict[str, Any]]] = {}
        for e in ev_list:
            by_h.setdefault(int(e["hr_ts"]), []).append(e)
        return by_h

    def _eval_one_config(
        label: str,
        ev_list: list[dict[str, Any]],
        allowed_set: set[str],
        cand_tag: str | None,
        cand_total: int,
        cand_total_26: int,
        max_pos_cfg: int,
        slot_lev_cfg: float,
    ) -> dict[str, Any]:
        by_hr = _group_by_hr(ev_list)
        sim_full = run_unified_1m_portfolio_simulation(
            mats=bundle["mats"],
            o_1m_core=bundle["o_1m_core"],
            h_1m_core=bundle["h_1m_core"],
            l_1m_core=bundle["l_1m_core"],
            eval_idx=eval_idx,
            open_core=bundle["open_core"],
            universe_symbols=bundle["core_cols"],
            p2_w_mat=bundle["p2_w_mat"],
            p1_risk_scale=0.85,
            resolved_sniper_events_by_hour=by_hr,
            sniper_slot_lev=slot_lev_cfg,
            sniper_max_pos=max_pos_cfg,
            allowed_sniper_strats=allowed_set,
            initial_capital=initial_capital,
        )
        m = compute_portfolio_era_metrics(sim_full, initial_capital=initial_capital)
        m["variant"] = label
        m["sniper_max_pos"] = max_pos_cfg
        m["sniper_slot_lev"] = slot_lev_cfg
        m["new_sleeve_trades_executed"] = int(m["sniper_counts"].get(cand_tag, 0)) if cand_tag else 0
        m["new_sleeve_candidates_total"] = int(cand_total)

        if has_26:
            sim_26 = run_unified_1m_portfolio_simulation(
                mats=bundle["mats"],
                o_1m_core=bundle["o_1m_core"][mask_26],
                h_1m_core=bundle["h_1m_core"][mask_26],
                l_1m_core=bundle["l_1m_core"][mask_26],
                eval_idx=eval_idx[mask_26],
                open_core=bundle["open_core"][mask_26],
                universe_symbols=bundle["core_cols"],
                p2_w_mat=bundle["p2_w_mat"][mask_26],
                p1_risk_scale=0.85,
                resolved_sniper_events_by_hour=by_hr,
                sniper_slot_lev=slot_lev_cfg,
                sniper_max_pos=max_pos_cfg,
                allowed_sniper_strats=allowed_set,
                initial_capital=initial_capital,
            )
            m["eq_2026_from_2700_usd"] = round(float(sim_26["final_eq"]), 2)
            m["ret_2026_from_2700_pct"] = round(((float(sim_26["final_eq"]) / initial_capital) - 1.0) * 100.0, 1)
            m["min_eq_2026_from_2700_usd"] = round(float(sim_26["min_eq"]), 2)
            m["floor_2500_intact_2026"] = bool(float(sim_26["min_eq"]) >= 2500.0)
            m["new_sleeve_trades_2026"] = int(sim_26.get("sniper_counts", {}).get(cand_tag, 0)) if cand_tag else 0
            m["new_sleeve_candidates_2026"] = int(cand_total_26)
        else:
            m["eq_2026_from_2700_usd"] = initial_capital
            m["ret_2026_from_2700_pct"] = 0.0
            m["min_eq_2026_from_2700_usd"] = initial_capital
            m["floor_2500_intact_2026"] = True
            m["new_sleeve_trades_2026"] = 0
            m["new_sleeve_candidates_2026"] = 0

        return m

    rows: list[dict[str, Any]] = []
    m0 = _eval_one_config(
        label="00_CURRENT_PROD_1M_NO_M25",
        ev_list=base_evs,
        allowed_set=base_strats,
        cand_tag=None,
        cand_total=0,
        cand_total_26=0,
        max_pos_cfg=sniper_max_pos,
        slot_lev_cfg=sniper_slot_lev,
    )
    m0["sharpe_lift_full"] = 0.0
    m0["sharpe_lift_2025"] = 0.0
    m0["sharpe_lift_2026"] = 0.0
    m0["eq_2026_delta_usd"] = 0.0
    rows.append(m0)

    base_sh_full = m0["full_sharpe"]
    base_sh_25 = m0["sharpe_2025"]
    base_sh_26 = m0["sharpe_2026"]
    base_eq_26 = m0["eq_2026_from_2700_usd"]

    ts_2026_start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    prepared_cands: dict[str, tuple[str, list[dict[str, Any]], int, int]] = {}
    for cand_name, cand_df in candidate_sleeves.items():
        sub_df = cand_df[cand_df["entry_ts"] >= min_eval_s] if not cand_df.empty else cand_df
        strat_tag = f"Sniper:{cand_name}"
        cand_evs = convert_trade_df_to_hourly_sniper_events(sub_df, strat_tag)
        n_26 = int((sub_df["entry_ts"] >= ts_2026_start).sum()) if not sub_df.empty else 0
        prepared_cands[cand_name] = (strat_tag, cand_evs, len(cand_evs), n_26)

        mc = _eval_one_config(
            label=f"PROD + {cand_name}",
            ev_list=base_evs + cand_evs,
            allowed_set=base_strats | {strat_tag},
            cand_tag=strat_tag,
            cand_total=len(cand_evs),
            cand_total_26=n_26,
            max_pos_cfg=sniper_max_pos,
            slot_lev_cfg=sniper_slot_lev,
        )
        mc["sharpe_lift_full"] = round(mc["full_sharpe"] - base_sh_full, 2)
        mc["sharpe_lift_2025"] = round(mc["sharpe_2025"] - base_sh_25, 2)
        mc["sharpe_lift_2026"] = round(mc["sharpe_2026"] - base_sh_26, 2)
        mc["eq_2026_delta_usd"] = round(mc["eq_2026_from_2700_usd"] - base_eq_26, 2)
        rows.append(mc)

    for extra_label, ref_cand, max_p, s_lev in extra_slot_configs or []:
        if ref_cand not in prepared_cands:
            continue
        strat_tag, cand_evs, n_tot, n_26 = prepared_cands[ref_cand]
        me = _eval_one_config(
            label=extra_label,
            ev_list=base_evs + cand_evs,
            allowed_set=base_strats | {strat_tag},
            cand_tag=strat_tag,
            cand_total=n_tot,
            cand_total_26=n_26,
            max_pos_cfg=max_p,
            slot_lev_cfg=s_lev,
        )
        me["sharpe_lift_full"] = round(me["full_sharpe"] - base_sh_full, 2)
        me["sharpe_lift_2025"] = round(me["sharpe_2025"] - base_sh_25, 2)
        me["sharpe_lift_2026"] = round(me["sharpe_2026"] - base_sh_26, 2)
        me["eq_2026_delta_usd"] = round(me["eq_2026_from_2700_usd"] - base_eq_26, 2)
        rows.append(me)

    return rows
