"""
Frente E1 Laboratory: Basis & Carry Harvester (`Manga P4`) + Definitive Resolution of
GitHub Issue `#64` (`M15-4: Extreme Funding Rate Harvester`) (`1-Minute` Physics, `2023–2026`).

Executes a three-stage forensic study with `0.000 ms` lookahead across `SNIPER_UNIVERSE_82`
using exact `1m` USD-M klines, `1h` `premiumIndexKlines` (`Spot-Perp Basis` in bps), and
realized `:00 / :08 / :16` funding coupons:

  1. **Part E1.1 — Issue `#64` (`M15-4`) Pre-Settlement Extreme Funding Coupon Snipers (`H1..H5`):**
     - Uses **strictly causal predicted funding** at `T - 60m` (`7/8` closed hours of the `8h` TWAP
       window from `premiumIndexKlines` + prior settled funding, `0.000 ms` peek into `[T-60m, T]`),
       while crediting/charging the **actual realized `:00:00` coupon** at settlement!
     - Tests `H1` (Shadow naive unfiltered baseline reproducing Issue `#64`'s failure), `H2` (exact
       Issue `#64` proposal: `$75M+` ADV, `T - 30m` entry, `4h` trend filter), `H3` (`Spot > Perp`
       basis-anchored negative-funding squeeze coupon), `H4` (`Perp > Spot` frothy positive-funding
       short coupon), and `H5` (`Beta-Neutral Pair: Alt Coupon Leg + Beta-Matched BTCUSDT Hedge`
       across `[T - 30m, T + 5m]`).

  2. **Part E1.2 — Continuous Multi-Cycle `Manga P4` Carry Sleeves (`P4_C1..P4_C4`, `1x` Standalone):**
     - `P4_C1_SPOT_PERP_CASH_AND_CARRY_MAJORS`: Honest `1x` Spot capital lockup (`0.87x` hedged
       notional per `$1` allocated equity, `10.05 bps` each way) on `BTC, ETH, SOL`.
     - `P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY`: `100%` Intra-USD-M (`0` Spot lockup) Short Overheated
       Liquid Alts (`FR_24h >= +8 bps / 8h`, `Perp > Spot`) + Long Beta-Matched `BTCUSDT`.
     - `P4_C3_USDM_SYMMETRIC_DISPERSION_POD`: Adds Long Deep-Negative-FR Alts (`Spot >= Perp`,
       `ret_24h > 0`) + Short Beta-Matched `BTCUSDT`.
     - `P4_C4_WIDE_TOP35_CARRY_BHS`: Cross-sectional pure carry across `Top-35` liquid perps (`BHS(3,8)`).

  3. **Part E1.3 — Unified `1m` Portfolio Integration (`Stack M17 D10/D11` + `Manga P4` Carry):**
     - Compares `20%` permanent Spot+Perp capital split vs Idle-Cash Spot+Perp parking vs
       One-Way Non-Colliding Intra-USD-M Carry Pod (`0.40x`) + surviving Coupon Sniper (`$2,700`
       `2026` start against `$2,500` CPPI hard floor).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import compute_cpcv_portfolio_sharpe_stability
from src.backtest.frente_d_architecture_synthesis_lab import (
    FRENTE_D_SPECS,
    simulate_frente_d_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    compute_portfolio_era_metrics,
    convert_trade_df_to_hourly_sniper_events,
)
from src.data.microstructure_p1_ingestor import (
    _build_pooled_session,
    parse_premium_index_zip_bytes,
)
from src.data.multimodal_vision_ingestor import PERP_TICKER_ALIASES, VISION_CDN_BASE
from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics
from src.strategies.event_snipers import (
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_M26_STRAT,
    SNIPER_UNIVERSE_82,
)


def fetch_1h_premium_index_klines_universe(
    symbols: list[str],
    all_yms: list[tuple[int, int]],
    recent_days: list[str],
    max_ts_sec: int,
    workers: int = 12,
) -> dict[str, pd.DataFrame]:
    """
    Fetches `1h` `premiumIndexKlines` (`p_open_bps, p_high_bps, p_low_bps, p_close_bps`)
    from Binance Vision CDN for `symbols` across `2023–2026` (`~15s` on Colab).
    """
    sess = _build_pooled_session(pool_size=64)

    def _fetch_sym(sym: str) -> tuple[str, pd.DataFrame | None]:
        tks = PERP_TICKER_ALIASES.get(sym, (sym,))
        parts: list[pd.DataFrame] = []
        for y, m in all_yms:
            ym_s = f"{y}-{m:02d}"
            for tk in tks:
                url = f"{VISION_CDN_BASE}/data/futures/um/monthly/premiumIndexKlines/{tk}/1h/{tk}-1h-{ym_s}.zip"
                try:
                    r = sess.get(url, timeout=15)
                    if r.status_code == 200 and r.content:
                        p_df = parse_premium_index_zip_bytes(r.content, sym)
                        if p_df is not None and not p_df.empty:
                            parts.append(p_df)
                        break
                except Exception:  # noqa: BLE001
                    continue
        for d_s in recent_days:
            for tk in tks:
                url = f"{VISION_CDN_BASE}/data/futures/um/daily/premiumIndexKlines/{tk}/1h/{tk}-1h-{d_s}.zip"
                try:
                    r = sess.get(url, timeout=12)
                    if r.status_code == 200 and r.content:
                        p_df = parse_premium_index_zip_bytes(r.content, sym)
                        if p_df is not None and not p_df.empty:
                            parts.append(p_df)
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not parts:
            return sym, None
        df_m = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        df_m = df_m[df_m["timestamp"] <= max_ts_sec + 3600].reset_index(drop=True)
        return sym, df_m

    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_fetch_sym, s): s for s in symbols}
        for fut in as_completed(futs):
            s_k, df_k = fut.result()
            if df_k is not None and not df_k.empty:
                out[s_k] = df_k
    sess.close()
    return out


@dataclass(frozen=True)
class PreSettleHarvesterSpec:
    mech_id: str
    description: str
    min_adv_usd: float
    entry_offset_min: int  # minutes BEFORE `:00:00` settlement, e.g. 60 (`T-60m`) or 30 (`T-30m`)
    exit_offset_min: int  # minutes AFTER `:00:00` settlement, e.g. 3 (`T+3m`) or 10 (`T+10m`)
    min_abs_fr_bps: float  # causal predicted |FR| threshold in bps (e.g. 15.0 or 20.0)
    mode: str  # "both_naive", "both_issue64", "neg_basis_only", "pos_froth_only", "beta_hedged_pair"
    sl_bps: float = 160.0


HARVESTER_ISSUE64_SPECS: tuple[PreSettleHarvesterSpec, ...] = (
    PreSettleHarvesterSpec(
        mech_id="H1_SHADOW_NAIVE_UNFILTERED_60M",
        description="Reproduces Shadow Issue #64 Failure (`ADV>=$10M`, `|FR|>=15bps`, `T-60m->T+5m`, No Trend/Basis Gate)",
        min_adv_usd=10_000_000.0,
        entry_offset_min=60,
        exit_offset_min=5,
        min_abs_fr_bps=15.0,
        mode="both_naive",
        sl_bps=220.0,
    ),
    PreSettleHarvesterSpec(
        mech_id="H2_ISSUE64_EXACT_RECALIB_30M",
        description="Exact Issue #64 Proposal (`ADV>=$75M`, `|FR|>=15bps`, `T-30m->T+3m`, `4h` Trend Aligned)",
        min_adv_usd=75_000_000.0,
        entry_offset_min=30,
        exit_offset_min=3,
        min_abs_fr_bps=15.0,
        mode="both_issue64",
        sl_bps=150.0,
    ),
    PreSettleHarvesterSpec(
        mech_id="H3_BASIS_ANCHORED_NEG_COUPON_30M",
        description="Spot-Shortage Long Coupon (`ADV>=$50M`, `FR<=-20bps`, `Spot>Perp >=25bps`, `ret_4h>0`, `T-30m->T+10m`)",
        min_adv_usd=50_000_000.0,
        entry_offset_min=30,
        exit_offset_min=10,
        min_abs_fr_bps=20.0,
        mode="neg_basis_only",
        sl_bps=160.0,
    ),
    PreSettleHarvesterSpec(
        mech_id="H4_FROTH_CONTANGO_SHORT_COUPON_30M",
        description="Blow-Off Contango Short Coupon (`ADV>=$50M`, `FR>=+20bps`, `Perp>Spot >=15bps`, `ret_4h<=+0.5%`, `T-30m->T+10m`)",
        min_adv_usd=50_000_000.0,
        entry_offset_min=30,
        exit_offset_min=10,
        min_abs_fr_bps=20.0,
        mode="pos_froth_only",
        sl_bps=160.0,
    ),
    PreSettleHarvesterSpec(
        mech_id="H5_BETA_NEUTRAL_COUPON_PAIR_30M",
        description="Pod Beta-Hedged Pair (`H3+H4` Alt Leg + Beta-Matched `BTCUSDT` Hedge across `[T-30m, T+5m]`)",
        min_adv_usd=50_000_000.0,
        entry_offset_min=30,
        exit_offset_min=5,
        min_abs_fr_bps=20.0,
        mode="beta_hedged_pair",
        sl_bps=180.0,
    ),
)


def compute_causal_predicted_funding_bps(
    prev_fr_bps: np.ndarray,
    prem_twap_7h_bps: np.ndarray,
) -> np.ndarray:
    """
    Computes strictly causal (`0.000 ms` lookahead at `T - 60m`) predicted funding rate in `bps`:
      - Uses the closed `7h` `premiumIndex` TWAP (`p_bps + clamp(1.0 - p_bps, -5.0, +5.0)`) when finite,
        blended `75% / 25%` with the prior settled funding rate (`prev_fr_bps`) to capture dynamic caps/intervals.
    """
    clamp_interest = np.clip(1.0 - prem_twap_7h_bps, -5.0, 5.0)
    est_from_prem = prem_twap_7h_bps + clamp_interest
    out = np.where(np.isfinite(prem_twap_7h_bps), 0.75 * est_from_prem + 0.25 * prev_fr_bps, prev_fr_bps)
    return np.clip(np.nan_to_num(out, nan=0.0), -250.0, 250.0)


def identify_issue64_candidate_windows(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    prem_1h_by_sym: dict[str, pd.DataFrame],
    eval_start: str = "2023-03-01",
    min_abs_fr_bps: float = 15.0,
    min_adv_usd: float = 10_000_000.0,
) -> dict[str, list[int]]:
    """Returns `{symbol: [t_settle_sec, ...]}` so `1m` streaming can slice `[t-4200, t+1200]` in a single pass."""
    syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    if not syms or "BTCUSDT" not in bars_1h_by_sym:
        return {}
    btc_1h = bars_1h_by_sym["BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    full_idx = pd.DatetimeIndex(pd.to_datetime(btc_1h["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
    eval_idx = full_idx[full_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, ns = len(eval_idx), len(syms)
    if n_b < 60:
        return {}
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()
    qv_1h = np.zeros((n_b, ns), dtype=np.float64)
    prem_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    settle_fr_bps = np.full((n_b, ns), np.nan, dtype=np.float64)
    for j_s, sym in enumerate(syms):
        bs = bars_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        qv_1h[:, j_s] = (
            pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        if sym in prem_1h_by_sym and not prem_1h_by_sym[sym].empty:
            ps = prem_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            p_idx = pd.DatetimeIndex(pd.to_datetime(ps["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
            prem_1h[:, j_s] = (
                pd.Series(ps["p_close_bps"].to_numpy(dtype=np.float64), index=p_idx).reindex(eval_idx).to_numpy()
            )
        if sym in funding_by_sym and not funding_by_sym[sym].empty:
            fd = funding_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_ts = ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600
            f_ser = pd.Series(
                fd["funding_rate"].to_numpy(dtype=np.float64) * 1e4,
                index=pd.DatetimeIndex(pd.to_datetime(f_ts, unit="s", utc=True)),
            )
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            settle_fr_bps[:, j_s] = f_ser.reindex(eval_idx).to_numpy()
    adv30 = (
        pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).sum().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
        / 30.0
    )
    prev_fr = pd.DataFrame(settle_fr_bps).ffill().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    prem_7h = pd.DataFrame(prem_1h).rolling(7, min_periods=2).mean().shift(1).to_numpy(dtype=np.float64)
    pred_fr = compute_causal_predicted_funding_bps(prev_fr, prem_7h)
    out: dict[str, list[int]] = {}
    all_t_btc: set[int] = set()
    for t in range(10, n_b - 2):
        t_dec = t - 1
        t_s = int(ts_eval[t])
        for j_s, sym in enumerate(syms):
            if sym == "BTCUSDT" or not np.isfinite(settle_fr_bps[t, j_s]):
                continue
            if abs(float(pred_fr[t_dec, j_s])) >= min_abs_fr_bps and float(adv30[t_dec, j_s]) >= min_adv_usd:
                out.setdefault(sym, []).append(t_s)
                all_t_btc.add(t_s)
    if all_t_btc:
        out["BTCUSDT"] = sorted(all_t_btc)
    return out


def run_issue64_coupon_harvester_1m_study(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    prem_1h_by_sym: dict[str, pd.DataFrame],
    symbol_1m_loader: Any,
    eval_start: str = "2023-03-01",
    specs: tuple[PreSettleHarvesterSpec, ...] = HARVESTER_ISSUE64_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates `H1..H5` Pre-Settlement Extreme Funding Coupon Harvesters on exact `1m` physics (`2023–2026`)."""
    syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    if not syms or "BTCUSDT" not in bars_1h_by_sym:
        return pd.DataFrame(), pd.DataFrame()

    btc_1h = bars_1h_by_sym["BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    full_idx = pd.DatetimeIndex(pd.to_datetime(btc_1h["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
    eval_idx = full_idx[full_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, ns = len(eval_idx), len(syms)
    if n_b < 60:
        return pd.DataFrame(), pd.DataFrame()
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    c_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    qv_1h = np.zeros((n_b, ns), dtype=np.float64)
    prem_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    settle_fr_bps = np.full((n_b, ns), np.nan, dtype=np.float64)

    for j_s, sym in enumerate(syms):
        bs = bars_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        c_1h[:, j_s] = pd.Series(bs["close"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        qv_1h[:, j_s] = (
            pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        if sym in prem_1h_by_sym and not prem_1h_by_sym[sym].empty:
            ps = prem_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            p_idx = pd.DatetimeIndex(pd.to_datetime(ps["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
            prem_1h[:, j_s] = (
                pd.Series(ps["p_close_bps"].to_numpy(dtype=np.float64), index=p_idx).reindex(eval_idx).to_numpy()
            )
        if sym in funding_by_sym and not funding_by_sym[sym].empty:
            fd = funding_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_ts = ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600
            f_idx = pd.DatetimeIndex(pd.to_datetime(f_ts, unit="s", utc=True))
            f_ser = pd.Series(fd["funding_rate"].to_numpy(dtype=np.float64) * 1e4, index=f_idx)
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            settle_fr_bps[:, j_s] = f_ser.reindex(eval_idx).to_numpy()

    adv30_daily = (
        pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).sum().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
        / 30.0
    )
    prev_fr = pd.DataFrame(settle_fr_bps).ffill().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    prem_7h_twap = pd.DataFrame(prem_1h).rolling(7, min_periods=2).mean().shift(1).to_numpy(dtype=np.float64)
    prem_prev = pd.DataFrame(prem_1h).shift(1).to_numpy(dtype=np.float64)
    pred_fr_bps = compute_causal_predicted_funding_bps(prev_fr, prem_7h_twap)

    ret_1h_df = pd.DataFrame(c_1h).pct_change(fill_method=None).fillna(0.0)
    ret_4h_prev = ret_1h_df.rolling(4, min_periods=2).sum().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    btc_j = syms.index("BTCUSDT")
    btc_r = ret_1h_df.iloc[:, btc_j]
    btc_var = btc_r.rolling(72, min_periods=12).var().replace(0, 1e-8)
    beta_72 = (
        ret_1h_df.rolling(72, min_periods=12)
        .cov(btc_r)
        .div(btc_var, axis=0)
        .clip(0.25, 2.40)
        .shift(1)
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
    )

    # Scan candidate settlement timestamps `t` where an actual funding settlement occurs (`isfinite(settle_fr_bps[t, j])`)
    # Note: at `T_settle - 60m` (`bar t-1` closed at `T_settle - 60m`? Wait: bar `t-1` opens at `T_settle - 3600` and CLOSES at `T_settle`!
    # So at `T_settle - 30m` or `T_settle - 60m`, the latest COMPLETED 1h bar is `t - 2`, which closed at `T_settle - 3600`!!).
    # Using index `t - 1` of `.shift(1)` arrays (`which equals raw bar t - 2`, closed at `T_settle - 3600`!) guarantees `0.000 ms` lookahead!
    cand_by_sym: dict[str, list[tuple[int, float, float, float, float, float, float]]] = {}
    min_thr = min(sp.min_abs_fr_bps for sp in specs)
    min_adv = min(sp.min_adv_usd for sp in specs)

    for t in range(10, n_b - 2):
        t_dec = t - 1  # Strictly closed at `ts_eval[t] - 3600` (`60 minutes` BEFORE `ts_eval[t]` settlement!)
        t_settle = int(ts_eval[t])
        for j_s, sym in enumerate(syms):
            if sym == "BTCUSDT":
                continue
            real_fr = settle_fr_bps[t, j_s]
            if not np.isfinite(real_fr):
                continue
            est_fr = float(pred_fr_bps[t_dec, j_s])
            adv_v = float(adv30_daily[t_dec, j_s])
            if abs(est_fr) < min_thr or adv_v < min_adv:
                continue
            pr_v = float(prem_prev[t_dec, j_s]) if np.isfinite(prem_prev[t_dec, j_s]) else 0.0
            r4_v = float(ret_4h_prev[t_dec, j_s])
            bt_v = float(beta_72[t_dec, j_s])
            cand_by_sym.setdefault(sym, []).append((t_settle, est_fr, float(real_fr), adv_v, pr_v, r4_v, bt_v))

    if not cand_by_sym:
        return pd.DataFrame(), pd.DataFrame()

    # Load `BTCUSDT` 1m once for `H5` beta-hedged pair leg
    btc_1m_df = symbol_1m_loader("BTCUSDT")
    btc_ts = (
        btc_1m_df["timestamp"].to_numpy(dtype=np.int64)
        if btc_1m_df is not None and not btc_1m_df.empty
        else np.empty(0, dtype=np.int64)
    )
    btc_o = (
        btc_1m_df["open"].to_numpy(dtype=np.float64)
        if btc_1m_df is not None and not btc_1m_df.empty
        else np.empty(0, dtype=np.float64)
    )
    btc_c = (
        btc_1m_df["close"].to_numpy(dtype=np.float64)
        if btc_1m_df is not None and not btc_1m_df.empty
        else np.empty(0, dtype=np.float64)
    )
    btc_fr_map = {
        int(ts_eval[t]): float(settle_fr_bps[t, btc_j]) for t in range(n_b) if np.isfinite(settle_fr_bps[t, btc_j])
    }

    all_trades: list[dict[str, Any]] = []
    for sym, ev_list in sorted(cand_by_sym.items()):
        df1 = symbol_1m_loader(sym)
        if df1 is None or df1.empty:
            continue
        d1 = df1.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        ts_a = d1["timestamp"].to_numpy(dtype=np.int64)
        o_a = d1["open"].to_numpy(dtype=np.float64)
        h_a = d1["high"].to_numpy(dtype=np.float64)
        l_a = d1["low"].to_numpy(dtype=np.float64)

        for t_settle, est_fr, real_fr, adv_v, pr_v, r4_v, bt_v in ev_list:
            for sp in specs:
                if adv_v < sp.min_adv_usd or abs(est_fr) < sp.min_abs_fr_bps:
                    continue
                side = 1 if est_fr < 0 else -1
                if sp.mode == "both_issue64":
                    if (side > 0 and r4_v <= 0.0) or (side < 0 and r4_v >= 0.0):
                        continue
                elif sp.mode == "neg_basis_only":
                    if side < 0 or pr_v > -25.0 or r4_v <= 0.0:
                        continue
                elif sp.mode == "pos_froth_only":
                    if side > 0 or pr_v < 15.0 or r4_v > 0.005:
                        continue
                elif sp.mode == "beta_hedged_pair":
                    ok_neg = side > 0 and pr_v <= -25.0 and r4_v > 0.0
                    ok_pos = side < 0 and pr_v >= 15.0 and r4_v <= 0.005
                    if not (ok_neg or ok_pos):
                        continue

                en_ts = t_settle - int(sp.entry_offset_min) * 60
                ex_ts_target = t_settle + int(sp.exit_offset_min) * 60
                i0 = int(np.searchsorted(ts_a, en_ts, side="left"))
                i1 = int(np.searchsorted(ts_a, ex_ts_target, side="left"))
                if i0 >= len(ts_a) or i1 >= len(ts_a) or ts_a[i0] != en_ts or i1 <= i0:
                    continue

                en_px = float(o_a[i0])
                if en_px <= 0.0 or not np.isfinite(en_px):
                    continue
                sl_px = en_px * (1.0 - side * (sp.sl_bps / 10_000.0))
                ex_px = float(o_a[i1])
                act_ex_ts = int(ts_a[i1])
                hit_sl = False
                for k_m in range(i0, i1):
                    if (side > 0 and float(l_a[k_m]) <= sl_px) or (side < 0 and float(h_a[k_m]) >= sl_px):
                        ex_px = min(float(o_a[k_m]), sl_px) if side > 0 else max(float(o_a[k_m]), sl_px)
                        act_ex_ts = int(ts_a[k_m]) + 60
                        hit_sl = True
                        break

                price_bps = side * ((ex_px - en_px) / en_px) * 10_000.0
                # Coupon is earned/paid ONLY if still open across `t_settle`!
                held_across_settle = act_ex_ts > t_settle
                coupon_bps = (-side * real_fr) if held_across_settle else 0.0
                one_way_fee = 3.10 if adv_v >= 75_000_000.0 else 4.50
                fee_bps = one_way_fee + (5.50 if hit_sl else one_way_fee)

                if sp.mode == "beta_hedged_pair" and len(btc_ts) > 0:
                    b0 = int(np.searchsorted(btc_ts, en_ts, side="left"))
                    b1 = int(np.searchsorted(btc_ts, act_ex_ts, side="left"))
                    if b0 < len(btc_ts) and b1 < len(btc_ts) and btc_o[b0] > 0:
                        btc_ret_bps = (
                            (-side * bt_v)
                            * ((float(btc_c[min(b1, len(btc_c) - 1)]) - float(btc_o[b0])) / float(btc_o[b0]))
                            * 10_000.0
                        )
                        btc_fnd_bps = (side * bt_v * btc_fr_map.get(t_settle, 0.0)) if held_across_settle else 0.0
                        # Normalize by `1 + bt_v` total gross so `net_bps` is strictly per `1x` deployed gross!
                        g_norm = 1.0 + bt_v
                        price_bps = (price_bps + btc_ret_bps) / g_norm
                        coupon_bps = (coupon_bps + btc_fnd_bps) / g_norm
                        fee_bps = (fee_bps + bt_v * 5.0) / g_norm

                gross_tot = price_bps + coupon_bps
                net_bps = gross_tot - fee_bps
                yr = datetime.fromtimestamp(en_ts, tz=timezone.utc).year
                era = "2023_2024" if yr <= 2024 else ("2025" if yr == 2025 else "2026")
                all_trades.append(
                    {
                        "mech_id": sp.mech_id,
                        "pillar": "E1_Coupon_Harvester_Issue64",
                        "exec_mode": sp.mode.upper(),
                        "symbol": sym,
                        "entry_ts": int(en_ts),
                        "exit_ts": int(max(en_ts + 60, act_ex_ts)),
                        "era": era,
                        "strength": round(abs(est_fr), 2),
                        "gross_bps": float(price_bps),
                        "funding_bps": float(coupon_bps),
                        "fee_bps": float(fee_bps),
                        "net_bps": float(net_bps),
                    }
                )
        del df1, d1

    tr_df = pd.DataFrame(all_trades)
    if tr_df.empty:
        return pd.DataFrame(), tr_df
    tr_df.sort_values(["mech_id", "entry_ts"], inplace=True)
    tr_df.reset_index(drop=True, inplace=True)
    lb = evaluate_infold_cpcv_and_treasure_metrics(tr_df, total_trials=max(1, len(specs)))
    return lb, tr_df


def build_and_evaluate_continuous_carry_sleeves(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    prem_1h_by_sym: dict[str, pd.DataFrame],
    eval_idx: pd.DatetimeIndex,
    core_p1_daily: pd.Series,
    manga_p2_daily: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Constructs and evaluates `4` continuous `8h`-rebalanced `Manga P4` Carry Pods (`1x` Gross):
      - `P4_C1_SPOT_PERP_CASH_AND_CARRY_MAJORS`: Honest `0.87x` hedged notional per `$1` capital (`Spot 1x` lockup!).
      - `P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY`: `100%` USD-M Short Frothy Alts (`FR_24h >= +8bps`) + Long Beta-Matched `BTC`.
      - `P4_C3_USDM_SYMMETRIC_DISPERSION_POD`: `P4_C2` + Long Neg-FR Squeeze Alts (`FR_24h <= -8bps`, `Spot>=Perp`) vs Short `BTC`.
      - `P4_C4_WIDE_TOP35_CARRY_BHS`: `Top-35` Liquid Perps Cross-Sectional Carry (`BHS(3,8)`, `disp >= 10bps`).
    Returns `(leaderboard_df, sleeve_streams_dict)`.
    """
    syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    n_b, ns = len(eval_idx), len(syms)
    c_mat = np.full((n_b, ns), np.nan, dtype=np.float64)
    qv_mat = np.zeros((n_b, ns), dtype=np.float64)
    f_settle = np.zeros((n_b, ns), dtype=np.float64)
    f_raw = np.zeros((n_b, ns), dtype=np.float64)
    p_mat = np.zeros((n_b, ns), dtype=np.float64)

    for j_s, sym in enumerate(syms):
        bs = bars_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        c_mat[:, j_s] = (
            pd.Series(bs["close"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).ffill().to_numpy()
        )
        qv_mat[:, j_s] = (
            pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        if sym in prem_1h_by_sym and not prem_1h_by_sym[sym].empty:
            ps = prem_1h_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            p_idx = pd.DatetimeIndex(pd.to_datetime(ps["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
            p_mat[:, j_s] = (
                pd.Series(ps["p_close_bps"].to_numpy(dtype=np.float64), index=p_idx)
                .reindex(eval_idx)
                .ffill()
                .fillna(0.0)
                .to_numpy()
            )
        if sym in funding_by_sym and not funding_by_sym[sym].empty:
            fd = funding_by_sym[sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_ts = ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600
            f_idx = pd.DatetimeIndex(pd.to_datetime(f_ts, unit="s", utc=True))
            f_ser = pd.Series(fd["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            f_settle[:, j_s] = f_ser.reindex(eval_idx).fillna(0.0).to_numpy()
            f_raw[:, j_s] = f_ser.reindex(eval_idx).ffill().fillna(0.0).to_numpy()

    r_perp = np.zeros_like(c_mat)
    r_perp[1:] = np.where(
        (c_mat[:-1] > 0) & np.isfinite(c_mat[:-1]) & np.isfinite(c_mat[1:]), (c_mat[1:] - c_mat[:-1]) / c_mat[:-1], 0.0
    )
    # Exact implied Spot Index price `S_t = P_t / (1 + p_bps * 1e-4)` for Spot+Perp Cash-and-Carry basis MTM!
    spot_mat = c_mat / np.maximum(0.5, 1.0 + p_mat * 1e-4)
    r_spot = np.zeros_like(spot_mat)
    r_spot[1:] = np.where(
        (spot_mat[:-1] > 0) & np.isfinite(spot_mat[:-1]) & np.isfinite(spot_mat[1:]),
        (spot_mat[1:] - spot_mat[:-1]) / spot_mat[:-1],
        0.0,
    )

    adv30 = (
        pd.DataFrame(qv_mat).rolling(24 * 30, min_periods=48).sum().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
        / 30.0
    )
    rk30 = pd.DataFrame(adv30).rank(axis=1, ascending=False, method="first").fillna(999.0).to_numpy(dtype=np.int64) - 1
    fr24_sh = pd.DataFrame(f_raw).rolling(24, min_periods=3).mean().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    fr72_sh = pd.DataFrame(f_raw).rolling(72, min_periods=8).mean().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    p12_sh = pd.DataFrame(p_mat).rolling(12, min_periods=2).mean().shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    r24_sh = pd.DataFrame(c_mat).pct_change(24, fill_method=None).shift(1).fillna(0.0).to_numpy(dtype=np.float64)

    btc_j = syms.index("BTCUSDT")
    r_df = pd.DataFrame(r_perp)
    b_var = r_df.iloc[:, btc_j].rolling(72, min_periods=12).var().replace(0, 1e-8)
    beta_sh = (
        r_df.rolling(72, min_periods=12)
        .cov(r_df.iloc[:, btc_j])
        .div(b_var, axis=0)
        .clip(0.30, 2.40)
        .shift(1)
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
    )
    hrs = eval_idx.hour.to_numpy()

    majors_idx = [syms.index(m) for m in ("BTCUSDT", "ETHUSDT", "SOLUSDT") if m in syms]
    sleeves_out: dict[str, dict[str, Any]] = {}

    # --- 1) P4_C1: Spot + Perp Cash-and-Carry on BTC/ETH/SOL (`0.87x` hedged notional per `$1` capital) ---
    hr_c1 = np.zeros(n_b, dtype=np.float64)
    act_c1 = np.zeros(len(majors_idx), dtype=np.float64)  # hedged notional per major (`sum <= 0.87`)
    fee_c1 = 0.0
    for t in range(n_b):
        f_t = 0.0
        if hrs[t] % 8 == 0:
            tgt = act_c1.copy()
            cands = [
                k
                for k, mj in enumerate(majors_idx)
                if fr24_sh[t, mj] >= 0.00025 or (act_c1[k] > 0 and fr72_sh[t, mj] >= 0.00005)
            ]
            if cands:
                each_w = 0.87 / float(len(cands))
                tgt = np.array([each_w if k in cands else 0.0 for k in range(len(majors_idx))], dtype=np.float64)
            else:
                tgt[:] = 0.0
            dw = np.abs(tgt - act_c1)
            mask_mv = dw >= 0.08
            if np.any(mask_mv):
                # 7.5 bps Spot + 2.55 bps Perp = 10.05 bps per unit of hedged notional changed!
                f_t = float(np.sum(dw[mask_mv])) * 0.001005
                act_c1 = np.where(mask_mv, tgt, act_c1)
                fee_c1 += f_t
        pnl_t = 0.0
        for k, mj in enumerate(majors_idx):
            if act_c1[k] > 1e-8:
                # Long Spot + Short Perp + Collect Perp Funding!
                pnl_t += act_c1[k] * (r_spot[t, mj] - r_perp[t, mj] + f_settle[t, mj])
        hr_c1[t] = pnl_t - f_t
    sleeves_out["P4_C1_SPOT_PERP_CASH_AND_CARRY_MAJORS"] = {"hr_ret": hr_c1, "w_usdm": None, "fee_drag": fee_c1 * 100.0}

    # --- Helper for USD-M Carry Pods (`P4_C2`, `P4_C3`, `P4_C4`) ---
    for pod_id in (
        "P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY",
        "P4_C3_USDM_SYMMETRIC_DISPERSION_POD",
        "P4_C4_WIDE_TOP35_CARRY_BHS",
    ):
        w_hist = np.zeros((n_b, ns), dtype=np.float64)
        curr_w = np.zeros(ns, dtype=np.float64)
        hr_pod = np.zeros(n_b, dtype=np.float64)
        fee_pod = 0.0
        for t in range(n_b):
            f_t = 0.0
            if hrs[t] % 8 == 0:
                tgt = np.zeros(ns, dtype=np.float64)
                btc_fr = fr24_sh[t, btc_j]
                if pod_id in ("P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY", "P4_C3_USDM_SYMMETRIC_DISPERSION_POD"):
                    # Frothy Short Alts (`ADV>=$50M`, `FR_24h >= +8bps`, spread vs BTC `>= +5bps`, `Perp >= Spot`)
                    froth_j = [
                        j
                        for j in range(ns)
                        if j != btc_j
                        and adv30[t, j] >= 50_000_000.0
                        and (
                            (fr24_sh[t, j] >= 0.0008 and (fr24_sh[t, j] - btc_fr) >= 0.0005 and p12_sh[t, j] >= 5.0)
                            or (curr_w[j] < -1e-6 and fr72_sh[t, j] >= 0.0003 and (fr72_sh[t, j] - btc_fr) >= 0.0002)
                        )
                    ]
                    froth_j.sort(key=lambda j: fr24_sh[t, j], reverse=True)
                    froth_j = froth_j[:3]

                    sqz_j: list[int] = []
                    if pod_id == "P4_C3_USDM_SYMMETRIC_DISPERSION_POD":
                        sqz_j = [
                            j
                            for j in range(ns)
                            if j != btc_j
                            and j not in froth_j
                            and adv30[t, j] >= 50_000_000.0
                            and (
                                (fr24_sh[t, j] <= -0.0008 and p12_sh[t, j] <= -5.0 and r24_sh[t, j] > 0.0)
                                or (curr_w[j] > 1e-6 and fr72_sh[t, j] <= -0.0003 and r24_sh[t, j] > -0.02)
                            )
                        ]
                        sqz_j.sort(key=lambda j: fr24_sh[t, j])
                        sqz_j = sqz_j[:2]

                    n_active = len(froth_j) + len(sqz_j)
                    if n_active > 0:
                        raw_alt = np.zeros(ns, dtype=np.float64)
                        for j in froth_j:
                            raw_alt[j] = -1.0 / float(n_active)
                        for j in sqz_j:
                            raw_alt[j] = +1.0 / float(n_active)
                        net_b = float(np.sum(raw_alt * beta_sh[t]))
                        raw_alt[btc_j] = -net_b
                        g_tot = float(np.sum(np.abs(raw_alt)))
                        if g_tot > 1e-8:
                            tgt = raw_alt / g_tot
                else:
                    # P4_C4: Top-35 Liquid Perps Cross-Sectional Carry with BHS(3, 8)
                    univ_j = [j for j in range(ns) if rk30[t, j] < 35]
                    if len(univ_j) >= 10:
                        frs_u = np.array([fr24_sh[t, j] for j in univ_j], dtype=np.float64)
                        disp = float(np.max(frs_u) - np.min(frs_u))
                        if disp >= 0.0010 or (np.any(np.abs(curr_w) > 1e-6) and disp >= 0.0004):
                            # Lowest FR -> Long, Highest FR -> Short
                            ord_asc = [univ_j[k] for k in np.argsort(frs_u)]
                            ord_desc = list(reversed(ord_asc))
                            longs = [j for j in ord_asc[:8] if curr_w[j] > 1e-6][:3]
                            for j in ord_asc[:3]:
                                if len(longs) < 3 and j not in longs:
                                    longs.append(j)
                            shorts = [j for j in ord_desc[:8] if curr_w[j] < -1e-6][:3]
                            for j in ord_desc[:3]:
                                if len(shorts) < 3 and j not in shorts and j not in longs:
                                    shorts.append(j)
                            if longs and shorts:
                                for j in longs:
                                    tgt[j] = 0.50 / float(len(longs))
                                for j in shorts:
                                    tgt[j] = -0.50 / float(len(shorts))

                dw = np.abs(tgt - curr_w)
                mask_mv = dw >= 0.04
                if np.any(mask_mv):
                    f_t = float(np.sum(dw[mask_mv])) * 0.00031
                    curr_w = np.where(mask_mv, tgt, curr_w)
                    fee_pod += f_t
            w_hist[t] = curr_w
            hr_pod[t] = float(np.sum(curr_w * r_perp[t])) - float(np.sum(curr_w * f_settle[t])) - f_t
        sleeves_out[pod_id] = {"hr_ret": hr_pod, "w_usdm": w_hist, "syms": syms, "fee_drag": fee_pod * 100.0}

    rows_lb: list[dict[str, Any]] = []
    for s_id, s_data in sleeves_out.items():
        hr_s = pd.Series(s_data["hr_ret"], index=eval_idx)
        d_s = hr_s.resample("1D").apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
        sim_mock = {
            "daily": d_s,
            "final_eq": 2700.0 * float(np.prod(1.0 + d_s.to_numpy())),
            "min_eq": 2700.0 * float(np.min(np.cumprod(1.0 + d_s.to_numpy()))),
            "worst_24h_dd": 0.0,
            "sniper_counts": {},
        }
        m_e = compute_portfolio_era_metrics(sim_mock, initial_capital=2700.0)
        cp = compute_cpcv_portfolio_sharpe_stability(d_s)
        active_pct = round(100.0 * float((hr_s.abs() > 1e-10).mean()), 1)
        al_p1 = pd.concat([d_s.rename("c"), core_p1_daily.rename("p1")], axis=1).dropna()
        al_p2 = pd.concat([d_s.rename("c"), manga_p2_daily.rename("p2")], axis=1).dropna()
        c_p1 = float(al_p1["c"].corr(al_p1["p1"])) if len(al_p1) > 20 and float(d_s.std()) > 1e-10 else 0.0
        c_p2 = float(al_p2["c"].corr(al_p2["p2"])) if len(al_p2) > 20 and float(d_s.std()) > 1e-10 else 0.0
        rows_lb.append(
            {
                "carry_id": s_id,
                "active_time_pct": active_pct,
                "sharpe_4y": m_e["full_sharpe"],
                "cagr_4y_pct": m_e["full_cagr_pct"],
                "max_dd_4y_pct": m_e["full_max_dd_pct"],
                "sharpe_23_24": m_e["sharpe_23_24"],
                "sharpe_2025": m_e["sharpe_2025"],
                "sharpe_2026": m_e["sharpe_2026"],
                "cpcv_mean_sharpe": cp["cpcv_mean_sharpe"],
                "cpcv_pos_splits_pct": cp["cpcv_pos_splits_pct"],
                "corr_vs_core_p1": round(c_p1 if np.isfinite(c_p1) else 0.0, 3),
                "corr_vs_manga_p2": round(c_p2 if np.isfinite(c_p2) else 0.0, 3),
                "fee_drag_pct": round(float(s_data["fee_drag"]), 1),
            }
        )
    _ = PROD_TOP_15
    return pd.DataFrame(rows_lb), sleeves_out


def evaluate_portfolio_carry_integrations(
    bundle: dict[str, Any],
    p2_inv_vol_mat: np.ndarray,
    sleeves_out: dict[str, dict[str, Any]],
    m26_trades_df: pd.DataFrame | None = None,
    coupon_trades_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Evaluates capital-allocation vs idle-margin integration of `Manga P4` Carry onto the
    validated `M17` stack (`D11`: `T-5m`, `2.55 bps` WS router, `Inv-Vol P2`, `Kelly Snipers`, `M04 <= -8bps`)
    across both `2023–2026` AND fresh `$2,700` `2026-01-01` start against the `$2,500` CPPI floor.
    """
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    mats: dict[str, Any] = bundle["mats"]
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]

    base_evs = list(bundle.get("base_sniper_events", []))
    events_by_hr: dict[int, list[dict[str, Any]]] = {}
    for ev_b in base_evs:
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

    d11_spec = next(s for s in FRENTE_D_SPECS if s.exp_id == "D11_M17_STACK_A_C_PLUS_KELLY_M04")
    sim_4y_base = simulate_frente_d_1m_portfolio(
        mats=mats,
        o_1m_core=o_1m_core,
        h_1m_core=h_1m_core,
        l_1m_core=l_1m_core,
        eval_idx=eval_idx,
        open_core=open_core,
        p2_w_mat=p2_inv_vol_mat,
        spec=d11_spec,
        resolved_sniper_events_by_hour=events_by_hr,
        allowed_sniper_strats=allowed_strats,
    )
    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    idx_2026 = eval_idx[mask_2026]
    ts_2026_start = int(idx_2026[0].timestamp())
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}
    sim_26_base = simulate_frente_d_1m_portfolio(
        mats=mats,
        o_1m_core=o_1m_core[mask_2026],
        h_1m_core=h_1m_core[mask_2026],
        l_1m_core=l_1m_core[mask_2026],
        eval_idx=idx_2026,
        open_core=open_core[mask_2026],
        p2_w_mat=p2_inv_vol_mat[mask_2026],
        spec=d11_spec,
        resolved_sniper_events_by_hour=evs_2026,
        allowed_sniper_strats=allowed_strats,
    )

    hr_base_4y = (
        sim_4y_base["eq_1h"]
        .pct_change()
        .fillna((float(sim_4y_base["eq_1h"].iloc[0]) / 2700.0) - 1.0)
        .to_numpy(dtype=np.float64)
    )
    hr_base_26 = (
        sim_26_base["eq_1h"]
        .pct_change()
        .fillna((float(sim_26_base["eq_1h"].iloc[0]) / 2700.0) - 1.0)
        .to_numpy(dtype=np.float64)
    )

    c1_hr = sleeves_out["P4_C1_SPOT_PERP_CASH_AND_CARRY_MAJORS"]["hr_ret"]
    c2_hr = sleeves_out["P4_C2_USDM_FROTH_ALT_VS_BTC_CARRY"]["hr_ret"]
    c3_hr = sleeves_out["P4_C3_USDM_SYMMETRIC_DISPERSION_POD"]["hr_ret"]
    c4_hr = sleeves_out["P4_C4_WIDE_TOP35_CARRY_BHS"]["hr_ret"]

    # Causal low-conviction / idle-margin regime gate (`conviction * vol_scalar < 0.70`)
    conv_sh = (mats["conviction"] * mats["vol_scalar"]).shift(1).fillna(1.0).loc[eval_idx].to_numpy(dtype=np.float64)
    idle_gate = np.clip((0.85 - conv_sh) / 0.35, 0.0, 1.0)

    def _eval_stream(port_id: str, desc: str, r4y: np.ndarray, r26: np.ndarray) -> dict[str, Any]:
        # Enforce CPPI cushion scaling on additive overlay when near `$2,500` floor on 2026 `$2,700` run!
        eq_4y = 2700.0 * np.cumprod(1.0 + np.clip(r4y, -0.90, None))
        d_4y = pd.Series(r4y, index=eval_idx).resample("1D").apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
        m4 = compute_portfolio_era_metrics(
            {
                "daily": d_4y,
                "final_eq": float(eq_4y[-1]),
                "min_eq": float(np.min(eq_4y)),
                "worst_24h_dd": 0.0,
                "sniper_counts": {},
            },
            initial_capital=2700.0,
        )
        cp = compute_cpcv_portfolio_sharpe_stability(d_4y)

        eq_26 = np.empty(len(r26), dtype=np.float64)
        cur_e = 2700.0
        for i_t in range(len(r26)):
            c_sc = float(np.clip((cur_e - 2510.0) / 240.0, 0.15, 1.0)) if cur_e < 2750.0 else 1.0
            base_r = hr_base_26[i_t]
            extra_r = r26[i_t] - base_r
            # If capital split (`E1_P1`), `r26` is already convex combination; if overlay, scale extra by CPPI cushion!
            eff_r = r26[i_t] if "SPLIT" in port_id else (base_r + c_sc * extra_r)
            cur_e = max(1.0, cur_e * (1.0 + eff_r))
            eq_26[i_t] = cur_e

        return {
            "port_id": port_id,
            "description": desc,
            "sharpe_4y": m4["full_sharpe"],
            "cagr_4y_pct": m4["full_cagr_pct"],
            "max_dd_4y_pct": m4["full_max_dd_pct"],
            "sharpe_23_24": m4["sharpe_23_24"],
            "sharpe_2025": m4["sharpe_2025"],
            "sharpe_2026": m4["sharpe_2026"],
            "cpcv_mean_sharpe": cp["cpcv_mean_sharpe"],
            "cpcv_pos_splits_pct": cp["cpcv_pos_splits_pct"],
            "fresh_2026_final_eq": round(float(eq_26[-1]), 2),
            "fresh_2026_min_eq": round(float(np.min(eq_26)), 2),
            "fresh_2026_floor_2500_ok": bool(float(np.min(eq_26)) >= 2500.0),
        }

    port_rows = [
        _eval_stream(
            "E1_P0_M17_STACK_D11_NO_CARRY",
            "Reference M17 Stack (`D11`: `T-5m + WS Router + InvVol P2 + Kelly + M04<=-8bps`)",
            hr_base_4y,
            hr_base_26,
        ),
        _eval_stream(
            "E1_P1_CAPITAL_SPLIT_20PCT_SPOT_PERP",
            "Carve `20%` Equity Out of USD-M into `1x` Spot+Perp Cash-and-Carry (`80% D11 + 20% P4_C1`)",
            0.80 * hr_base_4y + 0.20 * c1_hr,
            0.80 * hr_base_26 + 0.20 * c1_hr[mask_2026],
        ),
        _eval_stream(
            "E1_P2_USDM_FROTH_CARRY_OVERLAY_035X",
            "Intra-USD-M Froth Carry Overlay (`+0.35x P4_C2` on Idle USD-M Margin, `0` Spot Lockup)",
            hr_base_4y + 0.35 * c2_hr,
            hr_base_26 + 0.35 * c2_hr[mask_2026],
        ),
        _eval_stream(
            "E1_P3_USDM_SYMMETRIC_POD_OVERLAY_035X",
            "Intra-USD-M Symmetric Carry Pod (`+0.35x P4_C3` on Idle USD-M Margin)",
            hr_base_4y + 0.35 * c3_hr,
            hr_base_26 + 0.35 * c3_hr[mask_2026],
        ),
        _eval_stream(
            "E1_P4_LOW_CONVICTION_CARRY_PARKING_050X",
            "Regime-Gated Carry Parking (`+0.50x P4_C2` ONLY when `Core P1` Conviction `< 0.85`)",
            hr_base_4y + 0.50 * idle_gate * c2_hr,
            hr_base_26 + 0.50 * idle_gate[mask_2026] * c2_hr[mask_2026],
        ),
        _eval_stream(
            "E1_P5_WIDE_TOP35_CARRY_OVERLAY_035X",
            "Wide `Top-35` Cross-Sectional Carry Overlay (`+0.35x P4_C4` `BHS(3,8)`)",
            hr_base_4y + 0.35 * c4_hr,
            hr_base_26 + 0.35 * c4_hr[mask_2026],
        ),
    ]
    if coupon_trades_df is not None and not coupon_trades_df.empty:
        best_h_id = str(coupon_trades_df.groupby("mech_id")["net_bps"].mean().idxmax())
        sub_h = coupon_trades_df[coupon_trades_df["mech_id"] == best_h_id]
        h_evs = convert_trade_df_to_hourly_sniper_events(sub_h, "Harvester:Issue64")
        ts_arr = eval_idx.as_unit("s").astype(np.int64).to_numpy()
        hr_to_idx = {int(ts_arr[k]): k for k in range(len(ts_arr))}
        coup_hr = np.zeros(len(eval_idx), dtype=np.float64)
        active_end: list[int] = []
        for ev in sorted(h_evs, key=lambda x: int(x["entry_ts"])):
            en_t = int(ev["entry_ts"])
            active_end = [x for x in active_end if x > en_t]
            if len(active_end) >= 2:
                continue
            active_end.append(int(ev["exit_ts"]))
            for h_k, r_v in ev["hourly_ret"].items():
                pos_k = hr_to_idx.get(int(h_k))
                if pos_k is not None:
                    coup_hr[pos_k] += 0.20 * float(r_v)
        port_rows.append(
            _eval_stream(
                f"E1_P6_D11_PLUS_{best_h_id[:16]}",
                f"D11 + Best Issue #64 Coupon Harvester (`{best_h_id}`, `0.20x` max 2 slots)",
                hr_base_4y + coup_hr,
                hr_base_26 + coup_hr[mask_2026],
            )
        )
    return pd.DataFrame(port_rows)
