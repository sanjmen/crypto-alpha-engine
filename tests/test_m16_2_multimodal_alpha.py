"""
Unit & Strict Causality Tests for Issue #68 (M16-2):
Multi-Modal Microstructure Feature Library (`src/features/multimodal_alpha.py`).

Verifies:
  1. ~100 features generated across all 4 orthogonal families (A, B, C, D).
  2. Strict Zero-Lookahead Invariance under extreme future-bar mutation across all 4 streams
     (`perp_bars`, `spot_bars`, `metrics_5m` + `60s` safety lag, `funding` + causal clock projection)
     and cross-sectional panel transforms.
  3. Structural physics correctness (4-Quadrant OI/Price liquidation flush, Spot-Perp CVD divergence,
     compressed `2h` negative funding settlement clock).
  4. Walk-Forward (`2023–2024` Train / `2025` Val / `2026` Holdout) IC & Conditional Tail Atlas polarity locking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.multimodal_alpha import (
    MultiModalFeatureConfig,
    attach_forward_returns_for_audit,
    compute_cross_sectional_panel_features,
    compute_symbol_multimodal_features,
    evaluate_walk_forward_feature_atlas,
    list_multimodal_feature_columns,
)


def _make_synthetic_multimodal_symbol(
    symbol: str,
    start_ts: int,
    n_15m_bars: int = 260,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts_15m = np.arange(start_ts, start_ts + n_15m_bars * 900, 900, dtype=np.int64)
    rets = rng.normal(0.0001, 0.004, size=n_15m_bars)
    close_p = 100.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close_p, 1)
    open_p[0] = close_p[0]
    high_p = np.maximum(open_p, close_p) * (1.0 + rng.uniform(0.0005, 0.003, size=n_15m_bars))
    low_p = np.minimum(open_p, close_p) * (1.0 - rng.uniform(0.0005, 0.003, size=n_15m_bars))
    qvol_p = rng.uniform(300_000.0, 900_000.0, size=n_15m_bars)
    tb_share_p = rng.uniform(0.42, 0.58, size=n_15m_bars)

    perp_df = pd.DataFrame(
        {
            "timestamp": ts_15m,
            "symbol": symbol,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": qvol_p / close_p,
            "quote_volume": qvol_p,
            "trade_count": rng.integers(200, 1500, size=n_15m_bars),
            "taker_buy_vol": (qvol_p * tb_share_p) / close_p,
            "taker_buy_quote_volume": qvol_p * tb_share_p,
        }
    )

    basis_noise = rng.normal(0.0002, 0.0006, size=n_15m_bars)
    close_s = close_p / (1.0 + basis_noise)
    qvol_s = qvol_p * rng.uniform(0.3, 0.9, size=n_15m_bars)
    tb_share_s = rng.uniform(0.40, 0.62, size=n_15m_bars)
    spot_df = pd.DataFrame(
        {
            "timestamp": ts_15m,
            "symbol": symbol,
            "open": close_s * 0.9995,
            "high": close_s * 1.002,
            "low": close_s * 0.998,
            "close": close_s,
            "volume": qvol_s / close_s,
            "quote_volume": qvol_s,
            "trade_count": rng.integers(100, 900, size=n_15m_bars),
            "taker_buy_vol": (qvol_s * tb_share_s) / close_s,
            "taker_buy_quote_volume": qvol_s * tb_share_s,
        }
    )

    ts_5m = np.arange(start_ts, start_ts + n_15m_bars * 900, 300, dtype=np.int64)
    oi_series = 500_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, size=len(ts_5m))))
    metrics_df = pd.DataFrame(
        {
            "timestamp": ts_5m,
            "symbol": symbol,
            "sum_open_interest": oi_series,
            "sum_open_interest_value": oi_series * np.interp(ts_5m, ts_15m, close_p),
            "count_toptrader_long_short_ratio": rng.uniform(0.9, 1.6, size=len(ts_5m)),
            "sum_toptrader_long_short_ratio": rng.uniform(0.85, 1.9, size=len(ts_5m)),
            "count_long_short_ratio": rng.uniform(0.8, 2.4, size=len(ts_5m)),
            "sum_taker_long_short_vol_ratio": rng.uniform(0.7, 1.4, size=len(ts_5m)),
        }
    )

    ts_fnd = np.arange(start_ts, start_ts + n_15m_bars * 900, 4 * 3600, dtype=np.int64)
    funding_df = pd.DataFrame(
        {
            "timestamp": ts_fnd,
            "symbol": symbol,
            "interval_hours": np.where(np.arange(len(ts_fnd)) % 5 == 0, 2, 4),
            "funding_rate": rng.normal(-0.0005, 0.0025, size=len(ts_fnd)),
        }
    )
    return perp_df, spot_df, metrics_df, funding_df


def test_zero_lookahead_invariance_under_future_mutation_across_all_modalities() -> None:
    start_ts = 1_735_689_600  # 2025-01-01 00:00:00 UTC
    cfg = MultiModalFeatureConfig(decision_freq="15min", metrics_safety_lag_sec=60)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ALGOUSDT"]

    orig_frames: dict[str, pd.DataFrame] = {}
    raw_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for idx, sym in enumerate(symbols):
        p_df, s_df, m_df, f_df = _make_synthetic_multimodal_symbol(sym, start_ts, n_15m_bars=240, seed=10 + idx)
        raw_inputs[sym] = (p_df, s_df, m_df, f_df)
        orig_frames[sym] = compute_symbol_multimodal_features(
            perp_bars=p_df,
            spot_bars=s_df,
            metrics_5m=m_df,
            funding=f_df,
            symbol=sym,
            onboard_timestamp=start_ts - 10 * 86400,
            config=cfg,
        )

    orig_panel = compute_cross_sectional_panel_features(orig_frames)
    feat_cols = list_multimodal_feature_columns(orig_panel)
    assert len(feat_cols) >= 95, f"Expected >= 95 features, got {len(feat_cols)}"

    # Pick exact interior cutoff `decision_ts` at bar index 150 (`open_ts = start_ts + 150*900`, `decision_ts = open_ts + 900`)
    cutoff_decision_ts = int(orig_frames["BTCUSDT"]["decision_ts"].iloc[150])

    # Violently mutate ALL future bars (`open_ts >= cutoff_decision_ts`, `metrics > cutoff_decision_ts - 60s`, `funding > cutoff_decision_ts`)
    mut_frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        p_df, s_df, m_df, f_df = (df.copy() for df in raw_inputs[sym])

        fut_p = p_df["timestamp"] >= cutoff_decision_ts
        p_df.loc[fut_p, ["open", "high", "low", "close"]] *= 12.5
        p_df.loc[fut_p, "quote_volume"] *= 200.0
        p_df.loc[fut_p, "taker_buy_quote_volume"] = 0.0

        fut_s = s_df["timestamp"] >= cutoff_decision_ts
        s_df.loc[fut_s, ["open", "high", "low", "close"]] *= 0.05
        s_df.loc[fut_s, "quote_volume"] *= 500.0

        # Mutate metrics strictly inside the 60s safety lag window (`timestamp > cutoff_decision_ts - 60`) AND all future!
        fut_m = m_df["timestamp"] > (cutoff_decision_ts - 60)
        m_df.loc[fut_m, "sum_open_interest"] *= 80.0
        m_df.loc[fut_m, "sum_open_interest_value"] *= 800.0
        m_df.loc[fut_m, "sum_toptrader_long_short_ratio"] = 99.0
        m_df.loc[fut_m, "count_long_short_ratio"] = 0.01
        # Also inject an unlagged snapshot at `cutoff_decision_ts - 30` (must be blocked by 60s safety lag!)
        extra_m = m_df.iloc[[-1]].copy()
        extra_m["timestamp"] = cutoff_decision_ts - 30
        extra_m["sum_open_interest"] = 999_999_999.0
        m_df = pd.concat([m_df, extra_m], ignore_index=True)

        # Mutate all future funding settlements + interval changes (`timestamp > cutoff_decision_ts`)
        fut_f = f_df["timestamp"] > cutoff_decision_ts
        f_df.loc[fut_f, "funding_rate"] = -0.025
        f_df.loc[fut_f, "interval_hours"] = 1

        mut_frames[sym] = compute_symbol_multimodal_features(
            perp_bars=p_df,
            spot_bars=s_df,
            metrics_5m=m_df,
            funding=f_df,
            symbol=sym,
            onboard_timestamp=start_ts - 10 * 86400,
            config=cfg,
        )

    mut_panel = compute_cross_sectional_panel_features(mut_frames)

    past_orig = orig_panel[orig_panel["decision_ts"] <= cutoff_decision_ts].reset_index(drop=True)
    past_mut = mut_panel[mut_panel["decision_ts"] <= cutoff_decision_ts].reset_index(drop=True)
    assert len(past_orig) == len(past_mut)

    for col in feat_cols:
        np.testing.assert_allclose(
            past_orig[col].to_numpy(dtype=np.float64),
            past_mut[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Lookahead leakage detected in feature '{col}'!",
        )


def test_quadrant_liquidation_spot_cvd_and_compressed_funding_clock_physics() -> None:
    start_ts = 1_750_000_000
    p_df, s_df, m_df, f_df = _make_synthetic_multimodal_symbol("SOLUSDT", start_ts, n_15m_bars=160, seed=7)

    # Engineer a sharp Long Liquidation Flush + Spot Absorption + Compressed 2h Negative Funding Squeeze on final 4 bars (1h)
    p_df.loc[p_df.index[-4:], "close"] = p_df["close"].iloc[-5] * np.array([0.97, 0.95, 0.93, 0.91])
    p_df.loc[p_df.index[-4:], "low"] = p_df.loc[p_df.index[-4:], "close"] * 0.995
    p_df.loc[p_df.index[-4:], "taker_buy_quote_volume"] = (
        p_df.loc[p_df.index[-4:], "quote_volume"] * 0.15
    )  # Perp panic selling

    # Spot buyers aggressively absorb the dip (`85%` Spot taker buy share!)
    s_df.loc[s_df.index[-4:], "close"] = p_df.loc[p_df.index[-4:], "close"] * 1.0025  # Perp trades at -25 bps discount!
    s_df.loc[s_df.index[-4:], "taker_buy_quote_volume"] = s_df.loc[s_df.index[-4:], "quote_volume"] * 0.88

    # Open Interest collapses `-25%` (Long liquidations flushing out!)
    last_bar_close = int(p_df["timestamp"].iloc[-1]) + 900
    m_recent = m_df["timestamp"] >= (last_bar_close - 3600)
    m_df.loc[m_recent, "sum_open_interest"] = m_df["sum_open_interest"].iloc[0] * 0.72

    # Funding settled 90 minutes ago at `-55 bps` on a compressed `2h` (`120m`) clock -> 30 mins until next settlement!
    f_spike = pd.DataFrame(
        {
            "timestamp": [last_bar_close - 90 * 60],
            "symbol": ["SOLUSDT"],
            "interval_hours": [2],
            "funding_rate": [-0.0055],
        }
    )
    f_df = pd.concat([f_df[f_df["timestamp"] < last_bar_close - 4 * 3600], f_spike], ignore_index=True)

    feat = compute_symbol_multimodal_features(
        perp_bars=p_df,
        spot_bars=s_df,
        metrics_5m=m_df,
        funding=f_df,
        symbol="SOLUSDT",
        config=MultiModalFeatureConfig(decision_freq="15min", metrics_safety_lag_sec=60),
    )
    last = feat.iloc[-1]

    assert last["quad_long_liq_flush_1h"] > 1.0
    assert last["quad_new_shorts_1h"] == 0.0
    assert last["spot_vs_perp_cvd_div_1h"] > 1.0
    assert last["spot_buy_perp_liq_combo"] > 1.0
    assert last["basis_bps"] < -15.0
    assert last["is_compressed_funding_clock"] == 1.0
    assert last["clock_compression_ratio"] == pytest.approx(4.0)
    assert last["mins_to_next_settlement"] == pytest.approx(30.0)
    assert last["in_pre_settlement_window"] == 1.0
    assert last["pre_settle_neg_squeeze_40bps"] > 1.0


def test_walk_forward_atlas_polarity_locking_across_2024_2025_2026() -> None:
    rng = np.random.default_rng(123)
    # Build synthetic cross-sectional panel spanning 2024 (Train), 2025 (Val), and 2026 (Holdout)
    eras = [
        ("2024-06-01", 80),
        ("2025-06-01", 80),
        ("2026-06-01", 80),
    ]
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
    rows = []
    for start_str, n_steps in eras:
        t0 = int(pd.Timestamp(start_str, tz="UTC").timestamp())
        for step in range(n_steps):
            dts = t0 + step * 14400
            alpha_sig = rng.normal(0.0, 1.5, size=len(syms))
            noise_sig = rng.normal(0.0, 1.5, size=len(syms))
            # True predictive signal: negative relationship (`-0.0035 * alpha_sig`) consistent across all 3 eras!
            fwd_r = -0.0035 * alpha_sig + rng.normal(0.0, 0.002, size=len(syms))
            for i, s in enumerate(syms):
                rows.append(
                    {
                        "timestamp": dts - 14400,
                        "decision_ts": dts,
                        "symbol": s,
                        "close": 100.0,
                        "quote_volume": 50_000_000.0,
                        "tradable_adv_15m_gate": 1.0,
                        "basis_z_24h": alpha_sig[i],
                        "oi_vel_z_4h": noise_sig[i],
                        "fwd_ret_4h": fwd_r[i],
                    }
                )
    panel = pd.DataFrame(rows)
    atlas = evaluate_walk_forward_feature_atlas(panel, fwd_col="fwd_ret_4h", tail_z_threshold=1.5)
    assert not atlas.empty
    top = atlas.iloc[0]
    assert top["feature"] == "basis_z_24h"
    assert top["train_polarity"] == -1
    assert top["val_2025_ic_tstat"] > 3.0
    assert top["holdout_2026_ic_tstat"] > 3.0
    assert top["holdout_2026_tail_net_maker_bps"] > 20.0
    assert bool(top["sign_consistent_3_eras"]) is True

    # Also verify attach_forward_returns_for_audit shifts strictly forward per symbol
    chk = attach_forward_returns_for_audit(
        pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
                "decision_ts": [1000, 2000, 3000],
                "close": [100.0, 105.0, 110.25],
            }
        ),
        {"1b": 1},
    )
    assert pytest.approx(chk["fwd_ret_1b"].iloc[0], rel=1e-9) == np.log(1.05)
    assert np.isnan(chk["fwd_ret_1b"].iloc[-1])
