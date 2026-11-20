import numpy as np
import pandas as pd

from src.models.sniper_l2_and_cascade_short_lab import (
    WATERFALL_SHORT_SPECS,
    lookup_causal_l2_book_imbalance,
    scan_creative_1h_snipers_p9_p10,
    simulate_waterfall_short_on_window,
)


def test_waterfall_short_1ms_and_causal_l2_lookup():
    sig_ts = 1_760_000_000
    trig_ms = (sig_ts - 120) * 1000
    rows = [
        # 1-ms forced sell burst: $60k & 12 makers at $100.00
        {"ts_ms": trig_ms, "price": 100.0, "usd": 60_000.0, "n_makers": 12, "is_buyer_maker": True},
        # +100ms (before +250ms arm latency) — must be ignored!
        {"ts_ms": trig_ms + 100, "price": 99.40, "usd": 5_000.0, "n_makers": 2, "is_buyer_maker": True},
        # +400ms (after +250ms latency) breaks -35 bps (`<= 99.65`) at 99.60 -> triggers Taker SHORT!
        {"ts_ms": trig_ms + 400, "price": 99.60, "usd": 20_000.0, "n_makers": 4, "is_buyer_maker": True},
    ]
    # Secondary waterfall drops down to 97.50 (-210 bps) -> hits TP_MAKER!
    for s in range(1, 30):
        px = max(97.50, 99.60 - 0.12 * s)
        rows.append(
            {"ts_ms": trig_ms + 400 + s * 1000, "price": px, "usd": 8_000.0, "n_makers": 2, "is_buyer_maker": True}
        )

    agg_df = pd.DataFrame(rows)
    res = simulate_waterfall_short_on_window(agg_df, signal_ts_sec=sig_ts, spec=WATERFALL_SHORT_SPECS[2])
    assert res is not None
    assert res["exit_reason"] == "TP_MAKER"
    assert res["net_bps"] > 140.0

    # Strict post-anchor mode MUST reject bursts occurring before `sig_ts` (`0.000 ms` window conditioning!)
    assert (
        simulate_waterfall_short_on_window(
            agg_df, signal_ts_sec=sig_ts, spec=WATERFALL_SHORT_SPECS[2], post_anchor_only=True
        )
        is None
    )
    # And MUST accept bursts occurring strictly at or after `sig_ts`!
    res_post = simulate_waterfall_short_on_window(
        agg_df, signal_ts_sec=sig_ts - 180, spec=WATERFALL_SHORT_SPECS[2], post_anchor_only=True
    )
    assert res_post is not None and res_post["exit_reason"] == "TP_MAKER"

    # Strict causal L2 lookup (`t_book <= entry_ts`, zero lookahead)
    bd = pd.DataFrame(
        {
            "timestamp": [1000, 1060],
            "bid_1_usd": [300_000.0, 50_000.0],
            "ask_1_usd": [100_000.0, 400_000.0],
            "bid_2_usd": [500_000.0, 100_000.0],
            "ask_2_usd": [200_000.0, 600_000.0],
        }
    )
    imb1, _ = lookup_causal_l2_book_imbalance(bd, entry_ts=1059)
    assert imb1 is not None and abs(imb1 - 0.50) < 1e-6  # Uses 1000, NOT future 1060!

    # P9 / P10 creative 1h scanner test
    n_b = 70
    ts_eval = np.arange(1_760_000_000, 1_760_000_000 + n_b * 3600, 3600, dtype=np.int64)
    syms = ["BTCUSDT", "SOLUSDT", "FETUSDT"]
    c = np.full((n_b, 3), 100.0)
    h = np.full((n_b, 3), 100.5)
    qv = np.full((n_b, 3), 1e7)
    tb = np.full((n_b, 3), 5e6)
    fr = np.zeros((n_b, 3))
    rk = np.zeros((n_b, 3), dtype=np.int64)

    # Bar 50: BTC crashes -2% on 3x volume, while FET rallies +1.2% on 65% taker buy -> P10!
    c[50, 0] = 98.0
    qv[50, 0] = 3e7
    c[50, 2] = 101.2
    tb[50, 2] = 0.65 * qv[50, 2]
    evs = scan_creative_1h_snipers_p9_p10(ts_eval, syms, c, h, qv, tb, fr, rk)
    assert any(kind == "P10_BTC_FLUSH_IMMUNE_LONG" and s_i == 2 for _, s_i, _, kind in evs)

    from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics
    from src.models.sniper_l2_and_cascade_short_lab import build_causal_1h_panic_flush_anchors

    tr_df = pd.DataFrame([res])
    tr_df["symbol"] = "SOLUSDT"
    lb = evaluate_infold_cpcv_and_treasure_metrics(tr_df, total_trials=4)
    assert len(lb) == 1

    # Test build_causal_1h_panic_flush_anchors + evaluate_frente_b_sniper_portfolio_ablation end-to-end
    idx_h = pd.date_range("2025-11-01", periods=120, freq="1h", tz="UTC")
    ts_h = idx_h.as_unit("s").astype(np.int64).to_numpy()
    b_df = pd.DataFrame(
        {
            "timestamp": ts_h,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "quote_volume": 1e7,
            "taker_buy_quote_volume": 5e6,
        }
    )
    s_df = b_df.copy()
    s_df.loc[60, "close"] = 95.0
    s_df.loc[60, "quote_volume"] = 5e7
    s_df.loc[60, "taker_buy_quote_volume"] = 1e7
    anch = build_causal_1h_panic_flush_anchors({"BTCUSDT": b_df, "SOLUSDT": s_df}, eval_start="2025-11-01")
    assert len(anch) == 1 and int(anch["entry_ts"].iloc[0]) == int(ts_h[60]) + 3600
