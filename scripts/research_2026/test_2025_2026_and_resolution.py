"""
Comprehensive 2025-2026 (22 Months: 2025-01-01 to 2026-10-20) Quantitative Benchmark
& 1m / 5m / 15m / 30m / 1h Signal-to-Noise Resolution Study.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from explore_high_return_strategies import TOP15, load_matrices
from test_neutral_alphas import make_continuous_neutral_weights
from test_subhourly_frequencies import load_timeframe_matrices


def analyze_1m_5m_15m_1h_noise():
    print("=" * 125)
    print("PART 1: EMPIRICAL SIGNAL-TO-NOISE & MICROSTRUCTURE RESOLUTION STUDY (1m vs 5m vs 15m vs 30m vs 1h)")
    print("=" * 125)

    # Load 1m (7.5d), 5m (31d), 15m (6m), 1h (22m)
    _, o1, h1, l1, c1, v1, tb1, _, _ = load_timeframe_matrices("1m", "2026-10-13 06:00:00", "2026-10-20 18:00:00")
    _, o5, h5, l5, c5, v5, tb5, _, f5 = load_timeframe_matrices("5m", "2026-09-19 00:00:00", "2026-10-20 18:00:00")
    _, o15, h15, l15, c15, v15, tb15, _, f15 = load_timeframe_matrices("15m", "2026-04-20 00:00:00", "2026-10-20 18:00:00")

    # Resample 1m to 5m, 15m, 30m, 1h on the exact same 7.5-day window to compute exact Variance Ratio & Wick Noise
    r_1m = np.log(c1 / c1.shift(1)).dropna()
    var_1m = r_1m.var()

    print(f"{'Timeframe':<12} | {'Lag-1 AutoCorr':>15} | {'Variance Ratio VR(q)':>21} | {'Wick/Noise Ratio':>17} | {'Effective SNR':>14} | {'Microstructure Regime'}")
    print("-" * 125)

    for tf_lbl, rule, q_min in [
        ("1m  Bars", "1min", 1),
        ("5m  Bars", "5min", 5),
        ("15m Bars", "15min", 15),
        ("30m Bars", "30min", 30),
        ("1h  Bars", "1h", 60),
        ("4h  Bars", "4h", 240),
    ]:
        c_rs = c1.resample(rule).last().dropna()
        o_rs = o1.resample(rule).first().dropna()
        h_rs = h1.resample(rule).max().dropna()
        l_rs = l1.resample(rule).min().dropna()
        r_rs = np.log(c_rs / c_rs.shift(1)).dropna()

        # Average lag-1 autocorrelation across the 15 symbols
        ac1 = np.mean([r_rs[s].autocorr(lag=1) for s in TOP15])
        # Variance Ratio VR(q) = Var(r_q) / (q * Var(r_1m))
        vr = np.mean([(r_rs[s].var()) / (q_min * var_1m[s]) for s in TOP15])
        # Wick noise ratio = 1 - |C - O| / (H - L)
        hl_range = (h_rs - l_rs).replace(0, np.nan)
        body = (c_rs - o_rs).abs()
        wick_ratio = float((1.0 - (body / hl_range)).mean().mean() * 100.0)
        # Signal-to-Noise Ratio: |body| / sqrt(sum of 1m squared returns inside the bar)
        snr = float(((body / hl_range) / np.sqrt(1.0 + np.abs(1.0 - vr))).mean().mean())

        regime_desc = (
            "Severe Bid-Ask Bounce & Wick Noise" if q_min <= 5 else
            ("Transition (Microstructure Pullback Optimal)" if q_min <= 30 else "Macro Trend / Continuation Dominant")
        )
        print(f"{tf_lbl:<12} | {ac1:>+15.4f} | {vr:>21.4f} | {wick_ratio:>16.1f}% | {snr:>14.3f} | {regime_desc}")

    # Test on the 31-day 5m dataset (2026-09-20 to 2026-10-20):
    # Does using 5m bars to compute a 30m VWAP anchor + 5m Realized Bipower Volatility beat raw Close prices?
    print("\n--- Testing 5m Sub-Bar Resolution Features (30m VWAP Price + Bipower Jump-Filtered Vol) on 30-Day Window ---")
    # On 5m dataset: 1h = 12 bars, 4h = 48 bars, 12h = 144 bars, 24h = 288 bars, 48h = 576 bars
    dv5 = c5 * v5
    vwap_30m = dv5.rolling(6).sum() / v5.rolling(6).sum().replace(0, np.nan)
    vwap_30m = vwap_30m.fillna(c5)

    r5 = np.log(c5 / c5.shift(1)).fillna(0.0)
    # Realized Bipower Variation (Barndorff-Nielsen & Shephard): (pi/2) * mean(|r_t| * |r_{t-1}|) — immune to 5m jump spikes!
    bv_24h = np.sqrt((np.pi / 2.0) * (r5.abs() * r5.abs().shift(1)).rolling(288).mean()).bfill().fillna(0.005)
    std_24h = r5.rolling(288).std().bfill().fillna(0.005)

    for mode_lbl, price_ref, vol_ref in [
        ("Raw Last-Tick Close + Standard Vol", c5, std_24h),
        ("5m->30m Smoothed VWAP + Bipower Jump-Filtered Vol", vwap_30m, bv_24h),
    ]:
        lr = np.log(price_ref / price_ref.shift(1)).fillna(0.0)
        z_4h = lr.rolling(48).sum() / (vol_ref * np.sqrt(48))
        z_12h = lr.rolling(144).sum() / (vol_ref * np.sqrt(144))
        z_24h = lr.rolling(288).sum() / (vol_ref * np.sqrt(288))
        z_48h = lr.rolling(576).sum() / (vol_ref * np.sqrt(576))
        sc = (0.35 * z_12h + 0.45 * z_24h + 0.20 * z_48h) - 0.25 * z_4h
        w = make_continuous_neutral_weights(sc, 1.0 / vol_ref, gross_leverage=2.2, power=1.8)
        # Evaluate at 4h (48 bars of 5m)
        eval_idx = c5.index[c5.index >= pd.Timestamp("2026-09-22 00:00:00", tz="UTC")]
        w_sub = w.reindex(eval_idx).iloc[::48].reindex(eval_idx).ffill().fillna(0.0)
        exec_w = w_sub.shift(1).fillna(0.0)
        rets = c5.pct_change().reindex(eval_idx).fillna(0.0)
        turn = exec_w.diff().abs().sum(axis=1).fillna(0.0)
        net_r = (exec_w * rets).sum(axis=1) - (exec_w * f5.reindex(eval_idx).fillna(0.0)).sum(axis=1) - turn * 0.0002
        eq = 1000.0 * (1.0 + net_r).cumprod()
        d_rets = eq.resample("1D").last().pct_change().dropna()
        shp = (d_rets.mean() / d_rets.std()) * np.sqrt(365.0) if d_rets.std() > 0 else 0.0
        print(f"  {mode_lbl:<52} | 30d Net Ret: {(eq.iloc[-1]/1000 - 1)*100:+6.2f}% | Sharpe: {shp:+5.2f}")


def run_2025_2026_full_backtest():
    print("\n" + "=" * 125)
    print("PART 2: FULL 22-MONTH WALK-FORWARD BACKTEST (ALL 2025 + ALL 2026: 2025-01-01 to 2026-10-20, 658 DAYS)")
    print("=" * 125)

    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(
        TOP15, start_date="2024-11-15", end_date="2026-10-20 18:00:00"
    )
    print(f"Loaded 1h matrix: {len(idx):,} hourly bars ({idx[0]} -> {idx[-1]}) across {len(TOP15)} symbols.")

    rets_1h = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_24h).clip(upper=100.0)
    vol_surge = (dollar_vol.rolling(6).mean() / (dollar_vol.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)

    # Rolling 72h Beta to BTC & Idiosyncratic Returns
    btc_ret = rets_1h["BTCUSDT"]
    cov_btc = rets_1h.rolling(72).cov(btc_ret)
    var_btc = btc_ret.rolling(72).var().replace(0, 1e-8)
    beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5).fillna(1.0)
    resid_ret = rets_1h.sub(beta_btc.mul(btc_ret, axis=0))

    r_2h  = log_rets.rolling(2).sum() / (vol_24h * np.sqrt(2))
    r_4h  = log_rets.rolling(4).sum() / (vol_24h * np.sqrt(4))
    r_12h = log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12))
    r_24h = log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24))
    r_48h = log_rets.rolling(48).sum() / (vol_24h * np.sqrt(48))
    r_72h = log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72))

    ir_2h  = resid_ret.rolling(2).sum() / (vol_24h * np.sqrt(2))
    ir_4h  = resid_ret.rolling(4).sum() / (vol_24h * np.sqrt(4))
    ir_12h = resid_ret.rolling(12).sum() / (vol_24h * np.sqrt(12))
    ir_24h = resid_ret.rolling(24).sum() / (vol_24h * np.sqrt(24))
    ir_48h = resid_ret.rolling(48).sum() / (vol_24h * np.sqrt(48))
    ir_72h = resid_ret.rolling(72).sum() / (vol_24h * np.sqrt(72))

    mom_pullback_raw = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) - 0.25 * r_4h
    mom_pullback_idio = (0.30 * ir_12h + 0.40 * ir_24h + 0.30 * ir_72h) * np.sqrt(vol_surge) - 0.20 * ir_4h
    mom_pullback_blend = 0.60 * mom_pullback_raw + 0.40 * mom_pullback_idio

    # Cross-Sectional Autocorrelation Sensor (48h)
    ret_24h = log_rets.rolling(24).sum()
    prev_ret_24h = ret_24h.shift(24)
    m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
    m_curr = ret_24h.sub(ret_24h.mean(axis=1), axis=0)
    num = (m_prev * m_curr).sum(axis=1)
    den = np.sqrt((m_prev ** 2).sum(axis=1) * (m_curr ** 2).sum(axis=1)) + 1e-9
    raw_autocorr = num / den
    cs_autocorr_48h = raw_autocorr.rolling(48).mean().fillna(0.0)

    ann_fund_72h = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

    tilt = ((cs_autocorr_48h - (-0.14)) / (-0.04 - (-0.14))).clip(0.0, 1.0)
    lev_mult = (tilt.abs() + 0.35).clip(0.45, 1.0)
    dyn_score = mom_pullback_blend.mul(tilt, axis=0) + 0.30 * carry_score

    # Base unit-leverage weights (Gross = 1.0x before scaling)
    w_unit = make_continuous_neutral_weights(dyn_score, inv_vol, gross_leverage=1.0, power=1.8).mul(lev_mult, axis=0)

    eval_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]
    rets_eval = close.pct_change().reindex(eval_idx).fillna(0.0).values
    fund_eval = funding.reindex(eval_idx).fillna(0.0).values
    w_unit_arr = w_unit.reindex(eval_idx).fillna(0.0).values

    def sim_leverage_scheme(
        start_lev: float = 2.5,
        max_lev: float = 5.0,
        step_lev: float = 0.5,
        step_profit_pct: float = 0.25,  # Step up +0.5x every +25% equity gain above previous step HWM
        dd_reduce_thresh: float = 0.08, # If drawdown > 8%, step down leverage
        dd_reset_thresh: float = 0.14,  # If drawdown > 14%, reset leverage to start_lev (2.5x)
        reb_hours: int = 4,
        deadband_ratio: float = 0.018,  # proportional deadband (~4% at 2.2x)
        fee_rate: float = 0.0002,
    ):
        n_bars, n_cols = w_unit_arr.shape
        eq = 1000.0
        hwm = 1000.0
        last_step_eq = 1000.0
        cur_lev = start_lev
        prev_w = np.zeros(n_cols)

        eq_curve = np.zeros(n_bars)
        lev_curve = np.zeros(n_bars)
        total_fees_pct = 0.0

        for t in range(n_bars):
            # 1. Execute return of bar t using prev_w (set at bar t-1)
            if t > 0:
                p_ret = np.sum(prev_w * rets_eval[t])
                f_ret = -np.sum(prev_w * fund_eval[t])
                bar_ret = p_ret + f_ret
                # Check maintenance margin / liquidation safety (at 5x gross, -18% single-bar loss would liquidate)
                eq = eq * (1.0 + bar_ret)
                if eq > hwm:
                    hwm = eq

            # 2. Update Dynamic Stepped Leverage at rebalance points
            cur_dd = (hwm - eq) / hwm if hwm > 0 else 0.0
            if step_lev > 0.0:
                if cur_dd >= dd_reset_thresh:
                    cur_lev = start_lev
                    last_step_eq = eq
                elif cur_dd >= dd_reduce_thresh:
                    cur_lev = max(start_lev, cur_lev - step_lev)
                elif cur_dd <= 0.02 and eq >= last_step_eq * (1.0 + step_profit_pct):
                    if cur_lev < max_lev:
                        cur_lev = min(max_lev, cur_lev + step_lev)
                        last_step_eq = eq

            # 3. Rebalance portfolio every reb_hours
            if t % reb_hours == 0:
                max_single = 0.205 * cur_lev
                row = w_unit_arr[t] * cur_lev
                g_init = np.sum(np.abs(row))
                if g_init > 1e-8:
                    target_g = min(g_init, cur_lev)
                    row = np.clip(row, -max_single, max_single)
                    pos = np.maximum(row, 0.0)
                    neg = np.minimum(row, 0.0)
                    if pos.sum() > 0:
                        pos = pos * ((target_g / 2.0) / pos.sum())
                    if np.abs(neg.sum()) > 0:
                        neg = neg * ((target_g / 2.0) / np.abs(neg.sum()))
                    desired = np.clip(pos + neg, -max_single, max_single)
                else:
                    desired = np.zeros(n_cols)

                db = deadband_ratio * cur_lev
                delta = np.abs(desired - prev_w)
                upd = np.where((desired == 0.0) | (delta >= db), desired, prev_w)
                g = np.sum(np.abs(upd))
                if g > cur_lev:
                    upd = upd * (cur_lev / g)

                turn = np.sum(np.abs(upd - prev_w))
                fee_cost = turn * fee_rate
                eq = eq * (1.0 - fee_cost)
                total_fees_pct += fee_cost * 100.0
                prev_w = upd

            eq_curve[t] = eq
            lev_curve[t] = cur_lev

        eq_s = pd.Series(eq_curve, index=eval_idx)
        daily_eq = eq_s.resample("1D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna()
        mean_d = float(daily_rets.mean())
        std_d = float(daily_rets.std())
        sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
        down = daily_rets[daily_rets < 0]
        dd_std = float(np.sqrt((down ** 2).mean())) if len(down) > 0 else 1e-12
        sortino = (mean_d / dd_std) * np.sqrt(365.0) if dd_std > 1e-12 else 0.0
        peaks = eq_s.cummax()
        mdd = float(((eq_s - peaks) / peaks).min() * 100.0)
        win_days = float((daily_rets > 0).mean() * 100.0)

        monthly = daily_eq.resample("ME").last().pct_change()
        monthly.iloc[0] = (daily_eq.resample("ME").last().iloc[0] / 1000.0) - 1.0

        # Split metrics by 2025 (full year) and 2026 (Jan-Oct)
        eq_2025_end = float(eq_s.loc["2025-12-31"].iloc[-1])
        ret_2025 = (eq_2025_end / 1000.0 - 1.0) * 100.0
        ret_2026 = (float(eq_s.iloc[-1]) / eq_2025_end - 1.0) * 100.0

        return {
            "final_eq": float(eq_s.iloc[-1]),
            "total_ret_pct": (float(eq_s.iloc[-1]) / 1000.0 - 1.0) * 100.0,
            "ret_2025_pct": ret_2025,
            "ret_2026_pct": ret_2026,
            "avg_daily_pct": mean_d * 100.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd_pct": mdd,
            "win_days_pct": win_days,
            "avg_lev": float(np.mean(lev_curve)),
            "end_lev": float(lev_curve[-1]),
            "pos_months": int((monthly > 0).sum()),
            "total_months": len(monthly),
            "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
        }

    configs = [
        ("Fixed 2.2x Gross (Reb 4h)", 2.2, 2.2, 0.0, 0.25, 0.08, 0.14, 4),
        ("Fixed 2.5x Gross (Reb 4h)", 2.5, 2.5, 0.0, 0.25, 0.08, 0.14, 4),
        ("Fixed 3.0x Gross (Reb 4h)", 3.0, 3.0, 0.0, 0.25, 0.08, 0.14, 4),
        ("Fixed 3.5x Gross (Reb 4h)", 3.5, 3.5, 0.0, 0.25, 0.08, 0.14, 4),
        ("Fixed 4.0x Gross (Reb 4h)", 4.0, 4.0, 0.0, 0.25, 0.08, 0.14, 4),
        ("Fixed 5.0x Gross (Reb 4h)", 5.0, 5.0, 0.0, 0.25, 0.08, 0.14, 4),
        ("Stepped 2.5x->5.0x (+0.5x / +25% Eq, DD Guard)", 2.5, 5.0, 0.5, 0.25, 0.08, 0.14, 4),
        ("Stepped 2.5x->5.0x (+0.5x / +20% Eq, Tight DD)", 2.5, 5.0, 0.5, 0.20, 0.06, 0.11, 4),
        ("Stepped 2.5x->5.0x (Reb 8h, +0.5x / +25% Eq)", 2.5, 5.0, 0.5, 0.25, 0.08, 0.14, 8),
    ]

    print(
        f"{'Configuration (2025-01 to 2026-10)':<46} | {'2025 Ret':>9} | {'2026 Ret':>9} | {'Total 22M':>10} | "
        f"{'Final Eq':>10} | {'DailyAvg':>8} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7} | {'PosMos':>6}"
    )
    print("-" * 145)
    all_res = {}
    for name, s_lev, m_lev, stp, p_step, dd_red, dd_res, reb_h in configs:
        r = sim_leverage_scheme(s_lev, m_lev, stp, p_step, dd_red, dd_res, reb_h)
        all_res[name] = r
        pm = f"{r['pos_months']}/{r['total_months']}"
        print(
            f"{name:<46} | {r['ret_2025_pct']:>+8.1f}% | {r['ret_2026_pct']:>+8.1f}% | {r['total_ret_pct']:>+9.1f}% | "
            f"${r['final_eq']:>9,.0f} | {r['avg_daily_pct']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['max_dd_pct']:>6.2f}% | {pm:>6}"
        )

    print("\n--- Monthly Return Breakdown Across 2025 and 2026 (Selected Configurations) ---")
    for k in [
        "Fixed 2.5x Gross (Reb 4h)",
        "Fixed 5.0x Gross (Reb 4h)",
        "Stepped 2.5x->5.0x (+0.5x / +25% Eq, DD Guard)",
        "Stepped 2.5x->5.0x (+0.5x / +20% Eq, Tight DD)",
    ]:
        r = all_res[k]
        m25 = " ".join([f"{m[5:]}:{v:+.1f}%" for m, v in r["monthly"].items() if m.startswith("2025")])
        m26 = " ".join([f"{m[5:]}:{v:+.1f}%" for m, v in r["monthly"].items() if m.startswith("2026")])
        print(f"\n[{k}] (Avg Lev: {r['avg_lev']:.2f}x | Pos Months: {r['pos_months']}/{r['total_months']}):")
        print(f"  2025: {m25}")
        print(f"  2026: {m26}")


if __name__ == "__main__":
    analyze_1m_5m_15m_1h_noise()
    run_2025_2026_full_backtest()
