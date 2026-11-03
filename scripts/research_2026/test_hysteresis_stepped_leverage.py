"""
Tests the Hysteresis Stepped Leverage Controller (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x in +0.5x steps)
with Bracketed Drawdown De-escalation and Rolling Volatility Targeting across the full 22-month dataset
(All 2025 + All 2026: 2025-01-01 to 2026-10-20, 658 Days).
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


def main():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(
        TOP15, start_date="2024-10-01", end_date="2026-10-20 18:00:00"
    )
    rets_1h = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_24h).clip(upper=100.0)
    vol_surge = (dollar_vol.rolling(6).mean() / (dollar_vol.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)
    vol_surge_wk = (dollar_vol.rolling(24).mean() / (dollar_vol.rolling(168).mean() + 1e-8)).clip(0.5, 2.5)

    btc_ret = rets_1h["BTCUSDT"]
    beta_72 = (rets_1h.rolling(72).cov(btc_ret).div(btc_ret.rolling(72).var().replace(0, 1e-8), axis=0)).clip(0.2, 2.5).fillna(1.0)
    resid_ret = rets_1h.sub(beta_72.mul(btc_ret, axis=0))

    def z_h(s, h):
        return s.rolling(h).sum() / (vol_24h * np.sqrt(h))

    horizons = [4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720]
    zr = {h: z_h(log_rets, h) for h in horizons}
    zi = {h: z_h(resid_ret, h) for h in horizons}

    # 7 Orthogonal Factors
    f_macro = 0.70 * ((0.35 * zi[168] + 0.40 * zi[336] + 0.25 * zi[720]) * np.sqrt(vol_surge_wk) - (0.15 * zi[4] + 0.20 * zi[24])) + \
              0.30 * ((0.35 * zr[168] + 0.40 * zr[336] + 0.25 * zr[720]) * np.sqrt(vol_surge_wk) - (0.15 * zr[4] + 0.20 * zr[24]))
    f_med   = (0.40 * zi[72] + 0.60 * zi[168]) * np.sqrt(vol_surge_wk) - 0.25 * zi[12]
    f_fast  = 0.60 * ((0.35 * zr[12] + 0.45 * zr[24] + 0.20 * zr[48]) * np.sqrt(vol_surge) - 0.25 * zr[4]) + \
              0.40 * ((0.30 * zi[12] + 0.40 * zi[24] + 0.30 * zi[72]) * np.sqrt(vol_surge) - 0.20 * zi[4])
    f_rev   = -(0.40 * zi[24] + 0.60 * zi[72]) - 0.25 * zr[4]
    ann_fund = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    f_carry = -ann_fund.where(ann_fund.abs() >= 0.05, 0.0)
    f_liq   = np.log(dollar_vol.rolling(168).mean() + 1.0) - np.log(dollar_vol.rolling(720).mean() + 1.0)
    f_size  = np.log(dollar_vol.rolling(168).mean() + 1.0)

    factor_scores = {
        "macro": f_macro,
        "med":   f_med,
        "fast":  f_fast,
        "rev":   f_rev,
        "carry": f_carry,
        "liq":   f_liq,
        "size":  f_size,
    }

    w_unit = {k: make_continuous_neutral_weights(sc, inv_vol, gross_leverage=1.0, power=1.7) for k, sc in factor_scores.items()}
    f_bar_ret = {}
    for k, w in w_unit.items():
        w_exec = w.shift(1).fillna(0.0)
        turn = w_exec.diff().abs().sum(axis=1).fillna(0.0)
        f_bar_ret[k] = (w_exec * rets_1h).sum(axis=1) - (w_exec * funding).sum(axis=1) - turn * 0.0001
    df_frets = pd.DataFrame(f_bar_ret)

    lookbacks = (36, 96, 240)
    t_df = pd.DataFrame(0.0, index=idx, columns=df_frets.columns)
    for lb in lookbacks:
        m = df_frets.rolling(lb).mean()
        s = df_frets.rolling(lb).std().replace(0, 1e-8)
        t_df += (m / s * np.sqrt(lb)).fillna(0.0) / len(lookbacks)

    priors = pd.Series({"macro": 0.30, "med": 0.15, "fast": 0.15, "rev": 0.10, "carry": 0.15, "liq": 0.10, "size": 0.15})
    raw_fw = np.maximum(0.0, t_df + priors) ** 2.0
    fw_sum = raw_fw.sum(axis=1).replace(0, 1e-8)
    fw_norm = raw_fw.div(fw_sum, axis=0)
    conviction = (raw_fw.sum(axis=1) / (priors.sum() ** 2.0)).clip(0.35, 1.15)

    comb_w = pd.DataFrame(0.0, index=idx, columns=TOP15)
    for k in factor_scores.keys():
        comb_w += w_unit[k].mul(fw_norm[k], axis=0)

    w_base_1x = make_continuous_neutral_weights(comb_w, pd.DataFrame(1.0, index=idx, columns=TOP15), gross_leverage=1.0, power=1.3)
    w_base_1x = w_base_1x.mul(conviction, axis=0)

    # Compute causal 1x portfolio volatility over 72h to normalize extreme volatility regimes
    unit_ret_1h = (w_base_1x.shift(1).fillna(0.0) * rets_1h).sum(axis=1)
    port_vol_72h = unit_ret_1h.rolling(72).std().bfill().fillna(0.003)
    median_vol = float(port_vol_72h.median())
    vol_scalar = (median_vol / port_vol_72h.replace(0, np.nan)).clip(0.55, 1.15).fillna(1.0)

    eval_idx = close.index[close.index >= pd.Timestamp("2025-01-01 00:00:00", tz="UTC")]
    rets_eval = close.pct_change().reindex(eval_idx).fillna(0.0).values
    fund_eval = funding.reindex(eval_idx).fillna(0.0).values
    w_1x_arr = w_base_1x.reindex(eval_idx).fillna(0.0).values
    v_scale_arr = vol_scalar.reindex(eval_idx).fillna(1.0).values

    def run_hysteresis_sim(
        start_lev: float = 2.5,
        max_lev: float = 5.0,
        step_lev: float = 0.5,
        gain_per_step: float = 0.25,  # Each +25% compound gain in HWM unlocks +0.5x tier (2.5 -> 3.0 -> 3.5 -> 4.0 -> 4.5 -> 5.0)
        dd_bracket: float = 0.10,     # Each 10% drawdown from HWM temporarily reduces active tier by -0.5x
        use_vol_scalar: bool = True,
        reb_hours: int = 8,
        fee_rate: float = 0.0002,
    ):
        n_bars, n_cols = w_1x_arr.shape
        eq = 1000.0
        hwm = 1000.0
        unlocked_tier = start_lev
        prev_w = np.zeros(n_cols)

        eq_curve = np.zeros(n_bars)
        lev_curve = np.zeros(n_bars)

        for t in range(n_bars):
            if t > 0:
                bar_r = np.sum(prev_w * rets_eval[t]) - np.sum(prev_w * fund_eval[t])
                eq = eq * (1.0 + bar_r)
                if eq > hwm:
                    hwm = eq

            # 1. Calculate unlocked leverage tier from High-Water Mark (HWM)
            if step_lev > 0.0:
                # Number of steps unlocked by HWM growth: floor(log(hwm / 1000) / log(1 + gain_per_step))
                steps_unlocked = int(np.floor(np.log(max(hwm / 1000.0, 1.0)) / np.log(1.0 + gain_per_step)))
                unlocked_tier = min(max_lev, start_lev + steps_unlocked * step_lev)

                # 2. Bracketed Drawdown De-escalation (with hysteresis: steps down by -0.5x per dd_bracket)
                cur_dd = (hwm - eq) / hwm if hwm > 0 else 0.0
                steps_down = int(np.floor(cur_dd / dd_bracket)) if dd_bracket < 0.9 else 0
                active_lev = max(start_lev, unlocked_tier - steps_down * step_lev)
            else:
                active_lev = start_lev

            eff_lev = active_lev * (v_scale_arr[t] if use_vol_scalar else 1.0)

            if t % reb_hours == 0:
                max_single = (0.45 / 2.2) * eff_lev
                row = w_1x_arr[t] * eff_lev
                g_init = np.sum(np.abs(row))
                if g_init > 1e-8:
                    target_g = min(g_init, eff_lev)
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

                db = (0.04 / 2.2) * eff_lev
                delta = np.abs(desired - prev_w)
                upd = np.where((desired == 0.0) | (delta >= db), desired, prev_w)
                g = np.sum(np.abs(upd))
                if g > eff_lev:
                    upd = upd * (eff_lev / g)

                turn = np.sum(np.abs(upd - prev_w))
                eq = eq * (1.0 - turn * fee_rate)
                prev_w = upd

            eq_curve[t] = eq
            lev_curve[t] = active_lev

        eq_s = pd.Series(eq_curve, index=eval_idx)
        daily_eq = eq_s.resample("1D").last().dropna()
        d_rets = daily_eq.pct_change().dropna()
        mean_d = float(d_rets.mean())
        std_d = float(d_rets.std())
        sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
        down = d_rets[d_rets < 0]
        dd_std = float(np.sqrt((down ** 2).mean())) if len(down) > 0 else 1e-12
        sortino = (mean_d / dd_std) * np.sqrt(365.0) if dd_std > 1e-12 else 0.0
        peaks = eq_s.cummax()
        mdd = float(((eq_s - peaks) / peaks).min() * 100.0)
        win_days = float((d_rets > 0).mean() * 100.0)

        monthly = daily_eq.resample("ME").last().pct_change()
        monthly.iloc[0] = (daily_eq.resample("ME").last().iloc[0] / 1000.0) - 1.0

        eq_2025 = float(eq_s.loc["2025-12-31"].iloc[-1])
        ret_2025 = (eq_2025 / 1000.0 - 1.0) * 100.0
        ret_2026 = (float(eq_s.iloc[-1]) / eq_2025 - 1.0) * 100.0
        eq_apr20 = float(eq_s.loc["2026-04-20 00:00:00"])
        ret_last6m = (float(eq_s.iloc[-1]) / eq_apr20 - 1.0) * 100.0

        return {
            "ret_2025": ret_2025,
            "ret_2026": ret_2026,
            "ret_last6m": ret_last6m,
            "total_ret": (float(eq_s.iloc[-1]) / 1000.0 - 1.0) * 100.0,
            "final_eq": float(eq_s.iloc[-1]),
            "daily_avg": mean_d * 100.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "mdd": mdd,
            "win_days": win_days,
            "avg_lev": float(np.mean(lev_curve)),
            "end_lev": float(lev_curve[-1]),
            "pos_m": int((monthly > 0).sum()),
            "tot_m": len(monthly),
            "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
        }

    print("=" * 160)
    print("HYSTERESIS STEPPED LEVERAGE (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x) ACROSS ALL OF 2025 AND 2026 (22 MONTHS)")
    print("=" * 160)
    print(
        f"{'Configuration':<48} | {'2025 Ret':>9} | {'2026 Ret':>9} | {'Last 6M':>8} | {'Total 22M':>10} | "
        f"{'Final Eq':>10} | {'DailyAvg':>8} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7} | {'PosMos':>6}"
    )
    print("-" * 160)

    configs = [
        ("1. Fixed 2.5x (Vol-Targeted)", 2.5, 2.5, 0.0, 0.20, 0.10, True),
        ("2. Fixed 3.5x (Vol-Targeted)", 3.5, 3.5, 0.0, 0.20, 0.10, True),
        ("3. Fixed 5.0x (Vol-Targeted)", 5.0, 5.0, 0.0, 0.20, 0.10, True),
        ("4. Stepped 2.5x->5.0x (+0.5x / +20% HWM, DD=10%, VolTgt)", 2.5, 5.0, 0.5, 0.20, 0.10, True),
        ("5. Stepped 2.5x->5.0x (+0.5x / +15% HWM, DD=10%, VolTgt)", 2.5, 5.0, 0.5, 0.15, 0.10, True),
        ("6. Stepped 2.5x->5.0x (+0.5x / +20% HWM, DD=12%, Raw)", 2.5, 5.0, 0.5, 0.20, 0.12, False),
        ("7. Stepped 2.5x->5.0x (+0.5x / +15% HWM, DD=12%, Raw)", 2.5, 5.0, 0.5, 0.15, 0.12, False),
    ]

    for name, s_l, m_l, stp, g_pct, dd_b, v_t in configs:
        r = run_hysteresis_sim(s_l, m_l, stp, g_pct, dd_b, v_t)
        pm = f"{r['pos_m']}/{r['tot_m']}"
        print(
            f"{name:<48} | {r['ret_2025']:>+8.1f}% | {r['ret_2026']:>+8.1f}% | {r['ret_last6m']:>+7.1f}% | {r['total_ret']:>+9.1f}% | "
            f"${r['final_eq']:>9,.0f} | {r['daily_avg']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['mdd']:>6.2f}% | {pm:>6}"
        )


if __name__ == "__main__":
    main()


def print_detailed_22m_breakdown():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(
        TOP15, start_date="2024-10-01", end_date="2026-10-20 18:00:00"
    )
    rets_1h = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_24h).clip(upper=100.0)
    vol_surge = (dollar_vol.rolling(6).mean() / (dollar_vol.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)
    vol_surge_wk = (dollar_vol.rolling(24).mean() / (dollar_vol.rolling(168).mean() + 1e-8)).clip(0.5, 2.5)

    btc_ret = rets_1h["BTCUSDT"]
    beta_72 = (rets_1h.rolling(72).cov(btc_ret).div(btc_ret.rolling(72).var().replace(0, 1e-8), axis=0)).clip(0.2, 2.5).fillna(1.0)
    resid_ret = rets_1h.sub(beta_72.mul(btc_ret, axis=0))

    def z_h(s, h):
        return s.rolling(h).sum() / (vol_24h * np.sqrt(h))

    horizons = [4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720]
    zr = {h: z_h(log_rets, h) for h in horizons}
    zi = {h: z_h(resid_ret, h) for h in horizons}

    f_macro = 0.70 * ((0.35 * zi[168] + 0.40 * zi[336] + 0.25 * zi[720]) * np.sqrt(vol_surge_wk) - (0.15 * zi[4] + 0.20 * zi[24])) + \
              0.30 * ((0.35 * zr[168] + 0.40 * zr[336] + 0.25 * zr[720]) * np.sqrt(vol_surge_wk) - (0.15 * zr[4] + 0.20 * zr[24]))
    f_med   = (0.40 * zi[72] + 0.60 * zi[168]) * np.sqrt(vol_surge_wk) - 0.25 * zi[12]
    f_fast  = 0.60 * ((0.35 * zr[12] + 0.45 * zr[24] + 0.20 * zr[48]) * np.sqrt(vol_surge) - 0.25 * zr[4]) + \
              0.40 * ((0.30 * zi[12] + 0.40 * zi[24] + 0.30 * zi[72]) * np.sqrt(vol_surge) - 0.20 * zi[4])
    f_rev   = -(0.40 * zi[24] + 0.60 * zi[72]) - 0.25 * zr[4]
    ann_fund = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    f_carry = -ann_fund.where(ann_fund.abs() >= 0.05, 0.0)
    f_liq   = np.log(dollar_vol.rolling(168).mean() + 1.0) - np.log(dollar_vol.rolling(720).mean() + 1.0)
    f_size  = np.log(dollar_vol.rolling(168).mean() + 1.0)

    factor_scores = {"macro": f_macro, "med": f_med, "fast": f_fast, "rev": f_rev, "carry": f_carry, "liq": f_liq, "size": f_size}
    w_unit = {k: make_continuous_neutral_weights(sc, inv_vol, gross_leverage=1.0, power=1.7) for k, sc in factor_scores.items()}
    f_bar_ret = {
        k: (w.shift(1).fillna(0.0) * rets_1h).sum(axis=1) - (w.shift(1).fillna(0.0) * funding).sum(axis=1) - w.shift(1).fillna(0.0).diff().abs().sum(axis=1).fillna(0.0) * 0.0001
        for k, w in w_unit.items()
    }
    df_frets = pd.DataFrame(f_bar_ret)

    t_df = pd.DataFrame(0.0, index=idx, columns=df_frets.columns)
    for lb in (36, 96, 240):
        t_df += (df_frets.rolling(lb).mean() / df_frets.rolling(lb).std().replace(0, 1e-8) * np.sqrt(lb)).fillna(0.0) / 3.0

    priors = pd.Series({"macro": 0.30, "med": 0.15, "fast": 0.15, "rev": 0.10, "carry": 0.15, "liq": 0.10, "size": 0.15})
    raw_fw = np.maximum(0.0, t_df + priors) ** 2.0
    fw_norm = raw_fw.div(raw_fw.sum(axis=1).replace(0, 1e-8), axis=0)
    conviction = (raw_fw.sum(axis=1) / (priors.sum() ** 2.0)).clip(0.35, 1.15)

    comb_w = sum(w_unit[k].mul(fw_norm[k], axis=0) for k in factor_scores.keys())
    w_base_1x = make_continuous_neutral_weights(comb_w, pd.DataFrame(1.0, index=idx, columns=TOP15), gross_leverage=1.0, power=1.3).mul(conviction, axis=0)

    unit_ret_1h = (w_base_1x.shift(1).fillna(0.0) * rets_1h).sum(axis=1)
    port_vol_72h = unit_ret_1h.rolling(72).std().bfill().fillna(0.003)
    median_vol = float(port_vol_72h.median())
    vol_scalar = (median_vol / port_vol_72h.replace(0, np.nan)).clip(0.55, 1.15).fillna(1.0)

    eval_idx = close.index[close.index >= pd.Timestamp("2025-01-01 00:00:00", tz="UTC")]
    rets_eval = close.pct_change().reindex(eval_idx).fillna(0.0).values
    fund_eval = funding.reindex(eval_idx).fillna(0.0).values
    w_1x_arr = w_base_1x.reindex(eval_idx).fillna(0.0).values
    v_scale_arr = vol_scalar.reindex(eval_idx).fillna(1.0).values

    # Test Stepped 2.5x -> 5.0x (+0.5x per +15% HWM, Vol-Targeted, with DD circuit breaker at 20%)
    for lbl, s_l, m_l, stp, g_pct, dd_b in [
        ("Fixed 2.5x (Vol-Targeted)", 2.5, 2.5, 0.0, 0.15, 0.99),
        ("Stepped 2.5x->5.0x (+0.5x / +15% HWM, Vol-Targeted)", 2.5, 5.0, 0.5, 0.15, 0.22),
        ("Stepped 2.5x->5.0x (+0.5x / +15% HWM, DD=10% StepDown)", 2.5, 5.0, 0.5, 0.15, 0.10),
        ("Fixed 5.0x (Vol-Targeted)", 5.0, 5.0, 0.0, 0.15, 0.99),
    ]:
        n_bars, n_cols = w_1x_arr.shape
        eq = 1000.0
        hwm = 1000.0
        prev_w = np.zeros(n_cols)
        eq_curve = np.zeros(n_bars)
        step_events = []
        last_tier = s_l

        for t in range(n_bars):
            if t > 0:
                eq = eq * (1.0 + np.sum(prev_w * rets_eval[t]) - np.sum(prev_w * fund_eval[t]))
                if eq > hwm:
                    hwm = eq
            if stp > 0.0:
                steps_unlocked = int(np.floor(np.log(max(hwm / 1000.0, 1.0)) / np.log(1.0 + g_pct)))
                unlocked_tier = min(m_l, s_l + steps_unlocked * stp)
                if unlocked_tier > last_tier:
                    step_events.append((eval_idx[t].strftime("%Y-%m-%d"), unlocked_tier, eq))
                    last_tier = unlocked_tier
                cur_dd = (hwm - eq) / hwm if hwm > 0 else 0.0
                steps_down = int(np.floor(cur_dd / dd_b)) if dd_b < 0.9 else 0
                active_lev = max(s_l, unlocked_tier - steps_down * stp)
            else:
                active_lev = s_l

            eff_lev = active_lev * v_scale_arr[t]
            if t % 8 == 0:
                max_single = (0.45 / 2.2) * eff_lev
                row = w_1x_arr[t] * eff_lev
                g_init = np.sum(np.abs(row))
                if g_init > 1e-8:
                    target_g = min(g_init, eff_lev)
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
                db = (0.04 / 2.2) * eff_lev
                delta = np.abs(desired - prev_w)
                upd = np.where((desired == 0.0) | (delta >= db), desired, prev_w)
                g = np.sum(np.abs(upd))
                if g > eff_lev:
                    upd = upd * (eff_lev / g)
                eq = eq * (1.0 - np.sum(np.abs(upd - prev_w)) * 0.0002)
                prev_w = upd
            eq_curve[t] = eq

        eq_s = pd.Series(eq_curve, index=eval_idx)
        daily_eq = eq_s.resample("1D").last().dropna()
        d_rets = daily_eq.pct_change().dropna()
        mean_d = float(d_rets.mean()) * 100.0
        sharpe = float((d_rets.mean() / d_rets.std()) * np.sqrt(365.0))
        peaks = eq_s.cummax()
        mdd = float(((eq_s - peaks) / peaks).min() * 100.0)
        monthly = daily_eq.resample("ME").last().pct_change()
        monthly.iloc[0] = (daily_eq.resample("ME").last().iloc[0] / 1000.0) - 1.0
        eq_2025 = float(eq_s.loc["2025-12-31"].iloc[-1])
        r25 = (eq_2025 / 1000.0 - 1.0) * 100.0
        r26 = (float(eq_s.iloc[-1]) / eq_2025 - 1.0) * 100.0
        r_tot = (float(eq_s.iloc[-1]) / 1000.0 - 1.0) * 100.0

        print(f"\n=== {lbl} ===")
        print(f"  2025: {r25:+.1f}% | 2026: {r26:+.1f}% | Total 22M: {r_tot:+.1f}% (${eq_s.iloc[-1]:,.2f}) | DailyAvg: {mean_d:+.3f}% | Sharpe: {sharpe:.2f} | MaxDD: {mdd:.2f}% | PosMonths: {(monthly>0).sum()}/{len(monthly)}")
        if step_events:
            print("  Step-Up Milestones Unlocked:", ", ".join([f"{dt}: {lv:.1f}x (${e:,.0f})" for dt, lv, e in step_events]))
        m25 = " ".join([f"{k.strftime('%Y-%m')[5:]}:{v*100:+.1f}%" for k, v in monthly.items() if k.year == 2025])
        m26 = " ".join([f"{k.strftime('%Y-%m')[5:]}:{v*100:+.1f}%" for k, v in monthly.items() if k.year == 2026])
        print(f"  2025 Months: {m25}")
        print(f"  2026 Months: {m26}")


if __name__ == "__main__":
    print_detailed_22m_breakdown()
