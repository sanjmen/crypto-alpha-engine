"""
Evaluates the 7-Factor Causal Online Meta-Allocator (which achieved +403.5% at 2.2x across 2025-2026)
under Fixed Leverage (2.2x, 2.5x, 3.0x, 3.5x, 4.0x, 4.5x, 5.0x) and the User's Progressive Stepped Leverage
(starting at 2.5x and stepping up by +0.5x to 5.0x as equity milestones are achieved and drawdown is controlled).
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

    # Exact 7 Orthogonal Factors from test_online_factor_allocator_22m.py
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

    # Base weights normalized to 1.0x gross leverage * conviction
    w_base_1x = make_continuous_neutral_weights(comb_w, pd.DataFrame(1.0, index=idx, columns=TOP15), gross_leverage=1.0, power=1.3)
    w_base_1x = w_base_1x.mul(conviction, axis=0)

    eval_idx = close.index[close.index >= pd.Timestamp("2025-01-01 00:00:00", tz="UTC")]
    rets_eval = close.pct_change().reindex(eval_idx).fillna(0.0).values
    fund_eval = funding.reindex(eval_idx).fillna(0.0).values
    w_1x_arr = w_base_1x.reindex(eval_idx).fillna(0.0).values

    def run_sim(
        start_lev: float = 2.5,
        max_lev: float = 5.0,
        step_lev: float = 0.5,
        step_gain_pct: float = 0.30,   # Step up +0.5x every +30% new HWM gain
        dd_step_down: float = 0.10,    # Step down -0.5x if drawdown > 10%
        dd_reset: float = 0.18,        # Reset to 2.5x if drawdown > 18%
        reb_hours: int = 8,
        fee_rate: float = 0.0002,
    ):
        n_bars, n_cols = w_1x_arr.shape
        eq = 1000.0
        hwm = 1000.0
        step_anchor_eq = 1000.0
        cur_lev = start_lev
        prev_w = np.zeros(n_cols)

        eq_curve = np.zeros(n_bars)
        lev_curve = np.zeros(n_bars)

        for t in range(n_bars):
            # 1. Apply bar return on prev_w (strictly causal shift(1))
            if t > 0:
                bar_r = np.sum(prev_w * rets_eval[t]) - np.sum(prev_w * fund_eval[t])
                eq = eq * (1.0 + bar_r)
                if eq > hwm:
                    hwm = eq

            # 2. Stepped Leverage Escalation / De-escalation
            cur_dd = (hwm - eq) / hwm if hwm > 0 else 0.0
            if step_lev > 0.0:
                if cur_dd >= dd_reset:
                    cur_lev = start_lev
                    step_anchor_eq = eq
                elif cur_dd >= dd_step_down:
                    cur_lev = max(start_lev, cur_lev - step_lev)
                    step_anchor_eq = eq
                elif cur_dd <= 0.03 and eq >= step_anchor_eq * (1.0 + step_gain_pct):
                    if cur_lev < max_lev:
                        cur_lev = min(max_lev, cur_lev + step_lev)
                        step_anchor_eq = eq

            # 3. Rebalance every reb_hours
            if t % reb_hours == 0:
                max_single = (0.45 / 2.2) * cur_lev
                row = w_1x_arr[t] * cur_lev
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

                db = (0.04 / 2.2) * cur_lev
                delta = np.abs(desired - prev_w)
                upd = np.where((desired == 0.0) | (delta >= db), desired, prev_w)
                g = np.sum(np.abs(upd))
                if g > cur_lev:
                    upd = upd * (cur_lev / g)

                turn = np.sum(np.abs(upd - prev_w))
                eq = eq * (1.0 - turn * fee_rate)
                prev_w = upd

            eq_curve[t] = eq
            lev_curve[t] = cur_lev

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
            "max_lev": float(np.max(lev_curve)),
            "end_lev": float(lev_curve[-1]),
            "pos_m": int((monthly > 0).sum()),
            "tot_m": len(monthly),
            "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
        }

    print("=" * 158)
    print("7-FACTOR CAUSAL ONLINE META-ALLOCATOR: FIXED vs STEPPED LEVERAGE (2.5x -> 5.0x) OVER ALL OF 2025 AND 2026 (22 MONTHS)")
    print("=" * 158)
    print(
        f"{'Leverage Configuration':<44} | {'2025 Ret':>9} | {'2026 Ret':>9} | {'Last 6M':>8} | {'Total 22M':>10} | "
        f"{'Final Eq':>10} | {'DailyAvg':>8} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7} | {'PosMos':>6}"
    )
    print("-" * 158)

    configs = [
        ("1. Fixed 2.2x Gross", 2.2, 2.2, 0.0, 0.30, 0.10, 0.18),
        ("2. Fixed 2.5x Gross (Starting Level)", 2.5, 2.5, 0.0, 0.30, 0.10, 0.18),
        ("3. Fixed 3.0x Gross", 3.0, 3.0, 0.0, 0.30, 0.10, 0.18),
        ("4. Fixed 3.5x Gross", 3.5, 3.5, 0.0, 0.30, 0.10, 0.18),
        ("5. Fixed 4.0x Gross", 4.0, 4.0, 0.0, 0.30, 0.10, 0.18),
        ("6. Fixed 4.5x Gross", 4.5, 4.5, 0.0, 0.30, 0.10, 0.18),
        ("7. Fixed 5.0x Gross (Full 5x Target)", 5.0, 5.0, 0.0, 0.30, 0.10, 0.18),
        ("8. Stepped 2.5x->5.0x (+0.5x per +25% Gain, DD Guard)", 2.5, 5.0, 0.5, 0.25, 0.10, 0.18),
        ("9. Stepped 2.5x->5.0x (+0.5x per +20% Gain, DD Guard)", 2.5, 5.0, 0.5, 0.20, 0.08, 0.15),
        ("10. Stepped 2.5x->5.0x (+0.5x per +15% Gain, No Reset)", 2.5, 5.0, 0.5, 0.15, 0.99, 0.99),
    ]

    res_dict = {}
    for name, s_l, m_l, stp, g_pct, dd_d, dd_r in configs:
        r = run_sim(s_l, m_l, stp, g_pct, dd_d, dd_r)
        res_dict[name] = r
        pm = f"{r['pos_m']}/{r['tot_m']}"
        print(
            f"{name:<44} | {r['ret_2025']:>+8.1f}% | {r['ret_2026']:>+8.1f}% | {r['ret_last6m']:>+7.1f}% | {r['total_ret']:>+9.1f}% | "
            f"${r['final_eq']:>9,.0f} | {r['daily_avg']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['mdd']:>6.2f}% | {pm:>6}"
        )

    print("\n--- Monthly Breakdown Across All 22 Months (2025-01 to 2026-10) ---")
    for k in [
        "2. Fixed 2.5x Gross (Starting Level)",
        "7. Fixed 5.0x Gross (Full 5x Target)",
        "8. Stepped 2.5x->5.0x (+0.5x per +25% Gain, DD Guard)",
        "10. Stepped 2.5x->5.0x (+0.5x per +15% Gain, No Reset)",
    ]:
        r = res_dict[k]
        m25 = " ".join([f"{m[5:]}:{v:+.1f}%" for m, v in r["monthly"].items() if m.startswith("2025")])
        m26 = " ".join([f"{m[5:]}:{v:+.1f}%" for m, v in r["monthly"].items() if m.startswith("2026")])
        print(f"\n[{k}] (Avg Lev: {r['avg_lev']:.2f}x | End Lev: {r['end_lev']:.1f}x | Pos Months: {r['pos_m']}/{r['tot_m']}):")
        print(f"  2025: {m25}")
        print(f"  2026: {m26}")


if __name__ == "__main__":
    main()
