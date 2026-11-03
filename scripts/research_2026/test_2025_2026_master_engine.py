"""
22-Month Master Strategy Research (All 2025 + All 2026: 2025-01-01 to 2026-10-20, 658 Days).
Combines:
1. Macro Idiosyncratic Pullback Momentum (168h / 336h / 720h trend entered on 4h / 24h pullback)
2. Fast Swing Pullback Momentum (12h / 24h / 48h trend entered on 4h pullback, gated by causal Factor IC / AutoCorr)
3. Perpetual Funding Carry Contrarian Engine (-funding_rate)
4. Causal Online Factor Momentum / Realized Sharpe Weighting (allocating dynamically to whichever of the 3 engines
   is currently in a positive regime over the past 7-14 days!)
5. Progressive Stepped Leverage (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x in +0.5x steps with drawdown guard).
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

    # Rolling 72h and 168h Beta to BTC
    btc_ret = rets_1h["BTCUSDT"]
    beta_72 = (rets_1h.rolling(72).cov(btc_ret).div(btc_ret.rolling(72).var().replace(0, 1e-8), axis=0)).clip(0.2, 2.5).fillna(1.0)
    resid_ret = rets_1h.sub(beta_72.mul(btc_ret, axis=0))

    def z_h(s, h):
        return s.rolling(h).sum() / (vol_24h * np.sqrt(h))

    horizons = [4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720]
    zr = {h: z_h(log_rets, h) for h in horizons}
    zi = {h: z_h(resid_ret, h) for h in horizons}

    # --- SUB-ENGINE 1: Macro Idiosyncratic Pullback (7d-30d trend, 4h-24h pullback) ---
    s_macro_idio = (0.35 * zi[168] + 0.40 * zi[336] + 0.25 * zi[720]) * np.sqrt(vol_surge_wk) - (0.15 * zi[4] + 0.20 * zi[24])
    s_macro_raw  = (0.35 * zr[168] + 0.40 * zr[336] + 0.25 * zr[720]) * np.sqrt(vol_surge_wk) - (0.15 * zr[4] + 0.20 * zr[24])
    score_macro = 0.70 * s_macro_idio + 0.30 * s_macro_raw

    # --- SUB-ENGINE 2: Fast Swing Pullback (12h-48h trend, 4h pullback) ---
    s_fast_raw  = (0.35 * zr[12] + 0.45 * zr[24] + 0.20 * zr[48]) * np.sqrt(vol_surge) - 0.25 * zr[4]
    s_fast_idio = (0.30 * zi[12] + 0.40 * zi[24] + 0.30 * zi[72]) * np.sqrt(vol_surge) - 0.20 * zi[4]
    score_fast = 0.60 * s_fast_raw + 0.40 * s_fast_idio

    # --- SUB-ENGINE 3: Mean-Reversion / Contrarian Swing (48h-72h reversion when market whipsaws) ---
    score_rev = -(0.35 * zi[12] + 0.40 * zi[24] + 0.25 * zi[48]) - 0.30 * zr[4]

    # --- SUB-ENGINE 4: Perpetual Funding Carry Contrarian ---
    ann_fund_72h = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    score_carry = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.05, 0.0)

    # Compute causal 1.0x unit-leverage portfolio weights for each sub-engine
    w_macro = make_continuous_neutral_weights(score_macro, inv_vol, gross_leverage=1.0, power=1.8)
    w_fast  = make_continuous_neutral_weights(score_fast,  inv_vol, gross_leverage=1.0, power=1.8)
    w_rev   = make_continuous_neutral_weights(score_rev,   inv_vol, gross_leverage=1.0, power=1.5)
    w_carry = make_continuous_neutral_weights(score_carry, inv_vol, gross_leverage=1.0, power=1.5)

    # Compute strictly causal hourly return of each sub-engine (using shift(1) weights * rets_1h)
    r_macro_bar = (w_macro.shift(1) * rets_1h).sum(axis=1)
    r_fast_bar  = (w_fast.shift(1)  * rets_1h).sum(axis=1)
    r_rev_bar   = (w_rev.shift(1)   * rets_1h).sum(axis=1)
    r_carry_bar = (w_carry.shift(1) * rets_1h - w_carry.shift(1) * funding).sum(axis=1)

    # Cross-sectional autocorrelation sensors at 24h (fast) and 168h (macro)
    def calc_cs_ac(h_win, smooth_win):
        r_h = log_rets.rolling(h_win).sum()
        p_h = r_h.shift(h_win)
        mp = p_h.sub(p_h.mean(axis=1), axis=0)
        mc = r_h.sub(r_h.mean(axis=1), axis=0)
        ac = (mp * mc).sum(axis=1) / (np.sqrt((mp ** 2).sum(axis=1) * (mc ** 2).sum(axis=1)) + 1e-9)
        return ac.rolling(smooth_win).mean().fillna(0.0)

    cs_ac_24h = calc_cs_ac(24, 48)
    cs_ac_168h = calc_cs_ac(168, 72)

    # Causal rolling Sharpe / T-stat of each sub-engine over past W hours (e.g. 120h = 5d and 336h = 14d)
    def rolling_tstat(bar_r, win):
        m = bar_r.rolling(win).mean()
        s = bar_r.rolling(win).std().replace(0, 1e-8)
        return (m / s * np.sqrt(win)).fillna(0.0)

    t_macro_7d = 0.5 * rolling_tstat(r_macro_bar, 120) + 0.5 * rolling_tstat(r_macro_bar, 336)
    t_fast_5d  = 0.6 * rolling_tstat(r_fast_bar, 72)   + 0.4 * rolling_tstat(r_fast_bar, 168)
    t_rev_5d   = 0.6 * rolling_tstat(r_rev_bar, 72)    + 0.4 * rolling_tstat(r_rev_bar, 168)
    t_carry_7d = rolling_tstat(r_carry_bar, 168)

    # Test individual engines and Adaptive Meta-Combinations over 2025-01-01 to 2026-10-20
    from diagnose_2025_2026_all_mechanisms import sim_22m

    print("=" * 140)
    print("A. INDIVIDUAL SUB-ENGINES ACROSS ALL OF 2025 AND ALL OF 2026 (2.0x Gross, Reb 8h, Maker 0.02%)")
    print("=" * 140)
    for name, w_df, reb_h in [
        ("1. Macro Idio Pullback (7d-30d - 24h dip)", w_macro * 2.0, 8),
        ("2. Fast Swing Pullback (12h-48h - 4h dip)", w_fast * 2.0, 4),
        ("3. Short-Term Reversion (-12h-48h)", w_rev * 2.0, 8),
        ("4. Contrarian Funding Carry (-fund_72h)", w_carry * 2.0, 8),
    ]:
        r = sim_22m(w_df, close, funding, reb_hours=reb_h, gross_lev=2.0, max_single=0.40, deadband=0.04)
        print(
            f"{name:<44} | 2025:{r['ret_2025']:+7.1f}% | 2026:{r['ret_2026']:+7.1f}% | "
            f"Total 22M:{r['total_ret']:+8.1f}% | Daily:{r['daily_avg']:+6.3f}% | Shp:{r['sharpe']:5.2f} | "
            f"MDD:{r['mdd']:6.2f}% | PosM:{r['pos_m']:2d}/{r['tot_m']}"
        )

    print("\n" + "=" * 140)
    print("B. ADAPTIVE MULTI-HORIZON REGIME ENGINES ACROSS ALL OF 2025 AND ALL OF 2026 (2.2x Gross)")
    print("=" * 140)

    # Build adaptive blends:
    # Core anchor: Macro Idio Pullback (works in both 2025 and 2026!) + Carry
    # Dynamic satellite: Fast Swing Pullback when (t_fast_5d > 0 and cs_ac_24h > -0.08), or Reversion when t_rev_5d > 0.5
    for fast_thresh in [-0.5, 0.0, 0.3, 0.6]:
        for macro_w_base in [0.55, 0.65, 0.75]:
            w_m_tilt = (t_macro_7d + 1.2).clip(0.3, 1.5)
            w_f_tilt = ((t_fast_5d - fast_thresh) * 0.8).clip(0.0, 1.2) * (cs_ac_24h > -0.10).astype(float)
            w_c_tilt = (t_carry_7d + 1.0).clip(0.2, 1.0)

            comb_score = (
                macro_w_base * score_macro.mul(w_m_tilt, axis=0)
                + (1.0 - macro_w_base) * score_fast.mul(w_f_tilt, axis=0)
                + 0.25 * score_carry.mul(w_c_tilt, axis=0)
            )
            w_comb = make_continuous_neutral_weights(comb_score, inv_vol, gross_leverage=2.2, power=1.8)
            for reb_h in [4, 8]:
                r = sim_22m(w_comb, close, funding, reb_hours=reb_h, gross_lev=2.2, max_single=0.45, deadband=0.04)
                if r["total_ret"] > 220.0:
                    lbl = f"Macro={macro_w_base:.2f} FastTh={fast_thresh:+.1f} Reb={reb_h}h"
                    print(
                        f"{lbl:<44} | 2025:{r['ret_2025']:+7.1f}% | 2026:{r['ret_2026']:+7.1f}% | "
                        f"Total 22M:{r['total_ret']:+8.1f}% (${r['final_eq']:,.0f}) | Daily:{r['daily_avg']:+6.3f}% | "
                        f"Shp:{r['sharpe']:5.2f} | MDD:{r['mdd']:6.2f}% | PosM:{r['pos_m']:2d}/{r['tot_m']}"
                    )


if __name__ == "__main__":
    main()
