"""
Tests Causal Online Factor Allocation (Factor Momentum + Cross-Sectional Autocorrelation + Dispersion Regime Sensors)
across the full 22-month dataset (All 2025 + All 2026: 2025-01-01 to 2026-10-20, 658 Days).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from diagnose_2025_2026_all_mechanisms import sim_22m
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

    # Orthogonal Factor Library
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

    # Unit-leverage portfolio weights and strictly causal hourly net returns for each factor
    w_unit = {k: make_continuous_neutral_weights(sc, inv_vol, gross_leverage=1.0, power=1.7) for k, sc in factor_scores.items()}
    f_bar_ret = {}
    for k, w in w_unit.items():
        # Include approximate turnover fee drag in factor's realized return so noisy factors aren't over-rewarded!
        w_exec = w.shift(1).fillna(0.0)
        turn = w_exec.diff().abs().sum(axis=1).fillna(0.0)
        f_bar_ret[k] = (w_exec * rets_1h).sum(axis=1) - (w_exec * funding).sum(axis=1) - turn * 0.0001
    df_frets = pd.DataFrame(f_bar_ret)

    # Test multi-horizon causal factor t-stat / Sharpe allocators
    print("=" * 145)
    print("CAUSAL ONLINE FACTOR META-ALLOCATOR BENCHMARK (ALL 2025 + ALL 2026: 2025-01-01 to 2026-10-20, 22 MONTHS)")
    print("=" * 145)

    for lookbacks, temp, reb_h, prior_macro, prior_carry in [
        ((48, 120, 240), 1.5, 8, 0.35, 0.15),
        ((36, 96, 240),  2.0, 8, 0.30, 0.15),
        ((24, 72, 168),  2.0, 8, 0.25, 0.15),
        ((24, 72, 168),  2.5, 4, 0.25, 0.15),
        ((48, 168, 336), 2.0, 8, 0.40, 0.20),
        ((72, 168, 504), 2.0, 8, 0.45, 0.20),
    ]:
        # Compute blended rolling t-stat for each factor
        t_df = pd.DataFrame(0.0, index=idx, columns=df_frets.columns)
        for lb in lookbacks:
            m = df_frets.rolling(lb).mean()
            s = df_frets.rolling(lb).std().replace(0, 1e-8)
            t_df += (m / s * np.sqrt(lb)).fillna(0.0) / len(lookbacks)

        # Positive-part ReLU + temperature power + structural prior on Macro, Carry, and Size
        priors = pd.Series({"macro": prior_macro, "med": 0.15, "fast": 0.15, "rev": 0.10, "carry": prior_carry, "liq": 0.10, "size": 0.15})
        raw_fw = np.maximum(0.0, t_df + priors) ** temp
        # Normalize factor weights per row, but scale down total conviction when all factors have negative t-stats!
        fw_sum = raw_fw.sum(axis=1).replace(0, 1e-8)
        fw_norm = raw_fw.div(fw_sum, axis=0)
        conviction = (raw_fw.sum(axis=1) / (priors.sum() ** temp)).clip(0.35, 1.15)

        # Combine portfolio weights across factors
        comb_w = pd.DataFrame(0.0, index=idx, columns=TOP15)
        for k in factor_scores.keys():
            comb_w += w_unit[k].mul(fw_norm[k], axis=0)

        # Re-normalize combined weights to target gross leverage = 2.2 * conviction
        comb_score_w = make_continuous_neutral_weights(comb_w, pd.DataFrame(1.0, index=idx, columns=TOP15), gross_leverage=2.2, power=1.3)
        comb_score_w = comb_score_w.mul(conviction, axis=0)

        r = sim_22m(comb_score_w, close, funding, reb_hours=reb_h, gross_lev=2.2, max_single=0.45, deadband=0.04)
        lbl = f"LB={lookbacks} T={temp} Reb={reb_h}h PrM={prior_macro}"
        print(
            f"{lbl:<44} | 2025:{r['ret_2025']:+7.1f}% | 2026:{r['ret_2026']:+7.1f}% | "
            f"Total 22M:{r['total_ret']:+8.1f}% (${r['final_eq']:,.0f}) | Daily:{r['daily_avg']:+6.3f}% | "
            f"Shp:{r['sharpe']:5.2f} | MDD:{r['mdd']:6.2f}% | PosM:{r['pos_m']:2d}/{r['tot_m']}"
        )


if __name__ == "__main__":
    main()
