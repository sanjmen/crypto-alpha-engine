from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from test_7of7_months import *

# Smooth regime transition based on cs_autocorr_48h (which is 100% causal at bar t)
# Let alpha_mom_weight = clip((cs_autocorr_48h - (-0.10)) / (0.02 - (-0.10)), -0.4, 1.0)
# When cs_autocorr_48h >= +0.02 -> weight = +1.0 (Full Pullback Momentum)
# When cs_autocorr_48h == -0.06 -> weight = +0.33 (De-risked Momentum + Carry)
# When cs_autocorr_48h <= -0.12 -> weight = -0.35 (Cross-Sectional Mean Reversion + Carry!)

smooth_results = {}
for lo_ac, hi_ac, min_tilt in [
    (-0.10, 0.00, 0.0),
    (-0.10, 0.00, -0.35),
    (-0.12, -0.02, 0.0),
    (-0.12, -0.02, -0.35),
    (-0.08, 0.02, -0.30),
]:
    slope = (cs_autocorr_48h - lo_ac) / (hi_ac - lo_ac)
    tilt = slope.clip(min_tilt, 1.0)  # +1.0 in momentum regime, negative in strong mean-reversion regime

    # Dynamic score: tilt * PullbackMom_Blend + 0.30 * CarryScore
    dyn_score = mom_pullback_blend.mul(tilt, axis=0) + 0.30 * carry_score
    w_base = make_continuous_neutral_weights(dyn_score, inv_vol, gross_leverage=2.2, power=1.8)

    # Scale gross leverage proportionally to |tilt| + 0.35 so low-conviction transition zones use less leverage
    lev_mult = (tilt.abs() + 0.35).clip(0.45, 1.0)
    w_final = to_w19(w_base.mul(lev_mult, axis=0))

    label = f"SmoothAC[{lo_ac:.2f},{hi_ac:.2f}]_Min{min_tilt:+.2f}_Reb4h"
    smooth_results[label] = simulate_portfolio(
        w_final, close, funding,
        rebalance_every_n_hours=4,
        max_single_weight=0.45,
        max_gross_leverage=2.2,
        deadband=0.04,
    )

print("=" * 165)
print(f"{'Strategy / Engine':<44} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'Calmar':>6} | {'MaxDD':>7} | {'WinDays':>7} | {'PosMos':>6}")
print("-" * 165)
for name, r in smooth_results.items():
    pm = f"{r['positive_months']}/{r['total_months']}"
    print(
        f"{name:<44} | ${r['final_equity']:>8.2f} | {r['total_ret_pct']:>+7.2f}% | "
        f"{r['avg_daily_ret_pct']:>+8.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
        f"{r['calmar']:>6.2f} | {r['max_dd_pct']:>6.2f}% | {r['win_days_pct']:>6.1f}% | {pm:>6}"
    )

print("\n" + "=" * 165)
print("MONTHLY RETURN BREAKDOWN (%)")
print("-" * 165)
months = list(next(iter(smooth_results.values()))["monthly_rets"].keys())
print(f"{'Strategy / Engine':<44} | " + " | ".join(f"{m:>8}" for m in months))
print("-" * 165)
for name, r in smooth_results.items():
    m_str = " | ".join(f"{r['monthly_rets'].get(m, 0.0):>+7.2f}%" for m in months)
    print(f"{name:<44} | {m_str}")
