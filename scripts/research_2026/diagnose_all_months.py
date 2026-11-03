"""
Testing Dynamic Cross-Sectional IC Regime Switching, Volatility-Targeted Sizing,
and High-Frequency Short-Horizon Alphas across all 7 months (Apr-Oct 2026).
Strict out-of-sample causality: all signals and IC regime metrics at bar t use only data <= t,
and execute on bar t+1 returns.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))

from explore_high_return_strategies import load_matrices, TOP15, ALL19
from deep_strategy_research import simulate_portfolio
from test_neutral_alphas import make_dollar_neutral_weights, make_continuous_neutral_weights


def main():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(ALL19)
    c15 = close[TOP15]
    h15 = high[TOP15]
    l15 = low[TOP15]
    dv15 = dollar_vol[TOP15]
    rf15 = raw_funding[TOP15]

    rets_1h = c15.pct_change().fillna(0.0)
    log_rets = np.log(c15 / c15.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_24h).clip(upper=100.0)

    # Rolling 72h Beta to BTC
    btc_ret = rets_1h["BTCUSDT"]
    cov_btc = rets_1h.rolling(72).cov(btc_ret)
    var_btc = btc_ret.rolling(72).var().replace(0, 1e-8)
    beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5)
    resid_ret = rets_1h.sub(beta_btc.mul(btc_ret, axis=0))

    # Volume surge
    vol_surge = (dv15.rolling(6).mean() / (dv15.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)

    # 1. Core Alpha Factors
    mom_raw = (
        0.25 * (log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12)))
        + 0.35 * (log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24)))
        + 0.25 * (log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72)))
        + 0.15 * (log_rets.rolling(168).sum() / (vol_24h * np.sqrt(168)))
    ) * np.sqrt(vol_surge)

    mom_idio = (
        0.25 * (resid_ret.rolling(12).sum() / (vol_24h * np.sqrt(12)))
        + 0.35 * (resid_ret.rolling(24).sum() / (vol_24h * np.sqrt(24)))
        + 0.25 * (resid_ret.rolling(72).sum() / (vol_24h * np.sqrt(72)))
        + 0.15 * (resid_ret.rolling(168).sum() / (vol_24h * np.sqrt(168)))
    ) * np.sqrt(vol_surge)

    mom_blend = 0.55 * mom_raw + 0.45 * mom_idio

    # Short-horizon 8h-16h Reversal Factor (fading short-term overextensions)
    rev_short = -(
        0.50 * (resid_ret.rolling(8).sum() / (vol_24h * np.sqrt(8)))
        + 0.50 * (resid_ret.rolling(20).sum() / (vol_24h * np.sqrt(20)))
    )

    # Funding Carry Factor (with Volatility Targeting, not forced 2.0x leverage!)
    ann_fund_72h = rf15.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.10, 0.0)

    def to_w19(w15):
        w = pd.DataFrame(0.0, index=idx, columns=ALL19)
        w[TOP15] = w15
        return w

    # Base continuous weights (at 2.0x gross)
    w15_mom_raw = make_continuous_neutral_weights(mom_raw, inv_vol, gross_leverage=2.0, power=1.5)
    w15_mom_idio = make_continuous_neutral_weights(mom_idio, inv_vol, gross_leverage=2.0, power=1.2)
    w15_mom_blend = make_continuous_neutral_weights(mom_blend, inv_vol, gross_leverage=2.0, power=1.5)
    w15_rev = make_continuous_neutral_weights(rev_short, inv_vol, gross_leverage=1.5, power=1.2)

    # Volatility-Targeted Carry (scales down when asset volatility is high, max 0.8x gross)
    w15_carry_unit = make_continuous_neutral_weights(carry_score, inv_vol, gross_leverage=0.8, power=1.0)

    # -------------------------------------------------------------------------
    # Compute Realized Cross-Sectional Momentum IC / Performance Regime (Strictly Causal!)
    # At bar t, we know the hourly return of w15_mom_blend from t-1 to t:
    # r_mom(t) = sum_i ( w15_mom_blend(t-1, i) * rets_1h(t, i) )
    # This is 100% observable at time t before choosing weights for t -> t+1!
    # -------------------------------------------------------------------------
    realized_mom_1h = (w15_mom_blend.shift(1).fillna(0.0) * rets_1h).sum(axis=1)
    realized_rev_1h = (w15_rev.shift(1).fillna(0.0) * rets_1h).sum(axis=1)

    # Rolling Sharpe / t-stat of Momentum over past W hours (e.g. 48h, 72h, 120h)
    experiments = {}

    # Baseline pure momentum blends
    experiments["Base_RawMom_P1.5_Reb4h (2.0x)"] = (to_w19(w15_mom_raw), 4, 0.40, 2.0, 0.04)
    experiments["Base_BlendMom_P1.5_Reb4h (2.0x)"] = (to_w19(w15_mom_blend), 4, 0.40, 2.0, 0.04)
    experiments["Base_BlendMom_P1.5_Reb4h (2.3x)"] = (to_w19(w15_mom_blend * 1.15), 4, 0.45, 2.3, 0.04)

    # Portfolio Volatility Targeting on BlendMom:
    # Realized 48h annualized volatility of the long-short portfolio itself!
    port_vol_48h = realized_mom_1h.rolling(48).std() * np.sqrt(24 * 365)
    for target_pvol in [0.35, 0.45, 0.55]:
        vol_scalar = (target_pvol / (port_vol_48h + 1e-6)).clip(0.3, 1.25)
        w_vt = w15_mom_blend.mul(vol_scalar, axis=0) + 0.35 * w15_carry_unit
        experiments[f"VolTarget_BlendMom_TV{int(target_pvol*100)}_Reb4h"] = (to_w19(w_vt), 4, 0.45, 2.3, 0.04)

    # Dynamic Cross-Sectional IC / Equity-Slope Regime Gate:
    for ic_win in [48, 72, 96, 120]:
        mom_roll_ret = realized_mom_1h.rolling(ic_win).sum()
        mom_roll_std = realized_mom_1h.rolling(ic_win).std() * np.sqrt(ic_win) + 1e-8
        mom_z = mom_roll_ret / mom_roll_std  # Rolling t-stat of cross-sectional momentum

        # Mode A: Smooth sigmoid/linear scaling of Momentum + Carry + Reversal when Mom Z < -0.5
        mom_weight = np.where(mom_z > 0.5, 1.20, np.where(mom_z > -0.3, 0.90, np.where(mom_z > -1.0, 0.35, 0.0)))
        rev_weight = np.where(mom_z < -0.8, 0.60, 0.0)
        carry_weight = 0.35

        w_adaptive = (
            w15_mom_blend.mul(pd.Series(mom_weight, index=idx), axis=0)
            + w15_rev.mul(pd.Series(rev_weight, index=idx), axis=0)
            + w15_carry_unit * carry_weight
        )
        experiments[f"Adaptive_ICGate_W{ic_win}h_Reb4h (2.3x)"] = (to_w19(w_adaptive), 4, 0.45, 2.3, 0.04)
        experiments[f"Adaptive_ICGate_W{ic_win}h_Reb8h (2.3x)"] = (to_w19(w_adaptive), 8, 0.45, 2.3, 0.04)

    # Also test: Combining VolTarget + IC Gate
    for ic_win in [72, 96]:
        mom_roll_ret = realized_mom_1h.rolling(ic_win).sum()
        mom_roll_std = realized_mom_1h.rolling(ic_win).std() * np.sqrt(ic_win) + 1e-8
        mom_z = mom_roll_ret / mom_roll_std
        vol_scalar = (0.45 / (port_vol_48h + 1e-6)).clip(0.4, 1.25)

        mom_w = pd.Series(np.where(mom_z > 0.2, 1.15, np.where(mom_z > -0.5, 0.75, 0.15)), index=idx) * vol_scalar
        rev_w = pd.Series(np.where(mom_z < -0.7, 0.50, 0.0), index=idx)
        w_hybrid = (
            w15_mom_blend.mul(mom_w, axis=0)
            + w15_rev.mul(rev_w, axis=0)
            + w15_carry_unit * 0.35
        )
        experiments[f"Hybrid_VolTarget_ICGate_W{ic_win}h_Reb4h"] = (to_w19(w_hybrid), 4, 0.45, 2.4, 0.04)

    print("=" * 165)
    print(f"{'Strategy / Engine':<44} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'Calmar':>6} | {'MaxDD':>7} | {'WinDays':>7} | {'PosMos':>6}")
    print("-" * 165)

    res_map = {}
    for name, (w_df, reb, max_s, max_g, db) in experiments.items():
        r = simulate_portfolio(
            w_df, close, funding,
            rebalance_every_n_hours=reb,
            max_single_weight=max_s,
            max_gross_leverage=max_g,
            deadband=db,
        )
        res_map[name] = r
        pm = f"{r['positive_months']}/{r['total_months']}"
        print(
            f"{name:<44} | ${r['final_equity']:>8.2f} | {r['total_ret_pct']:>+7.2f}% | "
            f"{r['avg_daily_ret_pct']:>+8.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['calmar']:>6.2f} | {r['max_dd_pct']:>6.2f}% | {r['win_days_pct']:>6.1f}% | {pm:>6}"
        )

    print("\n" + "=" * 165)
    print("MONTHLY RETURN BREAKDOWN (%)")
    print("-" * 165)
    months = list(next(iter(res_map.values()))["monthly_rets"].keys())
    print(f"{'Strategy / Engine':<44} | " + " | ".join(f"{m:>8}" for m in months))
    print("-" * 165)
    for name, r in res_map.items():
        m_str = " | ".join(f"{r['monthly_rets'].get(m, 0.0):>+7.2f}%" for m in months)
        print(f"{name:<44} | {m_str}")


if __name__ == "__main__":
    main()
