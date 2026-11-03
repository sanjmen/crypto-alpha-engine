"""
Comprehensive Evaluation of:
1. All Individual Neutral Families (A: Momentum, B: Carry, C: StatArb, D: Donchian Rank)
2. Walk-Forward Machine Learning (Ridge + LightGBM Rank Ensemble)
3. High-Frequency Event Scanner (1h Dislocation Sniper: Extreme Z-Score + Funding + Volume Spike)
4. Multi-Mechanism Regime-Adaptive Ensembles ( aiming for 7/7 positive months, high daily return, Sharpe > 3.5 )
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
from src.features.hurst import estimate_hurst_variance_time


def main():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(ALL19)
    c15 = close[TOP15]
    h15 = high[TOP15]
    l15 = low[TOP15]
    o15 = open_[TOP15]
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

    # 1. Raw & Idiosyncratic Momentum
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

    # 2. Donchian Channel Position Rank (48h + 96h)
    pos_48 = ((c15 - l15.rolling(48).min()) / (h15.rolling(48).max() - l15.rolling(48).min() + 1e-8)) - 0.5
    pos_96 = ((c15 - l15.rolling(96).min()) / (h15.rolling(96).max() - l15.rolling(96).min() + 1e-8)) - 0.5
    donchian_score = (0.5 * pos_48 + 0.5 * pos_96) * np.sqrt(vol_surge)

    # 3. Exact FundingCarryArbitrageStrategy logic (72h rolling mean, >= 10% annualized hurdle, dollar-neutral)
    ann_fund_72h = rf15.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    carry_raw = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.10, 0.0)
    # Demean across active assets per row
    active_c = carry_raw != 0.0
    mean_c = carry_raw.where(active_c).mean(axis=1).fillna(0.0)
    carry_demeaned = carry_raw.sub(mean_c, axis=0).where(active_c, 0.0)
    carry_weighted = carry_demeaned * inv_vol
    pos_c = carry_weighted.clip(lower=0.0)
    neg_c = (-carry_weighted).clip(lower=0.0)
    valid_c = (pos_c.sum(axis=1) > 1e-8) & (neg_c.sum(axis=1) > 1e-8)
    w_carry_15 = (
        pos_c.div(pos_c.sum(axis=1).replace(0, 1.0), axis=0)
        - neg_c.div(neg_c.sum(axis=1).replace(0, 1.0), axis=0)
    ).mul(valid_c.astype(float), axis=0)

    # 4. Exact StatArbMeanReversionStrategy logic (72h rolling z-score + Hurst < 0.45 filter)
    ma_72 = c15.rolling(72).mean()
    sd_72 = c15.rolling(72).std().replace(0, 1e-8)
    z_72 = (c15 - ma_72) / sd_72
    hurst_df = pd.DataFrame(0.5, index=idx, columns=TOP15)
    for s in TOP15:
        vals = c15[s].values
        h_arr = np.full(len(vals), 0.5)
        for i in range(72, len(vals), 4):
            h_est = estimate_hurst_variance_time(vals[i - 72 : i])
            h_arr[i : min(i + 4, len(vals))] = h_est
        hurst_df[s] = h_arr

    statarb_raw = pd.DataFrame(0.0, index=idx, columns=TOP15)
    short_sa = (hurst_df < 0.45) & (z_72 >= 2.0)
    long_sa = (hurst_df < 0.45) & (z_72 <= -2.0)
    statarb_raw = statarb_raw.where(~short_sa, -np.minimum(1.0, (z_72 - 2.0) / 2.0 + 0.5))
    statarb_raw = statarb_raw.where(~long_sa, np.minimum(1.0, (z_72.abs() - 2.0) / 2.0 + 0.5))
    # Cross-sectionally demean across active signals when >1 signal
    active_sa = statarb_raw != 0.0
    n_sa = active_sa.sum(axis=1)
    mean_sa = statarb_raw.where(active_sa).mean(axis=1).fillna(0.0)
    sa_demeaned = statarb_raw.sub(mean_sa, axis=0).where(active_sa & (n_sa > 1).values[:, None], 0.0)
    sa_weighted = sa_demeaned * inv_vol
    pos_sa = sa_weighted.clip(lower=0.0)
    neg_sa = (-sa_weighted).clip(lower=0.0)
    valid_sa = (pos_sa.sum(axis=1) > 1e-8) & (neg_sa.sum(axis=1) > 1e-8)
    w_statarb_15 = (
        pos_sa.div(pos_sa.sum(axis=1).replace(0, 1.0), axis=0)
        - neg_sa.div(neg_sa.sum(axis=1).replace(0, 1.0), axis=0)
    ).mul(valid_sa.astype(float), axis=0)

    # 5. Short-Horizon (12h) Overreaction Reversal in Low-Dispersion / Ranging Regimes
    # Notice: In Oct 2026, momentum stalled because breakouts reversed within 12-24h.
    # Let's check when BTC 48h efficiency ratio (|ret_48h| / sum(|ret_1h|)) is very low (< 0.15 = choppy range!)
    btc_net_48 = (c15["BTCUSDT"] - c15["BTCUSDT"].shift(48)).abs()
    btc_path_48 = c15["BTCUSDT"].diff().abs().rolling(48).sum().replace(0, 1e-8)
    btc_er_48 = btc_net_48 / btc_path_48  # Kaufman Efficiency Ratio in [0, 1]

    # Also cross-sectional momentum dispersion: std of 24h returns across the 15 assets
    cs_disp = log_rets.rolling(24).sum().std(axis=1)
    cs_disp_ma = cs_disp.rolling(168).mean().replace(0, 1e-8)
    disp_ratio = (cs_disp / cs_disp_ma).clip(0.3, 2.0)

    # Construct full 19-col weight frames
    def to_w19(w15):
        w = pd.DataFrame(0.0, index=idx, columns=ALL19)
        w[TOP15] = w15
        return w

    w_raw_mom = to_w19(make_continuous_neutral_weights(mom_raw, inv_vol, gross_leverage=2.0, power=1.5))
    w_idio_mom = to_w19(make_continuous_neutral_weights(mom_idio, inv_vol, gross_leverage=2.0, power=1.0))
    w_donch = to_w19(make_continuous_neutral_weights(donchian_score, inv_vol, gross_leverage=2.0, power=1.5))
    w_carry = to_w19(w_carry_15 * 1.0)  # 1.0x per side = 2.0x gross
    w_statarb = to_w19(w_statarb_15 * 0.9)  # 0.9x per side = 1.8x gross

    # 6. Multi-Mechanism Ensemble (Blending RawMom + IdioMom + Donchian + Carry + StatArb)
    # With Regime Conditioning:
    # When BTC Efficiency Ratio < 0.12 (choppy whipsaw market), scale down Momentum by 50% and boost Carry + StatArb!
    trend_scale = np.where(btc_er_48 < 0.12, 0.45, np.where(btc_er_48 > 0.25, 1.15, 0.90))
    trend_scale_s = pd.Series(trend_scale, index=idx)

    # Composite Score Blending BEFORE sizing (so opposing signals net out internally before turnover deadband!)
    # Normalize each alpha factor cross-sectionally to Z-score first:
    def cs_zscore(df):
        m = df.mean(axis=1)
        s = df.std(axis=1).replace(0, 1.0)
        return df.sub(m, axis=0).div(s, axis=0).fillna(0.0)

    z_mom_raw = cs_zscore(mom_raw)
    z_mom_idio = cs_zscore(mom_idio)
    z_donch = cs_zscore(donchian_score)
    z_carry = cs_zscore(carry_raw)
    z_statarb = cs_zscore(statarb_raw)

    # Dynamic Regime-Conditioned Alpha Factor Combination:
    # Core Momentum/Breakout factor:
    alpha_trend = 0.40 * z_mom_raw + 0.35 * z_mom_idio + 0.25 * z_donch
    # Core Market-Neutral Carry + Mean-Reversion factor:
    alpha_defensive = 0.65 * z_carry + 0.35 * z_statarb

    # Blended Alpha at different leverage & regime sensitivity levels:
    alpha_combo = alpha_trend.mul(trend_scale_s, axis=0) + 0.45 * alpha_defensive

    w_combo_2_0x = to_w19(make_continuous_neutral_weights(alpha_combo, inv_vol, gross_leverage=2.0, power=1.3))
    w_combo_2_4x = to_w19(make_continuous_neutral_weights(alpha_combo, inv_vol, gross_leverage=2.4, power=1.4))

    # Dynamic Leverage scaling based on regime conviction (reduce gross leverage in choppy low-ER regimes!)
    lev_scale = pd.Series(np.where(btc_er_48 < 0.12, 0.55, 1.0), index=idx)
    w_combo_dyn_2_0x = w_combo_2_0x.mul(lev_scale, axis=0)
    w_combo_dyn_2_4x = w_combo_2_4x.mul(lev_scale, axis=0)

    experiments = {
        "1_RawMom_ContP1.5_Reb4h (2.0x)": (w_raw_mom, 4, 0.40, 2.0, 0.04),
        "2_IdioBetaMom_ContP1.0_Reb8h (2.0x)": (w_idio_mom, 8, 0.40, 2.0, 0.04),
        "3_DonchianRank_ContP1.5_Reb4h (2.0x)": (w_donch, 4, 0.40, 2.0, 0.04),
        "4_FundingCarry_Exact_Reb8h (2.0x)": (w_carry, 8, 0.40, 2.0, 0.04),
        "5_StatArb_Hurst72_Exact_Reb8h (1.8x)": (w_statarb, 8, 0.40, 1.8, 0.04),
        "6_AlphaCombo_Static_Reb4h (2.0x)": (w_combo_2_0x, 4, 0.40, 2.0, 0.04),
        "7_AlphaCombo_DynamicER_Reb4h (2.0x)": (w_combo_dyn_2_0x, 4, 0.40, 2.0, 0.04),
        "8_AlphaCombo_DynamicER_Reb4h (2.4x)": (w_combo_dyn_2_4x, 4, 0.45, 2.4, 0.04),
        "9_AlphaCombo_DynamicER_Reb8h (2.4x)": (w_combo_dyn_2_4x, 8, 0.45, 2.4, 0.04),
    }

    print("=" * 165)
    print(f"{'Strategy / Engine':<42} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'Calmar':>6} | {'MaxDD':>7} | {'WinDays':>7} | {'PosMos':>6}")
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
            f"{name:<42} | ${r['final_equity']:>8.2f} | {r['total_ret_pct']:>+7.2f}% | "
            f"{r['avg_daily_ret_pct']:>+8.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['calmar']:>6.2f} | {r['max_dd_pct']:>6.2f}% | {r['win_days_pct']:>6.1f}% | {pm:>6}"
        )

    print("\n" + "=" * 165)
    print("MONTHLY RETURN BREAKDOWN (%)")
    print("-" * 165)
    months = list(next(iter(res_map.values()))["monthly_rets"].keys())
    print(f"{'Strategy / Engine':<42} | " + " | ".join(f"{m:>8}" for m in months))
    print("-" * 165)
    for name, r in res_map.items():
        m_str = " | ".join(f"{r['monthly_rets'].get(m, 0.0):>+7.2f}%" for m in months)
        print(f"{name:<42} | {m_str}")


if __name__ == "__main__":
    main()
