"""
Full Monthly & Daily Verification of the Cross-Sectional Autocorrelation Regime-Switching
Pullback Momentum + Idiosyncratic Momentum + Carry + Intraday Dislocation Engine.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from explore_high_return_strategies import load_matrices, TOP15, ALL19
from deep_strategy_research import simulate_portfolio
from test_neutral_alphas import make_continuous_neutral_weights

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
vol_surge = (dv15.rolling(6).mean() / (dv15.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)

# Rolling 72h Beta to BTC & Idiosyncratic Returns
btc_ret = rets_1h["BTCUSDT"]
cov_btc = rets_1h.rolling(72).cov(btc_ret)
var_btc = btc_ret.rolling(72).var().replace(0, 1e-8)
beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5)
resid_ret = rets_1h.sub(beta_btc.mul(btc_ret, axis=0))

# 1. Raw Volatility-Normalized Returns
r_4h = log_rets.rolling(4).sum() / (vol_24h * np.sqrt(4))
r_12h = log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12))
r_24h = log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24))
r_48h = log_rets.rolling(48).sum() / (vol_24h * np.sqrt(48))
r_72h = log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72))

# 2. Idiosyncratic (Beta-Neutral) Volatility-Normalized Returns
ir_4h = resid_ret.rolling(4).sum() / (vol_24h * np.sqrt(4))
ir_12h = resid_ret.rolling(12).sum() / (vol_24h * np.sqrt(12))
ir_24h = resid_ret.rolling(24).sum() / (vol_24h * np.sqrt(24))
ir_48h = resid_ret.rolling(48).sum() / (vol_24h * np.sqrt(48))
ir_72h = resid_ret.rolling(72).sum() / (vol_24h * np.sqrt(72))

# 3. Pullback-Entry Momentum (Raw & Idiosyncratic Blend)
mom_pullback_raw = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) - 0.25 * r_4h
mom_pullback_idio = (0.30 * ir_12h + 0.40 * ir_24h + 0.30 * ir_72h) * np.sqrt(vol_surge) - 0.20 * ir_4h
mom_pullback_blend = 0.60 * mom_pullback_raw + 0.40 * mom_pullback_idio

# 4. Cross-Sectional Return Autocorrelation Sensor (Strictly Causal!)
ret_24h = log_rets.rolling(24).sum()
prev_ret_24h = ret_24h.shift(24)
m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
m_curr = ret_24h.sub(ret_24h.mean(axis=1), axis=0)
num = (m_prev * m_curr).sum(axis=1)
den = np.sqrt((m_prev**2).sum(axis=1) * (m_curr**2).sum(axis=1)) + 1e-9
raw_autocorr = num / den
cs_autocorr_48h = raw_autocorr.rolling(48).mean().fillna(0.0)

# 5. Mean-Reversion & Carry Factors (for Negative Autocorrelation Whipsaw Regime)
rev_score = -(0.50 * r_24h + 0.50 * r_48h)
ann_fund_72h = rf15.rolling(72, min_periods=1).mean() * 3.0 * 365.0
carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

def to_w19(w15):
    w = pd.DataFrame(0.0, index=idx, columns=ALL19)
    w[TOP15] = w15
    return w

models = {}

# Model 1: Pure Pullback-Entry Momentum (Raw, Reb 4h, 2.2x)
w1 = to_w19(make_continuous_neutral_weights(mom_pullback_raw, inv_vol, gross_leverage=2.2, power=1.8))
models["1_PullbackMom_Raw_Reb4h (2.2x)"] = (w1, 4, 0.45, 2.2, 0.04)

# Model 2: Pure Pullback-Entry Momentum (Raw + Idio Blend, Reb 4h, 2.2x)
w2 = to_w19(make_continuous_neutral_weights(mom_pullback_blend, inv_vol, gross_leverage=2.2, power=1.8))
models["2_PullbackMom_Blend_Reb4h (2.2x)"] = (w2, 4, 0.45, 2.2, 0.04)

# Model 3: Autocorr-Gated Pullback Momentum + Carry De-Risking (Conservative in Chop, Reb 4h)
ac_gate = (cs_autocorr_48h > -0.05).astype(float)
score_m3 = mom_pullback_raw.mul(ac_gate, axis=0) + 0.25 * carry_score
w3 = to_w19(make_continuous_neutral_weights(score_m3, inv_vol, gross_leverage=2.2, power=1.8))
scale_m3 = pd.Series(np.where(cs_autocorr_48h > -0.05, 1.0, 0.35), index=idx)
w3 = w3.mul(scale_m3, axis=0)
models["3_AutocorrGated_PullbackMom+Carry_Reb4h"] = (w3, 4, 0.45, 2.2, 0.04)

# Model 4: Autocorr-Gated Blend (Raw+Idio Pullback Mom in Trend + Mild Reversion & Carry in Chop)
# Smooth transition based on cs_autocorr_48h:
w_mom_part = make_continuous_neutral_weights(mom_pullback_blend, inv_vol, gross_leverage=2.2, power=1.8)
w_rev_part = make_continuous_neutral_weights(0.60 * rev_score + 0.40 * carry_score, inv_vol, gross_leverage=1.2, power=1.2)

w4_15 = w_mom_part.mul((cs_autocorr_48h > -0.05).astype(float), axis=0) + w_rev_part.mul((cs_autocorr_48h <= -0.05).astype(float), axis=0)
models["4_RegimeSwitch_BlendMom_To_RevCarry_Reb4h"] = (to_w19(w4_15), 4, 0.45, 2.2, 0.04)

# Model 5: Always-On Blend (75% Autocorr-Gated Pullback Mom + 25% IdioBetaMom_8h)
w_idio_base = make_continuous_neutral_weights(mom_pullback_idio, inv_vol, gross_leverage=2.0, power=1.2)
w5_15 = 0.70 * w4_15 + 0.30 * w_idio_base
models["5_MultiPillar_Apex_Ensemble_Reb4h (2.2x)"] = (to_w19(w5_15), 4, 0.45, 2.2, 0.04)

# Model 6: Fast 2h Rebalance Version of Model 3 & 4
models["6_AutocorrGated_PullbackMom_FastReb2h"] = (w3, 2, 0.45, 2.2, 0.04)

print("=" * 165)
print(f"{'Strategy / Engine':<44} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'Calmar':>6} | {'MaxDD':>7} | {'WinDays':>7} | {'PosMos':>6}")
print("-" * 165)

res_map = {}
for name, (w_df, reb, max_s, max_g, db) in models.items():
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
print("MONTHLY RETURN BREAKDOWN (%) — EVERY SINGLE MONTH FROM APR TO OCT 2026")
print("-" * 165)
months = list(next(iter(res_map.values()))["monthly_rets"].keys())
print(f"{'Strategy / Engine':<44} | " + " | ".join(f"{m:>8}" for m in months))
print("-" * 165)
for name, r in res_map.items():
    m_str = " | ".join(f"{r['monthly_rets'].get(m, 0.0):>+7.2f}%" for m in months)
    print(f"{name:<44} | {m_str}")
