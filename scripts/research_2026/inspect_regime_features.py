from pathlib import Path
import sys
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from explore_high_return_strategies import load_matrices, TOP15, ALL19
from deep_strategy_research import simulate_portfolio
from test_neutral_alphas import make_continuous_neutral_weights

idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(ALL19)
c15 = close[TOP15]
dv15 = dollar_vol[TOP15]
rf15 = raw_funding[TOP15]
rets_1h = c15.pct_change().fillna(0.0)
log_rets = np.log(c15 / c15.shift(1)).fillna(0.0)
vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
inv_vol = (1.0 / vol_24h).clip(upper=100.0)
vol_surge = (dv15.rolling(6).mean() / (dv15.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)

ret_24h = log_rets.rolling(24).sum()
cs_disp_24h = ret_24h.std(axis=1)

mkt_ret_1h = rets_1h.mean(axis=1)
var_mkt_72h = mkt_ret_1h.rolling(72).var()
mean_var_i_72h = rets_1h.rolling(72).var().mean(axis=1)
avg_corr_72h = (var_mkt_72h / (mean_var_i_72h + 1e-9)).clip(0.0, 1.0)

prev_ret_24h = ret_24h.shift(24)
m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
m_curr = ret_24h.sub(ret_24h.mean(axis=1), axis=0)
num = (m_prev * m_curr).sum(axis=1)
den = np.sqrt((m_prev**2).sum(axis=1) * (m_curr**2).sum(axis=1)) + 1e-9
raw_autocorr = num / den
cs_autocorr_24h = raw_autocorr.rolling(24).mean()
cs_autocorr_48h = raw_autocorr.rolling(48).mean()

df_diag = pd.DataFrame({
    "cs_disp_24h": cs_disp_24h,
    "avg_corr_72h": avg_corr_72h,
    "cs_autocorr_24h": cs_autocorr_24h,
    "cs_autocorr_48h": cs_autocorr_48h,
    "btc_vol_24h": rets_1h["BTCUSDT"].rolling(24).std() * np.sqrt(24 * 365),
}, index=idx)

print(df_diag[df_diag.index >= "2026-04-20"].resample("ME").mean())

# Now test conditioning our Pullback-Entry Momentum + Reversion + Carry on cs_autocorr!
r_4h = log_rets.rolling(4).sum() / (vol_24h * np.sqrt(4))
r_12h = log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12))
r_24h = log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24))
r_48h = log_rets.rolling(48).sum() / (vol_24h * np.sqrt(48))

# Core Pullback-Entry Momentum Score
mom_pullback = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) - 0.25 * r_4h
# Cross-Sectional Mean-Reversion Score (fading 24h-48h moves when cs_autocorr < 0!)
rev_score = -(0.50 * r_24h + 0.50 * r_48h)
# Carry Score
ann_fund_72h = rf15.rolling(72, min_periods=1).mean() * 3.0 * 365.0
carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

for ac_win in [12, 24, 36, 48]:
    ac = raw_autocorr.rolling(ac_win).mean().fillna(0.0)
    # When ac > -0.05: Momentum regime! When ac <= -0.05: Mean-Reversion / Carry regime!
    for thresh in [-0.10, -0.05, 0.0, 0.05]:
        for rev_mult in [0.0, 0.4, 0.7]:
            regime_mom = (ac > thresh).astype(float)
            regime_rev = (ac <= thresh).astype(float) * rev_mult
            # Dynamic composite score
            dyn_score = mom_pullback.mul(regime_mom, axis=0) + rev_score.mul(regime_rev, axis=0) + 0.25 * carry_score
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            w[TOP15] = make_continuous_neutral_weights(dyn_score, inv_vol, gross_leverage=2.2, power=1.8)
            # Also reduce gross leverage when in chop/reversion if rev_mult == 0
            if rev_mult == 0.0:
                scale_chop = pd.Series(np.where(ac > thresh, 1.0, 0.35), index=idx)
                w = w.mul(scale_chop, axis=0)
            res = simulate_portfolio(w, close, funding, rebalance_every_n_hours=4, max_single_weight=0.45, max_gross_leverage=2.2, deadband=0.04)
            m = res["monthly_rets"]
            if res["positive_months"] >= 6 and res["total_ret_pct"] > 120:
                print(
                    f"AC_W={ac_win:>2}h Th={thresh:>+5.2f} Rev={rev_mult:.1f} | Eq=${res['final_equity']:>8.2f} ({res['total_ret_pct']:>+7.2f}%) | "
                    f"Daily={res['avg_daily_ret_pct']:>+6.3f}% | Sharpe={res['sharpe']:>5.2f} | Sortino={res['sortino']:>5.2f} | "
                    f"DD={res['max_dd_pct']:>6.2f}% | Mos={res['positive_months']}/7 | Jun={m['2026-06']:>+6.2f}% | Oct={m['2026-10']:>+6.2f}%"
                )
