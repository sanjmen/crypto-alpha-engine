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
dv15 = dollar_vol[TOP15]
rets_1h = c15.pct_change().fillna(0.0)
log_rets = np.log(c15 / c15.shift(1)).fillna(0.0)
vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
inv_vol = (1.0 / vol_24h).clip(upper=100.0)
vol_surge = (dv15.rolling(6).mean() / (dv15.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)

r_3h = log_rets.rolling(3).sum() / (vol_24h * np.sqrt(3))
r_4h = log_rets.rolling(4).sum() / (vol_24h * np.sqrt(4))
r_12h = log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12))
r_24h = log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24))
r_48h = log_rets.rolling(48).sum() / (vol_24h * np.sqrt(48))
r_72h = log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72))

for w4 in [0.0, -0.25, -0.45, -0.65]:
    for power in [1.0, 1.5, 2.0]:
        for reb in [2, 4]:
            score = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) + w4 * r_4h
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            w[TOP15] = make_continuous_neutral_weights(score, inv_vol, gross_leverage=2.2, power=power)
            res = simulate_portfolio(w, close, funding, rebalance_every_n_hours=reb, max_single_weight=0.45, max_gross_leverage=2.2, deadband=0.04)
            m = res["monthly_rets"]
            print(
                f"w4={w4:>+5.2f} P={power:.1f} Reb={reb}h | Eq=${res['final_equity']:>8.2f} ({res['total_ret_pct']:>+7.2f}%) | "
                f"Daily={res['avg_daily_ret_pct']:>+6.3f}% | Sharpe={res['sharpe']:>5.2f} | Sortino={res['sortino']:>5.2f} | "
                f"DD={res['max_dd_pct']:>6.2f}% | Mos={res['positive_months']}/7 | Jun={m['2026-06']:>+6.2f}% | Oct={m['2026-10']:>+6.2f}%"
            )
