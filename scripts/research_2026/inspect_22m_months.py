"""
Inspects the month-by-month return across all 22 months (2025-01 to 2026-10) for a wide variety of orthogonal factors:
1. Macro Idio Pullback (7d-30d trend, 24h pullback)
2. Fast Swing Pullback (12h-48h trend, 4h pullback)
3. Medium Idio Momentum (3d-7d = 72h-168h trend, 12h pullback)
4. Contrarian Funding Carry (-funding_72h)
5. Cross-Sectional Volatility / Low-Beta Defensive Factor (Long low-vol/low-beta, Short high-vol/high-beta in risk-off!)
6. Time-Series Market Trend / BTC Trend Filter
7. Cointegrated Sector Pairs / Relative Value Mean Reversion (7d-14d Z-score reversion)
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
    vol_168h = rets_1h.rolling(168).std().replace(0, np.nan).ffill().fillna(0.02)
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

    # Factors
    f_macro = 0.70 * ((0.35 * zi[168] + 0.40 * zi[336] + 0.25 * zi[720]) * np.sqrt(vol_surge_wk) - (0.15 * zi[4] + 0.20 * zi[24])) + \
              0.30 * ((0.35 * zr[168] + 0.40 * zr[336] + 0.25 * zr[720]) * np.sqrt(vol_surge_wk) - (0.15 * zr[4] + 0.20 * zr[24]))
    f_med   = (0.40 * zi[72] + 0.60 * zi[168]) * np.sqrt(vol_surge_wk) - 0.25 * zi[12]
    f_fast  = 0.60 * ((0.35 * zr[12] + 0.45 * zr[24] + 0.20 * zr[48]) * np.sqrt(vol_surge) - 0.25 * zr[4]) + \
              0.40 * ((0.30 * zi[12] + 0.40 * zi[24] + 0.30 * zi[72]) * np.sqrt(vol_surge) - 0.20 * zi[4])
    f_rev72 = -(0.50 * zi[48] + 0.50 * zi[72])
    f_carry = -(raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0)
    # Low-Volatility / Betting-Against-Beta (BAB): Long low vol/beta, Short high vol/beta
    f_lowvol = -(0.5 * (vol_168h.div(vol_168h.mean(axis=1), axis=0)) + 0.5 * beta_72)
    # Volume-Price Divergence / Amihud Liquidity Factor
    f_liq = np.log(dollar_vol.rolling(168).mean() + 1.0)

    factors = {
        "F1_Macro_Idio_PB": f_macro,
        "F2_Med_Idio_PB":   f_med,
        "F3_Fast_Swing_PB": f_fast,
        "F4_Rev_48_72h":    f_rev72,
        "F5_Carry_Neg":     f_carry,
        "F6_LowVol_BAB":    f_lowvol,
        "F7_HighLiq_Size":  f_liq,
    }

    res_map = {}
    for k, sc in factors.items():
        w = make_continuous_neutral_weights(sc, inv_vol, gross_leverage=2.0, power=1.6)
        res_map[k] = sim_22m(w, close, funding, reb_hours=8, gross_lev=2.0, max_single=0.40, deadband=0.04)

    months = list(next(iter(res_map.values()))["monthly"].keys())
    print("MONTHLY BREAKDOWN BY FACTOR (2025-01 to 2026-10):")
    header = f"{'Month':<8} | {'BTC_Ret':>8} | " + " | ".join(f"{k:>16}" for k in factors.keys())
    print(header)
    print("-" * len(header))

    btc_m = close["BTCUSDT"].reindex(close.index[close.index >= pd.Timestamp("2025-01-01", tz="UTC")]).resample("ME").last().pct_change()
    btc_m.iloc[0] = close["BTCUSDT"].loc["2025-01-31"].iloc[-1] / close["BTCUSDT"].loc["2025-01-01"].iloc[0] - 1.0
    btc_m_dict = {k.strftime("%Y-%m"): float(v * 100.0) for k, v in btc_m.items()}

    for m in months:
        row_str = f"{m:<8} | {btc_m_dict.get(m, 0.0):>+7.1f}% | " + " | ".join(f"{res_map[k]['monthly'][m]:>+15.1f}%" for k in factors.keys())
        print(row_str)

    print("-" * len(header))
    tot_str = f"{'TOTAL':<8} | {'':>8} | " + " | ".join(f"{res_map[k]['total_ret']:>+15.1f}%" for k in factors.keys())
    mdd_str = f"{'MAX_DD':<8} | {'':>8} | " + " | ".join(f"{res_map[k]['mdd']:>+15.1f}%" for k in factors.keys())
    shp_str = f"{'SHARPE':<8} | {'':>8} | " + " | ".join(f"{res_map[k]['sharpe']:>16.2f}" for k in factors.keys())
    print(tot_str)
    print(mdd_str)
    print(shp_str)


if __name__ == "__main__":
    main()
