"""
Exact Sub-Hourly (15m, 30m, 5m, 1m) Evaluation of the Volatility-Normalized Pullback-Momentum
+ Idiosyncratic Beta-Neutral + Carry + CS-AutoCorr Regime Engine, plus Sub-Hourly Microstructure Enhancements.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from test_neutral_alphas import make_continuous_neutral_weights
from test_subhourly_frequencies import TOP15, load_timeframe_matrices, sim_subhourly


def build_exact_subhourly_features(close, high, low, volume, taker_buy, raw_funding, bars_per_hour: int):
    """
    Computes the exact volatility-normalized t-statistic features at sub-hourly resolution
    (bars_per_hour = 4 for 15m, 2 for 30m, 12 for 5m, 60 for 1m).
    """
    bph = bars_per_hour
    rets_1b = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)

    # 24h rolling per-bar volatility
    vol_24h = rets_1b.rolling(24 * bph).std().replace(0, np.nan).ffill().fillna(0.01)
    # Parkinson High-Low range volatility over 24h
    hl_sq = (np.log(high / np.maximum(low, 1e-8)) ** 2)
    park_vol_24h = np.sqrt(hl_sq.rolling(24 * bph).mean() / (4.0 * np.log(2.0))).replace(0, np.nan).ffill().fillna(0.01)

    inv_vol = (1.0 / vol_24h).clip(upper=500.0)
    inv_park_vol = (1.0 / park_vol_24h).clip(upper=500.0)

    dv = close * volume
    vol_surge = (dv.rolling(6 * bph).mean() / (dv.rolling(48 * bph).mean() + 1e-8)).clip(0.5, 2.5)

    # Rolling 72h Beta to BTC computed using 1-hour returns to avoid 15m microstructure lead-lag attenuation
    rets_1h = close.pct_change(bph).fillna(0.0)
    btc_ret_1h = rets_1h["BTCUSDT"]
    cov_btc = rets_1h.rolling(72 * bph).cov(btc_ret_1h)
    var_btc = btc_ret_1h.rolling(72 * bph).var().replace(0, 1e-8)
    beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5).fillna(1.0)
    resid_1b = rets_1b.sub(beta_btc.mul(rets_1b["BTCUSDT"], axis=0))

    def norm_ret(lr, hours):
        n_bars = int(round(hours * bph))
        return lr.rolling(n_bars).sum() / (vol_24h * np.sqrt(n_bars))

    # Raw Volatility-Normalized Returns (including sub-hourly 0.25h=15m, 0.5h=30m, 1h, 2h, 3h, 4h)
    r_15m = norm_ret(log_rets, 0.25) if bph >= 4 else norm_ret(log_rets, 0.5)
    r_30m = norm_ret(log_rets, 0.50) if bph >= 2 else norm_ret(log_rets, 1.0)
    r_1h = norm_ret(log_rets, 1.0)
    r_2h = norm_ret(log_rets, 2.0)
    r_3h = norm_ret(log_rets, 3.0)
    r_4h = norm_ret(log_rets, 4.0)
    r_12h = norm_ret(log_rets, 12.0)
    r_24h = norm_ret(log_rets, 24.0)
    r_48h = norm_ret(log_rets, 48.0)
    r_72h = norm_ret(log_rets, 72.0)

    # Idiosyncratic Volatility-Normalized Returns
    ir_1h = norm_ret(resid_1b, 1.0)
    ir_2h = norm_ret(resid_1b, 2.0)
    ir_3h = norm_ret(resid_1b, 3.0)
    ir_4h = norm_ret(resid_1b, 4.0)
    ir_12h = norm_ret(resid_1b, 12.0)
    ir_24h = norm_ret(resid_1b, 24.0)
    ir_48h = norm_ret(resid_1b, 48.0)
    ir_72h = norm_ret(resid_1b, 72.0)

    # Order Flow Imbalance (OFI) over 1h, 4h, 24h
    ofi_1h = ((2.0 * taker_buy.rolling(bph).sum() - volume.rolling(bph).sum()) / (volume.rolling(bph).sum() + 1e-8)).fillna(0.0)
    ofi_4h = ((2.0 * taker_buy.rolling(4 * bph).sum() - volume.rolling(4 * bph).sum()) / (volume.rolling(4 * bph).sum() + 1e-8)).fillna(0.0)
    ofi_24h = ((2.0 * taker_buy.rolling(24 * bph).sum() - volume.rolling(24 * bph).sum()) / (volume.rolling(24 * bph).sum() + 1e-8)).fillna(0.0)

    # Cross-Sectional Return Autocorrelation Sensor (48h window)
    ret_24h_raw = log_rets.rolling(24 * bph).sum()
    prev_ret_24h = ret_24h_raw.shift(24 * bph)
    m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
    m_curr = ret_24h_raw.sub(ret_24h_raw.mean(axis=1), axis=0)
    num = (m_prev * m_curr).sum(axis=1)
    den = np.sqrt((m_prev ** 2).sum(axis=1) * (m_curr ** 2).sum(axis=1)) + 1e-9
    raw_autocorr = num / den
    cs_autocorr_48h = raw_autocorr.rolling(48 * bph).mean().fillna(0.0)

    ann_fund_72h = raw_funding.rolling(72 * bph, min_periods=1).mean() * 3.0 * 365.0
    carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

    return {
        "vol_24h": vol_24h,
        "park_vol_24h": park_vol_24h,
        "inv_vol": inv_vol,
        "inv_park_vol": inv_park_vol,
        "vol_surge": vol_surge,
        "r_15m": r_15m, "r_30m": r_30m, "r_1h": r_1h, "r_2h": r_2h, "r_3h": r_3h, "r_4h": r_4h,
        "r_12h": r_12h, "r_24h": r_24h, "r_48h": r_48h, "r_72h": r_72h,
        "ir_1h": ir_1h, "ir_2h": ir_2h, "ir_3h": ir_3h, "ir_4h": ir_4h,
        "ir_12h": ir_12h, "ir_24h": ir_24h, "ir_48h": ir_48h, "ir_72h": ir_72h,
        "ofi_1h": ofi_1h, "ofi_4h": ofi_4h, "ofi_24h": ofi_24h,
        "cs_autocorr_48h": cs_autocorr_48h,
        "carry_score": carry_score,
    }


def main():
    idx, open_, high, low, close, volume, taker_buy, raw_funding, funding = load_timeframe_matrices(
        "15m", start_date="2026-03-20", end_date="2026-10-20 18:00:00"
    )
    feats = build_exact_subhourly_features(close, high, low, volume, taker_buy, raw_funding, bars_per_hour=4)

    # Baseline formula from test_smooth_regime.py evaluated on 15m grid
    mom_pullback_raw = (0.35 * feats["r_12h"] + 0.45 * feats["r_24h"] + 0.20 * feats["r_48h"]) * np.sqrt(feats["vol_surge"]) - 0.25 * feats["r_4h"]
    mom_pullback_idio = (0.30 * feats["ir_12h"] + 0.40 * feats["ir_24h"] + 0.30 * feats["ir_72h"]) * np.sqrt(feats["vol_surge"]) - 0.20 * feats["ir_4h"]
    mom_pullback_blend = 0.60 * mom_pullback_raw + 0.40 * mom_pullback_idio

    lo_ac, hi_ac = -0.14, -0.04
    tilt = ((feats["cs_autocorr_48h"] - lo_ac) / (hi_ac - lo_ac)).clip(0.0, 1.0)
    dyn_score = mom_pullback_blend.mul(tilt, axis=0) + 0.30 * feats["carry_score"]
    w_base = make_continuous_neutral_weights(dyn_score, feats["inv_vol"], gross_leverage=2.2, power=1.8).clip(-0.45, 0.45)
    lev_mult = (tilt.abs() + 0.35).clip(0.45, 1.0)
    w_smooth = w_base.mul(lev_mult, axis=0)

    print("=" * 125)
    print("1. EXACT SMOOTH-REGIME PULLBACK-MOMENTUM ENGINE ACROSS REBALANCE FREQUENCIES (15m to 8h) — 6 MONTHS")
    print("=" * 125)
    for freq_name, reb_bars, db in [
        ("15m (every 1 bar,  db=4%)", 1, 0.04),
        ("15m (every 1 bar,  db=10%)", 1, 0.10),
        ("30m (every 2 bars, db=4%)", 2, 0.04),
        ("30m (every 2 bars, db=8%)", 2, 0.08),
        ("30m (every 2 bars, db=12%)", 2, 0.12),
        ("1h  (every 4 bars, db=4%)", 4, 0.04),
        ("1h  (every 4 bars, db=8%)", 4, 0.08),
        ("2h  (every 8 bars, db=4%)", 8, 0.04),
        ("2h  (every 8 bars, db=6%)", 8, 0.06),
        ("3h  (every 12 bars, db=4%)", 12, 0.04),
        ("3h  (every 12 bars, db=5%)", 12, 0.05),
        ("4h  (every 16 bars, db=4%)", 16, 0.04),
        ("6h  (every 24 bars, db=4%)", 24, 0.04),
        ("8h  (every 32 bars, db=4%)", 32, 0.04),
    ]:
        r = sim_subhourly(w_smooth, close, funding, "2026-04-20 00:00:00", reb_every_bars=reb_bars, deadband=db, max_gross=2.2)
        neg_m = sum(1 for v in r["monthly"].values() if v < 0)
        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in r["monthly"].items()])
        print(
            f"{freq_name:<28} | Gross:{r['gross_ret_pct']:+7.1f}% | Fees:{r['sum_fee_pct']:5.1f}% | "
            f"Net:{r['net_ret_pct']:+7.2f}% (${r['final_eq']:,.0f}) | Daily:{r['avg_daily_pct']:+6.3f}% | "
            f"Shp:{r['sharpe']:4.2f} | MDD:{r['max_dd_pct']:6.2f}% | NegM:{neg_m} [{m_str}]"
        )

    print("\n" + "=" * 125)
    print("2. CAN SUB-HOURLY FEATURES (15m/30m Dip, 2h/3h/4h Multi-Pullback, Parkinson High-Low Vol, Order Flow) BEAT +155.38%?")
    print("=" * 125)
    # Test multi-horizon sub-hourly pullback (e.g., combining 30m, 1h, 2h, 4h dips + Parkinson High-Low Vol + Order Flow)
    sub_pb_raw = (
        (0.35 * feats["r_12h"] + 0.45 * feats["r_24h"] + 0.20 * feats["r_48h"]) * np.sqrt(feats["vol_surge"])
        - (0.05 * feats["r_1h"] + 0.08 * feats["r_2h"] + 0.17 * feats["r_4h"])
    )
    sub_pb_idio = (
        (0.30 * feats["ir_12h"] + 0.40 * feats["ir_24h"] + 0.30 * feats["ir_72h"]) * np.sqrt(feats["vol_surge"])
        - (0.05 * feats["ir_1h"] + 0.05 * feats["ir_2h"] + 0.15 * feats["ir_4h"])
    )
    sub_pb_blend = 0.60 * sub_pb_raw + 0.40 * sub_pb_idio

    for v_name, inv_v in [("CloseVol_15m", feats["inv_vol"]), ("ParkinsonHL_15m", feats["inv_park_vol"])]:
        for ofi_coef in [0.0, 0.15, -0.15]:
            score_c = (sub_pb_blend + ofi_coef * (feats["ofi_24h"] - feats["ofi_4h"])).mul(tilt, axis=0) + 0.30 * feats["carry_score"]
            for g_lev, max_w in [(2.2, 0.45), (2.4, 0.48), (2.5, 0.50)]:
                w_c = make_continuous_neutral_weights(score_c, inv_v, gross_leverage=g_lev, power=1.85).clip(-max_w, max_w)
                w_c = w_c.mul(lev_mult, axis=0)
                for reb_b, db, cad_lbl in [(12, 0.05, "3h_db5%"), (14, 0.04, "3.5h_db4%"), (16, 0.04, "4h_db4%")]:
                    r = sim_subhourly(w_c, close, funding, "2026-04-20 00:00:00", reb_every_bars=reb_b, deadband=db, max_gross=g_lev)
                    neg_m = sum(1 for v in r["monthly"].values() if v < 0)
                    if r["net_ret_pct"] >= 150.0 and neg_m == 0:
                        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in r["monthly"].items()])
                        print(
                            f"{v_name:<15} OFI={ofi_coef:+.2f} G={g_lev:.1f} {cad_lbl:<9} | "
                            f"Net:{r['net_ret_pct']:+7.2f}% (${r['final_eq']:,.0f}) | Daily:{r['avg_daily_pct']:+6.3f}% | "
                            f"Shp:{r['sharpe']:4.2f} | Sort:{r['sortino']:4.2f} | MDD:{r['max_dd_pct']:6.2f}% | [{m_str}]"
                        )


if __name__ == "__main__":
    main()
