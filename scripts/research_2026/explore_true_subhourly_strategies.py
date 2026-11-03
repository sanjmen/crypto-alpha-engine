"""
Empirical Exploration of Native Sub-Hourly Strategies (15m, 30m, 5m, 1m):
1. S1: Thresholded Sub-Hourly Mean-Reversion Sniper (capturing the +520% Gross 15m / +144% Gross 30m reversion
   while filtering out sub-fee noise via |r_cs| and Z-score thresholds).
2. S2: Sub-Hourly BTC-to-Altcoin Lead-Lag Impulse (15m / 30m / 5m catch-up trades when BTC jumps).
3. S3: Sub-Hourly Taker Order Flow Imbalance (OFI) + Volume Burst.
4. S4: Multi-Frequency Harmonic Ensemble (blending 30m/1h/3h Sub-Hourly Pullback & Dislocation with 4h/8h Macro Regime).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from test_neutral_alphas import make_continuous_neutral_weights
from test_subhourly_alpha_boost import simulate_on_grid
from test_subhourly_frequencies import TOP15, load_timeframe_matrices


def eval_direct_weights(w_df: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, eval_start: str = "2026-04-20 00:00:00", fee_rate: float = 0.0002):
    eval_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]
    rets = close.pct_change().reindex(eval_idx).fillna(0.0)
    fund = funding.reindex(eval_idx).fillna(0.0)
    exec_w = w_df.reindex(eval_idx).fillna(0.0).shift(1).fillna(0.0)

    turnover = exec_w.diff().abs().sum(axis=1).fillna(0.0)
    if len(exec_w) > 0:
        turnover.iloc[0] = exec_w.iloc[0].abs().sum()

    gross_r = (exec_w * rets).sum(axis=1) - (exec_w * fund).sum(axis=1)
    fee_r = turnover * fee_rate
    net_r = gross_r - fee_r

    eq_gross = 1000.0 * (1.0 + gross_r).cumprod()
    eq_net = 1000.0 * (1.0 + net_r).cumprod()
    daily_eq = eq_net.resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    mean_d = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0
    std_d = float(daily_rets.std()) if len(daily_rets) > 1 else 0.0
    sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
    peaks = eq_net.cummax()
    mdd = float(((eq_net - peaks) / peaks).min() * 100.0)

    monthly = daily_eq.resample("ME").last().pct_change()
    first_m_start = eq_net.loc[eq_net.index.month == eval_ts.month].iloc[0]
    first_m_end = daily_eq.resample("ME").last().iloc[0]
    monthly.iloc[0] = (first_m_end / first_m_start) - 1.0

    return {
        "gross_ret_pct": float((eq_gross.iloc[-1] / 1000.0 - 1.0) * 100.0),
        "net_ret_pct": float((eq_net.iloc[-1] / 1000.0 - 1.0) * 100.0),
        "final_eq": float(eq_net.iloc[-1]),
        "avg_daily_pct": mean_d * 100.0,
        "sharpe": sharpe,
        "max_dd_pct": mdd,
        "sum_fee_pct": float(fee_r.sum() * 100.0),
        "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
    }


def main():
    idx, open_, high, low, close, volume, taker_buy, raw_funding, funding = load_timeframe_matrices(
        "15m", start_date="2026-03-20", end_date="2026-10-20 18:00:00"
    )
    r_15m = close.pct_change(1).fillna(0.0)
    r_30m = close.pct_change(2).fillna(0.0)
    r_1h  = close.pct_change(4).fillna(0.0)
    r_2h  = close.pct_change(8).fillna(0.0)
    r_4h  = close.pct_change(16).fillna(0.0)

    # Cross-sectional demeaned returns (relative to equal-weight market move)
    cs_15m = r_15m.sub(r_15m.mean(axis=1), axis=0)
    cs_30m = r_30m.sub(r_30m.mean(axis=1), axis=0)
    cs_1h  = r_1h.sub(r_1h.mean(axis=1), axis=0)
    cs_2h  = r_2h.sub(r_2h.mean(axis=1), axis=0)
    cs_4h  = r_4h.sub(r_4h.mean(axis=1), axis=0)

    print("=" * 125)
    print("S1. THRESHOLDED SUB-HOURLY MEAN-REVERSION SNIPER (15m / 30m / 1h / 2h) — 6 MONTHS")
    print("=" * 125)
    for label, cs_df, thresh, hold_bars, reb_b in [
        ("15m Dislocation > 1.5% (hold 30m)", cs_15m, 0.015, 2, 2),
        ("15m Dislocation > 2.5% (hold 1h)",  cs_15m, 0.025, 4, 4),
        ("30m Dislocation > 2.0% (hold 1h)",  cs_30m, 0.020, 4, 4),
        ("30m Dislocation > 3.0% (hold 2h)",  cs_30m, 0.030, 8, 8),
        ("1h  Dislocation > 3.0% (hold 2h)",  cs_1h,  0.030, 8, 8),
        ("2h  Dislocation > 4.0% (hold 4h)",  cs_2h,  0.040, 16, 16),
        ("4h  Dislocation > 5.0% (hold 4h)",  cs_4h,  0.050, 16, 16),
    ]:
        # Fade extreme relative move, hedged against market average
        sig = -cs_df.where(cs_df.abs() >= thresh, 0.0)
        if hold_bars > 1:
            sig = sig.rolling(hold_bars).mean().fillna(0.0)
        # Dollar-neutralize whenever active
        pos = sig.clip(lower=0.0)
        neg = sig.clip(upper=0.0)
        active = (pos.sum(axis=1) > 0) & (neg.abs().sum(axis=1) > 0)
        w = pd.DataFrame(0.0, index=idx, columns=TOP15)
        w.loc[active] = (
            pos.loc[active].div(pos.loc[active].sum(axis=1), axis=0) * 0.75
            + neg.loc[active].div(neg.loc[active].abs().sum(axis=1), axis=0) * 0.75
        ).clip(-0.40, 0.40)
        res = simulate_on_grid(w, close, funding, rebalance_every_bars=reb_b, max_single_weight=0.40, max_gross_leverage=1.5, deadband=0.02)
        print(
            f"{label:<36} | Gross:{res['gross_ret_pct']:+7.2f}% | Fees:{res['sum_fee_pct']:6.2f}% | "
            f"Net:{res['net_ret_pct']:+7.2f}% | Daily:{res['avg_daily_pct']:+6.3f}% | Shp:{res['sharpe']:5.2f} | MDD:{res['max_dd_pct']:6.2f}%"
        )

    print("\n" + "=" * 125)
    print("S2. SUB-HOURLY BTC-TO-ALTCOIN LEAD-LAG IMPULSE (15m / 30m / 1h) — 6 MONTHS")
    print("=" * 125)
    btc_15m = r_15m["BTCUSDT"]
    btc_30m = r_30m["BTCUSDT"]
    btc_1h  = r_1h["BTCUSDT"]
    # Rolling beta of each coin to BTC
    vol_ratio = (r_1h.rolling(96).std().div(btc_1h.rolling(96).std(), axis=0)).clip(0.5, 2.5).fillna(1.0)
    for label, b_ret, c_ret, reb_b, db in [
        ("15m BTC Lead-Lag Catchup (Reb 15m)", btc_15m, r_15m, 1, 0.06),
        ("30m BTC Lead-Lag Catchup (Reb 30m)", btc_30m, r_30m, 2, 0.06),
        ("1h  BTC Lead-Lag Catchup (Reb 1h)",  btc_1h,  r_1h,  4, 0.06),
    ]:
        # Expected move = beta * r_BTC; Laggard gap = expected - actual
        expected = vol_ratio.mul(b_ret, axis=0)
        gap = expected - c_ret
        w = make_continuous_neutral_weights(gap, 1.0 / r_1h.rolling(96).std().bfill().fillna(0.02), gross_leverage=1.6, power=1.5)
        res = simulate_on_grid(w, close, funding, rebalance_every_bars=reb_b, max_single_weight=0.40, max_gross_leverage=1.6, deadband=db)
        print(
            f"{label:<36} | Gross:{res['gross_ret_pct']:+7.2f}% | Fees:{res['sum_fee_pct']:6.2f}% | "
            f"Net:{res['net_ret_pct']:+7.2f}% | Daily:{res['avg_daily_pct']:+6.3f}% | Shp:{res['sharpe']:5.2f} | MDD:{res['max_dd_pct']:6.2f}%"
        )

    print("\n" + "=" * 125)
    print("S3. SUB-HOURLY TAKER ORDER FLOW IMBALANCE (OFI) & VOLUME SURGE — 6 MONTHS")
    print("=" * 125)
    ofi_30m = ((2.0 * taker_buy.rolling(2).sum() - volume.rolling(2).sum()) / (volume.rolling(2).sum() + 1e-8)).fillna(0.0)
    ofi_2h  = ((2.0 * taker_buy.rolling(8).sum() - volume.rolling(8).sum()) / (volume.rolling(8).sum() + 1e-8)).fillna(0.0)
    ofi_12h = ((2.0 * taker_buy.rolling(48).sum() - volume.rolling(48).sum()) / (volume.rolling(48).sum() + 1e-8)).fillna(0.0)
    inv_v1h = (1.0 / r_1h.rolling(96).std().bfill().fillna(0.02)).clip(upper=100.0)
    for label, s_df, reb_b, db in [
        ("OFI 30m Mean-Reversion (-OFI_30m, 30m)", -ofi_30m, 2, 0.06),
        ("OFI 2h  Mean-Reversion (-OFI_2h,  2h)",  -ofi_2h,  8, 0.06),
        ("OFI 12h Continuation   (+OFI_12h, 4h)",  ofi_12h, 16, 0.04),
        ("OFI Pullback (+OFI_12h - 0.5*OFI_2h, 4h)", ofi_12h - 0.5 * ofi_2h, 16, 0.04),
    ]:
        w = make_continuous_neutral_weights(s_df, inv_v1h, gross_leverage=1.8, power=1.5)
        res = simulate_on_grid(w, close, funding, rebalance_every_bars=reb_b, max_single_weight=0.40, max_gross_leverage=1.8, deadband=db)
        print(
            f"{label:<36} | Gross:{res['gross_ret_pct']:+7.2f}% | Fees:{res['sum_fee_pct']:6.2f}% | "
            f"Net:{res['net_ret_pct']:+7.2f}% | Daily:{res['avg_daily_pct']:+6.3f}% | Shp:{res['sharpe']:5.2f} | MDD:{res['max_dd_pct']:6.2f}%"
        )

    print("\n" + "=" * 125)
    print("S4. MULTI-FREQUENCY HARMONIC ENSEMBLE (Sub-Hourly 30m/2h/3h Pullback + 4h + 8h Tranches on 15m Grid)")
    print("=" * 125)
    # Build the exact features from test_subhourly_alpha_boost
    log_rets_15m = np.log(close / close.shift(1)).fillna(0.0)
    vol_1h_24h = (log_rets_15m.rolling(4).sum().rolling(96).std()).replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_1h_24h).clip(upper=100.0)
    dv = close * volume
    vol_surge = (dv.rolling(24).mean() / (dv.rolling(192).mean() + 1e-8)).clip(0.5, 2.5)

    btc_r1h = r_1h["BTCUSDT"]
    beta_btc = (r_1h.rolling(288).cov(btc_r1h).div(btc_r1h.rolling(288).var().replace(0, 1e-8), axis=0)).clip(0.2, 2.5).fillna(1.0)
    resid_15m = log_rets_15m.sub(beta_btc.mul(log_rets_15m["BTCUSDT"], axis=0))

    def r_norm(series_15m, hours):
        bars = int(round(hours * 4))
        return series_15m.rolling(bars).sum() / (vol_1h_24h * np.sqrt(hours))

    rn_30m, rn_1h, rn_2h, rn_4h = r_norm(log_rets_15m, 0.5), r_norm(log_rets_15m, 1.0), r_norm(log_rets_15m, 2.0), r_norm(log_rets_15m, 4.0)
    rn_12h, rn_24h, rn_48h = r_norm(log_rets_15m, 12.0), r_norm(log_rets_15m, 24.0), r_norm(log_rets_15m, 48.0)
    irn_30m, irn_2h, irn_4h = r_norm(resid_15m, 0.5), r_norm(resid_15m, 2.0), r_norm(resid_15m, 4.0)
    irn_12h, irn_24h, irn_72h = r_norm(resid_15m, 12.0), r_norm(resid_15m, 24.0), r_norm(resid_15m, 72.0)

    ret_24h = log_rets_15m.rolling(96).sum()
    prev_ret_24h = ret_24h.shift(96)
    m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
    m_curr = ret_24h.sub(ret_24h.mean(axis=1), axis=0)
    cs_autocorr_48h = ((m_prev * m_curr).sum(axis=1) / (np.sqrt((m_prev ** 2).sum(axis=1) * (m_curr ** 2).sum(axis=1)) + 1e-9)).rolling(192).mean().fillna(0.0)

    ann_fund_72h = raw_funding.rolling(288, min_periods=1).mean() * 3.0 * 365.0
    carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

    shock_1h = rn_1h.where(rn_1h.abs() > 2.0, 0.0)
    sub_pb_raw = (0.35 * rn_12h + 0.45 * rn_24h + 0.20 * rn_48h) * np.sqrt(vol_surge) - (0.05 * rn_30m + 0.08 * rn_2h + 0.18 * rn_4h + 0.08 * shock_1h)
    sub_pb_idio = (0.30 * irn_12h + 0.40 * irn_24h + 0.30 * irn_72h) * np.sqrt(vol_surge) - (0.05 * irn_30m + 0.06 * irn_2h + 0.15 * irn_4h)
    sub_blend = 0.60 * sub_pb_raw + 0.40 * sub_pb_idio

    tilt = ((cs_autocorr_48h - (-0.14)) / (-0.04 - (-0.14))).clip(0.0, 1.0)
    lev_mult = (tilt.abs() + 0.35).clip(0.45, 1.0)
    sc = sub_blend.mul(tilt, axis=0) + 0.30 * carry_score

    for g_lev, max_w in [(2.2, 0.45), (2.5, 0.50), (2.8, 0.55)]:
        w_raw_target = make_continuous_neutral_weights(sc, inv_vol, gross_leverage=g_lev, power=1.85).mul(lev_mult, axis=0)

        # Evaluate 3h, 6h, 8h, and Staggered Tranche (50% 3h SubHourly + 50% 8h Macro)
        for name, reb_b, db in [
            (f"SubHourly_3h (G={g_lev:.1f}x, db=5%)", 12, 0.05),
            (f"Macro_8h     (G={g_lev:.1f}x, db=4%)", 32, 0.04),
        ]:
            res = simulate_on_grid(w_raw_target, close, funding, rebalance_every_bars=reb_b, max_single_weight=max_w, max_gross_leverage=g_lev, deadband=db)
            neg_m = sum(1 for v in res["monthly"].values() if v < 0)
            m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in res["monthly"].items()])
            print(
                f"{name:<36} | Gross:{res['gross_ret_pct']:+7.1f}% | Fees:{res['sum_fee_pct']:5.1f}% | "
                f"Net:{res['net_ret_pct']:+7.2f}% (${res['final_eq']:,.0f}) | Daily:{res['avg_daily_pct']:+6.3f}% | "
                f"Shp:{res['sharpe']:4.2f} | MDD:{res['max_dd_pct']:6.2f}% | NegM:{neg_m} [{m_str}]"
            )


if __name__ == "__main__":
    main()
