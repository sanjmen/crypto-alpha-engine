"""
Deep Empirical Investigation of Sub-Hourly (15m, 30m, 5m, 1m) Alpha Mechanisms:
1. Exact alignment between 15m grid and 1h grid (verifying why 4h cadence achieved +155.38% / +160.37%).
2. Testing whether 15m / 30m sub-hourly signals (Extreme Dislocation Z-Score, Sub-Hourly Pullback Curve,
   Taker Order Flow Imbalance, and High-Low Range Volatility) can push 6-month returns even higher (> +170% to +200%+).
3. Testing pure High-Frequency Sub-Hourly Sniper (only entering when |Z_15m/30m/1h| > threshold so expected
   reversion exceeds the 0.04% round-trip Maker fee barrier).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from test_neutral_alphas import make_continuous_neutral_weights
from test_subhourly_frequencies import TOP15, load_timeframe_matrices


def simulate_on_grid(
    weights_df: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    eval_start: str = "2026-04-20 00:00:00",
    rebalance_every_bars: int = 16,
    max_single_weight: float = 0.45,
    max_gross_leverage: float = 2.2,
    deadband: float = 0.04,
    fee_rate: float = 0.0002,
):
    eval_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]
    rets = close.pct_change().reindex(eval_idx).fillna(0.0)
    fund = funding.reindex(eval_idx).fillna(0.0)
    w_raw = weights_df.reindex(eval_idx).fillna(0.0)

    # Cap single asset weight and re-normalize long/short legs to preserve dollar neutrality
    w_arr = w_raw.values.copy()
    n_rows, n_cols = w_arr.shape
    for t in range(n_rows):
        row = w_arr[t]
        g_init = np.sum(np.abs(row))
        if g_init < 1e-8:
            continue
        target_g = min(g_init, max_gross_leverage)
        row = np.clip(row, -max_single_weight, max_single_weight)
        pos = np.maximum(row, 0.0)
        neg = np.minimum(row, 0.0)
        if pos.sum() > 0:
            pos = pos * ((target_g / 2.0) / pos.sum())
        if np.abs(neg.sum()) > 0:
            neg = neg * ((target_g / 2.0) / np.abs(neg.sum()))
        row = np.clip(pos + neg, -max_single_weight, max_single_weight)
        w_arr[t] = row

    eff_arr = np.zeros_like(w_arr)
    prev_w = np.zeros(n_cols)
    for t in range(n_rows):
        if t % rebalance_every_bars == 0:
            desired = w_arr[t]
            delta = np.abs(desired - prev_w)
            upd = np.where((desired == 0.0) | (delta >= deadband), desired, prev_w)
            g = np.sum(np.abs(upd))
            if g > max_gross_leverage:
                upd = upd * (max_gross_leverage / g)
            eff_arr[t] = upd
            prev_w = upd
        else:
            eff_arr[t] = prev_w

    eff_w = pd.DataFrame(eff_arr, index=eval_idx, columns=close.columns)
    exec_w = eff_w.shift(1).fillna(0.0)
    turnover = exec_w.diff().abs().sum(axis=1).fillna(0.0)
    if len(exec_w) > 0:
        turnover.iloc[0] = exec_w.iloc[0].abs().sum()

    price_ret = (exec_w * rets).sum(axis=1)
    fund_ret = -(exec_w * fund).sum(axis=1)
    fee_drag = turnover * fee_rate
    gross_ret = price_ret + fund_ret
    net_ret = gross_ret - fee_drag

    eq_gross = 1000.0 * (1.0 + gross_ret).cumprod()
    eq_net = 1000.0 * (1.0 + net_ret).cumprod()

    daily_eq = eq_net.resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    mean_d = float(daily_rets.mean())
    std_d = float(daily_rets.std())
    sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
    down = daily_rets[daily_rets < 0]
    dd_std = float(np.sqrt((down ** 2).mean())) if len(down) > 0 else 1e-12
    sortino = (mean_d / dd_std) * np.sqrt(365.0) if dd_std > 1e-12 else 0.0
    peaks = eq_net.cummax()
    mdd = float(((eq_net - peaks) / peaks).min() * 100.0)
    win_days = float((daily_rets > 0).mean() * 100.0)

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
        "sortino": sortino,
        "max_dd_pct": mdd,
        "win_days_pct": win_days,
        "sum_fee_pct": float(fee_drag.sum() * 100.0),
        "total_turnover": float(turnover.sum()),
        "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
    }


def main():
    idx, open_, high, low, close, volume, taker_buy, raw_funding, funding = load_timeframe_matrices(
        "15m", start_date="2026-03-20", end_date="2026-10-20 18:00:00"
    )
    # 1. Compute exact 1h-equivalent rolling metrics at every 15m bar so any rebalance cadence (15m, 30m, 1h, 2h, 4h)
    # uses the exact same normalized feature scale!
    log_rets_15m = np.log(close / close.shift(1)).fillna(0.0)
    rets_1h_rolling = close.pct_change(4).fillna(0.0)
    # 1h return sampled over last 24h (using 4-bar stride via rolling mean of squared 1h returns / 4)
    vol_1h_24h = (log_rets_15m.rolling(4).sum().rolling(96).std()).replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_1h_24h).clip(upper=100.0)

    dv = close * volume
    vol_surge = (dv.rolling(24).mean() / (dv.rolling(192).mean() + 1e-8)).clip(0.5, 2.5)

    # Rolling 72h BTC beta on 1h returns
    btc_r1h = rets_1h_rolling["BTCUSDT"]
    cov_btc = rets_1h_rolling.rolling(288).cov(btc_r1h)
    var_btc = btc_r1h.rolling(288).var().replace(0, 1e-8)
    beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5).fillna(1.0)
    resid_15m = log_rets_15m.sub(beta_btc.mul(log_rets_15m["BTCUSDT"], axis=0))

    # Volatility-normalized returns across sub-hourly (15m, 30m, 1h, 2h, 3h, 4h) and intermediate (12h, 24h, 48h, 72h)
    def r_norm(series_15m, hours):
        bars = int(round(hours * 4))
        return series_15m.rolling(bars).sum() / (vol_1h_24h * np.sqrt(hours))

    r_15m = r_norm(log_rets_15m, 0.25)
    r_30m = r_norm(log_rets_15m, 0.50)
    r_1h  = r_norm(log_rets_15m, 1.0)
    r_2h  = r_norm(log_rets_15m, 2.0)
    r_3h  = r_norm(log_rets_15m, 3.0)
    r_4h  = r_norm(log_rets_15m, 4.0)
    r_12h = r_norm(log_rets_15m, 12.0)
    r_24h = r_norm(log_rets_15m, 24.0)
    r_48h = r_norm(log_rets_15m, 48.0)
    r_72h = r_norm(log_rets_15m, 72.0)

    ir_30m = r_norm(resid_15m, 0.50)
    ir_1h  = r_norm(resid_15m, 1.0)
    ir_2h  = r_norm(resid_15m, 2.0)
    ir_3h  = r_norm(resid_15m, 3.0)
    ir_4h  = r_norm(resid_15m, 4.0)
    ir_12h = r_norm(resid_15m, 12.0)
    ir_24h = r_norm(resid_15m, 24.0)
    ir_48h = r_norm(resid_15m, 48.0)
    ir_72h = r_norm(resid_15m, 72.0)

    # Cross-sectional autocorrelation sensor (48h)
    ret_24h = log_rets_15m.rolling(96).sum()
    prev_ret_24h = ret_24h.shift(96)
    m_prev = prev_ret_24h.sub(prev_ret_24h.mean(axis=1), axis=0)
    m_curr = ret_24h.sub(ret_24h.mean(axis=1), axis=0)
    num = (m_prev * m_curr).sum(axis=1)
    den = np.sqrt((m_prev ** 2).sum(axis=1) * (m_curr ** 2).sum(axis=1)) + 1e-9
    raw_autocorr = num / den
    cs_autocorr_48h = raw_autocorr.rolling(192).mean().fillna(0.0)

    ann_fund_72h = raw_funding.rolling(288, min_periods=1).mean() * 3.0 * 365.0
    carry_score = -ann_fund_72h.where(ann_fund_72h.abs() >= 0.08, 0.0)

    # Exact baseline SmoothAC[-0.14, -0.04] formula evaluated on 15m grid across all rebalance frequencies
    mom_pullback_raw = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) - 0.25 * r_4h
    mom_pullback_idio = (0.30 * ir_12h + 0.40 * ir_24h + 0.30 * ir_72h) * np.sqrt(vol_surge) - 0.20 * ir_4h
    mom_pullback_blend = 0.60 * mom_pullback_raw + 0.40 * mom_pullback_idio

    tilt = ((cs_autocorr_48h - (-0.14)) / (-0.04 - (-0.14))).clip(0.0, 1.0)
    dyn_score = mom_pullback_blend.mul(tilt, axis=0) + 0.30 * carry_score
    w_base = make_continuous_neutral_weights(dyn_score, inv_vol, gross_leverage=2.2, power=1.8)
    lev_mult = (tilt.abs() + 0.35).clip(0.45, 1.0)
    w_smooth = w_base.mul(lev_mult, axis=0)

    print("=" * 130)
    print("A. FREQUENCY VS FEE DRAG TRADEOFF: EXACT SAME ALPHA EVALUATED AT 15m, 30m, 1h, 2h, 3h, 4h, 6h, 8h (6 MONTHS)")
    print("=" * 130)
    for freq_lbl, reb_b, db in [
        ("15m (every 15m, db=4%)", 1, 0.04),
        ("15m (every 15m, db=12%)", 1, 0.12),
        ("30m (every 30m, db=4%)", 2, 0.04),
        ("30m (every 30m, db=10%)", 2, 0.10),
        ("1h  (every 1h,  db=4%)", 4, 0.04),
        ("1h  (every 1h,  db=8%)", 4, 0.08),
        ("2h  (every 2h,  db=4%)", 8, 0.04),
        ("2h  (every 2h,  db=6%)", 8, 0.06),
        ("3h  (every 3h,  db=4%)", 12, 0.04),
        ("4h  (every 4h,  db=4%)", 16, 0.04),
        ("6h  (every 6h,  db=4%)", 24, 0.04),
        ("8h  (every 8h,  db=4%)", 32, 0.04),
    ]:
        res = simulate_on_grid(w_smooth, close, funding, rebalance_every_bars=reb_b, deadband=db)
        neg_m = sum(1 for v in res["monthly"].values() if v < 0)
        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in res["monthly"].items()])
        print(
            f"{freq_lbl:<25} | Gross:{res['gross_ret_pct']:+7.1f}% | Fees:{res['sum_fee_pct']:5.1f}% | "
            f"Net:{res['net_ret_pct']:+7.2f}% (${res['final_eq']:,.0f}) | Daily:{res['avg_daily_pct']:+6.3f}% | "
            f"Shp:{res['sharpe']:4.2f} | MDD:{res['max_dd_pct']:6.2f}% | NegM:{neg_m} [{m_str}]"
        )

    # B. Now test Sub-Hourly Enhancements:
    # 1) Adding 30m/1h/2h sub-hourly dip and extreme sub-hourly shock reversion (|r_1h| > 2.0)
    # 2) Testing staggered/multi-phase sub-hourly execution or higher gross leverage in high-autocorr regimes
    print("\n" + "=" * 130)
    print("B. SUB-HOURLY MICROSTRUCTURE & MULTI-HORIZON ENHANCEMENTS (15m/30m Pullback + Shock Sniper + Adaptive Leverage)")
    print("=" * 130)

    # Sub-hourly extreme shock filter: only triggers when a coin has a violent 30m-2h spike/drop (|z| > 2.0)
    shock_1h = r_1h.where(r_1h.abs() > 2.0, 0.0)
    shock_30m = r_30m.where(r_30m.abs() > 2.2, 0.0)

    # Sub-hourly refined pullback: using 30m + 1h + 2h + 4h microstructure dip
    sub_pb_raw = (0.35 * r_12h + 0.45 * r_24h + 0.20 * r_48h) * np.sqrt(vol_surge) - (0.05 * r_30m + 0.08 * r_2h + 0.18 * r_4h + 0.08 * shock_1h)
    sub_pb_idio = (0.30 * ir_12h + 0.40 * ir_24h + 0.30 * ir_72h) * np.sqrt(vol_surge) - (0.05 * ir_30m + 0.06 * ir_2h + 0.15 * ir_4h)
    sub_blend = 0.60 * sub_pb_raw + 0.40 * sub_pb_idio

    for cfg_name, score_base, g_lev, max_w, pwr, reb_b, db in [
        ("SubHourly_Pullback (2.2x, 4h)", sub_blend, 2.2, 0.45, 1.8, 16, 0.04),
        ("SubHourly_Pullback (2.2x, 3h, db=5%)", sub_blend, 2.2, 0.45, 1.8, 12, 0.05),
        ("SubHourly_Pullback (2.2x, 2h, db=7%)", sub_blend, 2.2, 0.45, 1.8, 8, 0.07),
        ("SubHourly_Pullback (2.4x, 4h)", sub_blend, 2.4, 0.48, 1.85, 16, 0.04),
        ("SubHourly_Pullback (2.5x, 4h)", sub_blend, 2.5, 0.50, 1.9, 16, 0.04),
        ("SubHourly_Pullback (2.6x, 4h)", sub_blend, 2.6, 0.52, 1.9, 16, 0.04),
        ("SubHourly_Pullback (2.8x, 4h)", sub_blend, 2.8, 0.55, 1.9, 16, 0.04),
    ]:
        sc = score_base.mul(tilt, axis=0) + 0.30 * carry_score
        w_c = make_continuous_neutral_weights(sc, inv_vol, gross_leverage=g_lev, power=pwr).mul(lev_mult, axis=0)
        res = simulate_on_grid(w_c, close, funding, rebalance_every_bars=reb_b, max_single_weight=max_w, max_gross_leverage=g_lev, deadband=db)
        neg_m = sum(1 for v in res["monthly"].values() if v < 0)
        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in res["monthly"].items()])
        print(
            f"{cfg_name:<36} | Gross:{res['gross_ret_pct']:+7.1f}% | Fees:{res['sum_fee_pct']:5.1f}% | "
            f"Net:{res['net_ret_pct']:+7.2f}% (${res['final_eq']:,.0f}) | Daily:{res['avg_daily_pct']:+6.3f}% | "
            f"Shp:{res['sharpe']:4.2f} | MDD:{res['max_dd_pct']:6.2f}% | NegM:{neg_m} [{m_str}]"
        )


if __name__ == "__main__":
    main()
