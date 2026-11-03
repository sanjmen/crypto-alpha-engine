"""
Verifies the Multi-Frequency Harmonic Ensemble (combining 3h Sub-Hourly Pullback Tranche + 4h + 8h Macro Tranches
computed from 15m/30m microstructure bars) with 95% Bootstrap Confidence Intervals.
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


def get_effective_weights(
    weights_df: pd.DataFrame,
    eval_idx: pd.DatetimeIndex,
    rebalance_every_bars: int,
    max_single_weight: float,
    max_gross_leverage: float,
    deadband: float,
) -> pd.DataFrame:
    w_raw = weights_df.reindex(eval_idx).fillna(0.0)
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

    return pd.DataFrame(eff_arr, index=eval_idx, columns=weights_df.columns)


def eval_eff_weights(eff_w: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, fee_rate: float = 0.0002):
    eval_idx = eff_w.index
    rets = close.pct_change().reindex(eval_idx).fillna(0.0)
    fund = funding.reindex(eval_idx).fillna(0.0)
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

    # 95% Bootstrap CI (2,000 resamples of daily returns)
    rng = np.random.default_rng(42)
    d_arr = daily_rets.values
    boot_means = []
    boot_sharpes = []
    for _ in range(2000):
        sample = rng.choice(d_arr, size=len(d_arr), replace=True)
        m = np.mean(sample)
        s = np.std(sample, ddof=1)
        boot_means.append(m * 100.0)
        boot_sharpes.append((m / s) * np.sqrt(365.0) if s > 1e-12 else 0.0)
    ci_daily = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    ci_sharpe = (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5)))

    monthly = daily_eq.resample("ME").last().pct_change()
    first_m_start = eq_net.iloc[0]
    first_m_end = daily_eq.resample("ME").last().iloc[0]
    monthly.iloc[0] = (first_m_end / first_m_start) - 1.0

    return {
        "gross_ret_pct": float((eq_gross.iloc[-1] / 1000.0 - 1.0) * 100.0),
        "net_ret_pct": float((eq_net.iloc[-1] / 1000.0 - 1.0) * 100.0),
        "final_eq": float(eq_net.iloc[-1]),
        "avg_daily_pct": mean_d * 100.0,
        "ci_daily": ci_daily,
        "sharpe": sharpe,
        "ci_sharpe": ci_sharpe,
        "sortino": sortino,
        "max_dd_pct": mdd,
        "win_days_pct": win_days,
        "sum_fee_pct": float(fee_drag.sum() * 100.0),
        "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
    }


def main():
    idx, open_, high, low, close, volume, taker_buy, raw_funding, funding = load_timeframe_matrices(
        "15m", start_date="2026-03-20", end_date="2026-10-20 18:00:00"
    )
    eval_ts = pd.Timestamp("2026-04-20 00:00:00", tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]

    r_1h = close.pct_change(4).fillna(0.0)
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
        w_target = make_continuous_neutral_weights(sc, inv_vol, gross_leverage=g_lev, power=1.85).mul(lev_mult, axis=0)
        eff_3h = get_effective_weights(w_target, eval_idx, 12, max_w, g_lev, 0.05)
        eff_8h = get_effective_weights(w_target, eval_idx, 32, max_w, g_lev, 0.04)
        eff_blend = 0.40 * eff_3h + 0.60 * eff_8h

        for lbl, ew in [
            (f"15m-SubHourly Pullback (Reb 3h, {g_lev}x)", eff_3h),
            (f"15m-SubHourly Pullback (Reb 8h, {g_lev}x)", eff_8h),
            (f"Harmonic Blend 40% 3h + 60% 8h ({g_lev}x)", eff_blend),
        ]:
            r = eval_eff_weights(ew, close, funding)
            m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in r["monthly"].items()])
            print(
                f"{lbl:<43} | Net:{r['net_ret_pct']:+7.2f}% (${r['final_eq']:,.0f}) | "
                f"Daily:{r['avg_daily_pct']:+6.3f}% [{r['ci_daily'][0]:+.2f}%,{r['ci_daily'][1]:+.2f}%] | "
                f"Shp:{r['sharpe']:4.2f} [{r['ci_sharpe'][0]:.2f},{r['ci_sharpe'][1]:.2f}] | "
                f"Sort:{r['sortino']:4.2f} | MDD:{r['max_dd_pct']:6.2f}% | WinD:{r['win_days_pct']:4.1f}% | [{m_str}]"
            )


if __name__ == "__main__":
    main()
