"""
Diagnoses all strategy mechanisms and factor return profiles across the ENTIRE 22-month dataset
(All of 2025 + All of 2026: 2025-01-01 to 2026-10-20, 15,787 hours).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/Users/santim/.gemini/antigravity-cli/brain/f2deb7ce-5226-4e45-937f-49a27050b4d0/scratch")

from explore_high_return_strategies import TOP15, load_matrices
from test_neutral_alphas import make_continuous_neutral_weights


def sim_22m(
    w_df: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    reb_hours: int = 8,
    gross_lev: float = 2.0,
    max_single: float = 0.40,
    deadband: float = 0.04,
    fee_rate: float = 0.0002,
):
    eval_idx = close.index[close.index >= pd.Timestamp("2025-01-01 00:00:00", tz="UTC")]
    rets = close.pct_change().reindex(eval_idx).fillna(0.0).values
    fund = funding.reindex(eval_idx).fillna(0.0).values
    w_raw = w_df.reindex(eval_idx).fillna(0.0).values

    n_bars, n_cols = w_raw.shape
    eff_arr = np.zeros_like(w_raw)
    prev_w = np.zeros(n_cols)

    for t in range(n_bars):
        if t % reb_hours == 0:
            row = w_raw[t].copy()
            g_init = np.sum(np.abs(row))
            if g_init > 1e-8:
                target_g = min(g_init, gross_lev)
                row = np.clip(row, -max_single, max_single)
                pos = np.maximum(row, 0.0)
                neg = np.minimum(row, 0.0)
                if pos.sum() > 0:
                    pos = pos * ((target_g / 2.0) / pos.sum())
                if np.abs(neg.sum()) > 0:
                    neg = neg * ((target_g / 2.0) / np.abs(neg.sum()))
                desired = np.clip(pos + neg, -max_single, max_single)
            else:
                desired = np.zeros(n_cols)

            delta = np.abs(desired - prev_w)
            upd = np.where((desired == 0.0) | (delta >= deadband), desired, prev_w)
            g = np.sum(np.abs(upd))
            if g > gross_lev:
                upd = upd * (gross_lev / g)
            eff_arr[t] = upd
            prev_w = upd
        else:
            eff_arr[t] = prev_w

    exec_w = np.vstack([np.zeros((1, n_cols)), eff_arr[:-1]])
    turn = np.sum(np.abs(np.diff(exec_w, axis=0, prepend=0.0)), axis=1)
    bar_net = np.sum(exec_w * rets, axis=1) - np.sum(exec_w * fund, axis=1) - turn * fee_rate
    eq = pd.Series(1000.0 * np.cumprod(1.0 + bar_net), index=eval_idx)

    daily_eq = eq.resample("1D").last().dropna()
    d_rets = daily_eq.pct_change().dropna()
    mean_d = float(d_rets.mean())
    std_d = float(d_rets.std())
    sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
    peaks = eq.cummax()
    mdd = float(((eq - peaks) / peaks).min() * 100.0)

    monthly = daily_eq.resample("ME").last().pct_change()
    monthly.iloc[0] = (daily_eq.resample("ME").last().iloc[0] / 1000.0) - 1.0

    eq_2025 = float(eq.loc["2025-12-31"].iloc[-1])
    ret_2025 = (eq_2025 / 1000.0 - 1.0) * 100.0
    ret_2026 = (float(eq.iloc[-1]) / eq_2025 - 1.0) * 100.0

    return {
        "ret_2025": ret_2025,
        "ret_2026": ret_2026,
        "total_ret": (float(eq.iloc[-1]) / 1000.0 - 1.0) * 100.0,
        "final_eq": float(eq.iloc[-1]),
        "daily_avg": mean_d * 100.0,
        "sharpe": sharpe,
        "mdd": mdd,
        "pos_m": int((monthly > 0).sum()),
        "tot_m": len(monthly),
        "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
        "bar_net": pd.Series(bar_net, index=eval_idx),
    }


def main():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(
        TOP15, start_date="2024-11-01", end_date="2026-10-20 18:00:00"
    )
    rets_1h = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)
    inv_vol = (1.0 / vol_24h).clip(upper=100.0)

    # BTC Beta & Idiosyncratic Residuals
    btc_ret = rets_1h["BTCUSDT"]
    beta_btc = (rets_1h.rolling(72).cov(btc_ret).div(btc_ret.rolling(72).var().replace(0, 1e-8), axis=0)).clip(0.2, 2.5).fillna(1.0)
    resid_ret = rets_1h.sub(beta_btc.mul(btc_ret, axis=0))

    def z_h(s, h):
        return s.rolling(h).sum() / (vol_24h * np.sqrt(h))

    # Horizons from 2h up to 720h (30 days)
    horizons = [2, 4, 8, 12, 24, 48, 72, 120, 168, 336, 720]
    z_raw = {h: z_h(log_rets, h) for h in horizons}
    z_idio = {h: z_h(resid_ret, h) for h in horizons}

    print("=" * 135)
    print("1. TERM STRUCTURE OF RETURNS ACROSS ALL OF 2025 vs ALL OF 2026 (2.0x Gross, Reb 8h, Maker 0.02%)")
    print("=" * 135)
    print(f"{'Factor / Horizon':<34} | {'2025 Ret':>9} | {'2026 Ret':>9} | {'Total 22M':>10} | {'DailyAvg':>8} | {'Sharpe':>6} | {'MaxDD':>7} | {'PosMos':>6}")
    print("-" * 135)

    factor_returns = {}
    for h in horizons:
        # Momentum (+z) and Mean Reversion (-z) for both Raw and Idiosyncratic
        for prefix, d_map, sign in [
            (f"Mom_Raw_{h}h", z_raw, 1.0),
            (f"Rev_Raw_{h}h", z_raw, -1.0),
            (f"Mom_Idio_{h}h", z_idio, 1.0),
            (f"Rev_Idio_{h}h", z_idio, -1.0),
        ]:
            w = make_continuous_neutral_weights(sign * d_map[h], inv_vol, gross_leverage=2.0, power=1.5)
            r = sim_22m(w, close, funding, reb_hours=8, gross_lev=2.0, max_single=0.40, deadband=0.04)
            factor_returns[prefix] = r
            if r["total_ret"] > 15.0 or h in [4, 12, 24, 48, 168, 336]:
                print(
                    f"{prefix:<34} | {r['ret_2025']:>+8.1f}% | {r['ret_2026']:>+8.1f}% | {r['total_ret']:>+9.1f}% | "
                    f"{r['daily_avg']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['mdd']:>6.2f}% | {r['pos_m']:>2d}/{r['tot_m']}"
                )

    # Also test Funding Carry, Breakout, Volume-Weighted Momentum, and Baseline Meta-Allocator signals
    ann_fund_72h = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    carry_neg = -ann_fund_72h
    carry_pos = ann_fund_72h  # Trend-following funding (long high positive funding? or short high funding?)
    for name, sc in [
        ("Funding_Carry_Contrarian (-fund)", carry_neg),
        ("Funding_Trend_Momentum (+fund)", carry_pos),
        ("Idio_Pullback (24h+48h - 0.3*4h)", 0.5 * z_idio[24] + 0.5 * z_idio[48] - 0.3 * z_idio[4]),
        ("Macro_Momentum (168h+336h - 0.3*24h)", 0.5 * z_raw[168] + 0.5 * z_raw[336] - 0.3 * z_raw[24]),
        ("Idio_Macro_Mom (168h+336h - 0.3*24h)", 0.5 * z_idio[168] + 0.5 * z_idio[336] - 0.3 * z_idio[24]),
        ("Short_Term_Reversion (-4h - 12h)", -0.5 * z_raw[4] - 0.5 * z_raw[12]),
        ("Idio_Short_Reversion (-4h - 12h)", -0.5 * z_idio[4] - 0.5 * z_idio[12]),
    ]:
        w = make_continuous_neutral_weights(sc, inv_vol, gross_leverage=2.0, power=1.5)
        r = sim_22m(w, close, funding, reb_hours=8, gross_lev=2.0, max_single=0.40, deadband=0.04)
        factor_returns[name] = r
        print(
            f"{name:<34} | {r['ret_2025']:>+8.1f}% | {r['ret_2026']:>+8.1f}% | {r['total_ret']:>+9.1f}% | "
            f"{r['daily_avg']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['mdd']:>6.2f}% | {r['pos_m']:>2d}/{r['tot_m']}"
        )


if __name__ == "__main__":
    main()
