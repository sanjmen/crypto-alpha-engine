"""
Testing Strictly Dollar-Neutral & Beta-Neutral Alpha Mechanisms and Regime-Adaptive Ensembles.
Goal: Maximize daily return, Sharpe/Sortino, and month-by-month consistency (Apr-Oct 2026).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))

from explore_high_return_strategies import load_matrices, TOP15, ALL19
from deep_strategy_research import simulate_portfolio


def make_dollar_neutral_weights(
    score_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame,
    top_k: int = 3,
    gross_leverage: float = 2.0,
    min_score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Converts raw cross-sectional scores into strictly dollar-neutral weights:
    sum(longs) = +gross_leverage / 2
    sum(shorts) = -gross_leverage / 2
    Net exposure = 0.0 at every single timestamp.
    """
    n_sym = score_df.shape[1]
    ranks = score_df.rank(axis=1, method="first")

    long_mask = (ranks > (n_sym - top_k)) & (score_df > min_score_threshold)
    short_mask = (ranks <= top_k) & (score_df < -min_score_threshold)

    # Only trade when we have at least 1 valid long AND 1 valid short so we stay 100% dollar-neutral
    valid_both = (long_mask.sum(axis=1) > 0) & (short_mask.sum(axis=1) > 0)

    wl = inv_vol_df.where(long_mask, 0.0)
    wl = wl.div(wl.sum(axis=1).replace(0, 1.0), axis=0) * (gross_leverage / 2.0)

    ws = inv_vol_df.where(short_mask, 0.0)
    ws = ws.div(ws.sum(axis=1).replace(0, 1.0), axis=0) * (gross_leverage / 2.0)

    net_w = (wl - ws).mul(valid_both.astype(float), axis=0)
    return net_w


def make_continuous_neutral_weights(
    score_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame,
    gross_leverage: float = 2.0,
    power: float = 1.5,
) -> pd.DataFrame:
    """
    Cross-sectionally demeans scores, raises to power to emphasize tails,
    weights by inverse volatility, and enforces exact dollar neutrality (Long = Short = gross/2).
    """
    # Standardize cross-sectionally per row
    mean_s = score_df.mean(axis=1)
    std_s = score_df.std(axis=1).replace(0, 1.0)
    z = score_df.sub(mean_s, axis=0).div(std_s, axis=0).fillna(0.0)

    # Non-linear tail emphasis
    tilted = np.sign(z) * (np.abs(z) ** power) * inv_vol_df

    pos = tilted.clip(lower=0.0)
    neg = (-tilted).clip(lower=0.0)

    valid = (pos.sum(axis=1) > 1e-8) & (neg.sum(axis=1) > 1e-8)
    wl = pos.div(pos.sum(axis=1).replace(0, 1.0), axis=0) * (gross_leverage / 2.0)
    ws = neg.div(neg.sum(axis=1).replace(0, 1.0), axis=0) * (gross_leverage / 2.0)

    return (wl - ws).mul(valid.astype(float), axis=0)


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

    candidates = {}

    # -------------------------------------------------------------------------
    # A. Multi-Horizon Cross-Sectional Momentum (Raw vs Beta-Neutral Idiosyncratic Momentum)
    # -------------------------------------------------------------------------
    mom_raw = (
        0.25 * (log_rets.rolling(12).sum() / (vol_24h * np.sqrt(12)))
        + 0.35 * (log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24)))
        + 0.25 * (log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72)))
        + 0.15 * (log_rets.rolling(168).sum() / (vol_24h * np.sqrt(168)))
    ) * np.sqrt(vol_surge)

    # Idiosyncratic (Beta-adjusted) Momentum: strips out BTC beta so we rank pure alt alpha!
    mom_idio = (
        0.25 * (resid_ret.rolling(12).sum() / (vol_24h * np.sqrt(12)))
        + 0.35 * (resid_ret.rolling(24).sum() / (vol_24h * np.sqrt(24)))
        + 0.25 * (resid_ret.rolling(72).sum() / (vol_24h * np.sqrt(72)))
        + 0.15 * (resid_ret.rolling(168).sum() / (vol_24h * np.sqrt(168)))
    ) * np.sqrt(vol_surge)

    for name_m, s_df in [("RawMom", mom_raw), ("IdioMom", mom_idio)]:
        for k in [3, 4]:
            for reb in [4, 8]:
                w = pd.DataFrame(0.0, index=idx, columns=ALL19)
                w[TOP15] = make_dollar_neutral_weights(s_df, inv_vol, top_k=k, gross_leverage=2.0)
                candidates[f"A_{name_m}_Top{k}_Reb{reb}h_Lev2.0"] = (w, reb, 0.40, 2.0)

        for power in [1.0, 1.5, 2.0]:
            for reb in [4, 8]:
                w = pd.DataFrame(0.0, index=idx, columns=ALL19)
                w[TOP15] = make_continuous_neutral_weights(s_df, inv_vol, gross_leverage=2.0, power=power)
                candidates[f"A_{name_m}_ContP{power}_Reb{reb}h_Lev2.0"] = (w, reb, 0.40, 2.0)

    # -------------------------------------------------------------------------
    # B. Strictly Dollar-Neutral Funding Carry Arbitrage (Multi-Window 24h + 72h)
    # -------------------------------------------------------------------------
    # Negative of annualized funding rate = carry alpha
    carry_24h = -rf15.rolling(24, min_periods=1).mean() * 3.0 * 365.0
    carry_72h = -rf15.rolling(72, min_periods=1).mean() * 3.0 * 365.0
    carry_blend = 0.5 * carry_24h + 0.5 * carry_72h

    for name_c, c_df in [("Carry72h", carry_72h), ("CarryBlend", carry_blend)]:
        for reb in [4, 8]:
            w1 = pd.DataFrame(0.0, index=idx, columns=ALL19)
            # Only trade assets where |annualized funding| >= 8%
            c_filtered = c_df.where(c_df.abs() >= 0.08, 0.0)
            w1[TOP15] = make_continuous_neutral_weights(c_filtered, inv_vol, gross_leverage=2.0, power=1.0)
            candidates[f"B_{name_c}_Cont_Reb{reb}h_Lev2.0"] = (w1, reb, 0.40, 2.0)

            w2 = pd.DataFrame(0.0, index=idx, columns=ALL19)
            w2[TOP15] = make_dollar_neutral_weights(c_df, inv_vol, top_k=3, gross_leverage=2.0, min_score_threshold=0.05)
            candidates[f"B_{name_c}_Top3_Reb{reb}h_Lev2.0"] = (w2, reb, 0.40, 2.0)

    # -------------------------------------------------------------------------
    # C. Strictly Dollar-Neutral Mean Reversion (72h Z-Score + Hurst Filter + Idiosyncratic Reversion)
    # -------------------------------------------------------------------------
    for win in [48, 72, 96]:
        # Price z-score
        ma_p = c15.rolling(win).mean()
        sd_p = c15.rolling(win).std().replace(0, 1e-8)
        z_p = (c15 - ma_p) / sd_p
        mr_score = -z_p.where(z_p.abs() >= 1.5, 0.0)

        w_mr = pd.DataFrame(0.0, index=idx, columns=ALL19)
        w_mr[TOP15] = make_continuous_neutral_weights(mr_score, inv_vol, gross_leverage=1.8, power=1.0)
        candidates[f"C_StatArb_Z{win}h_Cont_Reb8h_Lev1.8"] = (w_mr, 8, 0.35, 1.8)

        # Residual return z-score
        res_sum = resid_ret.rolling(win).sum()
        res_std = resid_ret.rolling(win).std() * np.sqrt(win) + 1e-8
        z_res = res_sum / res_std
        mr_res_score = -z_res.where(z_res.abs() >= 1.5, 0.0)
        w_mr_res = pd.DataFrame(0.0, index=idx, columns=ALL19)
        w_mr_res[TOP15] = make_continuous_neutral_weights(mr_res_score, inv_vol, gross_leverage=1.8, power=1.0)
        candidates[f"C_ResidStatArb_Z{win}h_Cont_Reb4h_Lev1.8"] = (w_mr_res, 4, 0.35, 1.8)

    # -------------------------------------------------------------------------
    # D. Breakout Closeness / Donchian Position Rank (Strictly Dollar-Neutral!)
    # Instead of binary breakout, rank all 15 coins by where Close sits in [Low_N, High_N]
    # Coins near 48h High -> Long; Coins near 48h Low -> Short (100% dollar-neutral!)
    # -------------------------------------------------------------------------
    for chan in [24, 48, 96]:
        hh = h15.rolling(chan).max()
        ll = l15.rolling(chan).min()
        pos_in_chan = ((c15 - ll) / (hh - ll + 1e-8)) - 0.5  # [-0.5, +0.5]
        brk_score = pos_in_chan * np.sqrt(vol_surge)
        for reb in [4, 8]:
            w_b = pd.DataFrame(0.0, index=idx, columns=ALL19)
            w_b[TOP15] = make_dollar_neutral_weights(brk_score, inv_vol, top_k=3, gross_leverage=2.0)
            candidates[f"D_DonchianRank_C{chan}h_Top3_Reb{reb}h_Lev2.0"] = (w_b, reb, 0.40, 2.0)

    # Evaluate all candidates
    all_res = {}
    for name, (w_df, reb, max_s, max_g) in candidates.items():
        all_res[name] = simulate_portfolio(
            w_df, close, funding,
            rebalance_every_n_hours=reb,
            max_single_weight=max_s,
            max_gross_leverage=max_g,
            deadband=0.04,
        )

    sorted_res = sorted(all_res.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print("=" * 165)
    print(f"{'Strategy Name':<44} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'Calmar':>6} | {'MaxDD':>7} | {'WinDays':>7} | {'PosMos':>6}")
    print("-" * 165)
    for name, r in sorted_res[:20]:
        pm = f"{r['positive_months']}/{r['total_months']}"
        print(
            f"{name:<44} | ${r['final_equity']:>8.2f} | {r['total_ret_pct']:>+7.2f}% | "
            f"{r['avg_daily_ret_pct']:>+8.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['calmar']:>6.2f} | {r['max_dd_pct']:>6.2f}% | {r['win_days_pct']:>6.1f}% | {pm:>6}"
        )

    print("\n" + "=" * 165)
    print("MONTHLY BREAKDOWN OF TOP 12 BY SHARPE")
    print("-" * 165)
    months = list(sorted_res[0][1]["monthly_rets"].keys())
    print(f"{'Strategy Name':<44} | " + " | ".join(f"{m:>8}" for m in months))
    print("-" * 165)
    for name, r in sorted_res[:12]:
        m_str = " | ".join(f"{r['monthly_rets'].get(m, 0.0):>+7.2f}%" for m in months)
        print(f"{name:<44} | {m_str}")


if __name__ == "__main__":
    main()
