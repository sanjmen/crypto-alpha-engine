"""
Systematic Quantitative Strategy Discovery & High-Frequency / Multi-Mechanism Benchmark.
Period: 2026-04-20 to 2026-10-20 (6 months, 4,411 hourly bars, 183.8 days).
Strict out-of-sample causality (weights at bar t execute on return at bar t+1),
realistic Maker (0.02%) and Taker (0.04%) transaction fees, and 8h funding settlements.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path("/Users/santim/Projects/trading/crypto-alpha-engine")
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = Path("/tmp/crypto_alpha_backtest_cache")
TOP15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "FETUSDT",
]
ALL19 = TOP15 + ["CTSIUSDT", "MAGICUSDT", "OGNUSDT", "RLCUSDT"]


def load_matrices(symbols=TOP15, start_date="2026-03-15", end_date="2026-10-20 18:00:00"):
    bars_dir = CACHE_DIR / "bars" / "1h"
    funding_dir = CACHE_DIR / "funding"

    dfs = {}
    f_dfs = {}
    for s in symbols:
        df = pd.read_parquet(bars_dir / f"{s}.parquet")
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
        dfs[s] = df

        fp = funding_dir / f"{s}.parquet"
        if fp.exists():
            fdf = pd.read_parquet(fp)
            fdf["dt"] = pd.to_datetime(fdf["timestamp"], unit="s", utc=True)
            fdf = fdf.sort_values("dt").drop_duplicates("dt").set_index("dt")
            f_dfs[s] = fdf

    idx = dfs["BTCUSDT"].index
    idx = idx[(idx >= pd.Timestamp(start_date, tz="UTC")) & (idx <= pd.Timestamp(end_date, tz="UTC"))]

    close = pd.DataFrame({s: dfs[s]["close"].reindex(idx).ffill().bfill() for s in symbols})
    open_ = pd.DataFrame({s: dfs[s]["open"].reindex(idx).ffill().bfill() for s in symbols})
    high = pd.DataFrame({s: dfs[s]["high"].reindex(idx).ffill().bfill() for s in symbols})
    low = pd.DataFrame({s: dfs[s]["low"].reindex(idx).ffill().bfill() for s in symbols})
    volume = pd.DataFrame({s: dfs[s]["volume"].reindex(idx).fillna(0.0) for s in symbols})
    dollar_vol = close * volume

    funding = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    is_settle = pd.Series(idx.hour.isin([0, 8, 16]), index=idx)
    raw_funding = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    for s in symbols:
        if s in f_dfs and not f_dfs[s].empty:
            fs = f_dfs[s]["funding_rate"].reindex(idx, method="ffill").fillna(0.0)
            raw_funding[s] = fs
            funding[s] = fs * is_settle
        else:
            raw_funding[s] = 0.0
            funding[s] = 0.0

    return idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding


def evaluate_weights(
    weights_df: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    eval_start: str = "2026-04-20 00:00:00",
    initial_capital: float = 1000.0,
    fee_rate: float = 0.0002,
    deadband: float = 0.02,
    rebalance_every_n_hours: int = 1,
    max_gross_leverage: float = 2.0,
):
    eval_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]
    rets = close.pct_change().reindex(eval_idx).fillna(0.0)
    fund = funding.reindex(eval_idx).fillna(0.0)
    w_raw = weights_df.reindex(eval_idx).fillna(0.0)

    # Enforce max gross leverage per row
    gross = w_raw.abs().sum(axis=1)
    scale = np.where(gross > max_gross_leverage, max_gross_leverage / np.maximum(gross, 1e-8), 1.0)
    w_raw = w_raw.mul(scale, axis=0)

    # Apply rebalance cadence & turnover deadband
    w_arr = w_raw.values
    eff_arr = np.zeros_like(w_arr)
    prev_w = np.zeros(w_arr.shape[1])

    for t in range(len(w_arr)):
        if t % rebalance_every_n_hours == 0:
            desired = w_arr[t]
            delta = np.abs(desired - prev_w)
            # If desired == 0 (closing position), always allow close; else apply deadband
            update_mask = (delta > deadband) | ((np.abs(desired) < 1e-6) & (np.abs(prev_w) > 0))
            prev_w = np.where(update_mask, desired, prev_w)
        eff_arr[t] = prev_w

    w_eff = pd.DataFrame(eff_arr, index=eval_idx, columns=close.columns)
    # Shift by 1 bar: signal at bar t-1 earns return at bar t
    w_held = w_eff.shift(1).fillna(0.0)

    gross_ret = (w_held * rets).sum(axis=1)
    turnover = (w_eff - w_held).abs().sum(axis=1)
    fees = turnover * fee_rate
    fund_ret = -(w_held * fund).sum(axis=1)
    net_ret = gross_ret - fees + fund_ret

    equity = initial_capital * (1.0 + net_ret).cumprod()
    daily_eq = equity.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    total_ret_pct = (equity.iloc[-1] / initial_capital - 1.0) * 100.0
    avg_daily_ret_pct = daily_ret.mean() * 100.0
    med_daily_ret_pct = daily_ret.median() * 100.0
    win_days_pct = (daily_ret > 0).mean() * 100.0
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(365)
    downside = daily_ret[daily_ret < 0].std() + 1e-9
    sortino = (daily_ret.mean() / downside) * np.sqrt(365)
    dd = (equity / equity.cummax() - 1.0).min() * 100.0
    total_fees_usd = (fees * equity.shift(1).fillna(initial_capital)).sum()
    total_fund_usd = (fund_ret * equity.shift(1).fillna(initial_capital)).sum()
    avg_lev = w_held.abs().sum(axis=1).mean()

    return {
        "final_equity": float(equity.iloc[-1]),
        "total_ret_pct": float(total_ret_pct),
        "avg_daily_ret_pct": float(avg_daily_ret_pct),
        "med_daily_ret_pct": float(med_daily_ret_pct),
        "win_days_pct": float(win_days_pct),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd_pct": float(dd),
        "avg_leverage": float(avg_lev),
        "fees_usd": float(total_fees_usd),
        "funding_usd": float(total_fund_usd),
        "equity_series": equity,
        "net_ret_series": net_ret,
    }


def main():
    idx, open_, high, low, close, volume, dollar_vol, raw_funding, funding = load_matrices(ALL19)
    rets_1h = close.pct_change().fillna(0.0)
    log_rets = np.log(close / close.shift(1)).fillna(0.0)
    vol_24h = rets_1h.rolling(24).std().replace(0, np.nan).ffill().fillna(0.02)

    results = {}

    # -------------------------------------------------------------------------
    # MECHANISM 1: High-Conviction Concentrated Cross-Sectional Momentum (Top K vs Bottom K)
    # Instead of diluting across all 15 coins, concentrate in Top K winners and Bottom K losers
    # -------------------------------------------------------------------------
    mom_fast = (
        0.35 * (log_rets.rolling(6).sum() / (vol_24h * np.sqrt(6)))
        + 0.35 * (log_rets.rolling(24).sum() / (vol_24h * np.sqrt(24)))
        + 0.30 * (log_rets.rolling(72).sum() / (vol_24h * np.sqrt(72)))
    )
    # Volume confirmation boost: relative 6h volume vs 48h volume
    vol_surge = (dollar_vol.rolling(6).mean() / (dollar_vol.rolling(48).mean() + 1e-8)).clip(0.5, 2.5)
    mom_score = mom_fast * np.sqrt(vol_surge)

    for k in [2, 3]:
        for reb in [2, 4, 8]:
            for lev in [1.5, 2.0]:
                ranks = mom_score[TOP15].rank(axis=1, method="first")
                n_sym = len(TOP15)
                w = pd.DataFrame(0.0, index=idx, columns=ALL19)
                long_mask = ranks > (n_sym - k)
                short_mask = ranks <= k
                # Inverse volatility weighting within top K and bottom K
                inv_vol = (1.0 / vol_24h[TOP15]).clip(upper=100.0)
                w_long = inv_vol.where(long_mask, 0.0)
                w_long = w_long.div(w_long.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
                w_short = inv_vol.where(short_mask, 0.0)
                w_short = w_short.div(w_short.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
                w[TOP15] = w_long - w_short
                res = evaluate_weights(w, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=lev, deadband=0.04)
                results[f"M1_Concentrated_CS_Mom_Top{k}_Reb{reb}h_Lev{lev}"] = res

    # -------------------------------------------------------------------------
    # MECHANISM 2: Time-Series Trend + Cross-Sectional Breakout (Directional + Relative)
    # Allows net directional exposure when market breadth & BTC trend align!
    # -------------------------------------------------------------------------
    # Individual asset trend: EMA(8) vs EMA(24) and EMA(24) vs EMA(72) normalized by vol
    ema8 = close.ewm(span=8).mean()
    ema24 = close.ewm(span=24).mean()
    ema72 = close.ewm(span=72).mean()
    ema168 = close.ewm(span=168).mean()

    ts_trend = (
        0.4 * ((close - ema24) / (close * vol_24h * np.sqrt(24)))
        + 0.4 * ((ema8 - ema72) / (close * vol_24h * np.sqrt(72)))
        + 0.2 * ((ema24 - ema168) / (close * vol_24h * np.sqrt(168)))
    )
    for reb in [2, 4, 8]:
        for lev in [1.5, 2.0]:
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            sub_t = ts_trend[TOP15]
            # Only trade strong signals |z| > 0.75
            active = sub_t.where(sub_t.abs() > 0.75, 0.0)
            denom = active.abs().sum(axis=1).replace(0, 1.0)
            w[TOP15] = active.div(denom, axis=0) * lev
            res = evaluate_weights(w, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=lev, deadband=0.05)
            results[f"M2_TimeSeries_Trend_Reb{reb}h_Lev{lev}"] = res

    # -------------------------------------------------------------------------
    # MECHANISM 3: Funding Squeeze & Carry Arbitrage (Double Alpha: Carry + Squeeze Momentum)
    # Short overheated positive funding WHEN momentum stalls/reverses;
    # Long deeply negative funding WHEN momentum turns positive (short squeeze!)
    # -------------------------------------------------------------------------
    ann_fund = raw_funding * 3.0 * 365.0
    ret_12h = log_rets.rolling(12).sum()
    ret_4h = log_rets.rolling(4).sum()

    for reb in [4, 8]:
        for lev in [1.5, 2.0]:
            # Pure concentrated Top-3 vs Bottom-3 Funding Carry
            f_sub = ann_fund[ALL19]
            ranks = f_sub.rank(axis=1, method="first")
            n_s = len(ALL19)
            k = 3
            # High funding -> Short; Low/negative funding -> Long
            short_mask = (ranks > (n_s - k)) & (f_sub > 0.08)
            long_mask = (ranks <= k) & (f_sub < 0.05)
            inv_v = (1.0 / vol_24h[ALL19]).clip(upper=100.0)
            wl = inv_v.where(long_mask, 0.0)
            wl = wl.div(wl.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
            ws = inv_v.where(short_mask, 0.0)
            ws = ws.div(ws.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
            w_carry = wl - ws
            res = evaluate_weights(w_carry, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=lev, deadband=0.04)
            results[f"M3_Concentrated_Carry_Top3_Reb{reb}h_Lev{lev}"] = res

            # Funding + Momentum alignment (Carry + Trend filter: don't short a coin exploding +10% in 4h, don't long a coin crashing -10% in 4h)
            safe_short = short_mask & (ret_4h[ALL19] < 0.04)
            safe_long = long_mask & (ret_4h[ALL19] > -0.04)
            wl2 = inv_v.where(safe_long, 0.0)
            wl2 = wl2.div(wl2.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
            ws2 = inv_v.where(safe_short, 0.0)
            ws2 = ws2.div(ws2.sum(axis=1).replace(0, 1.0), axis=0) * (lev / 2.0)
            w_carry_safe = wl2 - ws2
            res2 = evaluate_weights(w_carry_safe, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=lev, deadband=0.04)
            results[f"M3b_Carry_With_TrendGuard_Reb{reb}h_Lev{lev}"] = res2

    # -------------------------------------------------------------------------
    # MECHANISM 4: Intraday Relative-Value Mean-Reversion / Residual OU StatArb (1h / 2h / 4h)
    # Neutralize against BTC beta and fade short-horizon (12h-24h) idiosyncratic overextensions
    # when candle shows rejection wick!
    # -------------------------------------------------------------------------
    btc_ret = rets_1h["BTCUSDT"]
    # Rolling 72h beta to BTC
    cov_btc = rets_1h.rolling(72).cov(btc_ret)
    var_btc = btc_ret.rolling(72).var().replace(0, 1e-8)
    beta_btc = cov_btc.div(var_btc, axis=0).clip(0.2, 2.5)

    # Idiosyncratic (residual) return = r_i - beta_i * r_BTC
    resid_1h = rets_1h.sub(beta_btc.mul(btc_ret, axis=0))
    for win in [12, 24, 36]:
        resid_cum = resid_1h.rolling(win).sum()
        resid_std = resid_1h.rolling(win).std() * np.sqrt(win)
        resid_z = resid_cum / (resid_std + 1e-8)
        for reb in [1, 2, 4]:
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            z_sub = resid_z[TOP15]
            # Fade extreme idiosyncratic moves |z| > 1.5
            raw_sig = -z_sub.where(z_sub.abs() > 1.5, 0.0)
            # Cross-sectionally demean
            demeaned = raw_sig.sub(raw_sig.mean(axis=1), axis=0)
            denom = demeaned.abs().sum(axis=1).replace(0, 1.0)
            w[TOP15] = demeaned.div(denom, axis=0) * 1.5
            res = evaluate_weights(w, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=1.5, deadband=0.04)
            results[f"M4_Residual_BetaNeutral_StatArb_W{win}h_Reb{reb}h"] = res

    # -------------------------------------------------------------------------
    # MECHANISM 5: BTC Lead-Lag Beta Catch-Up (High-Frequency 1h-2h Horizon)
    # When BTC moves strongly over last 2h-4h, buy high-beta alts that lagged BTC's move,
    # or ride high-beta alts in direction of BTC impulse
    # -------------------------------------------------------------------------
    for horizon in [2, 4, 6]:
        btc_imp = log_rets["BTCUSDT"].rolling(horizon).sum()
        btc_vol = log_rets["BTCUSDT"].rolling(24).std() * np.sqrt(horizon)
        btc_z = btc_imp / (btc_vol + 1e-8)

        expected_alt_move = beta_btc[TOP15].mul(btc_imp, axis=0)
        actual_alt_move = log_rets[TOP15].rolling(horizon).sum()
        lag_gap = expected_alt_move - actual_alt_move  # Positive = alt lagged BTC rally

        for reb in [1, 2]:
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            # Only trigger when BTC has a meaningful impulse |btc_z| > 1.0
            active_btc = (btc_z.abs() > 1.0).astype(float)
            sig = lag_gap.mul(active_btc, axis=0)
            sig["BTCUSDT"] = 0.0
            denom = sig.abs().sum(axis=1).replace(0, 1.0)
            w[TOP15] = sig.div(denom, axis=0) * 1.5
            res = evaluate_weights(w, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=1.5, deadband=0.05)
            results[f"M5_BTC_LeadLag_CatchUp_H{horizon}h_Reb{reb}h"] = res

    # -------------------------------------------------------------------------
    # MECHANISM 6: Intraday Breakout + Volatility Compression Squeeze (Bollinger/Keltner Breakout)
    # Enters when price breaks 24h/48h channel after low-volatility compression + volume surge
    # -------------------------------------------------------------------------
    for chan in [24, 48]:
        hh = high[TOP15].shift(1).rolling(chan).max()
        ll = low[TOP15].shift(1).rolling(chan).min()
        brk_up = (close[TOP15] > hh) & (vol_surge[TOP15] > 1.3)
        brk_dn = (close[TOP15] < ll) & (vol_surge[TOP15] > 1.3)

        # Hold for 6 hours after breakout using rolling max of signal
        long_hold = brk_up.astype(float).rolling(6).max()
        short_hold = brk_dn.astype(float).rolling(6).max()
        net_brk = long_hold - short_hold

        for reb in [1, 2, 4]:
            w = pd.DataFrame(0.0, index=idx, columns=ALL19)
            denom = net_brk.abs().sum(axis=1).replace(0, 1.0)
            w[TOP15] = net_brk.div(denom, axis=0) * 1.5
            res = evaluate_weights(w, close, funding, rebalance_every_n_hours=reb, max_gross_leverage=1.5, deadband=0.05)
            results[f"M6_Channel_Breakout_C{chan}h_Reb{reb}h"] = res

    # Sort and print top results
    sorted_res = sorted(results.items(), key=lambda x: x[1]["total_ret_pct"], reverse=True)
    print(f"{'Strategy Name':<50} | {'Final Eq':>9} | {'6M Ret':>8} | {'Daily Avg':>9} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7} | {'WinDays':>7} | {'Fees':>7} | {'FundPnL':>8}")
    print("-" * 145)
    for name, r in sorted_res[:25]:
        print(
            f"{name:<50} | ${r['final_equity']:>8.2f} | {r['total_ret_pct']:>+7.2f}% | "
            f"{r['avg_daily_ret_pct']:>+8.3f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['max_dd_pct']:>6.2f}% | {r['win_days_pct']:>6.1f}% | ${r['fees_usd']:>6.2f} | ${r['funding_usd']:>+7.2f}"
        )


if __name__ == "__main__":
    main()
