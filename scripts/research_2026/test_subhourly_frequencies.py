"""
Empirical Benchmark of Sub-Hourly Frequencies: 1m, 5m, 15m, 30m vs 1h, 2h, 4h.
1. Full 6-Month Evaluation (2026-04-20 to 2026-10-20) using 15m (17,644 bars) and 30m (8,822 bars):
   - Naive high-frequency rebalancing (15m, 30m, 1h, 2h, 4h) -> Gross PnL vs Fee Drag vs Net PnL.
   - Sub-hourly microstructure enhancements (Taker Order Flow Imbalance, Parkinson High-Low Volatility,
     15m/30m pullback entry timing, and adaptive deadband filtering).
2. 31-Day Evaluation on 5m Bars (2026-09-20 to 2026-10-20, ~8,856 bars):
   - 5m vs 15m vs 30m vs 1h vs 4h cadence comparison + 5m Microstructure Sniper.
3. 7-Day Evaluation on 1m Bars (2026-10-13 to 2026-10-20, ~10,800 bars):
   - 1m vs 5m vs 15m vs 1h cadence comparison + 1m Order Flow / Reversal Sniper.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path("/tmp/crypto_alpha_backtest_cache")
TOP15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "FETUSDT",
]


def load_timeframe_matrices(tf: str, start_date: str, end_date: str = "2026-10-20 18:00:00"):
    bars_dir = CACHE_DIR / "bars" / tf
    funding_dir = CACHE_DIR / "funding"
    dfs = {}
    f_dfs = {}
    for s in TOP15:
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

    close = pd.DataFrame({s: dfs[s]["close"].reindex(idx).ffill().bfill() for s in TOP15})
    open_ = pd.DataFrame({s: dfs[s]["open"].reindex(idx).ffill().bfill() for s in TOP15})
    high = pd.DataFrame({s: dfs[s]["high"].reindex(idx).ffill().bfill() for s in TOP15})
    low = pd.DataFrame({s: dfs[s]["low"].reindex(idx).ffill().bfill() for s in TOP15})
    volume = pd.DataFrame({s: dfs[s]["volume"].reindex(idx).fillna(0.0) for s in TOP15})
    taker_buy = pd.DataFrame({
        s: dfs[s]["taker_buy_vol"].reindex(idx).fillna(0.0) if "taker_buy_vol" in dfs[s].columns else pd.Series(0.0, index=idx)
        for s in TOP15
    })

    # Funding settlement only occurs exactly at 00:00, 08:00, 16:00 UTC (minute == 0)
    is_settle = pd.Series((idx.hour.isin([0, 8, 16])) & (idx.minute == 0), index=idx)
    raw_funding = pd.DataFrame(index=idx, columns=TOP15, dtype=float)
    funding = pd.DataFrame(index=idx, columns=TOP15, dtype=float)
    for s in TOP15:
        if s in f_dfs and not f_dfs[s].empty:
            fs = f_dfs[s]["funding_rate"].reindex(idx, method="ffill").fillna(0.0)
            raw_funding[s] = fs
            funding[s] = fs * is_settle
        else:
            raw_funding[s] = 0.0
            funding[s] = 0.0

    return idx, open_, high, low, close, volume, taker_buy, raw_funding, funding


def cs_rank_centered(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df.rank(axis=1, pct=True)
    return ranks.sub(ranks.mean(axis=1), axis=0).fillna(0.0)


def build_neutral_weights(
    score_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    gross_target: pd.Series | float = 2.0,
    power: float = 1.8,
    max_single_weight: float = 0.45,
) -> pd.DataFrame:
    centered = cs_rank_centered(score_df)
    raw = np.sign(centered) * (np.abs(centered) ** power)
    inv_vol = 1.0 / np.maximum(vol_df, 0.002)
    raw = raw * inv_vol
    pos = raw.clip(lower=0.0)
    neg = raw.clip(upper=0.0)
    pos_sum = pos.sum(axis=1).replace(0.0, np.nan)
    neg_sum = neg.abs().sum(axis=1).replace(0.0, np.nan)
    if isinstance(gross_target, (int, float)):
        half_g = gross_target / 2.0
        w = pos.div(pos_sum, axis=0).fillna(0.0) * half_g + neg.div(neg_sum, axis=0).fillna(0.0) * half_g
    else:
        half_g = (gross_target / 2.0).reindex(score_df.index).fillna(0.0)
        w = pos.div(pos_sum, axis=0).fillna(0.0).mul(half_g, axis=0) + neg.div(neg_sum, axis=0).fillna(0.0).mul(half_g, axis=0)
    return w.clip(lower=-max_single_weight, upper=max_single_weight)


def sim_subhourly(
    weights_df: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    eval_start: str,
    reb_every_bars: int = 1,
    deadband: float = 0.04,
    max_gross: float = 2.2,
    fee_rate: float = 0.0002,
):
    eval_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_idx = close.index[close.index >= eval_ts]
    rets = close.pct_change().reindex(eval_idx).fillna(0.0)
    fund = funding.reindex(eval_idx).fillna(0.0)
    w_raw = weights_df.reindex(eval_idx).fillna(0.0)

    gross = w_raw.abs().sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / np.maximum(gross, 1e-8), 1.0)
    w_raw = w_raw.mul(scale, axis=0)

    w_arr = w_raw.values
    eff_arr = np.zeros_like(w_arr)
    prev_w = np.zeros(w_arr.shape[1])

    for t in range(len(w_arr)):
        if t % reb_every_bars == 0:
            desired = w_arr[t]
            delta = np.abs(desired - prev_w)
            upd = np.where((desired == 0.0) | (delta >= deadband), desired, prev_w)
            g = np.sum(np.abs(upd))
            if g > max_gross:
                upd = upd * (max_gross / g)
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
    gross_bar_ret = price_ret + fund_ret
    net_bar_ret = gross_bar_ret - fee_drag

    eq_gross = 1000.0 * (1.0 + gross_bar_ret).cumprod()
    eq_net = 1000.0 * (1.0 + net_bar_ret).cumprod()

    daily_eq = eq_net.resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    mean_d = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0
    std_d = float(daily_rets.std()) if len(daily_rets) > 1 else 0.0
    sharpe = (mean_d / std_d) * np.sqrt(365.0) if std_d > 1e-12 else 0.0
    down = daily_rets[daily_rets < 0]
    dd_std = float(np.sqrt((down ** 2).mean())) if len(down) > 0 else 1e-12
    sortino = (mean_d / dd_std) * np.sqrt(365.0) if dd_std > 1e-12 else 0.0
    peaks = eq_net.cummax()
    mdd = float(((eq_net - peaks) / peaks).min() * 100.0)
    win_rate = float((daily_rets > 0).mean() * 100.0) if len(daily_rets) > 0 else 0.0

    # Fee total in % of initial capital (compounded difference) and sum of fee_drag
    total_turnover = float(turnover.sum())
    sum_fee_pct = float(fee_drag.sum() * 100.0)
    gross_ret_pct = float((eq_gross.iloc[-1] / 1000.0 - 1.0) * 100.0)
    net_ret_pct = float((eq_net.iloc[-1] / 1000.0 - 1.0) * 100.0)

    monthly = daily_eq.resample("ME").last().pct_change()
    first_m_start = eq_net.loc[eq_net.index.month == eval_ts.month].iloc[0]
    first_m_end = daily_eq.resample("ME").last().iloc[0]
    monthly.iloc[0] = (first_m_end / first_m_start) - 1.0

    return {
        "gross_ret_pct": gross_ret_pct,
        "net_ret_pct": net_ret_pct,
        "final_eq": float(eq_net.iloc[-1]),
        "avg_daily_pct": mean_d * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": mdd,
        "win_days_pct": win_rate,
        "total_turnover": total_turnover,
        "sum_fee_pct": sum_fee_pct,
        "monthly": {k.strftime("%Y-%m"): float(v * 100.0) for k, v in monthly.items()},
    }


def run_15m_6month_experiments():
    print("=" * 115)
    print("PART 1: FULL 6-MONTH SUB-HOURLY BENCHMARK ON 15m & 30m BARS (2026-04-20 to 2026-10-20, 17,644 bars)")
    print("=" * 115)
    idx, open_, high, low, close, volume, taker_buy, raw_funding, funding = load_timeframe_matrices(
        "15m", start_date="2026-03-20", end_date="2026-10-20 18:00:00"
    )
    print(f"Loaded 15m matrix: {len(idx):,} bars ({idx[0]} -> {idx[-1]})")

    # Note: on 15m bars, 1 hour = 4 bars
    ret_1bar = close.pct_change(1)   # 15m
    ret_2bar = close.pct_change(2)   # 30m
    ret_4bar = close.pct_change(4)   # 1h
    ret_8bar = close.pct_change(8)   # 2h
    ret_12bar = close.pct_change(12) # 3h
    ret_16bar = close.pct_change(16) # 4h
    ret_48bar = close.pct_change(48) # 12h
    ret_96bar = close.pct_change(96) # 24h
    ret_192bar = close.pct_change(192) # 48h
    ret_288bar = close.pct_change(288) # 72h

    vol_15m = ret_1bar.rolling(96).std().bfill().fillna(0.01) # 24h rolling 15m vol
    # Parkinson high-low range estimator over 24h (96 bars)
    hl_log = (np.log(high / np.maximum(low, 1e-8)) ** 2)
    parkinson_vol = np.sqrt(hl_log.rolling(96).mean() / (4.0 * np.log(2.0))).bfill().fillna(0.01)

    dollar_vol = close * volume
    vol_surge = (
        dollar_vol.rolling(24).mean() / dollar_vol.rolling(192).mean().replace(0.0, np.nan)
    ).clip(0.5, 3.0).fillna(1.0)

    # Order Flow Imbalance (OFI): net taker buy volume divided by total volume
    ofi_1bar = ((2.0 * taker_buy - volume) / volume.replace(0.0, np.nan)).fillna(0.0)
    ofi_4h = ((2.0 * taker_buy.rolling(16).sum() - volume.rolling(16).sum()) / volume.rolling(16).sum().replace(0.0, np.nan)).fillna(0.0)
    ofi_24h = ((2.0 * taker_buy.rolling(96).sum() - volume.rolling(96).sum()) / volume.rolling(96).sum().replace(0.0, np.nan)).fillna(0.0)

    # Idiosyncratic residuals vs BTC
    btc_r1 = ret_1bar["BTCUSDT"]
    btc_var = btc_r1.rolling(288).var().replace(0.0, np.nan)
    betas = pd.DataFrame(index=idx, columns=TOP15, dtype=float)
    for s in TOP15:
        if s == "BTCUSDT":
            betas[s] = 1.0
        else:
            betas[s] = (ret_1bar[s].rolling(288).cov(btc_r1) / btc_var).clip(0.2, 2.5).fillna(1.0)

    ir_2h = ret_8bar.sub(betas.mul(ret_8bar["BTCUSDT"], axis=0))
    ir_4h = ret_16bar.sub(betas.mul(ret_16bar["BTCUSDT"], axis=0))
    ir_12h = ret_48bar.sub(betas.mul(ret_48bar["BTCUSDT"], axis=0))
    ir_24h = ret_96bar.sub(betas.mul(ret_96bar["BTCUSDT"], axis=0))
    ir_48h = ret_192bar.sub(betas.mul(ret_192bar["BTCUSDT"], axis=0))
    ir_72h = ret_288bar.sub(betas.mul(ret_288bar["BTCUSDT"], axis=0))

    # 1. Sub-hourly Term Structure Isolation (evaluated at 1h = 4 bars cadence)
    print("\n--- A. SUB-HOURLY TERM STRUCTURE OF CROSS-SECTIONAL RETURNS (15m to 48h) ---")
    for label, s_df, reb_b in [
        ("Mom_15m (1 bar)", ret_1bar, 1),
        ("Mom_30m (2 bars)", ret_2bar, 2),
        ("Mom_1h  (4 bars)", ret_4bar, 4),
        ("Mom_2h  (8 bars)", ret_8bar, 8),
        ("Mom_4h  (16 bars)", ret_16bar, 16),
        ("Mom_12h (48 bars)", ret_48bar, 16),
        ("Mom_24h (96 bars)", ret_96bar, 16),
        ("OFI_24h (Taker Order Flow)", ofi_24h, 16),
        ("Rev_15m (-ret_15m)", -ret_1bar, 1),
        ("Rev_30m (-ret_30m)", -ret_2bar, 2),
        ("Rev_1h  (-ret_1h)", -ret_4bar, 4),
        ("Rev_4h  (-ret_4h)", -ret_16bar, 16),
    ]:
        w = build_neutral_weights(s_df, vol_15m, gross_target=2.0, power=1.5, max_single_weight=0.40)
        r = sim_subhourly(w, close, funding, "2026-04-20 00:00:00", reb_every_bars=reb_b, deadband=0.04, max_gross=2.0)
        print(
            f"{label:<28} | Gross: {r['gross_ret_pct']:+8.2f}% | Fees: {r['sum_fee_pct']:6.2f}% | "
            f"Net: {r['net_ret_pct']:+7.2f}% | Daily: {r['avg_daily_pct']:+6.3f}% | Sharpe: {r['sharpe']:5.2f}"
        )

    # 2. Core Pullback-Momentum + Regime Sensor across 15m, 30m, 1h, 2h, 3h, 4h Rebalance Cadences
    raw_pullback = (0.35 * ret_48bar + 0.45 * ret_96bar + 0.20 * ret_192bar) * np.sqrt(vol_surge) - 0.25 * ret_16bar
    idio_pullback = (0.35 * ir_12h + 0.45 * ir_24h + 0.20 * ir_72h) * np.sqrt(vol_surge) - 0.20 * ir_4h
    mom_score = 0.60 * cs_rank_centered(raw_pullback) + 0.40 * cs_rank_centered(idio_pullback)
    carry_score = cs_rank_centered(-raw_funding.rolling(96).mean().fillna(0.0))

    # CS-AutoCorr 48h (192 bars of 15m)
    cs_ac = ret_96bar.corrwith(ret_96bar.shift(96), axis=1).rolling(192).mean().fillna(0.0)
    mom_tilt = ((cs_ac - (-0.14)) / (-0.04 - (-0.14))).clip(0.0, 1.0)
    gross_s = 1.1 + 1.1 * mom_tilt

    base_blend = mom_score.mul(mom_tilt, axis=0) + 0.30 * carry_score
    w_base = build_neutral_weights(base_blend, vol_15m, gross_target=gross_s, power=1.8, max_single_weight=0.45)

    print("\n--- B. REBALANCE FREQUENCY SWEEP (15m vs 30m vs 1h vs 2h vs 3h vs 4h) ON SAME ALPHA ---")
    for freq_label, reb_bars, db in [
        ("15m Rebalance (db=2%)", 1, 0.02),
        ("15m Rebalance (db=8%)", 1, 0.08),
        ("15m Rebalance (db=14%)", 1, 0.14),
        ("30m Rebalance (db=2%)", 2, 0.02),
        ("30m Rebalance (db=8%)", 2, 0.08),
        ("30m Rebalance (db=12%)", 2, 0.12),
        ("1h  Rebalance (db=4%)", 4, 0.04),
        ("1h  Rebalance (db=8%)", 4, 0.08),
        ("1h  Rebalance (db=12%)", 4, 0.12),
        ("2h  Rebalance (db=4%)", 8, 0.04),
        ("2h  Rebalance (db=8%)", 8, 0.08),
        ("3h  Rebalance (db=4%)", 12, 0.04),
        ("3h  Rebalance (db=6%)", 12, 0.06),
        ("4h  Rebalance (db=4%)", 16, 0.04),
        ("4h  Rebalance (db=6%)", 16, 0.06),
    ]:
        r = sim_subhourly(w_base, close, funding, "2026-04-20 00:00:00", reb_every_bars=reb_bars, deadband=db, max_gross=2.2)
        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in r["monthly"].items()])
        neg_m = sum(1 for v in r["monthly"].values() if v < 0)
        print(
            f"{freq_label:<25} | Gross: {r['gross_ret_pct']:+7.1f}% | Fees: {r['sum_fee_pct']:5.1f}% | "
            f"Net: {r['net_ret_pct']:+7.2f}% | Daily: {r['avg_daily_pct']:+6.3f}% | Shp: {r['sharpe']:4.2f} | "
            f"MDD: {r['max_dd_pct']:6.2f}% | NegM:{neg_m} [{m_str}]"
        )

    # 3. Sub-Hourly Microstructure Alpha (Using 15m/30m Pullback Dip + Parkinson Vol + Taker Order Flow)
    print("\n--- C. SUB-HOURLY MICROSTRUCTURE ALPHA ENHANCEMENTS (15m/30m Pullback + Order Flow + Parkinson Vol) ---")
    # Test fine-grained pullback windows (e.g. 2h + 3h + 4h pullback + 15m/30m Taker Order Flow confirmation)
    sub_pullback_raw = (
        0.35 * ret_48bar + 0.45 * ret_96bar + 0.20 * ret_192bar
    ) * np.sqrt(vol_surge) - (0.10 * ret_8bar + 0.20 * ret_16bar)
    sub_pullback_idio = (
        0.35 * ir_12h + 0.45 * ir_24h + 0.20 * ir_72h
    ) * np.sqrt(vol_surge) - (0.10 * ir_2h + 0.15 * ir_4h)

    for ofi_w in [0.0, 0.10, 0.15, -0.10]:
        for use_parkinson in [False, True]:
            v_est = parkinson_vol if use_parkinson else vol_15m
            v_lbl = "ParkinsonVol" if use_parkinson else "CloseVol"
            score_enh = (
                0.55 * cs_rank_centered(sub_pullback_raw)
                + 0.45 * cs_rank_centered(sub_pullback_idio)
                + ofi_w * cs_rank_centered(ofi_24h - ofi_4h)
            )
            blend_enh = score_enh.mul(mom_tilt, axis=0) + 0.30 * carry_score
            for gross_max in [2.2, 2.4, 2.5]:
                g_target = (gross_max / 2.0) + (gross_max / 2.0) * mom_tilt
                w_enh = build_neutral_weights(blend_enh, v_est, gross_target=g_target, power=1.85, max_single_weight=0.45 * (gross_max / 2.2))
                for reb_b, db, f_lbl in [(4, 0.10, "1h_db10"), (8, 0.08, "2h_db08"), (12, 0.06, "3h_db06"), (16, 0.04, "4h_db04")]:
                    r = sim_subhourly(w_enh, close, funding, "2026-04-20 00:00:00", reb_every_bars=reb_b, deadband=db, max_gross=gross_max)
                    neg_m = sum(1 for v in r["monthly"].values() if v < 0)
                    if r["net_ret_pct"] > 155.0 and neg_m == 0:
                        m_str = " ".join([f"{k[5:]}:{v:+.1f}%" for k, v in r["monthly"].items()])
                        print(
                            f"OFI={ofi_w:+.2f} {v_lbl:<12} G={gross_max:.1f} {f_lbl:<8} | "
                            f"Net: {r['net_ret_pct']:+7.2f}% (${r['final_eq']:,.0f}) | Daily: {r['avg_daily_pct']:+6.3f}% | "
                            f"Shp: {r['sharpe']:4.2f} | Sort: {r['sortino']:4.2f} | MDD: {r['max_dd_pct']:6.2f}% | [{m_str}]"
                        )


def run_5m_and_1m_experiments():
    print("\n" + "=" * 115)
    print("PART 2: ULTRA-HIGH FREQUENCY 5m (31 DAYS) AND 1m (7.5 DAYS) EMPIRICAL TESTS")
    print("=" * 115)

    # 5m Evaluation (2026-09-20 to 2026-10-20)
    idx5, o5, h5, l5, c5, v5, tb5, rf5, f5 = load_timeframe_matrices(
        "5m", start_date="2026-09-19", end_date="2026-10-20 18:00:00"
    )
    print(f"Loaded 5m matrix: {len(idx5):,} bars ({idx5[0]} -> {idx5[-1]})")
    # 1h = 12 bars, 4h = 48 bars, 12h = 144 bars, 24h = 288 bars, 48h = 576 bars
    r5_1 = c5.pct_change(1)     # 5m
    r5_3 = c5.pct_change(3)     # 15m
    r5_6 = c5.pct_change(6)     # 30m
    r5_12 = c5.pct_change(12)   # 1h
    r5_48 = c5.pct_change(48)   # 4h
    r5_144 = c5.pct_change(144) # 12h
    r5_288 = c5.pct_change(288) # 24h
    vol5 = r5_1.rolling(288).std().bfill().fillna(0.005)
    ofi5_15m = ((2.0 * tb5.rolling(3).sum() - v5.rolling(3).sum()) / v5.rolling(3).sum().replace(0.0, np.nan)).fillna(0.0)

    # Compare 5m vs 15m vs 30m vs 1h vs 4h over the last 30 days (2026-09-21 to 2026-10-20)
    score_5m_rev = -r5_3 + 0.3 * ofi5_15m
    score_pullback = (0.4 * r5_144 + 0.6 * r5_288) - 0.25 * r5_48
    w_5m_rev = build_neutral_weights(score_5m_rev, vol5, gross_target=2.0, power=1.8, max_single_weight=0.40)
    w_5m_pb = build_neutral_weights(score_pullback, vol5, gross_target=2.0, power=1.8, max_single_weight=0.40)

    print("\n--- 5m Dataset (2026-09-21 to 2026-10-20, 30 Days): Gross vs Fee Drag vs Net ---")
    for name, w_df, reb_b, db in [
        ("5m Fast Reversion (Reb 5m, db=2%)", w_5m_rev, 1, 0.02),
        ("5m Fast Reversion (Reb 15m, db=6%)", w_5m_rev, 3, 0.06),
        ("Pullback Mom (Reb 5m, db=2%)", w_5m_pb, 1, 0.02),
        ("Pullback Mom (Reb 15m, db=4%)", w_5m_pb, 3, 0.04),
        ("Pullback Mom (Reb 30m, db=4%)", w_5m_pb, 6, 0.04),
        ("Pullback Mom (Reb 1h, db=6%)", w_5m_pb, 12, 0.06),
        ("Pullback Mom (Reb 4h, db=4%)", w_5m_pb, 48, 0.04),
    ]:
        res = sim_subhourly(w_df, c5, f5, "2026-09-21 00:00:00", reb_every_bars=reb_b, deadband=db, max_gross=2.0)
        print(
            f"{name:<38} | Gross: {res['gross_ret_pct']:+7.2f}% | Fees: {res['sum_fee_pct']:6.2f}% | "
            f"Net: {res['net_ret_pct']:+7.2f}% | Daily: {res['avg_daily_pct']:+6.3f}% | Shp: {res['sharpe']:5.2f}"
        )

    # 1m Evaluation (2026-10-13 to 2026-10-20)
    idx1, o1, h1, l1, c1, v1, tb1, rf1, f1 = load_timeframe_matrices(
        "1m", start_date="2026-10-13 06:00:00", end_date="2026-10-20 18:00:00"
    )
    print(f"\nLoaded 1m matrix: {len(idx1):,} bars ({idx1[0]} -> {idx1[-1]})")
    r1_1 = c1.pct_change(1)     # 1m
    r1_5 = c1.pct_change(5)     # 5m
    r1_15 = c1.pct_change(15)   # 15m
    r1_60 = c1.pct_change(60)   # 1h
    r1_240 = c1.pct_change(240) # 4h
    r1_1440 = c1.pct_change(1440) # 24h
    vol1 = r1_1.rolling(240).std().bfill().fillna(0.002)

    w_1m_rev = build_neutral_weights(-r1_5, vol1, gross_target=2.0, power=1.8, max_single_weight=0.40)
    w_1m_pb = build_neutral_weights(r1_1440 - 0.25 * r1_240, vol1, gross_target=2.0, power=1.8, max_single_weight=0.40)

    print("\n--- 1m Dataset (2026-10-14 to 2026-10-20, 6.75 Days): Gross vs Fee Drag vs Net ---")
    for name, w_df, reb_b, db in [
        ("1m Micro-Reversion (Reb 1m, db=2%)", w_1m_rev, 1, 0.02),
        ("1m Micro-Reversion (Reb 5m, db=4%)", w_1m_rev, 5, 0.04),
        ("Pullback Mom (Reb 1m, db=2%)", w_1m_pb, 1, 0.02),
        ("Pullback Mom (Reb 5m, db=4%)", w_1m_pb, 5, 0.04),
        ("Pullback Mom (Reb 15m, db=4%)", w_1m_pb, 15, 0.04),
        ("Pullback Mom (Reb 1h, db=6%)", w_1m_pb, 60, 0.06),
        ("Pullback Mom (Reb 4h, db=4%)", w_1m_pb, 240, 0.04),
    ]:
        res = sim_subhourly(w_df, c1, f1, "2026-10-14 00:00:00", reb_every_bars=reb_b, deadband=db, max_gross=2.0)
        print(
            f"{name:<38} | Gross: {res['gross_ret_pct']:+7.2f}% | Fees: {res['sum_fee_pct']:6.2f}% | "
            f"Net: {res['net_ret_pct']:+7.2f}% | Daily: {res['avg_daily_pct']:+6.3f}% | Shp: {res['sharpe']:5.2f}"
        )


if __name__ == "__main__":
    run_15m_6month_experiments()
    run_5m_and_1m_experiments()
