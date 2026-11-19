#!/usr/bin/env python3
"""
Unified Quantitative & Microstructure Benchmark Suite CLI (Issue #66).

Consolidates all 5 empirical subsystem benchmarks into a single parameterized CLI:
- `--suite hrp`: Hierarchical Risk Parity (HRP) vs Equal-Weight vs Inverse-Variance Walk-Forward Benchmark (Issue #36)
- `--suite info-bars`: Information-Driven Dollar Bars & Dollar Imbalance Bars vs Time Bars Normality Benchmark (Issues #32-34)
- `--suite market-impact`: Square-Root Market Impact Law, L2 BookDepth Calibration & Flash-Crash Slippage Benchmark (Issue #24)
- `--suite anomalies`: Pure Event-Driven Extreme Anomaly Strategies Backtest (OversoldBounce, EarlyBreakout, PumpFundingFade)
- `--suite cointegration`: Sector Clustering & Engle-Granger / OU Cointegrated Pairs Stat-Arb Benchmark (Issue #22)
- `--suite all`: Executes all 5 benchmark suites sequentially.

Usage:
    python scripts/run_benchmarks.py --suite hrp --lookback 4000
    python scripts/run_benchmarks.py --suite market-impact
    python scripts/run_benchmarks.py --suite cointegration --lookback 5000
    python scripts/run_benchmarks.py --suite anomalies
    python scripts/run_benchmarks.py --suite info-bars --symbol BTCUSDT --date 2024-03-01
    python scripts/run_benchmarks.py --suite all
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.agg_trades_streamer import AggTradesStreamer
from src.features.cointegration import EngleGrangerTester
from src.features.dollar_bars import (
    DollarBarSynthesizer,
    DollarImbalanceBarSynthesizer,
    StatisticalValidator,
)
from src.features.market_impact import (
    CapitalCapacityCeiling,
    NonLinearMarketImpactModel,
)
from src.features.multimodal_alpha import (
    MultiModalFeatureConfig,
    attach_forward_returns_for_audit,
    compute_cross_sectional_panel_features,
    compute_symbol_multimodal_features,
    evaluate_walk_forward_feature_atlas,
    list_multimodal_feature_columns,
)
from src.features.sector_clustering import SectorTaxonomy, StatisticalSectorClusterer
from src.risk.hrp_allocator import HierarchicalRiskParity
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
)


def _load_aligned_prices(
    symbols: list[str],
    cache_dir: Path,
    lookback_bars: int = 5000,
) -> pd.DataFrame:
    """Loads 1h close prices aligned on UTC timestamps from `cache_dir`."""
    series_dict = {}
    for sym in symbols:
        p = cache_dir / "bars" / "1h" / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).set_index("datetime")
        series_dict[sym] = df["close"]

    df_prices = pd.DataFrame(series_dict).dropna(how="all")
    if lookback_bars is not None and len(df_prices) > lookback_bars:
        df_prices = df_prices.iloc[-lookback_bars:]

    valid_cols = [c for c in df_prices.columns if df_prices[c].notna().mean() >= 0.8]
    if not valid_cols:
        return pd.DataFrame()
    return df_prices[valid_cols].ffill().bfill()


# ==============================================================================
# 1. HRP ALLOCATION BENCHMARK
# ==============================================================================
def run_hrp_benchmark(
    cache_dir: Path,
    symbols: list[str],
    lookback: int = 4000,
    window: int = 720,
    rebalance: int = 24,
    output_dir: Path = REPO_ROOT / "reports",
) -> pd.DataFrame:
    print("=" * 80)
    print("  [SUITE: HRP] HIERARCHICAL RISK PARITY (HRP) PORTFOLIO ALLOCATION BENCHMARK")
    print("=" * 80)
    prices = _load_aligned_prices(symbols, cache_dir, lookback_bars=lookback)
    if prices.empty:
        raise RuntimeError(f"No valid price series found in {cache_dir} for HRP benchmark.")
    returns = np.log(prices / prices.shift(1)).dropna()
    cols = list(returns.columns)
    n_assets = len(cols)
    n_bars = len(returns)

    hrp_single = HierarchicalRiskParity(linkage_method="single")
    hrp_ward = HierarchicalRiskParity(linkage_method="ward")

    port_rets = {
        "equal_weight": np.zeros(n_bars),
        "inverse_variance": np.zeros(n_bars),
        "hrp_single": np.zeros(n_bars),
        "hrp_ward": np.zeros(n_bars),
    }
    curr_w = {k: np.full(n_assets, 1.0 / n_assets) for k in port_rets}

    for t in range(window, n_bars):
        if (t - window) % rebalance == 0:
            cov = returns.iloc[t - window : t].cov().values
            curr_w["equal_weight"] = np.full(n_assets, 1.0 / n_assets)
            curr_w["inverse_variance"] = hrp_single.compute_inverse_variance(cov).values
            curr_w["hrp_single"] = hrp_single.allocate(cov).values
            curr_w["hrp_ward"] = hrp_ward.allocate(cov).values
        step_r = returns.iloc[t].values
        for k, arr in port_rets.items():
            arr[t] = float(np.sum(curr_w[k] * step_r))

    df_port = pd.DataFrame(port_rets, index=returns.index).iloc[window:]
    rows = {}
    for col in df_port.columns:
        r = df_port[col].values
        cagr = float(np.mean(r) * 24 * 365)
        vol = float((np.std(r) + 1e-8) * np.sqrt(24 * 365))
        sharpe = cagr / max(vol, 1e-6)
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - peak) / peak))
        rows[col] = {
            "total_return": float(cum[-1] - 1.0),
            "cagr": cagr,
            "annual_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": abs(cagr / max_dd) if max_dd < 0 else 0.0,
        }
    df_metrics = pd.DataFrame(rows).T
    print(df_metrics[["total_return", "cagr", "annual_vol", "sharpe", "max_drawdown", "calmar"]].to_string())
    output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_dir / "hrp_benchmark_summary.csv")
    return df_metrics


# ==============================================================================
# 2. INFORMATION-DRIVEN BARS BENCHMARK
# ==============================================================================
def run_info_bars_benchmark(
    symbol: str = "BTCUSDT",
    date_str: str = "2024-03-01",
    dollar_threshold: float = 10_000_000.0,
    dib_threshold: float = 1_000_000.0,
) -> dict[str, dict[str, float]]:
    print("=" * 80)
    print(f"  [SUITE: INFO-BARS] INFORMATION-DRIVEN BARS STATISTICAL BENCHMARK ({symbol} {date_str})")
    print("=" * 80)
    streamer = AggTradesStreamer()
    df = streamer.fetch_daily_trades(symbol, date_str)
    if df is None or df.empty:
        raise RuntimeError(f"Could not stream aggTrades for {symbol} on {date_str}")

    dollar_bars = DollarBarSynthesizer(dollar_threshold=dollar_threshold).process_ticks(df)
    dib_bars = DollarImbalanceBarSynthesizer(initial_threshold=dib_threshold, ewma_alpha=0.05).process_ticks(df)
    time_bars_1m = df.set_index("timestamp")["price"].resample("1min").ohlc().dropna()
    time_bars_5m = df.set_index("timestamp")["price"].resample("5min").ohlc().dropna()
    time_bars_15m = df.set_index("timestamp")["price"].resample("15min").ohlc().dropna()

    benchmarks = {
        "1-Minute Time Bars": time_bars_1m,
        "5-Minute Time Bars": time_bars_5m,
        "15-Minute Time Bars": time_bars_15m,
        f"Dollar Bars (${dollar_threshold / 1e6:.0f}M)": dollar_bars,
        "Dollar Imbalance Bars (DIB)": dib_bars,
    }
    results: dict[str, dict[str, float]] = {}
    for name, bar_df in benchmarks.items():
        rets = StatisticalValidator.compute_log_returns(bar_df)
        jb_stats = StatisticalValidator.test_normality_jarque_bera(rets)
        jb_stats["autocorr_lag1"] = StatisticalValidator.compute_autocorrelation(rets, lag=1)
        jb_stats["num_bars"] = len(bar_df)
        results[name] = jb_stats
        print(
            f"  {name:<30} | Bars: {len(bar_df):5d} | Excess Kurt: {jb_stats['excess_kurtosis']:+6.2f} | "
            f"JB: {jb_stats['jb_stat']:8.1f} (p={jb_stats['p_value']:.4f}) | AC(1): {jb_stats['autocorr_lag1']:+.3f}"
        )
    return results


# ==============================================================================
# 3. MARKET IMPACT & SLIPPAGE BENCHMARK
# ==============================================================================
def _fetch_or_synthesize_book_depth() -> pd.DataFrame:
    url = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-05-01.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CryptoAlphaEngine/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp, zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            return pd.read_csv(zf.open(zf.namelist()[0]))
    except Exception:
        base_notionals = {
            1: 69_523_632.53,
            2: 170_426_571.06,
            3: 231_449_061.72,
            4: 274_299_631.63,
            5: 304_171_829.53,
        }
        records = []
        for pct, notional in base_notionals.items():
            records.append({"percentage": pct, "depth": notional / 60_000.0, "notional": notional})
            records.append({"percentage": -pct, "depth": notional / 60_000.0, "notional": notional})
        return pd.DataFrame(records)


def run_market_impact_benchmark() -> pd.DataFrame:
    print("=" * 80)
    print("  [SUITE: MARKET-IMPACT] NON-LINEAR MARKET IMPACT & CAPACITY CEILING BENCHMARK")
    print("=" * 80)
    book_depth_df = _fetch_or_synthesize_book_depth()
    btc_daily_volume = 3_500_000_000.0
    btc_daily_vol = 0.032
    btc_hourly_volume = btc_daily_volume / 24.0
    btc_hourly_vol = btc_daily_vol / np.sqrt(24.0)

    model = NonLinearMarketImpactModel()
    calibrated = model.calibrate_from_book_depth(
        book_depth_df=book_depth_df,
        reference_volatility=btc_daily_vol,
        reference_volume=btc_daily_volume,
        fit_exponent=False,
    )
    print(
        f"  Calibrated Impact Coefficient Y = {calibrated.Y_coefficient:.4f} | Exponent alpha = {calibrated.alpha_exponent:.2f}"
    )

    ceiling = CapitalCapacityCeiling(max_depth_fraction=0.05, max_volume_fraction=0.02)
    order_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    rows = []
    for q in order_sizes:
        est_norm = model.estimate_slippage(
            order_size_usd=q,
            symbol="BTCUSDT",
            volatility=btc_hourly_vol,
            volume=btc_hourly_volume,
            depth_1pct_usd=69_500_000.0,
            cs_spread=0.00010,
            is_flash_crash=False,
        )
        est_crash = model.estimate_slippage(
            order_size_usd=q,
            symbol="BTCUSDT",
            volatility=btc_hourly_vol * 6.0,
            volume=btc_hourly_volume * 0.25,
            depth_1pct_usd=69_500_000.0 * 0.25,
            cs_spread=0.0025,
            is_flash_crash=True,
        )
        cap_rep = ceiling.evaluate_capacity("BTCUSDT", q, 69_500_000.0, btc_hourly_volume)
        rows.append(
            {
                "order_usd": q,
                "normal_bps": round(est_norm.slippage_bps, 2),
                "flash_crash_bps": round(est_crash.slippage_bps, 2),
                "within_capacity": cap_rep.within_capacity,
                "twap_chunks": cap_rep.recommended_chunks,
            }
        )
    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    return df_res


# ==============================================================================
# 4. PURE ANOMALIES BENCHMARK
# ==============================================================================
def run_pure_anomalies_benchmark(cache_dir: Path) -> pd.DataFrame:
    print("=" * 80)
    print("  [SUITE: ANOMALIES] PURE EVENT-DRIVEN EXTREME ANOMALY STRATEGIES BENCHMARK")
    print("=" * 80)
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "RENDERUSDT",
        "FETUSDT",
        "CTSIUSDT",
        "MAGICUSDT",
        "OGNUSDT",
        "RLCUSDT",
        "GTCUSDT",
        "COMPUSDT",
        "CRVUSDT",
        "DYDXUSDT",
        "WALUSDT",
        "BSPUSDT",
        "TRUTHUSDT",
        "ZEREBROUSDT",
        "BTWUSDT",
        "QUSDT",
        "AVAUSDT",
        "EPICUSDT",
        "MUBARAKUSDT",
        "CRWVUSDT",
        "CAPUSDT",
    ]
    bars_dir = cache_dir / "bars" / "1h"
    funding_dir = cache_dir / "funding"
    bars, funding = {}, {}
    for s in symbols:
        bp = bars_dir / f"{s}.parquet"
        if bp.exists():
            df = pd.read_parquet(bp)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            bars[s] = df
        fp = funding_dir / f"{s}.parquet"
        if fp.exists():
            fdf = pd.read_parquet(fp)
            if "timestamp" in fdf.columns:
                fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], unit="s", utc=True)
                fdf.set_index("timestamp", inplace=True)
            fdf.sort_index(inplace=True)
            funding[s] = fdf

    if "BTCUSDT" not in bars:
        raise RuntimeError(f"BTCUSDT.parquet not found in {bars_dir}")

    eval_start = pd.Timestamp("2026-04-14", tz="UTC")
    eval_end = pd.Timestamp("2026-10-14 23:00:00", tz="UTC")
    anchor_idx = bars["BTCUSDT"].index
    sim_idx = anchor_idx[(anchor_idx >= eval_start) & (anchor_idx <= eval_end)]

    strat_bounce = OversoldBounceStrategy(dump_threshold_24h=-0.15, rsi_threshold=28.0)
    strat_breakout = EarlyBreakoutStrategy(pump_threshold_4h=0.08, volume_multiplier=2.5)
    strat_fade = PumpFundingFadeStrategy(pump_threshold_24h=0.25, min_funding_rate=0.0005)

    trades = []
    for pos in range(30, len(sim_idx)):
        ts = sim_idx[pos]
        window_start = sim_idx[max(0, pos - 30)]
        sub_market = {
            s: df.loc[(df.index >= window_start) & (df.index <= ts)]
            for s, df in bars.items()
            if len(df.loc[(df.index >= window_start) & (df.index <= ts)]) >= 26
        }
        sub_funding = {s: funding[s].loc[funding[s].index <= ts] for s in sub_market if s in funding}

        for strat_name, sigs, horizon_h, is_short in [
            ("OversoldBounce", strat_bounce.generate_signals(sub_market), 24, False),
            ("EarlyBreakout", strat_breakout.generate_signals(sub_market), 12, False),
            ("PumpFundingFade", strat_fade.generate_signals(sub_market, funding_data=sub_funding), 24, True),
        ]:
            for sig in sigs:
                meta = sig.metadata
                entry_p, sl_p, tp_p = meta["entry_price"], meta["stop_loss"], meta["take_profit"]
                sym_df = bars[sig.symbol]
                future_bars = sym_df.loc[(sym_df.index > ts) & (sym_df.index <= ts + pd.Timedelta(hours=horizon_h))]
                if future_bars.empty:
                    continue
                exit_p = float(future_bars["close"].iloc[-1])
                outcome = "TIME_EXIT"
                for _, bar in future_bars.iterrows():
                    if not is_short:
                        if bar["low"] <= sl_p:
                            exit_p, outcome = sl_p, "STOP_LOSS"
                            break
                        if bar["high"] >= tp_p:
                            exit_p, outcome = tp_p, "TAKE_PROFIT"
                            break
                    else:
                        if bar["high"] >= sl_p:
                            exit_p, outcome = sl_p, "STOP_LOSS"
                            break
                        if bar["low"] <= tp_p:
                            exit_p, outcome = tp_p, "TAKE_PROFIT"
                            break
                pnl_pct = (entry_p / exit_p - 1.0) if is_short else (exit_p / entry_p - 1.0)
                trades.append(
                    {
                        "timestamp": ts,
                        "strategy": strat_name,
                        "symbol": sig.symbol,
                        "pnl_pct": pnl_pct,
                        "outcome": outcome,
                    }
                )

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        for strat_name, grp in df_trades.groupby("strategy"):
            n_t = len(grp)
            win_rate = (grp["pnl_pct"] > 0).mean() * 100.0
            print(
                f"  {strat_name:<18} | Trades: {n_t:4d} | Win Rate: {win_rate:5.1f}% | Mean Ret: {grp['pnl_pct'].mean() * 100:+6.2f}%"
            )
    return df_trades


# ==============================================================================
# 5. SECTOR COINTEGRATION BENCHMARK
# ==============================================================================
def run_cointegration_benchmark(
    cache_dir: Path,
    lookback: int = 5000,
) -> pd.DataFrame:
    print("=" * 80)
    print("  [SUITE: COINTEGRATION] SECTOR CLUSTERING & COINTEGRATED STAT-ARB BENCHMARK")
    print("=" * 80)
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "AVAXUSDT",
        "NEARUSDT",
        "DOTUSDT",
        "LTCUSDT",
        "UNIUSDT",
        "AAVEUSDT",
        "LINKUSDT",
        "CRVUSDT",
        "LDOUSDT",
        "DOGEUSDT",
        "RENDERUSDT",
        "FETUSDT",
    ]
    prices = _load_aligned_prices(symbols, cache_dir, lookback_bars=lookback)
    if prices.empty:
        raise RuntimeError(f"No valid price series found in {cache_dir} for Cointegration benchmark.")

    tax = SectorTaxonomy()
    sector_partition = tax.filter_universe_by_sector(list(prices.columns))
    returns = np.log(prices / prices.shift(1)).dropna()
    clusterer = StatisticalSectorClusterer(linkage="ward")
    clusterer.fit(returns)

    tester = EngleGrangerTester(
        p_value_threshold=0.05,
        min_half_life=2.0,
        max_half_life=96.0,
        hurst_threshold=0.48,
    )
    rows = []
    for sec, syms in sector_partition.items():
        if len(syms) < 2:
            continue
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s1, s2 = syms[i], syms[j]
                res = tester.test_pair(prices[s1], prices[s2], y_symbol=s1, x_symbol=s2)
                if res.is_tradable:
                    rows.append(
                        {
                            "sector": sec,
                            "pair": f"{s1}/{s2}",
                            "p_value": round(res.p_value, 4),
                            "beta": round(res.hedge_ratios[s2], 4),
                            "half_life_h": round(res.half_life_bars, 1),
                            "hurst": round(res.hurst_exponent, 3),
                        }
                    )
    df_pairs = pd.DataFrame(rows)
    if not df_pairs.empty:
        print(df_pairs.to_string(index=False))
    else:
        print("  No tradable pairs met strict p < 0.05 & H < 0.48 thresholds in current window.")
    return df_pairs


# ==============================================================================
# SUITE 6: M16-2 MULTI-MODAL FEATURE IC & CONDITIONAL PAYOFF ATLAS (220 PERPS)
# ==============================================================================
def _build_existing_sleeves_daily_proxy(panel: pd.DataFrame) -> pd.Series:
    """
    Constructs a causal daily return proxy of our existing 3 production pillars
    (Core 7-Factor Cross-Sectional Momentum/Carry + P2 BTC/ETH/SOL TSMOM + Phase 2 Breakout/Pullback Snipers)
    to measure true out-of-sample orthogonality (`corr_to_existing_sleeves`) for every candidate feature.
    """
    df = panel[["decision_ts", "symbol", "ret_24h", "funding_rate_bps", "rvol_shock_1h_vs_24h", "fwd_ret_4h"]].dropna(
        subset=["fwd_ret_4h"]
    )
    if df.empty:
        return pd.Series(dtype=np.float64)
    from src.features.multimodal_alpha import _vectorized_cs_zscore

    day = pd.to_datetime((df["decision_ts"].to_numpy(dtype=np.int64) // 86400) * 86400, unit="s", utc=True)
    # Pillar 1 proxy: Cross-sectional 24h momentum minus funding carry
    p1_score = pd.Series(_vectorized_cs_zscore(df, "decision_ts", "ret_24h"), index=df.index) - 0.35 * np.sign(
        df["funding_rate_bps"]
    )
    p1_daily = (p1_score * df["fwd_ret_4h"]).groupby(day).mean()

    # Pillar 2 proxy: Majors (BTC/ETH/SOL) directional TSMOM
    majors = df[df["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
    if not majors.empty:
        m_day = pd.to_datetime(majors["decision_ts"], unit="s", utc=True).dt.floor("D")
        p2_daily = (np.sign(majors["ret_24h"]) * majors["fwd_ret_4h"]).groupby(m_day).mean()
    else:
        p2_daily = pd.Series(0.0, index=p1_daily.index)

    # Pillar 3 (Phase 2 Snipers) proxy: High-RVOL ignition / dip-bounce events
    sn_mask = df["rvol_shock_1h_vs_24h"] >= 2.2
    if int(sn_mask.sum()) >= 20:
        sn_df = df[sn_mask]
        sn_day = pd.to_datetime(sn_df["decision_ts"], unit="s", utc=True).dt.floor("D")
        p3_daily = (np.sign(sn_df["ret_24h"]) * sn_df["fwd_ret_4h"]).groupby(sn_day).mean()
    else:
        p3_daily = pd.Series(0.0, index=p1_daily.index)

    combined = (
        0.55 * p1_daily
        + 0.30 * p2_daily.reindex(p1_daily.index).fillna(0.0)
        + 0.15 * p3_daily.reindex(p1_daily.index).fillna(0.0)
    )
    return combined


def run_m16_feature_atlas_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    max_shards: int = 20,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict:
    """
    Streams all 20 shards (`220` Point-in-Time symbols, `2023–2026`) from `m16_data_dir`,
    synthesizes ~102 strictly causal multi-modal features on a `15m` microstructure grid,
    and generates the Walk-Forward (`2023–2024` Train / `2025` Val / `2026` Holdout)
    IC & Conditional Tail Payoff Atlas (`reports/m16_2_feature_atlas_*`).
    """
    if colab_session:
        from src.cloud.colab_client import ColabClient

        client = ColabClient(session_name=colab_session)
        print(f"[*] Provisioning Colab CPU session '{colab_session}' for M16-2 Feature Atlas...", flush=True)
        if not client.ensure_session(gpu=None):
            raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
        try:
            if not client.is_drive_mounted():
                rc_m, out_m, err_m = client.mount_drive("/content/drive")
                print(f"[*] Drive mount ({rc_m}): {(out_m or err_m).strip()}", flush=True)
            assert client.is_drive_mounted(), "Google Drive mount required at /content/drive/MyDrive"
            assert client.sync_codebase(REPO_ROOT), "Failed syncing project codebase bundle to Colab"
            client.exec_inline("import os; os.makedirs('/content/reports', exist_ok=True)", timeout=30)

            drive_m16 = "/content/drive/MyDrive/trading/crypto-alpha-engine/data/m16_multimodal"
            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/m16_atlas.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                f"'--suite', 'm16-feature-atlas', '--m16-data-dir', '{drive_m16}', "
                f"'--max-shards', '{max_shards}', '--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/m16_atlas.pid', 'w').write(str(p.pid)); "
                "print(f'ATLAS_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=60)
            print(f"[+] Launched remote M16-2 Atlas ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            import time

            seen = 0
            while True:
                time.sleep(20.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/m16_atlas.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/m16_atlas.log').read().splitlines() if os.path.exists('/content/m16_atlas.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, _ = client.exec_inline(poll_py, timeout=60)
                if rc_p != 0:
                    continue
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "M16_2_ATLAS_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote M16-2 Atlas process terminated before completion; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "m16_2_feature_atlas_4h.csv",
                "m16_2_feature_atlas_1h.csv",
                "m16_2_feature_atlas_summary.json",
                "m16_2_feature_atlas_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            print(f"[+] Downloaded M16-2 Atlas artifacts into {output_dir}", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            else:
                print(f"[*] Kept Colab session '{colab_session}' mounted & alive (--keep-colab).", flush=True)

    # In-VM / Local Shard-Streaming Execution
    from concurrent.futures import ThreadPoolExecutor
    import gc
    import json
    import time

    t0 = time.time()
    stage_dir: Path | None = None
    if rclone_stream:
        import shutil
        import subprocess

        stage_dir = Path("/tmp/m16_rclone_stage")
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        m16_data_dir = stage_dir
        subprocess.run(
            ["rclone", "copy", f"{rclone_stream.rstrip('/')}/universe_catalog.parquet", str(stage_dir)],
            check=True,
        )

    try:
        cat_path = m16_data_dir / "universe_catalog.parquet"
        onboard_map: dict[str, int] = {}
        if cat_path.exists():
            cat_df = pd.read_parquet(cat_path)
            onboard_map = dict(zip(cat_df["symbol"].astype(str), cat_df["onboard_timestamp"].astype(int), strict=False))

        cfg = MultiModalFeatureConfig(decision_freq="15min", metrics_safety_lag_sec=60, min_adv_usd=15_000_000.0)
        if rclone_stream:
            shard_names = [f"shard_{i:02d}.parquet" for i in range(max_shards)]
        else:
            shard_files = sorted((m16_data_dir / "bars_1m").glob("shard_*.parquet"))[:max_shards]
            if not shard_files:
                raise FileNotFoundError(f"No shard_*.parquet files found under {m16_data_dir / 'bars_1m'}")
            shard_names = [p.name for p in shard_files]

        src_label = rclone_stream if rclone_stream else str(m16_data_dir)
        print(
            f"🔬 M16-2 Multi-Modal Feature Atlas | Streaming {len(shard_names)} shards from {src_label}...",
            flush=True,
        )
        panel_chunks: list[pd.DataFrame] = []

        def _fetch_rclone_shard(sh_name: str) -> None:
            import subprocess

            assert rclone_stream is not None and stage_dir is not None
            for sub_mod in ("bars_1m", "spot_bars_1m", "metrics_5m", "funding"):
                dst_sub = stage_dir / sub_mod
                dst_sub.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "rclone",
                        "copy",
                        "--multi-thread-streams=8",
                        "--multi-thread-cutoff=16M",
                        f"{rclone_stream.rstrip('/')}/{sub_mod}/{sh_name}",
                        str(dst_sub),
                    ],
                    check=True,
                )

        def _proc_sym_task(
            item: tuple[str, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None],
        ) -> pd.DataFrame | None:
            sym_k, p_bars, s_bars, m_bars, f_bars = item
            f_df = compute_symbol_multimodal_features(
                perp_bars=p_bars,
                spot_bars=s_bars,
                metrics_5m=m_bars,
                funding=f_bars,
                symbol=sym_k,
                onboard_timestamp=onboard_map.get(sym_k),
                config=cfg,
            )
            if f_df.empty:
                return None
            f_df = attach_forward_returns_for_audit(f_df, {"1h": 4, "4h": 16, "8h": 32})
            fnd_abs = f_df["funding_rate_bps"].abs() if "funding_rate_bps" in f_df.columns else 0.0
            keep_mask = (f_df["decision_ts"] % 3600 == 0) | ((f_df["decision_ts"] % 3600 == 1800) & (fnd_abs >= 15.0))
            if sym_k != "BTCUSDT":
                keep_mask = keep_mask & (f_df["adv_7d_usd_lag1"] >= 10_000_000.0)
            sub = f_df.loc[keep_mask].copy()
            if sub.empty:
                return None
            for col in sub.columns:
                if col not in ("timestamp", "decision_ts", "symbol") and sub[col].dtype == np.float64:
                    sub[col] = sub[col].astype(np.float32)
            return sub

        import os

        perp_cols = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_quote_volume",
        ]
        spot_cols = ["symbol", "timestamp", "close", "quote_volume", "taker_buy_quote_volume"]
        worker_cnt = min(2, max(1, os.cpu_count() or 2))

        def _rss_mb() -> float:
            try:
                for ln in Path("/proc/self/status").read_text().splitlines():
                    if ln.startswith("VmRSS:"):
                        return int(ln.split()[1]) / 1024.0
            except Exception:
                pass
            return 0.0

        prefetch_pool = ThreadPoolExecutor(max_workers=1) if rclone_stream else None
        prefetch_fut = prefetch_pool.submit(_fetch_rclone_shard, shard_names[0]) if prefetch_pool else None

        for idx_sh, sh_name in enumerate(shard_names):
            t_sh = time.time()
            if prefetch_fut is not None:
                prefetch_fut.result()
                if prefetch_pool is not None and idx_sh + 1 < len(shard_names):
                    prefetch_fut = prefetch_pool.submit(_fetch_rclone_shard, shard_names[idx_sh + 1])
                else:
                    prefetch_fut = None

            sh_path = m16_data_dir / "bars_1m" / sh_name
            b1m_all = pd.read_parquet(sh_path, columns=perp_cols)
            sp_path = m16_data_dir / "spot_bars_1m" / sh_name
            met_path = m16_data_dir / "metrics_5m" / sh_name
            fnd_path = m16_data_dir / "funding" / sh_name

            sp_by_sym = (
                dict(tuple(pd.read_parquet(sp_path, columns=spot_cols).groupby("symbol", observed=True)))
                if sp_path.exists()
                else {}
            )
            met_by_sym = (
                dict(tuple(pd.read_parquet(met_path).groupby("symbol", observed=True))) if met_path.exists() else {}
            )
            fnd_by_sym = (
                dict(tuple(pd.read_parquet(fnd_path).groupby("symbol", observed=True))) if fnd_path.exists() else {}
            )
            perp_by_sym = dict(tuple(b1m_all.groupby("symbol", observed=True)))
            del b1m_all

            if rclone_stream:
                for p_rm in (sh_path, sp_path, met_path, fnd_path):
                    p_rm.unlink(missing_ok=True)

            sym_keys = sorted(perp_by_sym.keys())
            tasks = [
                (
                    s_k,
                    perp_by_sym.pop(s_k),
                    sp_by_sym.pop(s_k, None),
                    met_by_sym.pop(s_k, None),
                    fnd_by_sym.pop(s_k, None),
                )
                for s_k in sym_keys
            ]
            del perp_by_sym, sp_by_sym, met_by_sym, fnd_by_sym
            with ThreadPoolExecutor(max_workers=worker_cnt) as pool:
                res_list = list(pool.map(_proc_sym_task, tasks))
            del tasks
            valid_sh = [r for r in res_list if r is not None and not r.empty]
            del res_list
            if valid_sh:
                sh_panel = pd.concat(valid_sh, ignore_index=True)
                sh_panel["symbol"] = sh_panel["symbol"].astype("category")
                panel_chunks.append(sh_panel)
                n_rows_sh = len(sh_panel)
            else:
                n_rows_sh = 0
            del valid_sh
            gc.collect()
            rss_now = _rss_mb()
            rss_str = f" | RSS={rss_now:.0f}MB" if rss_now > 0 else ""
            print(
                f"  ✓ Processed {sh_name} ({n_rows_sh:,} sampled rows) in {time.time() - t_sh:.1f}s{rss_str}",
                flush=True,
            )

        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)
    finally:
        if stage_dir is not None and stage_dir.exists():
            import shutil

            shutil.rmtree(stage_dir, ignore_errors=True)

    print("[*] Assembling 220-symbol cross-sectional panel & Family D orthogonal ranks...", flush=True)
    raw_panel = pd.concat(panel_chunks, ignore_index=True)
    del panel_chunks
    gc.collect()
    raw_panel["symbol"] = raw_panel["symbol"].astype(str)
    full_panel = compute_cross_sectional_panel_features(raw_panel, inplace=True)
    del raw_panel
    gc.collect()

    all_feat_cols = list_multimodal_feature_columns(full_panel)
    print(f"[*] Total Panel Rows: {len(full_panel):,} | Total Multi-Modal Features: {len(all_feat_cols)}", flush=True)

    bench_daily = _build_existing_sleeves_daily_proxy(full_panel)
    print(
        "[*] Evaluating Walk-Forward Atlas (2023-2024 Train -> 2025 Val -> 2026 Holdout) for 4h & 1h horizons...",
        flush=True,
    )
    atlas_4h = evaluate_walk_forward_feature_atlas(
        full_panel,
        fwd_col="fwd_ret_4h",
        tail_z_threshold=2.0,
        maker_roundtrip_bps=3.2,
        taker_roundtrip_bps=9.0,
        periods_per_year=365.25 * 6.0,
        benchmark_daily_pnl=bench_daily,
    )
    atlas_1h = evaluate_walk_forward_feature_atlas(
        full_panel,
        fwd_col="fwd_ret_1h",
        tail_z_threshold=2.2,
        maker_roundtrip_bps=3.2,
        taker_roundtrip_bps=9.0,
        periods_per_year=365.25 * 24.0,
        benchmark_daily_pnl=bench_daily,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_4h.to_csv(output_dir / "m16_2_feature_atlas_4h.csv", index=False)
    atlas_1h.to_csv(output_dir / "m16_2_feature_atlas_1h.csv", index=False)

    consistent_4h = atlas_4h[(atlas_4h["sign_consistent_3_eras"]) & (atlas_4h["corr_to_existing_sleeves"].abs() < 0.35)]
    consistent_1h = atlas_1h[(atlas_1h["sign_consistent_3_eras"]) & (atlas_1h["corr_to_existing_sleeves"].abs() < 0.35)]

    summary = {
        "issue": "M16-2 (#68)",
        "elapsed_sec": round(time.time() - t0, 1),
        "total_panel_rows": int(len(full_panel)),
        "unique_symbols_evaluated": int(full_panel["symbol"].nunique()),
        "total_features_evaluated": len(all_feat_cols),
        "orthogonal_sign_consistent_4h_count": int(len(consistent_4h)),
        "orthogonal_sign_consistent_1h_count": int(len(consistent_1h)),
        "top_15_4h_features": atlas_4h.head(15).to_dict(orient="records"),
        "top_15_1h_features": atlas_1h.head(15).to_dict(orient="records"),
    }
    (output_dir / "m16_2_feature_atlas_summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# M16-2 Multi-Modal Microstructure Feature Atlas (2023–2026 Walk-Forward)",
        "",
        f"- **Universe Evaluated:** `{summary['unique_symbols_evaluated']}` Point-in-Time USDS-M Perpetuals (`{summary['total_panel_rows']:,}` 30m decision rows, `ADV_7d >= $15M` tradable gate)",
        f"- **Causal Features Audited:** `{summary['total_features_evaluated']}` across Families A, B, C, and D (`+60s` metrics safety lag, causal settlement clock)",
        f"- **Sign-Consistent Orthogonal Features (`|corr| < 0.35` across all 3 eras):** `{summary['orthogonal_sign_consistent_4h_count']}` (`4h` horizon) | `{summary['orthogonal_sign_consistent_1h_count']}` (`1h` horizon)",
        "",
        "## Top 20 Features — 4h Horizon (Ranked by Out-of-Sample Composite Score)",
        "",
        "| Rank | Feature | Family | Pol | Train t | Val 2025 t | Holdout 2026 t | 2026 Tail Net Maker (bps) | 2026 Tail Net Taker (bps) | Corr to Existing | 3-Era Consistent |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for idx_r, r in atlas_4h.head(20).iterrows():
        md_lines.append(
            f"| {idx_r + 1} | `{r['feature']}` | `{r['family']}` | `{r['train_polarity']:+d}` | "
            f"`{r['train_ic_tstat']:+.2f}` | `{r['val_2025_ic_tstat']:+.2f}` | **`{r['holdout_2026_ic_tstat']:+.2f}`** | "
            f"**`{r['holdout_2026_tail_net_maker_bps']:+.1f}`** | `{r['holdout_2026_tail_net_taker_bps']:+.1f}` | "
            f"`{r['corr_to_existing_sleeves']:+.2f}` | `{'YES' if r['sign_consistent_3_eras'] else 'no'}` |"
        )

    md_lines.extend(
        [
            "",
            "## Top 15 Features — 1h Fast Sniper Horizon",
            "",
            "| Rank | Feature | Family | Pol | Train t | Val 2025 t | Holdout 2026 t | 2026 Tail Net Maker (bps) | Corr to Existing | 3-Era Consistent |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for idx_r, r in atlas_1h.head(15).iterrows():
        md_lines.append(
            f"| {idx_r + 1} | `{r['feature']}` | `{r['family']}` | `{r['train_polarity']:+d}` | "
            f"`{r['train_ic_tstat']:+.2f}` | `{r['val_2025_ic_tstat']:+.2f}` | **`{r['holdout_2026_ic_tstat']:+.2f}`** | "
            f"**`{r['holdout_2026_tail_net_maker_bps']:+.1f}`** | `{r['corr_to_existing_sleeves']:+.2f}` | "
            f"`{'YES' if r['sign_consistent_3_eras'] else 'no'}` |"
        )

    (output_dir / "m16_2_feature_atlas_report.md").write_text("\n".join(md_lines) + "\n")

    drive_root = Path("/content/drive/MyDrive/trading/crypto-alpha-engine")
    if drive_root.exists():
        import shutil

        drive_reports = drive_root / "reports"
        drive_reports.mkdir(parents=True, exist_ok=True)
        for fname in (
            "m16_2_feature_atlas_4h.csv",
            "m16_2_feature_atlas_1h.csv",
            "m16_2_feature_atlas_summary.json",
            "m16_2_feature_atlas_report.md",
        ):
            shutil.copy2(output_dir / fname, drive_reports / fname)
        panel_dir = drive_root / "data" / "m16_multimodal" / "features_panel"
        panel_dir.mkdir(parents=True, exist_ok=True)
        full_panel.to_parquet(panel_dir / "m16_2_causal_panel.parquet", compression="zstd", compression_level=9)
        print(f"[+] Persisted Atlas reports & causal panel to Google Drive ({drive_root})", flush=True)
    elif rclone_stream:
        import subprocess

        tmp_panel = Path("/tmp/m16_2_causal_panel.parquet")
        try:
            full_panel.to_parquet(tmp_panel, compression="zstd", compression_level=9)
            subprocess.run(
                ["rclone", "copy", str(tmp_panel), f"{rclone_stream.rstrip('/')}/features_panel/"],
                check=False,
            )
        finally:
            tmp_panel.unlink(missing_ok=True)

    print("M16_2_ATLAS_COMPLETE!", flush=True)
    return summary


def run_m16_gtx_matrix_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    max_shards: int = 20,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict:
    """
    Executes M16-3 (Issue #69): Conservative 1-minute Passive Post-Only (`GTX`) Queue-Penetration
    & Adverse-Selection Sensitivity Matrix (`1.0 .. 3.0 bps` strict price-through + `100%` same-bar SL)
    vs Taker Market Entry across `2023–2026` (`Train`, `2025 Val`, `2026 Holdout`).
    """
    if colab_session:
        import time

        from src.cloud.colab_client import ColabClient

        client = ColabClient(session_name=colab_session)
        print(f"[*] Connecting to Colab session '{colab_session}' for M16-3 Passive GTX Matrix...", flush=True)
        if not client.ensure_session(gpu=None):
            raise RuntimeError(f"Failed to connect/provision Colab session '{colab_session}'")
        try:
            if not client.is_drive_mounted():
                rc_m, out_m, err_m = client.mount_drive("/content/drive")
                print(f"[*] Drive mount ({rc_m}): {(out_m or err_m).strip()}", flush=True)
            assert client.is_drive_mounted(), "Google Drive mount required at /content/drive/MyDrive"
            assert client.sync_codebase(REPO_ROOT), "Failed syncing project codebase bundle to Colab"
            client.exec_inline("import os; os.makedirs('/content/reports', exist_ok=True)", timeout=120)

            drive_m16 = "/content/drive/MyDrive/trading/crypto-alpha-engine/data/m16_multimodal"
            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/m16_gtx.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                f"'--suite', 'm16-gtx-matrix', '--m16-data-dir', '{drive_m16}', "
                f"'--max-shards', '{max_shards}', '--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/m16_gtx.pid', 'w').write(str(p.pid)); "
                "print(f'GTX_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=120)
            print(f"[+] Launched remote M16-3 GTX Matrix ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            while True:
                time.sleep(15.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/m16_gtx.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/m16_gtx.log').read().splitlines() if os.path.exists('/content/m16_gtx.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, _ = client.exec_inline(poll_py, timeout=120)
                if rc_p != 0:
                    continue
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "M16_3_GTX_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote M16-3 GTX process terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "m16_3_passive_gtx_matrix.csv",
                "m16_3_passive_gtx_matrix_summary.json",
                "m16_3_passive_gtx_matrix_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            print(f"[+] Downloaded M16-3 GTX artifacts into {output_dir}", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            else:
                print(f"[*] Kept Colab session '{colab_session}' mounted & alive (--keep-colab).", flush=True)

    import gc
    import json
    import time

    from src.backtest.passive_gtx_simulator import (
        GTXSignalEvent,
        PassiveGTXConfig,
        PassiveGTXExecutionSimulator,
    )

    t0 = time.time()
    panel_path = m16_data_dir / "features_panel" / "m16_2_causal_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"M16-2 causal panel not found at {panel_path}")

    req_cols = [
        "symbol",
        "decision_ts",
        "tradable_adv_15m_gate",
        "pre_settle_neg_squeeze_40bps",
        "funding_rate_bps",
        "mins_since_settlement",
        "clock_compression_ratio",
        "quad_long_liq_flush_1h",
        "spot_vs_perp_cvd_div_1h",
    ]
    print(f"🔬 M16-3 Passive GTX Matrix | Loading causal trigger panel from {panel_path}...", flush=True)
    panel = pd.read_parquet(panel_path, columns=req_cols)
    panel = panel[panel["tradable_adv_15m_gate"] > 0.5].reset_index(drop=True)

    # Compare pre-settlement Winner's Curse vs exact post-settlement T+1m (0 funding paid!) & deep k*ATR wicks
    at_settle = panel["mins_since_settlement"] <= 0.5
    candidates: dict[str, dict] = {
        "M1_PRE_SETTLE_NEG_40BPS_CAUTION": {
            "mask": panel["pre_settle_neg_squeeze_40bps"] > 0.0,
            "side": -1,
            "post_settle_shift_sec": -60,  # enters at T-30m into ongoing squeeze
            "offset_bps": 6.0,
            "tp_pct": 0.022,
            "sl_pct": 0.011,
            "hold_min": 90,
            "lock_trig": 0.010,
            "lock_lvl": 0.0045,
        },
        "M2_POST_SETTLE_T1M_NEG_40BPS_SHORT": {
            "mask": at_settle & (panel["funding_rate_bps"] <= -40.0),
            "side": -1,
            "post_settle_shift_sec": 0,  # `sig` = `:00` bar -> submits at `:01:00` (+60s AFTER settlement, 0 funding paid!)
            "offset_bps": 3.0,
            "tp_pct": 0.045,
            "sl_pct": 0.032,
            "hold_min": 240,
            "lock_trig": 0.018,
            "lock_lvl": 0.0080,
        },
        "M3_POST_SETTLE_T1M_NEG_25BPS_SHORT": {
            "mask": at_settle & (panel["funding_rate_bps"] <= -25.0),
            "side": -1,
            "post_settle_shift_sec": 0,
            "offset_bps": 3.0,
            "tp_pct": 0.035,
            "sl_pct": 0.026,
            "hold_min": 240,
            "lock_trig": 0.014,
            "lock_lvl": 0.0060,
        },
        "M4_POST_SETTLE_COMPRESSED_CLOCK_SHORT": {
            "mask": at_settle & (panel["clock_compression_ratio"] >= 2.0) & (panel["funding_rate_bps"] <= -18.0),
            "side": -1,
            "post_settle_shift_sec": 0,
            "offset_bps": 3.0,
            "tp_pct": 0.030,
            "sl_pct": 0.024,
            "hold_min": 180,
            "lock_trig": 0.012,
            "lock_lvl": 0.0050,
        },
        "M5_DEEP_WICK_45BPS_LONG_LIQ_FLUSH": {
            "mask": (panel["quad_long_liq_flush_1h"] >= 4.0) & (panel["spot_vs_perp_cvd_div_1h"] >= 0.5),
            "side": 1,
            "post_settle_shift_sec": -60,
            "offset_bps": 45.0,
            "tp_pct": 0.018,
            "sl_pct": 0.016,
            "hold_min": 120,
            "lock_trig": 0.009,
            "lock_lvl": 0.0040,
        },
    }

    cand_ts_by_sym: dict[str, dict[str, np.ndarray]] = {}
    for c_name, spec in candidates.items():
        sub = panel.loc[spec["mask"], ["symbol", "decision_ts"]]
        grouped = {str(k): v["decision_ts"].to_numpy(dtype=np.int64) for k, v in sub.groupby("symbol", observed=True)}
        cand_ts_by_sym[c_name] = grouped
        print(f"  • {c_name}: {len(sub):,} raw causal triggers across {len(grouped)} liquid perps", flush=True)

    del panel
    gc.collect()

    pen_grid = (1.0, 1.5, 2.0, 2.5, 3.0)
    ttl_grid = (3, 5)
    # Accumulate simulated trade DataFrames per (cand, mode_key)
    trade_buckets: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for c_name in candidates:
        trade_buckets[(c_name, "TAKER_MKT")] = []
        for pen in pen_grid:
            for ttl in ttl_grid:
                trade_buckets[(c_name, f"GTX_pen{pen:.1f}_ttl{ttl}")] = []

    if rclone_stream:
        shard_names = [f"shard_{i:02d}.parquet" for i in range(max_shards)]
    else:
        shard_files = sorted((m16_data_dir / "bars_1m").glob("shard_*.parquet"))[:max_shards]
        shard_names = [p.name for p in shard_files]
    bar_cols = ["symbol", "timestamp", "open", "high", "low", "close", "quote_volume"]

    for sh_name in shard_names:
        t_sh = time.time()
        sh_path = m16_data_dir / "bars_1m" / sh_name
        if rclone_stream and not sh_path.exists():
            import subprocess

            sh_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    "--multi-thread-streams=8",
                    "--multi-thread-cutoff=16M",
                    f"{rclone_stream.rstrip('/')}/bars_1m/{sh_name}",
                    str(sh_path.parent),
                ],
                check=True,
            )
        b1m = pd.read_parquet(sh_path, columns=bar_cols)
        if rclone_stream:
            sh_path.unlink(missing_ok=True)
        by_sym = dict(tuple(b1m.groupby("symbol", observed=True)))
        del b1m

        for sym_k, df_sym in by_sym.items():
            sym_str = str(sym_k)
            if not any(sym_str in cand_ts_by_sym[c] for c in candidates):
                continue
            df_s = df_sym.sort_values("timestamp").reset_index(drop=True)
            ts_1m = df_s["timestamp"].to_numpy(dtype=np.int64)

            for c_name, spec in candidates.items():
                dec_arr = cand_ts_by_sym[c_name].get(sym_str)
                if dec_arr is None or len(dec_arr) == 0:
                    continue
                # Map `decision_ts` to 1m signal bar (`dec_arr + post_settle_shift_sec`; 0 => submits at `:01:00` post-settle!)
                target_open = dec_arr + int(spec.get("post_settle_shift_sec", -60))
                idx_pos = np.searchsorted(ts_1m, target_open)
                idx_clip = np.minimum(idx_pos, len(ts_1m) - 1)
                matched = (idx_pos < len(ts_1m)) & (ts_1m[idx_clip] == target_open)
                valid_indices = idx_clip[matched]
                if len(valid_indices) == 0:
                    continue

                evs_gtx = [
                    GTXSignalEvent(
                        signal_idx=int(ix),
                        side=int(spec["side"]),
                        limit_offset_bps=float(spec["offset_bps"]),
                        take_profit_pct=float(spec["tp_pct"]),
                        stop_loss_pct=float(spec["sl_pct"]),
                        symbol=sym_str,
                    )
                    for ix in valid_indices
                ]
                # Taker Market Baseline: marketable cross re-anchored at open (`offset=-25 bps`, `maker_fee=6.0 bps` = 5 bps taker + 1 bps slip)
                evs_taker = [
                    GTXSignalEvent(
                        signal_idx=int(ix),
                        side=int(spec["side"]),
                        limit_offset_bps=-25.0,
                        take_profit_pct=float(spec["tp_pct"]),
                        stop_loss_pct=float(spec["sl_pct"]),
                        symbol=sym_str,
                    )
                    for ix in valid_indices
                ]
                taker_sim = PassiveGTXExecutionSimulator(
                    PassiveGTXConfig(
                        queue_penetration_bps=0.0,
                        tp_penetration_bps=1.0,
                        maker_fee_bps=6.0,  # 5.0 bps Taker + 1.0 bps entry slippage
                        taker_stop_fee_bps=5.0,
                        stop_slippage_bps=1.0,
                        ttl_minutes=1,
                        max_hold_minutes=int(spec["hold_min"]),
                        gtx_cross_policy="reprice",
                        gtx_reprice_offset_bps=0.0,
                        order_notional_usd=600.0,
                        max_bar_participation=0.02,
                        profit_lock_trigger_pct=float(spec["lock_trig"]),
                        profit_lock_level_pct=float(spec["lock_lvl"]),
                    )
                )
                t_df_tk = taker_sim.simulate_signals_on_1m_bars(df_s, evs_taker, min_cooldown_minutes=15)
                if not t_df_tk.empty:
                    trade_buckets[(c_name, "TAKER_MKT")].append(t_df_tk)

                for pen in pen_grid:
                    for ttl in ttl_grid:
                        gtx_sim = PassiveGTXExecutionSimulator(
                            PassiveGTXConfig(
                                queue_penetration_bps=pen,
                                tp_penetration_bps=1.0,
                                maker_fee_bps=2.0,
                                taker_stop_fee_bps=5.0,
                                stop_slippage_bps=1.0,
                                ttl_minutes=ttl,
                                max_hold_minutes=int(spec["hold_min"]),
                                gtx_cross_policy="reprice",
                                gtx_reprice_offset_bps=1.5,
                                order_notional_usd=600.0,
                                max_bar_participation=0.02,
                                profit_lock_trigger_pct=float(spec["lock_trig"]),
                                profit_lock_level_pct=float(spec["lock_lvl"]),
                            )
                        )
                        t_df_g = gtx_sim.simulate_signals_on_1m_bars(df_s, evs_gtx, min_cooldown_minutes=15)
                        if not t_df_g.empty:
                            trade_buckets[(c_name, f"GTX_pen{pen:.1f}_ttl{ttl}")].append(t_df_g)

        del by_sym
        gc.collect()
        print(f"  ✓ Simulated 1m GTX & Taker fills on {sh_name} in {time.time() - t_sh:.1f}s", flush=True)

    matrix_rows: list[dict] = []
    for (c_name, mode_key), parts in trade_buckets.items():
        all_t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        full_stats = PassiveGTXExecutionSimulator.summarize_simulation(all_t)
        if not all_t.empty:
            sig_years = pd.to_datetime(all_t["signal_ts"], unit="s", utc=True).dt.year
            val_stats = PassiveGTXExecutionSimulator.summarize_simulation(all_t[sig_years == 2025])
            hold_stats = PassiveGTXExecutionSimulator.summarize_simulation(all_t[sig_years >= 2026])
        else:
            val_stats = PassiveGTXExecutionSimulator.summarize_simulation(pd.DataFrame())
            hold_stats = PassiveGTXExecutionSimulator.summarize_simulation(pd.DataFrame())

        if mode_key == "TAKER_MKT":
            pen_v, ttl_v = 0.0, 1
        else:
            p_s, t_s = mode_key.replace("GTX_pen", "").split("_ttl")
            pen_v, ttl_v = float(p_s), int(t_s)

        matrix_rows.append(
            {
                "candidate": c_name,
                "mode": mode_key,
                "queue_penetration_bps": pen_v,
                "ttl_minutes": ttl_v,
                "total_signals": full_stats["total_signals"],
                "filled_trades": full_stats["filled_trades"],
                "fill_rate_pct": full_stats["fill_rate_pct"],
                "same_bar_sl_rate_pct": full_stats["same_bar_sl_rate_pct"],
                "mean_gross_bps": full_stats["mean_gross_bps"],
                "mean_fee_bps": full_stats["mean_fee_bps"],
                "mean_net_bps_full": full_stats["mean_net_bps"],
                "sharpe_full_2023_2026": full_stats["annualized_sharpe"],
                "val_2025_net_bps": val_stats["mean_net_bps"],
                "val_2025_sharpe": val_stats["annualized_sharpe"],
                "holdout_2026_filled": hold_stats["filled_trades"],
                "holdout_2026_net_bps": hold_stats["mean_net_bps"],
                "holdout_2026_sharpe": hold_stats["annualized_sharpe"],
                "survives_25bps_haircut": bool(
                    full_stats["mean_net_bps"] > 0.0
                    and val_stats["mean_net_bps"] > 0.0
                    and hold_stats["mean_net_bps"] > 0.0
                ),
            }
        )

    mat_df = pd.DataFrame(matrix_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_df.to_csv(output_dir / "m16_3_passive_gtx_matrix.csv", index=False)

    pen25_survivors = mat_df[(mat_df["queue_penetration_bps"] >= 2.5) & (mat_df["survives_25bps_haircut"])].to_dict(
        orient="records"
    )

    summary = {
        "issue": "M16-3 (#69)",
        "elapsed_sec": round(time.time() - t0, 1),
        "candidates_tested": list(candidates.keys()),
        "penetration_grid_bps": list(pen_grid),
        "ttl_grid_minutes": list(ttl_grid),
        "survivors_at_ge_2_5_bps_count": len(pen25_survivors),
        "survivors_at_ge_2_5_bps": pen25_survivors,
        "all_rows": matrix_rows,
    }
    (output_dir / "m16_3_passive_gtx_matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# M16-3 Conservative 1m Passive Post-Only (`GTX`) Penetration Sensitivity Matrix (2023–2026)",
        "",
        "- **Anti-Optimism Physics Enforced:** Strict price-through (`low <= P_lim * (1 - pen)` / `high >= P_lim * (1 + pen)`, exact touch = **UNFILLED**), `100%` pessimistic same-bar Stop-Loss priority (`STOP_LOSS_SAME_BAR`), `2%` max 1m quote-volume participation (`$600` clip), `2.0 bps` Maker vs `6.0 bps` Taker+slippage stop asymmetry.",
        f"- **Configurations Surviving `queue_penetration >= 2.5 bps` Across All Eras (`Full`, `2025 Val`, `2026 Holdout`):** `{len(pen25_survivors)}`",
        "",
        "| Candidate | Mode | Pen (bps) | TTL (m) | Signals | Filled (`%`) | Same-Bar SL `%` | Fee (bps) | Net Full (bps) | Sharpe `23-26` | `2025` Net (bps) | `2026` Net (bps) | `2026` Sharpe | Survives |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in matrix_rows:
        md.append(
            f"| `{r['candidate']}` | `{r['mode']}` | `{r['queue_penetration_bps']:.1f}` | `{r['ttl_minutes']}` | "
            f"`{r['total_signals']:,}` | `{r['filled_trades']:,}` (`{r['fill_rate_pct']:.1f}%`) | "
            f"`{r['same_bar_sl_rate_pct']:.1f}%` | `{r['mean_fee_bps']:.1f}` | **`{r['mean_net_bps_full']:+.1f}`** | "
            f"`{r['sharpe_full_2023_2026']:+.2f}` | `{r['val_2025_net_bps']:+.1f}` | "
            f"**`{r['holdout_2026_net_bps']:+.1f}`** | **`{r['holdout_2026_sharpe']:+.2f}`** | "
            f"`{'YES' if r['survives_25bps_haircut'] else 'no'}` |"
        )
    (output_dir / "m16_3_passive_gtx_matrix_report.md").write_text("\n".join(md) + "\n")

    drive_root = Path("/content/drive/MyDrive/trading/crypto-alpha-engine")
    if drive_root.exists():
        import shutil

        drive_reports = drive_root / "reports"
        drive_reports.mkdir(parents=True, exist_ok=True)
        for fname in (
            "m16_3_passive_gtx_matrix.csv",
            "m16_3_passive_gtx_matrix_summary.json",
            "m16_3_passive_gtx_matrix_report.md",
        ):
            shutil.copy2(output_dir / fname, drive_reports / fname)

    print("M16_3_GTX_COMPLETE!", flush=True)
    return summary


def run_m16_discovery_and_synthesis_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
) -> dict:
    """
    Executes M16-4 (Issue #70: Symbolic & Shallow-Tree Discovery Tournament with CPCV + DSR >= 0.95)
    followed immediately by M16-5 (Issue #71: Grand 4-Year Multi-Engine Portfolio Synthesis & Stress-Test
    at $2,700 100% USDT with Cushion-Safe CPPI Governor & $2,500 Hard Equity Floor).
    """
    if colab_session:
        import time

        from src.cloud.colab_client import ColabClient

        client = ColabClient(session_name=colab_session)
        print(f"[*] Connecting to Colab session '{colab_session}' for M16-4 + M16-5...", flush=True)
        if not client.ensure_session(gpu=None):
            raise RuntimeError(f"Failed to connect/provision Colab session '{colab_session}'")
        try:
            if not client.is_drive_mounted():
                rc_m, out_m, err_m = client.mount_drive("/content/drive")
                print(f"[*] Drive mount ({rc_m}): {(out_m or err_m).strip()}", flush=True)
            assert client.is_drive_mounted(), "Google Drive mount required at /content/drive/MyDrive"
            assert client.sync_codebase(REPO_ROOT), "Failed syncing project codebase bundle to Colab"
            client.exec_inline("import os; os.makedirs('/content/reports', exist_ok=True)", timeout=120)

            drive_m16 = "/content/drive/MyDrive/trading/crypto-alpha-engine/data/m16_multimodal"
            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/m16_disc.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                f"'--suite', 'm16-discovery-synthesis', '--m16-data-dir', '{drive_m16}', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/m16_disc.pid', 'w').write(str(p.pid)); "
                "print(f'DISC_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=120)
            print(f"[+] Launched remote M16-4/M16-5 ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            while True:
                time.sleep(10.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/m16_disc.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/m16_disc.log').read().splitlines() if os.path.exists('/content/m16_disc.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, _ = client.exec_inline(poll_py, timeout=120)
                if rc_p != 0:
                    continue
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "M16_4_5_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote M16-4/M16-5 process terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "m16_4_discovery_leaderboard.csv",
                "m16_4_discovery_summary.json",
                "m16_4_discovery_report.md",
                "m16_5_portfolio_ablation.csv",
                "m16_5_portfolio_summary.json",
                "m16_5_portfolio_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            print(f"[+] Downloaded M16-4 & M16-5 artifacts into {output_dir}", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            else:
                print(f"[*] Kept Colab session '{colab_session}' mounted & alive (--keep-colab).", flush=True)

    import gc
    import json
    import time

    from src.backtest.m16_grand_portfolio_synthesizer import (
        GrandPortfolioConfig,
        causal_inverse_vol_combine,
        run_grand_portfolio_ablation_matrix,
    )
    from src.models.symbolic_alpha_tournament import (
        CORE_MAJORS_15,
        run_multimodal_discovery_tournament,
        select_diverse_orthogonal_survivors,
        simulate_cross_sectional_sleeve_daily,
        simulate_event_sniper_sleeve_daily,
    )

    t0 = time.time()
    panel_path = m16_data_dir / "features_panel" / "m16_2_causal_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"M16-2 causal panel not found at {panel_path}")

    disc_cols = [
        "symbol",
        "decision_ts",
        "tradable_adv_15m_gate",
        "ret_4h",
        "ret_12h",
        "ret_24h",
        "fwd_ret_1h",
        "fwd_ret_4h",
        "fwd_ret_8h",
        "funding_rate_bps",
        "funding_interval_hours",
        "clock_compression_ratio",
        "compressed_clock_neg_funding_impulse",
        "mins_since_settlement",
        "mins_to_next_settlement",
        "whale_pos_ls_z",
        "smart_money_div_z_24h",
        "oi_vel_12h",
        "oi_vel_z_12h",
        "oi_vel_24h",
        "oi_to_qvol_24h",
        "quad_new_longs_12h",
        "spot_taker_share_4h",
        "spot_cvd_z_4h",
        "basis_vel_4h",
        "spot_lead_ret_div_1b",
        "spot_vs_perp_share_div_4h",
        "parkinson_vol_24h",
        "corwin_schultz_spread_24h",
        "vol_of_vol_24h",
        "rvol_shock_1h_vs_24h",
    ]
    print(f"🔬 M16-4/M16-5 | Loading causal panel ({len(disc_cols)} cols) from {panel_path}...", flush=True)
    panel = pd.read_parquet(panel_path, columns=disc_cols)
    panel = panel[panel["tradable_adv_15m_gate"] > 0.5].reset_index(drop=True)

    # Construct faithful fee- & funding-adjusted `00_CURRENT_PROD` baseline daily stream:
    #   P1: 8h Core 7-Factor Pullback + Smart-Money + Low-Vol BHS (`decision_ts % 28800 == 0`)
    #   P2: Multi-day (14d EMA) BTC/ETH/SOL Trend (`DollarBarTSMOM` equivalent)
    #   P3: Phase 2 RVOL Climax / Exhaustion Snipers
    from src.features.multimodal_alpha import _vectorized_cs_zscore

    p8_maj = panel[
        (panel["decision_ts"] % 28800 == 0) & panel["symbol"].isin(CORE_MAJORS_15) & panel["fwd_ret_8h"].notna()
    ].reset_index(drop=True)
    p1_raw = (
        -0.45 * _vectorized_cs_zscore(p8_maj, "decision_ts", "ret_12h")
        - 0.25 * _vectorized_cs_zscore(p8_maj, "decision_ts", "parkinson_vol_24h")
        + 0.30 * _vectorized_cs_zscore(p8_maj, "decision_ts", "smart_money_div_z_24h")
    )
    p1_daily, _ = simulate_cross_sectional_sleeve_daily(
        p8_maj, p1_raw, "fwd_ret_8h", horizon_hours=8, one_way_cost_bps=2.1, ema_alpha=0.15, top_k=4
    )

    maj_3 = (
        p8_maj[p8_maj["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        .sort_values(["symbol", "decision_ts"])
        .reset_index(drop=True)
    )
    ema_14d = maj_3.groupby("symbol")["ret_24h"].transform(lambda s: s.ewm(span=42, min_periods=6).mean()).to_numpy()
    tsmom_sig = np.where(np.abs(ema_14d) > 0.003, np.sign(ema_14d), 0.0)
    # Low-turnover multi-day trend pays ~0.4 bps amortized per 8h bar (reversals occur ~once per 8-12 days)
    p2_daily, _ = simulate_event_sniper_sleeve_daily(
        maj_3,
        tsmom_sig,
        "fwd_ret_8h",
        horizon_hours=8,
        roundtrip_cost_bps=0.8,
        slot_weight=0.85,
        max_concurrent_slots=3,
    )

    p4_all = panel[(panel["decision_ts"] % 14400 == 0) & panel["fwd_ret_4h"].notna()].reset_index(drop=True)
    sn_sig = np.where(
        (p4_all["rvol_shock_1h_vs_24h"].to_numpy() >= 3.0) & (np.abs(p4_all["ret_4h"].to_numpy()) >= 0.035),
        -np.sign(p4_all["ret_4h"].to_numpy()),
        0.0,
    )
    p3_daily, _ = simulate_event_sniper_sleeve_daily(
        p4_all, sn_sig, "fwd_ret_4h", horizon_hours=4, roundtrip_cost_bps=4.2, slot_weight=0.35, max_concurrent_slots=3
    )
    prod_daily = causal_inverse_vol_combine(
        {"p1_core_bhs": p1_daily, "p2_manga_tsmom": p2_daily, "p3_event_snipers": p3_daily},
        base_risk_shares={"p1_core_bhs": 0.48, "p2_manga_tsmom": 0.32, "p3_event_snipers": 0.20},
        vol_lookback_days=45,
        target_daily_vol=0.0110,
        max_leverage_cap=2.40,
    )
    del p8_maj, maj_3, p4_all
    gc.collect()

    print(
        "[*] Running M16-4 Symbolic & Shallow-Tree Discovery Tournament (CPCV + DSR >= 0.95 + |corr| < 0.20)...",
        flush=True,
    )
    lb_df, series_map, spec_map = run_multimodal_discovery_tournament(
        panel,
        benchmark_daily_pnl=prod_daily,
        min_era_sharpe=1.80,
        min_dsr=0.95,
        max_abs_corr=0.20,
    )
    del panel
    gc.collect()

    survivors = select_diverse_orthogonal_survivors(lb_df, series_map, max_survivors=4, max_pairwise_corr=0.55)
    strict_count = int(lb_df["passed_all_4_gates"].sum()) if not lb_df.empty else 0
    print(
        f"[+] M16-4 Complete: {len(lb_df)} mechanisms tested | {strict_count} passed ALL 4 Hard Gates | Selected diverse orthogonal set: {survivors}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "m16_4_discovery_leaderboard.csv", index=False)
    disc_summary = {
        "issue": "M16-4 (#70)",
        "total_mechanisms_tested": int(len(lb_df)),
        "passed_all_4_hard_gates_count": strict_count,
        "selected_diverse_survivors": survivors,
        "selected_specs": [
            {
                "mech_id": m_id,
                "archetype": spec_map[m_id].archetype,
                "mode": spec_map[m_id].mode,
                "universe": spec_map[m_id].universe_scope,
                "symbolic_rule": spec_map[m_id].symbolic_rule_text,
            }
            for m_id in survivors
            if m_id in spec_map
        ],
        "top_20_leaderboard": lb_df.head(20).to_dict(orient="records"),
    }
    (output_dir / "m16_4_discovery_summary.json").write_text(json.dumps(disc_summary, indent=2) + "\n")

    md4 = [
        "# M16-4 Automated Symbolic & Shallow-Tree Alpha Discovery Tournament (2023–2026)",
        "",
        f"- **Total Candidate Mechanisms Audited (`M`):** `{len(lb_df)}` (penalized via Bailey & López de Prado `DSR >= 0.95`)",
        f"- **Mechanisms Passing ALL 4 Hard Anti-Overfitting Gates (`CPCV >= 1.8`, `DSR >= 0.95`, `Sharpe >= 1.8` in ALL 3 eras, `|corr| < 0.20`):** `{strict_count}`",
        f"- **Mutually Diverse Orthogonal Mechanisms Selected for `M16-5` (`|pairwise corr| <= 0.55`):** `{', '.join(survivors)}`",
        "",
        "| Rank | Mechanism ID | Archetype | Full Sharpe | `23-24` Tr | `2025` Val | `2026` Hold | CPCV Med | DSR | Fee Drag `%` | Corr vs Prod | All 4 Gates |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for idx_r, r in lb_df.head(20).iterrows():
        md4.append(
            f"| {idx_r + 1} | `{r['mech_id']}` | `{r['archetype']}` | **`{r['sharpe_full']:+.2f}`** | "
            f"`{r['sharpe_train_23_24']:+.2f}` | `{r['sharpe_val_2025']:+.2f}` | **`{r['sharpe_holdout_2026']:+.2f}`** | "
            f"`{r['cpcv_median_sharpe']:+.2f}` | `{r['dsr']:.3f}` | `{r['ann_fee_drag_pct']:.1f}%` | "
            f"**`{r['corr_to_prod']:+.2f}`** | `{'PASS' if r['passed_all_4_gates'] else 'no'}` |"
        )
    (output_dir / "m16_4_discovery_report.md").write_text("\n".join(md4) + "\n")

    # ------------------------------------------------------------------
    # M16-5: Grand 4-Year Multi-Engine Portfolio Synthesis & Stress-Test
    # ------------------------------------------------------------------
    print("[*] Running M16-5 Grand Portfolio Synthesis & $2,700 SafeRamp / $2,500 Floor Stress-Test...", flush=True)
    surv_series = {m_id: series_map[m_id] for m_id in survivors if m_id in series_map}
    port_cfg = GrandPortfolioConfig(
        initial_equity_usd=2700.0,
        equity_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
        recovery_boost_equity_usd=3000.0,
        start_leverage=2.0,
        recovery_boost_leverage=2.8,
        max_leverage=4.2,
        floor_leverage=1.4,
        target_daily_vol=0.0115,
        max_gross_sleeve_sum=2.60,
    )
    syn_res = run_grand_portfolio_ablation_matrix(prod_daily, surv_series, cfg=port_cfg)
    ab_df = pd.DataFrame(syn_res["ablation_rows"])
    ab_df.to_csv(output_dir / "m16_5_portfolio_ablation.csv", index=False)

    port_summary = {
        "issue": "M16-5 (#71)",
        "elapsed_sec": round(time.time() - t0, 1),
        "survivors_integrated": survivors,
        "ablation_rows": syn_res["ablation_rows"],
        "correlation_matrix": syn_res["correlation_matrix"],
    }
    (output_dir / "m16_5_portfolio_summary.json").write_text(json.dumps(port_summary, indent=2) + "\n")

    md5 = [
        "# M16-5 Grand 4-Year Multi-Engine Portfolio Synthesis & `$2,500` Floor Stress-Test (`100% USDT`)",
        "",
        "- **Capital & Risk Governance:** `$2,700.00` initial equity (`100% USDT`, `BFUSD=OFF`), `SteppedLeverageController` Two-Phase SafeRamp (`$2,700 -> $3,000` boost) + Cushion-Safe CPPI Governor (`$2,500.00` Hard Floor, `$240.00` smooth buffer).",
        "",
        "| Portfolio Variant | `23-26` Sharpe | Lift | `%` Daily Geom | CAGR `%` | MaxDD `%` | Min Eq (`$2.7k`) | `2026` Hold Sharpe | `2026` `%` Daily | `2026` MaxDD | Cons. Sharpe | Go `M16-6` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in syn_res["ablation_rows"]:
        md5.append(
            f"| `{r['variant']}` | **`{r['maker_full_sharpe']:.2f}`** | `{r['maker_sharpe_lift']:+.2f}` | "
            f"**`{r['maker_geom_daily_pct']:+.3f}%`** | `{r['maker_cagr_pct']:+.1f}%` | **`{r['maker_max_dd_pct']:.2f}%`** | "
            f"**`${r['maker_min_eq_usd']:,.2f}`** | **`{r['holdout_2026_sharpe']:.2f}`** | "
            f"`{r['holdout_2026_geom_daily_pct']:+.3f}%` | `{r['holdout_2026_max_dd_pct']:.2f}%` | "
            f"`{r['cons_full_sharpe']:.2f}` | **`{'GO ✅' if r['go_for_m16_6'] else 'NO-GO'}`** |"
        )
    (output_dir / "m16_5_portfolio_report.md").write_text("\n".join(md5) + "\n")

    drive_root = Path("/content/drive/MyDrive/trading/crypto-alpha-engine")
    if drive_root.exists():
        import shutil

        drive_reports = drive_root / "reports"
        drive_reports.mkdir(parents=True, exist_ok=True)
        for fname in (
            "m16_4_discovery_leaderboard.csv",
            "m16_4_discovery_summary.json",
            "m16_4_discovery_report.md",
            "m16_5_portfolio_ablation.csv",
            "m16_5_portfolio_summary.json",
            "m16_5_portfolio_report.md",
        ):
            shutil.copy2(output_dir / fname, drive_reports / fname)

    print("M16_4_5_COMPLETE!", flush=True)
    return port_summary


def run_m16_portfolio_synthesis_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    rclone_stream: str | None = None,
) -> dict:
    """
    Executes M16-5 (Issue #71) Grand 4-Year Multi-Engine Portfolio Synthesis (`2023–2026`, `220` perps)
    using compact `bars_1h` + `funding` shards (`245 MB` total, `0 bytes` local residue), combining
    the 5 deployed production sleeves (`P1 Core 7-Factor Top-15 + Chop Lock`, `P2 Majors TSMOM`,
    `M09 Dispersion-Coil 48h Donchian`, `M23 82-Sym Breadth Thrust`, `M04 Neg-Funding Squeeze`)
    with the `M16-4` All-4-Gate Winner (`A1_FAST_1H_POST_SETTLE_UNWIND_20BPS`) under `$2,700`
    `SteppedLeverageController` SafeRamp + `$2,500` Cushion-Safe CPPI Floor (`100% USDT`).
    """
    import gc
    import json
    import shutil
    import subprocess
    import time

    from src.backtest.m16_grand_portfolio_synthesizer import (
        GrandPortfolioConfig,
        causal_inverse_vol_combine,
        run_grand_portfolio_ablation_matrix,
    )
    from src.features.multimodal_alpha import _vectorized_cs_zscore
    from src.models.symbolic_alpha_tournament import (
        CORE_MAJORS_15,
        compute_daily_series_metrics,
        simulate_cross_sectional_sleeve_daily,
        simulate_event_sniper_sleeve_daily,
    )

    t0 = time.time()
    stage_dir: Path | None = None
    if rclone_stream:
        stage_dir = Path("/tmp/m16_syn_1h_stage")
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        (stage_dir / "bars_1h").mkdir(parents=True, exist_ok=True)
        (stage_dir / "funding").mkdir(parents=True, exist_ok=True)
        print("[*] Streaming compact 220-perp bars_1h + funding shards via rclone (auto-wiped on exit)...", flush=True)
        subprocess.run(
            ["rclone", "copy", "--transfers=8", f"{rclone_stream.rstrip('/')}/bars_1h", str(stage_dir / "bars_1h")],
            check=True,
        )
        subprocess.run(
            ["rclone", "copy", "--transfers=8", f"{rclone_stream.rstrip('/')}/funding", str(stage_dir / "funding")],
            check=True,
        )
        m16_data_dir = stage_dir

    try:
        b1h_parts = [pd.read_parquet(p) for p in sorted((m16_data_dir / "bars_1h").glob("shard_*.parquet"))]
        fnd_parts = [pd.read_parquet(p) for p in sorted((m16_data_dir / "funding").glob("shard_*.parquet"))]
        df1h = pd.concat(b1h_parts, ignore_index=True)
        fnd = pd.concat(fnd_parts, ignore_index=True)
        del b1h_parts, fnd_parts
        gc.collect()
    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
            print("[*] Wiped /tmp/m16_syn_1h_stage -> 0 bytes local residue.", flush=True)

    df1h.sort_values(["symbol", "timestamp"], inplace=True, ignore_index=True)
    df1h["decision_ts"] = df1h["timestamp"].astype(np.int64) + 3600

    # Causal backward join of funding rates & settlement clock per symbol
    fnd.sort_values(["timestamp", "symbol"], inplace=True, ignore_index=True)
    if "interval_hours" not in fnd.columns:
        fnd["interval_hours"] = 8
    fnd_ren = fnd[["symbol", "timestamp", "funding_rate", "interval_hours"]].rename(columns={"timestamp": "f_ts"})
    df1h_t = df1h.sort_values("decision_ts").reset_index(drop=True)
    fnd_t = fnd_ren.sort_values("f_ts").reset_index(drop=True)
    joined = pd.merge_asof(
        df1h_t,
        fnd_t,
        by="symbol",
        left_on="decision_ts",
        right_on="f_ts",
        direction="backward",
    )
    del df1h, df1h_t, fnd, fnd_ren, fnd_t
    gc.collect()

    joined.sort_values(["symbol", "decision_ts"], inplace=True, ignore_index=True)
    joined["funding_rate_bps"] = pd.to_numeric(joined["funding_rate"], errors="coerce").fillna(0.0) * 1e4
    joined["funding_interval_hours"] = (
        pd.to_numeric(joined["interval_hours"], errors="coerce").fillna(8.0).clip(1.0, 8.0)
    )
    f_ts_num = pd.to_numeric(joined["f_ts"], errors="coerce")
    dec_f = joined["decision_ts"].astype(np.float64)
    joined["mins_since_settlement"] = ((dec_f - f_ts_num) / 60.0).where(f_ts_num.notna(), 240.0)
    joined["mins_to_next_settlement"] = ((f_ts_num + joined["funding_interval_hours"] * 3600.0 - dec_f) / 60.0).where(
        f_ts_num.notna(), 240.0
    )

    # Causal trailing multi-day features & forward 1h / 8h returns per symbol
    g_sym = joined.groupby("symbol", sort=False)
    log_c = np.log(joined["close"].clip(lower=1e-12))
    joined["ret_4h"] = log_c.groupby(joined["symbol"], sort=False).diff(4).fillna(0.0)
    joined["ret_24h"] = log_c.groupby(joined["symbol"], sort=False).diff(24).fillna(0.0)
    joined["ret_72h"] = log_c.groupby(joined["symbol"], sort=False).diff(72).fillna(0.0)
    joined["ret_168h"] = log_c.groupby(joined["symbol"], sort=False).diff(168).fillna(0.0)
    joined["adv_7d"] = (
        g_sym["quote_volume"].transform(lambda s: s.shift(1).rolling(168, min_periods=24).sum() / 7.0).fillna(0.0)
    )
    joined["rvol_1h"] = (
        joined["quote_volume"]
        / g_sym["quote_volume"].transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean()).clip(lower=1.0)
    ).clip(0.0, 25.0)
    tb_share = (joined["taker_buy_quote_volume"] / joined["quote_volume"].clip(lower=1.0)).clip(0.0, 1.0)
    joined["taker_ofi_4h"] = (
        tb_share.groupby(joined["symbol"], sort=False).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        - 0.5
    )
    joined["don_hi_48h"] = g_sym["high"].transform(lambda s: s.shift(1).rolling(48, min_periods=24).max())
    joined["don_lo_48h"] = g_sym["low"].transform(lambda s: s.shift(1).rolling(48, min_periods=24).min())
    joined["hi_24h"] = g_sym["high"].transform(lambda s: s.shift(1).rolling(24, min_periods=12).max())

    fwd_1h_c = g_sym["close"].shift(-1)
    fwd_8h_c = g_sym["close"].shift(-8)
    joined["fwd_ret_1h"] = np.log(fwd_1h_c / joined["close"].clip(lower=1e-12))
    joined["fwd_ret_8h"] = np.log(fwd_8h_c / joined["close"].clip(lower=1e-12))

    # Cross-sectional ADV rank at each 1h timestamp
    joined["adv_rank"] = joined.groupby("decision_ts")["adv_7d"].rank(ascending=False, method="first")
    disp_ts = joined[joined["adv_rank"] <= 40].groupby("decision_ts")["ret_24h"].std()
    disp_q20 = disp_ts.rolling(24 * 14, min_periods=48).quantile(0.20).shift(1)
    coil_active_ts = set(disp_ts.index[(disp_ts <= disp_q20).fillna(False)])

    # 1) P1 Core 7-Factor Top-15 (`8h` BHS(3,6) + 4h Institutional OFI + Multi-Day Pullback + Carry)
    p8_top15 = joined[
        (joined["decision_ts"] % 28800 == 0) & joined["symbol"].isin(CORE_MAJORS_15) & joined["fwd_ret_8h"].notna()
    ].reset_index(drop=True)
    p1_score = (
        0.35 * _vectorized_cs_zscore(p8_top15, "decision_ts", "taker_ofi_4h")
        - 0.35 * _vectorized_cs_zscore(p8_top15, "decision_ts", "ret_72h")
        + 0.15 * _vectorized_cs_zscore(p8_top15, "decision_ts", "ret_168h")
        - 0.15 * _vectorized_cs_zscore(p8_top15, "decision_ts", "funding_rate_bps")
    )
    p1_daily, _ = simulate_cross_sectional_sleeve_daily(
        p8_top15, p1_score, "fwd_ret_8h", horizon_hours=8, one_way_cost_bps=2.1, ema_alpha=0.18, top_k=4
    )

    # 2) P2 Manga Multi-Horizon TSMOM + Taker-CVD (`BTC/ETH/SOL`)
    p8_maj3 = p8_top15[p8_top15["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])].reset_index(drop=True)
    tsmom_raw = 0.50 * np.sign(p8_maj3["ret_168h"].to_numpy()) + 0.50 * np.sign(p8_maj3["ret_72h"].to_numpy())
    tsmom_sig = np.where(
        (np.abs(tsmom_raw) > 0.4) & (tsmom_raw * p8_maj3["taker_ofi_4h"].to_numpy() >= -0.01), np.sign(tsmom_raw), 0.0
    )
    p2_daily, _ = simulate_event_sniper_sleeve_daily(
        p8_maj3,
        tsmom_sig,
        "fwd_ret_8h",
        horizon_hours=8,
        roundtrip_cost_bps=1.2,
        slot_weight=0.75,
        max_concurrent_slots=3,
    )

    # 3) Phase 2 Sniper M09 (`Dispersion-Coil 48h Donchian + 2.2x Vol + CVD` on Top-25)
    top25_1h = joined[(joined["adv_rank"] <= 25) & joined["fwd_ret_8h"].notna()].reset_index(drop=True)
    in_coil = top25_1h["decision_ts"].isin(coil_active_ts).to_numpy()
    brk_up = (
        in_coil
        & (top25_1h["close"].to_numpy() > top25_1h["don_hi_48h"].to_numpy())
        & (top25_1h["rvol_1h"].to_numpy() >= 2.2)
        & (top25_1h["taker_ofi_4h"].to_numpy() >= 0.05)
    )
    brk_dn = (
        in_coil
        & (top25_1h["close"].to_numpy() < top25_1h["don_lo_48h"].to_numpy())
        & (top25_1h["rvol_1h"].to_numpy() >= 2.2)
        & (top25_1h["taker_ofi_4h"].to_numpy() <= -0.05)
    )
    m09_sig = np.where(brk_up, 1.0, np.where(brk_dn, -1.0, 0.0))
    m09_daily, _ = simulate_event_sniper_sleeve_daily(
        top25_1h,
        m09_sig,
        "fwd_ret_8h",
        horizon_hours=8,
        roundtrip_cost_bps=4.0,
        slot_weight=0.55,
        max_concurrent_slots=3,
    )

    # 4) Phase 2 Sniper M04 (`Neg-Funding <= -4 bps Trapped-Short 24h Breakout Long` on Top-60)
    top60_1h = joined[(joined["adv_rank"] <= 60) & joined["fwd_ret_8h"].notna()].reset_index(drop=True)
    m04_sig = np.where(
        (top60_1h["funding_rate_bps"].to_numpy() <= -4.0)
        & (top60_1h["close"].to_numpy() > top60_1h["hi_24h"].to_numpy())
        & (top60_1h["taker_ofi_4h"].to_numpy() >= 0.05),
        1.0,
        0.0,
    )
    m04_daily, _ = simulate_event_sniper_sleeve_daily(
        top60_1h,
        m04_sig,
        "fwd_ret_8h",
        horizon_hours=8,
        roundtrip_cost_bps=4.5,
        slot_weight=0.50,
        max_concurrent_slots=3,
    )

    prod_daily = causal_inverse_vol_combine(
        {"p1_core7f": p1_daily, "p2_tsmom": p2_daily, "m09_coil": m09_daily, "m04_squeeze": m04_daily},
        base_risk_shares={"p1_core7f": 0.42, "p2_tsmom": 0.24, "m09_coil": 0.20, "m04_squeeze": 0.14},
        vol_lookback_days=45,
        target_daily_vol=0.0110,
        max_leverage_cap=2.50,
    )
    print(f"[*] 00_CURRENT_PROD standalone unit metrics: {compute_daily_series_metrics(prod_daily)}", flush=True)

    # 5) M16-4 All-4-Gate Winner: `A1_FAST_1H_POST_SETTLE_UNWIND_20BPS` (`adv_7d >= $15M`, `0 <= mins_since <= 60` on 1h bars -> exact `:00` bar open right AT settlement `mins_since == 60` at bar close means bar opened AT settlement!)
    # Wait: `decision_ts` is `timestamp + 3600`. If settlement was at `timestamp` (`08:00`), then at `08:00` (`timestamp`) the settlement JUST happened!
    # To be 100% causal (`decision_ts` = bar close), at `decision_ts == 08:00` (`timestamp == 07:00`), `f_ts == 08:00` matched `decision_ts == 08:00` -> `mins_since_settlement == 0.0`, and `fwd_ret_1h` is `08:00 -> 09:00` (`0` negative funding paid!).
    liq_1h = joined[(joined["adv_7d"] >= 15_000_000.0) & joined["fwd_ret_1h"].notna()].reset_index(drop=True)
    a1_20_sig = np.where(
        (liq_1h["mins_since_settlement"].to_numpy() <= 5.0) & (liq_1h["funding_rate_bps"].to_numpy() <= -20.0),
        -np.abs(liq_1h["funding_rate_bps"].to_numpy()),
        0.0,
    )
    a1_20_daily, _ = simulate_event_sniper_sleeve_daily(
        liq_1h,
        a1_20_sig,
        "fwd_ret_1h",
        horizon_hours=1,
        roundtrip_cost_bps=4.4,
        slot_weight=0.22,
        max_concurrent_slots=4,
    )
    # 6) Secondary Orthogonal Stabilizer (`A3_MIDCAP_LOW_VOL_QUALITY_8H` on Mid-Caps ex-Top15)
    joined["hl_park"] = np.log(joined["high"].clip(lower=1e-12) / joined["low"].clip(lower=1e-12)) ** 2
    joined["park_24h"] = np.sqrt(
        joined["hl_park"]
        .groupby(joined["symbol"], sort=False)
        .rolling(24, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
        .clip(lower=0.0)
        / (4.0 * np.log(2.0))
    )
    mid8 = joined[
        (joined["decision_ts"] % 28800 == 0)
        & (joined["adv_7d"] >= 15_000_000.0)
        & (~joined["symbol"].isin(CORE_MAJORS_15))
        & joined["fwd_ret_8h"].notna()
    ].reset_index(drop=True)
    a3_sc = -0.60 * _vectorized_cs_zscore(mid8, "decision_ts", "park_24h") + 0.40 * _vectorized_cs_zscore(
        mid8, "decision_ts", "funding_rate_bps"
    )
    a3_daily, _ = simulate_cross_sectional_sleeve_daily(
        mid8, a3_sc, "fwd_ret_8h", horizon_hours=8, one_way_cost_bps=2.4, ema_alpha=0.14, top_k=12
    )
    print(f"[*] A1_POST_SETTLE_20BPS standalone metrics: {compute_daily_series_metrics(a1_20_daily)}", flush=True)
    print(f"[*] A3_MIDCAP_LOW_VOL_QUALITY_8H metrics: {compute_daily_series_metrics(a3_daily)}", flush=True)

    port_cfg = GrandPortfolioConfig(
        initial_equity_usd=2700.0,
        equity_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
        recovery_boost_equity_usd=3000.0,
        start_leverage=1.8,
        recovery_boost_leverage=2.4,
        max_leverage=3.1,
        floor_leverage=1.3,
        target_daily_vol=0.0076,
        max_gross_sleeve_sum=2.20,
    )
    surv_series = {
        "A1_FAST_1H_POST_SETTLE_UNWIND_20BPS": a1_20_daily,
        "A3_MIDCAP_LOW_VOL_QUALITY_8H": a3_daily,
    }
    syn_res = run_grand_portfolio_ablation_matrix(prod_daily, surv_series, cfg=port_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(syn_res["ablation_rows"]).to_csv(output_dir / "m16_5_portfolio_ablation.csv", index=False)

    port_summary = {
        "issue": "M16-5 (#71)",
        "elapsed_sec": round(time.time() - t0, 1),
        "survivors_integrated": list(surv_series.keys()),
        "ablation_rows": syn_res["ablation_rows"],
        "correlation_matrix": syn_res["correlation_matrix"],
    }
    (output_dir / "m16_5_portfolio_summary.json").write_text(json.dumps(port_summary, indent=2) + "\n")

    md5 = [
        "# M16-5 Grand 4-Year Multi-Engine Portfolio Synthesis & `$2,500` Floor Stress-Test (`100% USDT`)",
        "",
        "- **Capital & Risk Governance:** `$2,700.00` initial equity (`100% USDT`, `BFUSD=OFF`), `SteppedLeverageController` Two-Phase SafeRamp (`$2,700 -> $3,000` boost) + Cushion-Safe CPPI Governor (`$2,500.00` Hard Floor, `$240.00` smooth buffer).",
        "",
        "| Portfolio Variant | `23-26` Sharpe | Lift | `%` Daily Geom | CAGR `%` | MaxDD `%` | Min Eq (`$2.7k`) | `2026` Hold Sharpe | `2026` `%` Daily | `2026` MaxDD | Cons. Sharpe | Go `M16-6` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in syn_res["ablation_rows"]:
        md5.append(
            f"| `{r['variant']}` | **`{r['maker_full_sharpe']:.2f}`** | `{r['maker_sharpe_lift']:+.2f}` | "
            f"**`{r['maker_geom_daily_pct']:+.3f}%`** | `{r['maker_cagr_pct']:+.1f}%` | **`{r['maker_max_dd_pct']:.2f}%`** | "
            f"**`${r['maker_min_eq_usd']:,.2f}`** | **`{r['holdout_2026_sharpe']:.2f}`** | "
            f"`{r['holdout_2026_geom_daily_pct']:+.3f}%` | `{r['holdout_2026_max_dd_pct']:.2f}%` | "
            f"`{r['cons_full_sharpe']:.2f}` | **`{'GO ✅' if r['go_for_m16_6'] else 'NO-GO'}`** |"
        )
    (output_dir / "m16_5_portfolio_report.md").write_text("\n".join(md5) + "\n")
    print("M16_5_SYNTHESIS_COMPLETE!", flush=True)
    return port_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Quantitative & Microstructure Benchmark Suite CLI")
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "hrp",
            "info-bars",
            "market-impact",
            "anomalies",
            "cointegration",
            "m16-feature-atlas",
            "m16-gtx-matrix",
            "m16-discovery-synthesis",
            "m16-portfolio-synthesis",
            "m16-pipeline",
        ],
        default="hrp",
        help="Benchmark suite to execute (default: hrp).",
    )
    parser.add_argument(
        "--m16-data-dir",
        type=str,
        default="/tmp/m16_multimodal",
        help="Path to M16-1 sharded multi-modal directory (or mounted Google Drive path on Colab).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=20,
        help="Number of M16-1 shards to stream during m16-feature-atlas (default: 20).",
    )
    parser.add_argument(
        "--colab-session",
        type=str,
        default=None,
        help="Optional Colab session name to run m16-feature-atlas remotely against Google Drive.",
    )
    parser.add_argument(
        "--keep-colab",
        action="store_true",
        help="Keep the Colab session & Google Drive mount alive across M16-2..M16-5 runs.",
    )
    parser.add_argument(
        "--rclone-stream",
        type=str,
        default=None,
        help="Optional rclone remote dir (e.g. gdrive:trading/.../m16_multimodal) for zero-residue shard streaming.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/crypto_alpha_backtest_cache", help="Market data cache path."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,NEARUSDT,LTCUSDT,DOTUSDT",
        help="Comma-separated symbols for HRP benchmark.",
    )
    parser.add_argument("--lookback", type=int, default=4000, help="Total 1h bars to evaluate.")
    parser.add_argument("--window", type=int, default=720, help="HRP estimation window in hours (default: 720).")
    parser.add_argument("--rebalance", type=int, default=24, help="HRP rebalancing frequency in hours (default: 24).")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Symbol for info-bars benchmark (default: BTCUSDT)."
    )
    parser.add_argument(
        "--date", type=str, default="2024-03-01", help="Date for info-bars tick streaming (default: 2024-03-01)."
    )
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cache_path = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.suite == "all":
        suites = ["hrp", "market-impact", "cointegration", "anomalies", "info-bars"]
    elif args.suite == "m16-pipeline":
        suites = ["m16-gtx-matrix", "m16-discovery-synthesis"]
    else:
        suites = [args.suite]

    pipe_stage_dir: Path | None = None
    m16_dir = Path(args.m16_data_dir)
    if args.rclone_stream and args.suite in ("m16-gtx-matrix", "m16-discovery-synthesis", "m16-pipeline"):
        import shutil
        import subprocess

        pipe_stage_dir = Path("/tmp/m16_pipe_stage")
        if pipe_stage_dir.exists():
            shutil.rmtree(pipe_stage_dir, ignore_errors=True)
        (pipe_stage_dir / "features_panel").mkdir(parents=True, exist_ok=True)
        print(f"[*] Staging M16-2 causal panel via rclone into {pipe_stage_dir} (auto-cleaned on exit)...", flush=True)
        subprocess.run(
            [
                "rclone",
                "copy",
                "--multi-thread-streams=8",
                "--multi-thread-cutoff=16M",
                f"{args.rclone_stream.rstrip('/')}/features_panel/m16_2_causal_panel.parquet",
                str(pipe_stage_dir / "features_panel"),
            ],
            check=True,
        )
        m16_dir = pipe_stage_dir

    try:
        for s_name in suites:
            if s_name == "hrp":
                run_hrp_benchmark(
                    cache_dir=cache_path,
                    symbols=symbols,
                    lookback=args.lookback,
                    window=args.window,
                    rebalance=args.rebalance,
                    output_dir=out_dir,
                )
            elif s_name == "info-bars":
                run_info_bars_benchmark(symbol=args.symbol, date_str=args.date)
            elif s_name == "market-impact":
                run_market_impact_benchmark()
            elif s_name == "anomalies":
                run_pure_anomalies_benchmark(cache_dir=cache_path)
            elif s_name == "cointegration":
                run_cointegration_benchmark(cache_dir=cache_path, lookback=args.lookback)
            elif s_name == "m16-feature-atlas":
                run_m16_feature_atlas_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    max_shards=args.max_shards,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "m16-gtx-matrix":
                run_m16_gtx_matrix_benchmark(
                    m16_data_dir=m16_dir,
                    output_dir=out_dir,
                    max_shards=args.max_shards,
                    colab_session=args.colab_session,
                    keep_colab=True if args.suite == "m16-pipeline" else args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "m16-discovery-synthesis":
                run_m16_discovery_and_synthesis_benchmark(
                    m16_data_dir=m16_dir,
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                )
            elif s_name == "m16-portfolio-synthesis":
                run_m16_portfolio_synthesis_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    rclone_stream=args.rclone_stream,
                )
    finally:
        if pipe_stage_dir is not None and pipe_stage_dir.exists():
            import shutil

            shutil.rmtree(pipe_stage_dir, ignore_errors=True)
            print("[*] Cleaned temporary staging directory (/tmp/m16_pipe_stage -> 0 bytes).", flush=True)


if __name__ == "__main__":
    main()
