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


def run_counterparty_mechanics_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    max_shards: int = 20,
    colab_session: str | None = None,
    keep_colab: bool = False,
) -> dict:
    """
    Executes the 1-Minute Linear-Return Counterparty-Mechanics & Forced-Flow Laboratory
    (`docs/research/08_auditoria_critica_m16_y_nueva_ciencia_microestructura.md` Parts II & III):
      - Zero `np.log` returns (`100%` exact linear `(P_exit - P_entry) / P_entry`)
      - Native `1m` Perp + `1m` Spot + `+60s`-lagged `5m` Contract-OI/Whale/Retail + exact `:00:00` Funding
      - Side-by-side `TAKER_OPEN` vs `TOUCH_LIMIT` vs `GTX_1.5BPS` with `100%` pessimistic same-bar SL priority
      - True Out-of-Sample In-Fold Refit `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits) + Era Audit (`23-24`, `2025`, `2026`)
    """
    if colab_session:
        import time

        from src.cloud.colab_client import ColabClient

        client = ColabClient(session_name=colab_session)
        print(
            f"[*] Provisioning/connecting to Colab session '{colab_session}' (CPU, isolated from user sessions)...",
            flush=True,
        )
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
                "lf = open('/content/counterparty_lab.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                f"'--suite', 'counterparty-mechanics-lab', '--m16-data-dir', '{drive_m16}', "
                f"'--max-shards', '{max_shards}', '--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/counterparty_lab.pid', 'w').write(str(p.pid)); "
                "print(f'LAB_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=120)
            print(f"[+] Launched remote 1m Counterparty-Mechanics Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            while True:
                time.sleep(12.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/counterparty_lab.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/counterparty_lab.log').read().splitlines() "
                    "if os.path.exists('/content/counterparty_lab.log') else []; "
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
                        if "COUNTERPARTY_MECHANICS_LAB_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Counterparty-Mechanics Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "counterparty_mechanics_1m_leaderboard.csv",
                "counterparty_mechanics_1m_summary.json",
                "counterparty_mechanics_1m_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            print(f"[+] Downloaded 1m Counterparty-Mechanics Lab artifacts into {output_dir}", flush=True)
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

    from src.models.counterparty_mechanics_lab import (
        COUNTERPARTY_MECHANISM_SPECS,
        evaluate_infold_cpcv_and_treasure_metrics,
        extract_sovereign_btc_shocks,
        scan_symbol_counterparty_mechanisms_1m,
    )

    t0 = time.time()
    bars_dir = m16_data_dir / "bars_1m"
    spot_dir = m16_data_dir / "spot_bars_1m"
    met_dir = m16_data_dir / "metrics_5m"
    fnd_dir = m16_data_dir / "funding"
    shard_files = sorted(bars_dir.glob("shard_*.parquet"))[:max_shards]
    if not shard_files:
        raise FileNotFoundError(f"No 1m shards found in {bars_dir}")

    perp_cols = ["symbol", "timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    spot_cols = ["symbol", "timestamp", "close", "quote_volume", "taker_buy_quote_volume"]
    met_cols = [
        "symbol",
        "timestamp",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "count_long_short_ratio",
    ]
    fnd_cols = ["symbol", "timestamp", "funding_rate", "interval_hours"]

    # Step 1: Locate & extract Sovereign 1m BTC shocks (`C5`) + BTC daily benchmark via Parquet predicate pushdown
    print(f"🔬 1m Linear Counterparty-Mechanics Lab | Scanning {len(shard_files)} shards for BTCUSDT...", flush=True)
    btc_perp_parts: list[pd.DataFrame] = []
    btc_spot_parts: list[pd.DataFrame] = []
    for sh_p in shard_files:
        bp = pd.read_parquet(sh_p, columns=perp_cols, filters=[("symbol", "==", "BTCUSDT")])
        if not bp.empty:
            btc_perp_parts.append(bp)
            sp_p = spot_dir / sh_p.name
            if sp_p.exists():
                bs = pd.read_parquet(sp_p, columns=spot_cols, filters=[("symbol", "==", "BTCUSDT")])
                if not bs.empty:
                    btc_spot_parts.append(bs)
            break

    btc_perp_df = pd.concat(btc_perp_parts, ignore_index=True) if btc_perp_parts else pd.DataFrame()
    btc_spot_df = pd.concat(btc_spot_parts, ignore_index=True) if btc_spot_parts else pd.DataFrame()
    btc_shocks = extract_sovereign_btc_shocks(btc_perp_df, btc_spot_df)
    n_btc_shocks = int(np.sum(btc_shocks["shock_dir"] != 0)) if len(btc_shocks["shock_dir"]) else 0
    print(
        f"  • Extracted {n_btc_shocks:,} discrete 3m BTC Cash+Perp shock impulses ({time.time() - t0:.1f}s)", flush=True
    )

    btc_daily_ret = pd.Series(dtype=float)
    if not btc_perp_df.empty:
        b_sorted = btc_perp_df.sort_values("timestamp")
        b_dt = pd.to_datetime(b_sorted["timestamp"], unit="s", utc=True)
        b_close_d = pd.Series(b_sorted["close"].to_numpy(dtype=np.float64), index=b_dt).resample("1D").last().dropna()
        btc_daily_ret = b_close_d.pct_change().fillna(0.0)
    del btc_perp_parts, btc_spot_parts, btc_perp_df, btc_spot_df
    gc.collect()

    # Step 2: Stream shards sequentially with bounded RSS (< 2.5 GB)
    all_trade_records: list[dict] = []
    symbols_scanned = 0

    for idx_s, sh_p in enumerate(shard_files):
        t_sh = time.time()
        sh_name = sh_p.name
        b1m = pd.read_parquet(sh_p, columns=perp_cols)
        perp_by_sym = dict(tuple(b1m.groupby("symbol", observed=True)))
        del b1m

        sp_path = spot_dir / sh_name
        if sp_path.exists():
            s1m = pd.read_parquet(sp_path, columns=spot_cols)
            spot_by_sym = dict(tuple(s1m.groupby("symbol", observed=True)))
            del s1m
        else:
            spot_by_sym = {}

        m_path = met_dir / sh_name
        if m_path.exists():
            # Read available subset of met_cols safely
            m_raw = pd.read_parquet(m_path)
            keep_m = [c for c in met_cols if c in m_raw.columns]
            m_raw = m_raw[keep_m]
            met_by_sym = dict(tuple(m_raw.groupby("symbol", observed=True)))
            del m_raw
        else:
            met_by_sym = {}

        f_path = fnd_dir / sh_name
        if f_path.exists():
            f_raw = pd.read_parquet(f_path)
            keep_f = [c for c in fnd_cols if c in f_raw.columns]
            f_raw = f_raw[keep_f]
            fnd_by_sym = dict(tuple(f_raw.groupby("symbol", observed=True)))
            del f_raw
        else:
            fnd_by_sym = {}

        sh_trades = 0
        for sym_k, p_sym_df in perp_by_sym.items():
            sym_s = str(sym_k)
            symbols_scanned += 1
            recs = scan_symbol_counterparty_mechanisms_1m(
                symbol=sym_s,
                perp_1m=p_sym_df,
                spot_1m=spot_by_sym.get(sym_k),
                metrics_5m=met_by_sym.get(sym_k),
                funding_df=fnd_by_sym.get(sym_k),
                btc_shocks=btc_shocks,
            )
            if recs:
                all_trade_records.extend(recs)
                sh_trades += len(recs)

        del perp_by_sym, spot_by_sym, met_by_sym, fnd_by_sym
        gc.collect()
        print(
            f"  [{idx_s + 1:02d}/{len(shard_files):02d}] {sh_name}: +{sh_trades:,} 1m linear trade records "
            f"(cum={len(all_trade_records):,}, syms={symbols_scanned}, dt={time.time() - t_sh:.1f}s)",
            flush=True,
        )

    trades_df = pd.DataFrame(all_trade_records)
    del all_trade_records
    gc.collect()

    print(f"[*] Evaluating In-Fold Refit CPCV & Treasure Metrics across {len(trades_df):,} trades...", flush=True)
    lb_df = evaluate_infold_cpcv_and_treasure_metrics(trades_df, total_trials=30)

    # Attach daily correlation vs BTCUSDT linear returns + exit reason shares
    corr_map: dict[tuple[str, str], float] = {}
    exit_breakdown: dict[str, dict[str, float]] = {}
    if not trades_df.empty and not lb_df.empty:
        trades_df["day"] = pd.to_datetime(trades_df["entry_ts"], unit="s", utc=True).dt.floor("D")
        for (m_id, ex_m), grp in trades_df.groupby(["mech_id", "exec_mode"], sort=False):
            key_str = f"{m_id}|{ex_m}"
            vc = grp["exit_reason"].value_counts(normalize=True) * 100.0
            exit_breakdown[key_str] = {str(k): round(float(v), 1) for k, v in vc.items()}
            if not btc_daily_ret.empty:
                d_s = grp.groupby("day")["net_bps"].sum().reindex(btc_daily_ret.index, fill_value=0.0)
                if float(d_s.std()) > 1e-9 and float(btc_daily_ret.std()) > 1e-9:
                    corr_map[(str(m_id), str(ex_m))] = round(float(d_s.corr(btc_daily_ret)), 3)
                else:
                    corr_map[(str(m_id), str(ex_m))] = 0.0
        lb_df["corr_vs_btc"] = [
            corr_map.get((str(r["mech_id"]), str(r["exec_mode"])), 0.0) for _, r in lb_df.iterrows()
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "counterparty_mechanics_1m_leaderboard.csv", index=False)

    treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
    summary = {
        "benchmark": "counterparty_mechanics_1m_linear_lab",
        "elapsed_sec": round(time.time() - t0, 1),
        "symbols_scanned": symbols_scanned,
        "total_simulated_trades": int(len(trades_df)),
        "btc_sovereign_shocks": n_btc_shocks,
        "treasure_count": len(treasures),
        "treasures": treasures,
        "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
        "exit_reason_pct_by_variant": exit_breakdown,
    }
    (output_dir / "counterparty_mechanics_1m_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# 1-Minute Linear-Return Counterparty-Mechanics Laboratory (`2023–2026`, `220` Perps)",
        "",
        f"- **Universe Scanned:** `{symbols_scanned}` Point-in-Time USDS-M Perps (`{len(shard_files)}` shards, `{len(trades_df):,}` total simulated 1m trades, `{summary['elapsed_sec']}s` runtime).",
        "- **Return Accounting:** `100%` Exact Linear Perpetual Return `side * (P_exit - P_entry) / P_entry + Realized_Funding - Fees` (**zero `np.log` convexity bias**).",
        "- **Intrabar Execution Matrix:** `TAKER_OPEN` (`open[t+1]` + `6.5 bps` fee+slip) vs `TOUCH_LIMIT` (`0 bps` offset, `1.5 bps` price-through, `2.0 bps` Maker) vs `GTX_1.5BPS` (`+1.5 bps` offset, `2.0 bps` price-through, `2.0 bps` Maker) with **`100%` pessimistic same-bar Stop-Loss priority**.",
        f"- **Strict Treasure Survivors (`>= +25 bps` Full, `>= +15 bps` in `23-24` & `2025`, `>= +20 bps` in `2026`, `Fee Share <= 30%`, True In-Fold CPCV `>= 80%`):** **`{len(treasures)}`**",
        "",
        "| Mechanism | Mode | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | `2025` | `2026` | In-Fold CPCV bps | CPCV`%` | BTC Corr | Treasure |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in lb_df.iterrows():
        md.append(
            f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
            f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
            f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['med_net_bps_full']:+.1f}` | `{r['win_rate_pct']:.1f}%` | "
            f"`{r['fee_share_of_gross_pct']:.1f}%` | `{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | "
            f"**`{r['net_bps_2026']:+.1f}`** | `{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
            f"`{r.get('corr_vs_btc', 0.0):+.2f}` | **`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
        )
    md.append("")
    md.append("## Mechanism Specifications")
    for m_k, sp_v in COUNTERPARTY_MECHANISM_SPECS.items():
        md.append(f"- **`{m_k}`** (`{sp_v.pillar}`): {sp_v.description}")
    (output_dir / "counterparty_mechanics_1m_report.md").write_text("\n".join(md) + "\n")

    print("COUNTERPARTY_MECHANICS_LAB_COMPLETE!", flush=True)
    return summary


def run_p1_microstructure_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    max_shards: int = 20,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict:
    """
    Executes Phase-P1 Microstructure Data Ingestion (`1m premiumIndexKlines` + `L2 bookDepth` +
    On-Chain `Hyperliquid L1` Funding Dispersion) & Unchoked Convex Monetization Laboratory
    across the `220` Point-in-Time perps (`2023–2026`).
    """
    if colab_session:
        import time

        from src.cloud.colab_client import ColabClient

        client = ColabClient(session_name=colab_session)
        print(
            f"[*] Provisioning/connecting to Colab session '{colab_session}' (CPU, isolated from user sessions)...",
            flush=True,
        )
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
                "lf = open('/content/p1_micro_lab.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                f"'--suite', 'p1-microstructure-lab', '--m16-data-dir', '{drive_m16}', "
                f"'--max-shards', '{max_shards}', '--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/p1_micro_lab.pid', 'w').write(str(p.pid)); "
                "print(f'P1_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=120)
            print(f"[+] Launched remote Phase-P1 Microstructure Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(12.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/p1_micro_lab.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/p1_micro_lab.log').read().splitlines() "
                    "if os.path.exists('/content/p1_micro_lab.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=120)
                if rc_p != 0:
                    consec_err += 1
                    print(f"[!] Poll warning ({consec_err}/4): {(err_p or out_p).strip()[:160]}", flush=True)
                    if consec_err >= 4:
                        raise RuntimeError("Colab session disconnected or reclaimed during polling.")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "P1_MICROSTRUCTURE_LAB_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Phase-P1 Microstructure Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "p1_microstructure_1m_leaderboard.csv",
                "p1_microstructure_1m_summary.json",
                "p1_microstructure_1m_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            print(f"[+] Downloaded Phase-P1 Microstructure Lab artifacts into {output_dir}", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            else:
                print(f"[*] Kept Colab session '{colab_session}' mounted & alive (--keep-colab).", flush=True)

    import gc
    import json
    import shutil
    import subprocess
    import time

    from src.data.microstructure_p1_ingestor import (
        ensure_cross_venue_funding_parquet,
        ensure_premium_index_1m_shards,
        fetch_shard_event_book_depth,
    )
    from src.models.microstructure_monetization_lab import (
        P1_MECHANISM_PROFILES,
        evaluate_symbol_p1_with_book_depth,
        prepare_symbol_p1_base_state,
        summarize_p1_lab_results,
    )

    t0 = time.time()
    rclone_stage_dir: Path | None = None
    if rclone_stream:
        rclone_base = rclone_stream.rstrip("/")
        rclone_stage_dir = Path("/tmp/p1_rclone_stage")
        if rclone_stage_dir.exists():
            shutil.rmtree(rclone_stage_dir, ignore_errors=True)
        for sub_d in (
            "bars_1m",
            "spot_bars_1m",
            "premium_1m",
            "metrics_5m",
            "funding",
            "cross_venue",
            "book_depth_events",
            "p1_trades_ckpt_v1",
            "bars_1h",
        ):
            (rclone_stage_dir / sub_d).mkdir(parents=True, exist_ok=True)
        print("[*] Syncing P1 checkpoints + HL cross-venue funding via rclone (0 permanent local bytes)...", flush=True)
        subprocess.run(
            ["rclone", "copy", f"{rclone_base}/p1_trades_ckpt_v1", str(rclone_stage_dir / "p1_trades_ckpt_v1")],
            check=False,
        )
        subprocess.run(
            ["rclone", "copy", f"{rclone_base}/cross_venue", str(rclone_stage_dir / "cross_venue")], check=False
        )
        subprocess.run(
            ["rclone", "copy", "--transfers=8", f"{rclone_base}/bars_1h", str(rclone_stage_dir / "bars_1h")],
            check=False,
        )
        m16_data_dir = rclone_stage_dir

    try:
        bars_dir = m16_data_dir / "bars_1m"
        spot_dir = m16_data_dir / "spot_bars_1m"
        prem_dir = m16_data_dir / "premium_1m"
        met_dir = m16_data_dir / "metrics_5m"
        fnd_dir = m16_data_dir / "funding"
        ckpt_dir = m16_data_dir / "p1_trades_ckpt_v1"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if rclone_stream:
            shard_names = [f"shard_{i:02d}.parquet" for i in range(max_shards)]
        else:
            shard_files = sorted(bars_dir.glob("shard_*.parquet"))[:max_shards]
            if not shard_files:
                raise FileNotFoundError(f"No 1m shards found in {bars_dir}")
            shard_names = [p.name for p in shard_files]
            print(
                f"🔬 Phase-P1 Microstructure Lab | Step 1/3: Ensuring 1m premiumIndexKlines across {len(shard_names)} shards...",
                flush=True,
            )
            prem_stats = ensure_premium_index_1m_shards(m16_data_dir, max_shards=max_shards, workers=48)
            print(f"  • Premium1m ready: {prem_stats}", flush=True)

        perp_cols = ["symbol", "timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
        spot_cols = ["symbol", "timestamp", "close", "quote_volume", "taker_buy_quote_volume"]
        met_cols = ["symbol", "timestamp", "sum_open_interest", "sum_open_interest_value"]
        fnd_cols = ["symbol", "timestamp", "funding_rate", "interval_hours"]

        all_syms: list[str] = []
        btc_daily_ret = pd.Series(dtype=float)
        b1h_dir = m16_data_dir / "bars_1h"
        src_meta_files = (
            sorted(b1h_dir.glob("shard_*.parquet"))
            if b1h_dir.exists() and list(b1h_dir.glob("shard_*.parquet"))
            else sorted(bars_dir.glob("shard_*.parquet"))
        )
        for sh_p in src_meta_files:
            s_col = pd.read_parquet(sh_p, columns=["symbol"])["symbol"].unique()
            syms_here = [str(x) for x in s_col]
            all_syms.extend(syms_here)
            if "BTCUSDT" in syms_here and btc_daily_ret.empty:
                bp = pd.read_parquet(sh_p, columns=["timestamp", "close"], filters=[("symbol", "==", "BTCUSDT")])
                if not bp.empty:
                    b_s = bp.sort_values("timestamp")
                    b_dt = pd.to_datetime(b_s["timestamp"], unit="s", utc=True)
                    btc_daily_ret = (
                        pd.Series(b_s["close"].to_numpy(dtype=np.float64), index=b_dt)
                        .resample("1D")
                        .last()
                        .dropna()
                        .pct_change()
                        .fillna(0.0)
                    )
        if rclone_stream and b1h_dir.exists():
            shutil.rmtree(b1h_dir, ignore_errors=True)

        print(
            f"🔬 Step 2/3: Ensuring On-Chain Hyperliquid L1 funding across {len(set(all_syms))} universe perps...",
            flush=True,
        )
        hl_path = ensure_cross_venue_funding_parquet(m16_data_dir, symbols=all_syms, workers=3)
        hl_all_df = pd.read_parquet(hl_path) if hl_path.exists() else pd.DataFrame()
        hl_by_sym = dict(tuple(hl_all_df.groupby("symbol", observed=True))) if not hl_all_df.empty else {}
        del hl_all_df
        gc.collect()

        print("🔬 Step 3/3: Streaming 20 shards with causal L2 bookDepth + 3 Monetization Geometries...", flush=True)
        shard_trade_dfs: list[pd.DataFrame] = []
        symbols_scanned = 0
        total_book_days = 0

        for idx_s, sh_name in enumerate(shard_names):
            t_sh = time.time()
            sh_ckpt = ckpt_dir / sh_name
            if sh_ckpt.exists() and sh_ckpt.stat().st_size > 256:
                cached_tdf = pd.read_parquet(sh_ckpt)
                shard_trade_dfs.append(cached_tdf)
                symbols_scanned += 11
                print(
                    f"  [{idx_s + 1:02d}/{len(shard_names):02d}] {sh_name}: REUSED +{len(cached_tdf):,} cached P1 trades "
                    f"(dt={time.time() - t_sh:.1f}s)",
                    flush=True,
                )
                continue

            if rclone_stream:
                rclone_base = rclone_stream.rstrip("/")
                for mod_d in ("bars_1m", "spot_bars_1m", "premium_1m", "metrics_5m", "funding", "book_depth_events"):
                    subprocess.run(
                        [
                            "rclone",
                            "copy",
                            "--multi-thread-streams=8",
                            "--multi-thread-cutoff=16M",
                            f"{rclone_base}/{mod_d}/{sh_name}",
                            str(m16_data_dir / mod_d),
                        ],
                        check=(mod_d == "bars_1m"),
                    )

            sh_p = bars_dir / sh_name
            b1m = pd.read_parquet(sh_p, columns=perp_cols)
            perp_by_sym = dict(tuple(b1m.groupby("symbol", observed=True)))
            del b1m

            sp_by_sym = (
                dict(tuple(pd.read_parquet(spot_dir / sh_name, columns=spot_cols).groupby("symbol", observed=True)))
                if (spot_dir / sh_name).exists()
                else {}
            )
            pr_by_sym = (
                dict(tuple(pd.read_parquet(prem_dir / sh_name).groupby("symbol", observed=True)))
                if (prem_dir / sh_name).exists()
                else {}
            )
            if (met_dir / sh_name).exists():
                m_raw = pd.read_parquet(met_dir / sh_name)
                m_raw = m_raw[[c for c in met_cols if c in m_raw.columns]]
                met_by_sym = dict(tuple(m_raw.groupby("symbol", observed=True)))
                del m_raw
            else:
                met_by_sym = {}

            if (fnd_dir / sh_name).exists():
                f_raw = pd.read_parquet(fnd_dir / sh_name)
                f_raw = f_raw[[c for c in fnd_cols if c in f_raw.columns]]
                fnd_by_sym = dict(tuple(f_raw.groupby("symbol", observed=True)))
                del f_raw
            else:
                fnd_by_sym = {}

            states: dict[str, dict] = {}
            needed_dates_by_sym: dict[str, set[str]] = {}
            for sym_k, p_df in perp_by_sym.items():
                sym_s = str(sym_k)
                symbols_scanned += 1
                st = prepare_symbol_p1_base_state(
                    symbol=sym_s,
                    perp_1m=p_df,
                    spot_1m=sp_by_sym.get(sym_k),
                    prem_1m=pr_by_sym.get(sym_k),
                    metrics_5m=met_by_sym.get(sym_k),
                    funding_df=fnd_by_sym.get(sym_k),
                    hl_funding_df=hl_by_sym.get(sym_s),
                )
                if st is not None:
                    states[sym_s] = st
                    if st["needed_dates"]:
                        needed_dates_by_sym[sym_s] = st["needed_dates"]
                        total_book_days += len(st["needed_dates"])

            del perp_by_sym, sp_by_sym, pr_by_sym, met_by_sym, fnd_by_sym
            gc.collect()

            book_df_sh = (
                fetch_shard_event_book_depth(m16_data_dir, sh_name, needed_dates_by_sym, workers=12)
                if needed_dates_by_sym
                else pd.DataFrame()
            )
            book_by_sym = dict(tuple(book_df_sh.groupby("symbol", observed=True))) if not book_df_sh.empty else {}
            del book_df_sh

            sh_recs: list[dict] = []
            for sym_s, st in states.items():
                recs = evaluate_symbol_p1_with_book_depth(st, book_by_sym.get(sym_s))
                if recs:
                    sh_recs.extend(recs)

            del states, book_by_sym
            sh_tdf = pd.DataFrame(sh_recs)
            if not sh_tdf.empty:
                sh_tdf.to_parquet(sh_ckpt, index=False, compression="zstd")
                shard_trade_dfs.append(sh_tdf)

            if rclone_stream:
                rclone_base = rclone_stream.rstrip("/")
                if sh_ckpt.exists():
                    subprocess.run(["rclone", "copy", str(sh_ckpt), f"{rclone_base}/p1_trades_ckpt_v1"], check=False)
                bd_f = m16_data_dir / "book_depth_events" / sh_name
                if bd_f.exists():
                    subprocess.run(["rclone", "copy", str(bd_f), f"{rclone_base}/book_depth_events"], check=False)
                for mod_d in ("bars_1m", "spot_bars_1m", "premium_1m", "metrics_5m", "funding", "book_depth_events"):
                    (m16_data_dir / mod_d / sh_name).unlink(missing_ok=True)

            cum_n = sum(len(x) for x in shard_trade_dfs)
            gc.collect()
            print(
                f"  [{idx_s + 1:02d}/{len(shard_names):02d}] {sh_name}: +{len(sh_tdf):,} P1 trades "
                f"(cum={cum_n:,}, syms={symbols_scanned}, dt={time.time() - t_sh:.1f}s)",
                flush=True,
            )

        trades_df = pd.concat(shard_trade_dfs, ignore_index=True) if shard_trade_dfs else pd.DataFrame()
        del shard_trade_dfs
        gc.collect()

        print(
            f"[*] Evaluating In-Fold Refit CPCV & Treasure Metrics across {len(trades_df):,} P1 trades...", flush=True
        )
        lb_df = summarize_p1_lab_results(trades_df)

        corr_map: dict[tuple[str, str], float] = {}
        exit_breakdown: dict[str, dict[str, float]] = {}
        if not trades_df.empty and not lb_df.empty:
            trades_df["day"] = pd.to_datetime(trades_df["entry_ts"], unit="s", utc=True).dt.floor("D")
            for (m_id, ex_m), grp in trades_df.groupby(["mech_id", "exec_mode"], sort=False):
                vc = grp["exit_reason"].value_counts(normalize=True) * 100.0
                exit_breakdown[f"{m_id}|{ex_m}"] = {str(k): round(float(v), 1) for k, v in vc.items()}
                if not btc_daily_ret.empty:
                    d_s = grp.groupby("day")["net_bps"].sum().reindex(btc_daily_ret.index, fill_value=0.0)
                    if float(d_s.std()) > 1e-9 and float(btc_daily_ret.std()) > 1e-9:
                        corr_map[(str(m_id), str(ex_m))] = round(float(d_s.corr(btc_daily_ret)), 3)
                    else:
                        corr_map[(str(m_id), str(ex_m))] = 0.0
            lb_df["corr_vs_btc"] = [
                corr_map.get((str(r["mech_id"]), str(r["exec_mode"])), 0.0) for _, r in lb_df.iterrows()
            ]

        output_dir.mkdir(parents=True, exist_ok=True)
        lb_df.to_csv(output_dir / "p1_microstructure_1m_leaderboard.csv", index=False)

        treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
        summary = {
            "benchmark": "p1_microstructure_monetization_1m_lab",
            "elapsed_sec": round(time.time() - t0, 1),
            "symbols_scanned": symbols_scanned,
            "candidate_book_depth_symbol_days": total_book_days,
            "total_simulated_trades": int(len(trades_df)),
            "treasure_count": len(treasures),
            "treasures": treasures,
            "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
            "exit_reason_pct_by_variant": exit_breakdown,
        }
        (output_dir / "p1_microstructure_1m_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        md = [
            "# Phase-P1 Next-Gen Microstructure (`L2 bookDepth` + `1m PremiumIndex` + `Hyperliquid L1`) & Convex Monetization Lab",
            "",
            f"- **Universe Scanned:** `{symbols_scanned}` Point-in-Time Perps (`{len(trades_df):,}` simulated 1m trades, `{summary['elapsed_sec']}s` runtime).",
            "- **Monetization Geometries Compared:** `MONO_A_CONVEX_TRAIL` (unchoked right-tail) vs `MONO_B_COUPON_VACUUM` (exact `:00:00` coupon bank + conditional L2 continuation) vs `MONO_C_SPLIT_50_50` (`50%` Maker TP @ `+1.8%` + `50%` risk-free runner @ `+9.0%`).",
            f"- **Strict Treasure Survivors (`>= +25 bps` Full, `>= +15 bps` in `23-24` & `2025`, `>= +20 bps` in `2026`, `Fee Share <= 30%`, True In-Fold CPCV `>= 80%`):** **`{len(treasures)}`**",
            "",
            "| Mechanism | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | `2025` | `2026` | In-Fold CPCV bps | CPCV`%` | BTC Corr | Treasure |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, r in lb_df.iterrows():
            md.append(
                f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
                f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
                f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['med_net_bps_full']:+.1f}` | `{r['win_rate_pct']:.1f}%` | "
                f"`{r['fee_share_of_gross_pct']:.1f}%` | `{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | "
                f"**`{r['net_bps_2026']:+.1f}`** | `{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
                f"`{r.get('corr_vs_btc', 0.0):+.2f}` | **`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
            )
        md.append("")
        md.append("## Mechanism Profiles")
        for m_k, pf in P1_MECHANISM_PROFILES.items():
            md.append(f"- **`{m_k}`** (`{pf.family}`): {pf.description}")
        (output_dir / "p1_microstructure_1m_report.md").write_text("\n".join(md) + "\n")

        print("P1_MICROSTRUCTURE_LAB_COMPLETE!", flush=True)
        return summary
    finally:
        if rclone_stage_dir is not None and rclone_stage_dir.exists():
            shutil.rmtree(rclone_stage_dir, ignore_errors=True)
            print("[*] Wiped /tmp/p1_rclone_stage -> 0 bytes local residue.", flush=True)


def run_p2_aggtrades_sweep_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict:
    """
    Executes Phase-P2 (`aggTrades` 1-Millisecond Book-Sweep & Anti-Spoofing Tomography) across all
    candidate triggers of `P6A_L2_ASK_AIR_POCKET_LONG`, `P2_C2A_BOOK_VACUUM_PAIN_LONG`, and `B2_C2A`:
      - When `colab_session` + `rclone_stream` are passed: pulls the compact `~295 KB` `p1_trades_ckpt_v1`
        via `rclone`, uploads it directly to a fresh CPU Colab VM (**zero `colab drivemount` needed!**),
        streams all `~450` daily `aggTrades` ZIPs from `data.binance.vision` inside Colab at `500 MB/s`,
        computes causal `1-ms` sweep features, downloads results, persists features back to Google Drive
        via `rclone`, stops Colab, and wipes `/tmp` (`0 bytes` local residue).
    """
    import gc
    import json
    import shutil
    import subprocess
    import time

    if colab_session:
        from src.cloud.colab_client import ColabClient

        stage_p2 = Path("/tmp/p2_colab_bridge")
        if stage_p2.exists():
            shutil.rmtree(stage_p2, ignore_errors=True)
        (stage_p2 / "p1_trades_ckpt_v1").mkdir(parents=True, exist_ok=True)
        (stage_p2 / "p2_aggtrades").mkdir(parents=True, exist_ok=True)

        client = ColabClient(session_name=colab_session)
        try:
            if rclone_stream:
                rb = rclone_stream.rstrip("/")
                print("[*] Pulling compact P1 trade checkpoints (~300 KB) via rclone...", flush=True)
                subprocess.run(
                    ["rclone", "copy", f"{rb}/p1_trades_ckpt_v1", str(stage_p2 / "p1_trades_ckpt_v1")],
                    check=True,
                )
                subprocess.run(
                    ["rclone", "copy", f"{rb}/p2_aggtrades", str(stage_p2 / "p2_aggtrades")],
                    check=False,
                )
            else:
                src_ck = m16_data_dir / "p1_trades_ckpt_v1"
                for fp in sorted(src_ck.glob("shard_*.parquet")):
                    shutil.copy2(fp, stage_p2 / "p1_trades_ckpt_v1" / fp.name)

            ck_files = sorted((stage_p2 / "p1_trades_ckpt_v1").glob("shard_*.parquet"))
            if not ck_files:
                raise FileNotFoundError("No P1 trade checkpoints found to run Phase-P2 aggTrades Sweep Lab.")
            merged_p1_path = stage_p2 / "p1_all_trades.parquet"
            pd.concat([pd.read_parquet(f) for f in ck_files], ignore_index=True).to_parquet(
                merged_p1_path, index=False, compression="zstd"
            )

            print(
                f"[*] Provisioning headless Colab CPU session '{colab_session}' (zero Drive mount needed)...",
                flush=True,
            )
            if not client.ensure_session(gpu=None):
                raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
            assert client.sync_codebase(REPO_ROOT), "Failed syncing codebase to Colab"
            client.exec_inline(
                "import os; os.makedirs('/content/p2_data/p1_trades_ckpt_v1', exist_ok=True); "
                "os.makedirs('/content/p2_data/p2_aggtrades', exist_ok=True); "
                "os.makedirs('/content/reports', exist_ok=True)",
                timeout=90,
            )
            assert client.upload(merged_p1_path, "/content/p2_data/p1_trades_ckpt_v1/shard_00.parquet")
            cached_feat = stage_p2 / "p2_aggtrades" / "p2_aggtrades_event_features.parquet"
            if cached_feat.exists():
                client.upload(cached_feat, "/content/p2_data/p2_aggtrades/p2_aggtrades_event_features.parquet")

            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/p2_sweep.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                "'--suite', 'p2-aggtrades-sweep', '--m16-data-dir', '/content/p2_data', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/p2_sweep.pid', 'w').write(str(p.pid)); "
                "print(f'P2_SWEEP_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=90)
            print(f"[+] Launched remote Phase-P2 aggTrades Sweep Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(10.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/p2_sweep.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/p2_sweep.log').read().splitlines() "
                    "if os.path.exists('/content/p2_sweep.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=90)
                if rc_p != 0:
                    consec_err += 1
                    if consec_err >= 4:
                        raise RuntimeError(f"Colab session lost during P2 Sweep Lab: {(err_p or out_p)[:160]}")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "P2_AGGTRADES_SWEEP_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Phase-P2 aggTrades Sweep Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "p2_aggtrades_sweep_leaderboard.csv",
                "p2_aggtrades_sweep_summary.json",
                "p2_aggtrades_sweep_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)

            if rclone_stream:
                rb = rclone_stream.rstrip("/")
                for p2_f in ("p2_aggtrades_event_features.parquet", "p2_all_variants_trades.parquet"):
                    loc_f = stage_p2 / "p2_aggtrades" / p2_f
                    if client.download(f"/content/p2_data/p2_aggtrades/{p2_f}", loc_f) and loc_f.exists():
                        subprocess.run(["rclone", "copy", str(loc_f), f"{rb}/p2_aggtrades"], check=False)
                print("[+] Synced Phase-P2 aggTrades sweep features & survivor trades to Google Drive!", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            if stage_p2.exists():
                shutil.rmtree(stage_p2, ignore_errors=True)
                print("[*] Wiped /tmp/p2_colab_bridge -> 0 bytes local residue.", flush=True)

    from src.features.tick_sweep_physics import (
        enrich_events_with_aggtrades_sweeps,
        evaluate_p2_tick_sweep_ablation,
    )

    t0 = time.time()
    ck_dir = m16_data_dir / "p1_trades_ckpt_v1"
    ck_files = sorted(ck_dir.glob("shard_*.parquet"))
    if not ck_files:
        raise FileNotFoundError(f"No P1 trade checkpoints in {ck_dir}")

    p1_df = pd.concat([pd.read_parquet(f) for f in ck_files], ignore_index=True)
    target_mechs = ("P6A_L2_ASK_AIR_POCKET_LONG", "P2_C2A_BOOK_VACUUM_PAIN_LONG", "B2_C2A_BASE_UNCHOKED_LONG")
    cand_sub = p1_df[p1_df["mech_id"].isin(target_mechs)]
    print(
        f"🔬 Phase-P2 aggTrades 1ms Sweep Lab | {len(cand_sub):,} candidate trade rows "
        f"({cand_sub[['symbol', 'signal_ts']].drop_duplicates().shape[0]} unique causal signals)...",
        flush=True,
    )

    p2_out_dir = m16_data_dir / "p2_aggtrades"
    p2_out_dir.mkdir(parents=True, exist_ok=True)
    feat_cache = p2_out_dir / "p2_aggtrades_event_features.parquet"

    if feat_cache.exists() and feat_cache.stat().st_size > 512:
        existing_feats = pd.read_parquet(feat_cache)
        done_keys = set(
            zip(existing_feats["symbol"].astype(str), existing_feats["signal_ts"].astype(int), strict=False)
        )
        missing_mask = [
            (str(s), int(ts_v)) not in done_keys
            for s, ts_v in zip(cand_sub["symbol"], cand_sub["signal_ts"], strict=False)
        ]
        missing_df = cand_sub.loc[missing_mask]
        if not missing_df.empty:
            new_feats = enrich_events_with_aggtrades_sweeps(missing_df, workers=8)
            feats_df = pd.concat([existing_feats, new_feats], ignore_index=True).drop_duplicates(
                subset=["symbol", "signal_ts"], keep="last"
            )
            feats_df.to_parquet(feat_cache, index=False, compression="zstd")
        else:
            feats_df = existing_feats
    else:
        feats_df = enrich_events_with_aggtrades_sweeps(cand_sub, workers=8)
        feats_df.to_parquet(feat_cache, index=False, compression="zstd")

    cov_pct = float((feats_df["has_agg"] > 0.5).mean() * 100.0) if not feats_df.empty else 0.0
    print(
        f"  • Extracted 1ms sweep telemetry across {len(feats_df)} signals (aggTrades coverage={cov_pct:.1f}%)",
        flush=True,
    )

    lb_df, variants_df = evaluate_p2_tick_sweep_ablation(p1_df, feats_df)
    variants_df.to_parquet(p2_out_dir / "p2_all_variants_trades.parquet", index=False, compression="zstd")
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "p2_aggtrades_sweep_leaderboard.csv", index=False)

    treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
    summary = {
        "benchmark": "p2_aggtrades_1ms_sweep_tomography",
        "elapsed_sec": round(time.time() - t0, 1),
        "unique_signals_analyzed": int(len(feats_df)),
        "aggtrades_coverage_pct": round(cov_pct, 1),
        "treasure_count": len(treasures),
        "treasures": treasures,
        "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
    }
    (output_dir / "p2_aggtrades_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Phase-P2 `aggTrades` 1-Millisecond Book-Sweep & Anti-Spoofing Tomography (`2023–2026`)",
        "",
        f"- **Unique Causal Signals Analyzed:** `{len(feats_df)}` (`aggTrades` tick-slice coverage: `{cov_pct:.1f}%`, `{summary['elapsed_sec']}s` runtime).",
        "- **Causality Guarantee:** `transact_time_ms <= signal_ts * 1000 - 1` (`0.000 ms` lookahead into entry bar!).",
        f"- **Strict 4-Gate Universal Treasure Survivors (`>= +25 bps` Full, `>= +15 bps` in `23-24` & `2025`, `>= +20 bps` in `2026`, `Fee Share <= 30%`, True In-Fold CPCV `>= 80%`):** **`{len(treasures)}`**",
        "",
        "| Variant | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** | Treasure |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in lb_df.iterrows():
        md.append(
            f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
            f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
            f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['med_net_bps_full']:+.1f}` | `{r['win_rate_pct']:.1f}%` | "
            f"`{r['fee_share_of_gross_pct']:.1f}%` | `{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | "
            f"**`{r['net_bps_2026']:+.1f}`** | `{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
            f"**`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
        )
    (output_dir / "p2_aggtrades_sweep_report.md").write_text("\n".join(md) + "\n")

    print("P2_AGGTRADES_SWEEP_COMPLETE!", flush=True)
    return summary


def run_p2_unified_portfolio_1m_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict[str, object]:
    """
    Executes Step 2: Unified `1-Minute` One-Way Production Portfolio Simulation (`2023–2026` +
    Fresh `$2,700` `2026-01-01` Cushion Stress-Test against the `$2,500` CPPI Hard Floor),
    measuring the exact impact of integrating Phase-P1/P2 L2-Book + 1ms Tick-Sweep survivors
    alongside `Core P1 + Manga P2 + M09/M23/M04` (`M25=OFF`).
    """
    import gc
    import json
    import shutil
    import subprocess
    import time

    if colab_session:
        from src.backtest.p2_portfolio_integration import PROD_TOP_15
        from src.cloud.colab_client import ColabClient
        from src.strategies.event_snipers import SNIPER_UNIVERSE_82

        stage_port = Path("/tmp/p2_port_bridge")
        if stage_port.exists():
            shutil.rmtree(stage_port, ignore_errors=True)
        raw_stage = stage_port / "raw"
        (raw_stage / "bars_1h").mkdir(parents=True, exist_ok=True)
        (raw_stage / "funding").mkdir(parents=True, exist_ok=True)
        (raw_stage / "p2_aggtrades").mkdir(parents=True, exist_ok=True)

        client = ColabClient(session_name=colab_session)
        try:
            if not rclone_stream:
                raise ValueError("--rclone-stream is required for zero-click Colab portfolio bridge.")
            rb = rclone_stream.rstrip("/")
            print("[*] Staging compact 1h + funding + P2 trades via rclone (auto-cleaned on exit)...", flush=True)
            for sub_p in ("bars_1h", "funding"):
                subprocess.run(
                    [
                        "rclone",
                        "copy",
                        "--multi-thread-streams=8",
                        f"{rb}/{sub_p}",
                        str(raw_stage / sub_p),
                    ],
                    check=True,
                )
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    f"{rb}/p2_aggtrades/p2_all_variants_trades.parquet",
                    str(raw_stage / "p2_aggtrades"),
                ],
                check=True,
            )

            keep_syms = set(PROD_TOP_15) | set(SNIPER_UNIVERSE_82)
            b1h_cols = [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_quote_volume",
            ]
            b1h_parts = []
            for bf in sorted((raw_stage / "bars_1h").glob("shard_*.parquet")):
                df_b = pd.read_parquet(bf, columns=b1h_cols)
                sub_b = df_b[df_b["symbol"].isin(keep_syms)]
                if not sub_b.empty:
                    b1h_parts.append(sub_b)
            b1h_all = pd.concat(b1h_parts, ignore_index=True)

            fnd_parts = []
            for ff in sorted((raw_stage / "funding").glob("shard_*.parquet")):
                df_f = pd.read_parquet(ff, columns=["timestamp", "symbol", "funding_rate"])
                sub_f = df_f[df_f["symbol"].isin(keep_syms)]
                if not sub_f.empty:
                    fnd_parts.append(sub_f)
            fnd_all = pd.concat(fnd_parts, ignore_index=True)

            slim_dir = stage_port / "slim"
            (slim_dir / "bars_1h").mkdir(parents=True, exist_ok=True)
            (slim_dir / "funding").mkdir(parents=True, exist_ok=True)
            sorted_syms = sorted(b1h_all["symbol"].unique().tolist())
            chunk_sz = max(1, (len(sorted_syms) + 3) // 4)
            slim_b1h_paths: list[Path] = []
            for idx_c in range(0, len(sorted_syms), chunk_sz):
                s_slice = set(sorted_syms[idx_c : idx_c + chunk_sz])
                out_p = slim_dir / "bars_1h" / f"shard_{idx_c // chunk_sz:02d}.parquet"
                b1h_all[b1h_all["symbol"].isin(s_slice)].to_parquet(out_p, index=False, compression="zstd")
                slim_b1h_paths.append(out_p)
            slim_fnd = slim_dir / "funding" / "shard_00.parquet"
            fnd_all.to_parquet(slim_fnd, index=False, compression="zstd")

            print(
                f"[*] Provisioning headless Colab CPU VM '{colab_session}' (zero Drive mount needed)...",
                flush=True,
            )
            if not client.ensure_session(gpu=None):
                raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
            assert client.sync_codebase(REPO_ROOT), "Failed syncing codebase to Colab"
            client.exec_inline(
                "import os; "
                "os.makedirs('/content/p2_port_data/bars_1h', exist_ok=True); "
                "os.makedirs('/content/p2_port_data/funding', exist_ok=True); "
                "os.makedirs('/content/p2_port_data/p2_aggtrades', exist_ok=True); "
                "os.makedirs('/content/reports', exist_ok=True)",
                timeout=90,
            )
            for sp_b in slim_b1h_paths:
                assert client.upload(sp_b, f"/content/p2_port_data/bars_1h/{sp_b.name}")
            assert client.upload(slim_fnd, "/content/p2_port_data/funding/shard_00.parquet")
            assert client.upload(
                raw_stage / "p2_aggtrades" / "p2_all_variants_trades.parquet",
                "/content/p2_port_data/p2_aggtrades/p2_all_variants_trades.parquet",
            )

            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/p2_port.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                "'--suite', 'p2-unified-portfolio-1m', '--m16-data-dir', '/content/p2_port_data', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/p2_port.pid', 'w').write(str(p.pid)); "
                "print(f'P2_PORT_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=90)
            print(f"[+] Launched remote Unified 1m Portfolio Sim ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(10.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/p2_port.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/p2_port.log').read().splitlines() "
                    "if os.path.exists('/content/p2_port.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=90)
                if rc_p != 0:
                    consec_err += 1
                    if consec_err >= 4:
                        raise RuntimeError(f"Colab session lost during Portfolio Sim: {(err_p or out_p)[:160]}")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "P2_UNIFIED_PORTFOLIO_1M_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Unified 1m Portfolio Sim terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "p2_unified_portfolio_1m_leaderboard.csv",
                "p2_unified_portfolio_1m_summary.json",
                "p2_unified_portfolio_1m_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            if stage_port.exists():
                shutil.rmtree(stage_port, ignore_errors=True)
                print("[*] Wiped /tmp/p2_port_bridge -> 0 bytes local residue.", flush=True)

    from datetime import date, timedelta

    import pyarrow.parquet as pq

    from src.backtest.p2_portfolio_integration import (
        build_core_1m_tensors_and_base_snipers,
        identify_required_1m_symbols_for_portfolio,
        run_p2_portfolio_ablation_1m,
    )
    from src.data.multimodal_vision_ingestor import (
        PERP_TICKER_ALIASES,
        _make_pooled_session,
        fetch_gap_free_1m_klines,
    )

    t0 = time.time()
    b1h_files = sorted((m16_data_dir / "bars_1h").glob("shard_*.parquet"))
    fnd_files = sorted((m16_data_dir / "funding").glob("shard_*.parquet"))
    p2_trades_path = m16_data_dir / "p2_aggtrades" / "p2_all_variants_trades.parquet"
    if not b1h_files or not p2_trades_path.exists():
        raise FileNotFoundError(f"Missing bars_1h or {p2_trades_path} in {m16_data_dir}")

    bars_1h_by_sym: dict[str, pd.DataFrame] = {}
    for bf in b1h_files:
        bdf = pd.read_parquet(bf)
        for s_k, g_df in bdf.groupby("symbol", sort=False):
            bars_1h_by_sym[str(s_k)] = g_df.reset_index(drop=True)

    funding_by_sym: dict[str, pd.DataFrame] = {}
    for ff in fnd_files:
        fdf = pd.read_parquet(ff)
        for s_k, g_df in fdf.groupby("symbol", sort=False):
            funding_by_sym[str(s_k)] = g_df.reset_index(drop=True)

    needed_1m_syms = identify_required_1m_symbols_for_portfolio(
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=funding_by_sym,
        eval_start="2023-03-01",
    )
    print(
        f"🏦 Unified 1m Portfolio Sim | Loaded {len(bars_1h_by_sym)} 1h perps; "
        f"exact required 1m symbols (Core15 + M09 + M23 + M04): {len(needed_1m_syms)}",
        flush=True,
    )

    b1m_files = sorted((m16_data_dir / "bars_1m").glob("shard_*.parquet"))
    req_cols_1m = ["timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    if b1m_files:

        def _load_sym_1m_from_shards(sym_c: str) -> pd.DataFrame | None:
            parts = []
            for mf in b1m_files:
                tbl = pq.read_table(mf, columns=req_cols_1m, filters=[("symbol", "==", sym_c)])
                if tbl.num_rows > 0:
                    parts.append(tbl.to_pandas())
            if not parts:
                return None
            return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        sym_loader = _load_sym_1m_from_shards
        n_workers_1m = 2
    else:
        btc_max_ts = int(bars_1h_by_sym["BTCUSDT"]["timestamp"].max())
        end_dt = pd.Timestamp(btc_max_ts, unit="s", tz="UTC") + timedelta(days=1)
        all_yms = [
            (y, m) for y in range(2023, end_dt.year + 1) for m in range(1, 13) if (y, m) < (end_dt.year, end_dt.month)
        ]
        recent_days = [date(end_dt.year, end_dt.month, d).isoformat() for d in range(1, end_dt.day + 1)]
        sess = _make_pooled_session(64)

        def _stream_sym_1m_vision(sym_c: str) -> pd.DataFrame | None:
            tks = PERP_TICKER_ALIASES.get(sym_c, (sym_c,))
            df_c = fetch_gap_free_1m_klines("futures/um", tks, sym_c, all_yms, recent_days, session=sess)
            if df_c is not None and not df_c.empty:
                df_c = df_c[df_c["timestamp"] <= btc_max_ts + 3600][req_cols_1m].reset_index(drop=True)
            return df_c

        sym_loader = _stream_sym_1m_vision
        n_workers_1m = 2

    print("  • Streaming Core P1 7-Factor + Manga P2 Dollar-Bar + M09/M23/M04 1m bundle (O(1) RAM)...", flush=True)
    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=funding_by_sym,
        eval_start="2023-03-01",
        symbol_1m_loader=sym_loader,
        symbols_to_stream=needed_1m_syms,
        loader_workers=n_workers_1m,
    )
    print(
        f"  • Resolved {len(bundle['base_sniper_events'])} baseline live sniper triggers (M09/M23/M04) on 1m bars.",
        flush=True,
    )

    p2_all = pd.read_parquet(p2_trades_path)

    def _slice_cand(mech: str, mode: str) -> pd.DataFrame:
        return p2_all[(p2_all["mech_id"] == mech) & (p2_all["exec_mode"] == mode)].copy()

    cand_sleeves: dict[str, pd.DataFrame] = {
        "P6A_RAW_MONO_A": _slice_cand("P6A_L2_0_RAW_CONTROL", "MONO_A_CONVEX_TRAIL"),
        "P6A_RAW_MONO_C_50_50": _slice_cand("P6A_L2_0_RAW_CONTROL", "MONO_C_SPLIT_50_50"),
        "P6A_INST_SWEEP_MONO_A": _slice_cand("P6A_L2_3_INST_SWEEP_CONC", "MONO_A_CONVEX_TRAIL"),
        "P6A_INST_SWEEP_MONO_B": _slice_cand("P6A_L2_3_INST_SWEEP_CONC", "MONO_B_COUPON_VACUUM"),
        "P6A_INST_SWEEP_MONO_C": _slice_cand("P6A_L2_3_INST_SWEEP_CONC", "MONO_C_SPLIT_50_50"),
        "P6A_KYLE_1MS_MONO_A": _slice_cand("P6A_L2_2_KYLE_1MS_JUMP", "MONO_A_CONVEX_TRAIL"),
        "UNIFIED_L2_INST_CONC_MONO_A": _slice_cand("UNIFIED_L2_VACUUM_INST_CONC", "MONO_A_CONVEX_TRAIL"),
    }
    extra_cfgs = [
        ("PROD + P6A_INST_SWEEP_MONO_A (4-Slots)", "P6A_INST_SWEEP_MONO_A", 4, 0.24),
        ("PROD + P6A_INST_SWEEP_MONO_A (Lev=0.30)", "P6A_INST_SWEEP_MONO_A", 3, 0.30),
    ]

    rows = run_p2_portfolio_ablation_1m(
        bundle=bundle,
        candidate_sleeves=cand_sleeves,
        initial_capital=2700.0,
        sniper_slot_lev=0.24,
        sniper_max_pos=3,
        extra_slot_configs=extra_cfgs,
    )
    gc.collect()

    lb_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "p2_unified_portfolio_1m_leaderboard.csv", index=False)

    summary = {
        "benchmark": "p2_unified_portfolio_1m_ablation",
        "elapsed_sec": round(time.time() - t0, 1),
        "required_1m_symbols_count": len(needed_1m_syms),
        "base_sniper_triggers_1m": len(bundle["base_sniper_events"]),
        "results": rows,
    }
    (output_dir / "p2_unified_portfolio_1m_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Unified `1-Minute` One-Way Production Portfolio Ablation (`2023–2026` + `$2,700` `2026` Cushion Stress-Test)",
        "",
        f"- **Exact 1-Minute Production Stack:** `Core P1` (`0.85x` Top-15 + `1m` Chop Lock) + `Manga P2` (`1m` Dollar-Bar TSMOM) + `M09/M23/M04` (`M25=OFF`) + Candidate L2/Tick-Sweep Sleeve (`{summary['elapsed_sec']}s` runtime).",
        "- **Account Physics:** One-Way non-collision (`Core` vs `Sniper` symbol lock), `$2,700.00` Two-Phase SafeRamp (`2.5x -> 3.2x -> 5.0x`), `$2,500.00` Cushion-Safe CPPI Hard Floor.",
        "",
        "| Portfolio Configuration | `4Y Sharpe` (`Δ`) | `4Y CAGR` | `4Y MaxDD` | `SR 23-24 / 25 / 26` | `Exec / Cand` | **`2026` `$2,700 -> Final` (`Δ$`)** | **`2026` MinEq (`>= $2,500`)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        fl_ok = "✅" if r["floor_2500_intact_2026"] else "⚠️"
        md.append(
            f"| `{r['variant']}` | **`{r['full_sharpe']:.2f}`** (`{r['sharpe_lift_full']:+.2f}`) | "
            f"`{r['full_cagr_pct']:+.1f}%` | `{r['full_max_dd_pct']:.2f}%` | "
            f"`{r['sharpe_23_24']:.2f} / {r['sharpe_2025']:.2f} / {r['sharpe_2026']:.2f}` | "
            f"`{r['new_sleeve_trades_executed']} / {r['new_sleeve_candidates_total']}` | "
            f"**`${r['eq_2026_from_2700_usd']:,.2f}`** (`{r['eq_2026_delta_usd']:+.2f}`) | "
            f"`${r['min_eq_2026_from_2700_usd']:,.2f}` {fl_ok} |"
        )
    (output_dir / "p2_unified_portfolio_1m_report.md").write_text("\n".join(md) + "\n")

    print("P2_UNIFIED_PORTFOLIO_1M_COMPLETE!", flush=True)
    return summary


def run_p3_p4_cross_venue_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict[str, object]:
    """
    Executes Phase P3/P4: Cross-Exchange (`Bybit V5` / `OKX Swap`) & Full-Universe On-Chain
    (`Hyperliquid L1` ~140+ coins) Dispersion, Oracle-Lead & Solvency-Anchored Liquidation Lab
    across `2023–2026` on exact `1-minute` linear bars (`O(1)` streaming RAM, `0` OAuth prompts,
    `0 bytes` local residue).
    """
    import gc
    import json
    import shutil
    import subprocess
    import time

    if colab_session:
        from src.cloud.colab_client import ColabClient

        stage_p3 = Path("/tmp/p3_p4_bridge")
        if stage_p3.exists():
            shutil.rmtree(stage_p3, ignore_errors=True)
        raw_p3 = stage_p3 / "raw"
        for sub_d in ("bars_1h", "funding", "metrics_5m", "cross_venue"):
            (raw_p3 / sub_d).mkdir(parents=True, exist_ok=True)

        client = ColabClient(session_name=colab_session)
        try:
            if not rclone_stream:
                raise ValueError("--rclone-stream is required for zero-click Colab P3/P4 bridge.")
            rb = rclone_stream.rstrip("/")
            print("[*] Staging compact 1h + funding + metrics_5m + cross_venue via rclone...", flush=True)
            for sub_d in ("bars_1h", "funding", "metrics_5m", "cross_venue"):
                subprocess.run(
                    ["rclone", "copy", "--multi-thread-streams=8", f"{rb}/{sub_d}", str(raw_p3 / sub_d)],
                    check=(sub_d in ("bars_1h", "funding")),
                )

            # Exact mathematical filter: symbols that EVER reached 7d ADV >= $15M (`liq_gate.any() == True`)
            liq_syms: set[str] = set()
            max_1h_ts = 0
            for bf in sorted((raw_p3 / "bars_1h").glob("shard_*.parquet")):
                df_b = pd.read_parquet(bf, columns=["timestamp", "symbol", "quote_volume"])
                if not df_b.empty:
                    max_1h_ts = max(max_1h_ts, int(df_b["timestamp"].max()))
                    for sym_k, grp in df_b.groupby("symbol", observed=True):
                        q_s = grp.sort_values("timestamp")["quote_volume"].to_numpy(dtype=float)
                        if len(q_s) >= 24 * 14:
                            adv7 = float(pd.Series(q_s).rolling(168, min_periods=48).sum().max() or 0.0) / 7.0
                            if adv7 >= 15_000_000.0:
                                liq_syms.add(str(sym_k))

            slim_p3 = stage_p3 / "slim"
            for sub_d in ("funding_by_sym", "metrics_by_sym", "cross_venue"):
                (slim_p3 / sub_d).mkdir(parents=True, exist_ok=True)
            (slim_p3 / "meta.json").write_text(
                json.dumps({"liq_symbols": sorted(liq_syms), "max_1h_ts": max_1h_ts}, indent=2) + "\n"
            )

            for cv_f in sorted((raw_p3 / "cross_venue").glob("*.parquet")):
                if cv_f.stat().st_size > 1024:
                    shutil.copy2(cv_f, slim_p3 / "cross_venue" / cv_f.name)

            # Pre-ingest full 2023-2026 Bybit V5 / OKX + Hyperliquid L1 before Colab (avoids US Colab Bybit geo-block!)
            from src.data.cross_venue_p3_ingestor import (
                ensure_cex_peer_funding_parquet as _ensure_peer,
                ensure_full_hyperliquid_onchain_parquet as _ensure_hl,
            )

            hl_pre = _ensure_hl(slim_p3, sorted(liq_syms), workers=6)
            peer_pre = _ensure_peer(slim_p3, sorted(liq_syms), workers=12)
            for cv_sync_p in (hl_pre, peer_pre):
                if cv_sync_p.exists() and cv_sync_p.stat().st_size > 1024:
                    subprocess.run(["rclone", "copy", str(cv_sync_p), f"{rb}/cross_venue"], check=False)

            f_parts = [
                pd.read_parquet(fp)[lambda d: d["symbol"].isin(liq_syms)]
                for fp in sorted((raw_p3 / "funding").glob("shard_*.parquet"))
            ]
            f_all = pd.concat([x for x in f_parts if not x.empty], ignore_index=True)
            for sym_f, grp_f in f_all.groupby("symbol", observed=True):
                grp_f.to_parquet(slim_p3 / "funding_by_sym" / f"{sym_f}.parquet", index=False, compression="zstd")
            del f_all, f_parts

            m_parts = [
                pd.read_parquet(mp, columns=["timestamp", "symbol", "sum_open_interest"])[
                    lambda d: d["symbol"].isin(liq_syms)
                ]
                for mp in sorted((raw_p3 / "metrics_5m").glob("shard_*.parquet"))
            ]
            m_all = pd.concat([x for x in m_parts if not x.empty], ignore_index=True)
            for sym_m, grp_m in m_all.groupby("symbol", observed=True):
                grp_m[["timestamp", "sum_open_interest"]].to_parquet(
                    slim_p3 / "metrics_by_sym" / f"{sym_m}.parquet",
                    index=False,
                    compression="zstd",
                )
            del m_all, m_parts
            gc.collect()

            import tarfile

            tar_p3 = stage_p3 / "p3_data_bundle.tar"
            with tarfile.open(tar_p3, "w") as tf:
                for p_item in sorted(slim_p3.rglob("*")):
                    if p_item.is_file():
                        tf.add(p_item, arcname=str(p_item.relative_to(slim_p3)))
            bundle_mb = tar_p3.stat().st_size / 1e6

            print(
                f"[*] Provisioning headless Colab CPU VM '{colab_session}' "
                f"({len(liq_syms)} liquid perps, {bundle_mb:.1f} MB O(1)-RAM bundle, 0 OAuth prompts)...",
                flush=True,
            )
            if not client.ensure_session(gpu=None):
                raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
            assert client.sync_codebase(REPO_ROOT), "Failed syncing codebase to Colab"
            assert client.upload(tar_p3, "/content/_p3_data_bundle.tar"), "Failed uploading P3 slim bundle to Colab"
            rc_u, out_u, err_u = client.exec_inline(
                "import os, tarfile; "
                "[os.makedirs(f'/content/p3_data/{d}', exist_ok=True) "
                "for d in ('funding_by_sym', 'metrics_by_sym', 'cross_venue', 'book_depth_events', 'p3_p4_out')]; "
                "os.makedirs('/content/reports', exist_ok=True); "
                "tarfile.open('/content/_p3_data_bundle.tar', 'r').extractall('/content/p3_data'); "
                "os.remove('/content/_p3_data_bundle.tar'); "
                "print('P3_BUNDLE_READY')",
                timeout=180,
            )
            if rc_u != 0 or "P3_BUNDLE_READY" not in out_u:
                raise RuntimeError(f"Failed unpacking P3 bundle on Colab: {(err_u or out_u)[:200]}")

            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/p3_p4.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                "'--suite', 'p3-p4-cross-venue-lab', '--m16-data-dir', '/content/p3_data', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/p3_p4.pid', 'w').write(str(p.pid)); "
                "print(f'P3_P4_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=90)
            print(f"[+] Launched remote Phase P3/P4 Cross-Venue Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(10.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/p3_p4.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/p3_p4.log').read().splitlines() "
                    "if os.path.exists('/content/p3_p4.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=90)
                if rc_p != 0:
                    consec_err += 1
                    if consec_err >= 4:
                        raise RuntimeError(f"Colab session lost during P3/P4 Lab: {(err_p or out_p)[:160]}")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "P3_P4_CROSS_VENUE_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Phase P3/P4 Cross-Venue Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "p3_p4_cross_venue_leaderboard.csv",
                "p3_p4_cross_venue_summary.json",
                "p3_p4_cross_venue_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)

            for cv_sync in ("hl_funding_all_v2.parquet", "cex_peer_funding_1h.parquet"):
                loc_cv = stage_p3 / cv_sync
                if client.download(f"/content/p3_data/cross_venue/{cv_sync}", loc_cv) and loc_cv.exists():
                    subprocess.run(["rclone", "copy", str(loc_cv), f"{rb}/cross_venue"], check=False)
            loc_tr = stage_p3 / "p3_p4_all_trades.parquet"
            if client.download("/content/p3_data/p3_p4_out/p3_p4_all_trades.parquet", loc_tr) and loc_tr.exists():
                subprocess.run(["rclone", "copy", str(loc_tr), f"{rb}/p3_p4_cross_venue"], check=False)
            print("[+] Synced Phase P3/P4 cross-venue datasets & simulated trades to Google Drive!", flush=True)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            if stage_p3.exists():
                shutil.rmtree(stage_p3, ignore_errors=True)
                print("[*] Wiped /tmp/p3_p4_bridge -> 0 bytes local residue.", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import date, datetime, timedelta, timezone

    from src.data.cross_venue_p3_ingestor import (
        ensure_cex_peer_funding_parquet,
        ensure_full_hyperliquid_onchain_parquet,
    )
    from src.data.microstructure_p1_ingestor import (
        _fetch_single_premium_archive,
        fetch_shard_event_book_depth,
    )
    from src.data.multimodal_vision_ingestor import (
        PERP_TICKER_ALIASES,
        _make_pooled_session,
        fetch_gap_free_1m_klines,
    )
    from src.models.cross_venue_p3_p4_lab import (
        evaluate_p3_p4_leaderboard,
        evaluate_symbol_p3_p4_mechanisms,
        prepare_symbol_p3_p4_state,
    )

    t0 = time.time()

    if rclone_stream and not colab_session:
        rb = rclone_stream.rstrip("/")
        stream_dir = Path("/tmp/p3_p4_rclone_stream")
        if stream_dir.exists():
            shutil.rmtree(stream_dir, ignore_errors=True)
        for sub_d in (
            "bars_1m",
            "premium_1m",
            "metrics_5m",
            "funding",
            "book_depth_events",
            "cross_venue",
            "p3_p4_trades_ckpt_v1",
            "bars_1h",
        ):
            (stream_dir / sub_d).mkdir(parents=True, exist_ok=True)
        try:
            print(
                "[*] Streaming Phase P3/P4 shard-by-shard from Google Drive (O(1) RAM/disk, 0 local bytes on exit)...",
                flush=True,
            )
            for sync_sub in ("cross_venue", "p3_p4_trades_ckpt_v1", "bars_1h"):
                subprocess.run(
                    ["rclone", "copy", "--multi-thread-streams=8", f"{rb}/{sync_sub}", str(stream_dir / sync_sub)],
                    check=(sync_sub == "bars_1h"),
                )

            liq_syms_st: set[str] = set()
            for bf in sorted((stream_dir / "bars_1h").glob("shard_*.parquet")):
                df_b = pd.read_parquet(bf, columns=["timestamp", "symbol", "quote_volume"])
                for sym_k, grp in df_b.groupby("symbol", observed=True):
                    q_s = grp.sort_values("timestamp")["quote_volume"].to_numpy(dtype=float)
                    if len(q_s) >= 24 * 14:
                        adv7 = float(pd.Series(q_s).rolling(168, min_periods=48).sum().max() or 0.0) / 7.0
                        if adv7 >= 15_000_000.0:
                            liq_syms_st.add(str(sym_k))
            shutil.rmtree(stream_dir / "bars_1h", ignore_errors=True)

            cand_syms = sorted(liq_syms_st)
            hl_path = ensure_full_hyperliquid_onchain_parquet(stream_dir, cand_syms, workers=6)
            peer_path = ensure_cex_peer_funding_parquet(stream_dir, cand_syms, workers=12)
            for cv_p in (hl_path, peer_path):
                if cv_p.exists() and cv_p.stat().st_size > 1024:
                    subprocess.run(["rclone", "copy", str(cv_p), f"{rb}/cross_venue"], check=False)

            hl_df = pd.read_parquet(hl_path)
            peer_df = pd.read_parquet(peer_path)
            hl_by_sym = (
                {str(k): g.reset_index(drop=True) for k, g in hl_df.groupby("symbol", observed=True)}
                if not hl_df.empty
                else {}
            )
            peer_by_sym = (
                {str(k): g.reset_index(drop=True) for k, g in peer_df.groupby("symbol", observed=True)}
                if not peer_df.empty
                else {}
            )
            del hl_df, peer_df
            gc.collect()

            eval_set = {s for s in cand_syms if (s in hl_by_sym or s in peer_by_sym)}
            eval_syms = sorted(eval_set)
            print(
                f"🌐 Phase P3/P4 Cross-Venue Lab | Streaming 20 pre-built 1m shards across {len(eval_syms)} "
                f"liquid cross-venue perps (HL={len(hl_by_sym)}, PeerCEX={len(peer_by_sym)})...",
                flush=True,
            )

            perp_cols = [
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "quote_volume",
                "taker_buy_quote_volume",
            ]
            shard_dfs: list[pd.DataFrame] = []
            ckpt_dir = stream_dir / "p3_p4_trades_ckpt_v1"

            for idx_s in range(20):
                t_sh = time.time()
                sh_name = f"shard_{idx_s:02d}.parquet"
                sh_ckpt = ckpt_dir / sh_name
                if sh_ckpt.exists() and sh_ckpt.stat().st_size > 128:
                    c_df = pd.read_parquet(sh_ckpt)
                    if not c_df.empty and "mech_id" in c_df.columns:
                        shard_dfs.append(c_df)
                    print(
                        f"  [{idx_s + 1:02d}/20] {sh_name}: REUSED +{len(c_df):,} cached P3/P4 trades ({time.time() - t_sh:.1f}s)",
                        flush=True,
                    )
                    continue

                for mod_d in ("bars_1m", "premium_1m", "metrics_5m", "funding", "book_depth_events"):
                    subprocess.run(
                        [
                            "rclone",
                            "copy",
                            "--multi-thread-streams=8",
                            "--multi-thread-cutoff=16M",
                            f"{rb}/{mod_d}/{sh_name}",
                            str(stream_dir / mod_d),
                        ],
                        check=(mod_d == "bars_1m"),
                    )

                b1m = pd.read_parquet(stream_dir / "bars_1m" / sh_name, columns=perp_cols)
                perp_map = {str(k): g for k, g in b1m.groupby("symbol", observed=True) if str(k) in eval_set}
                del b1m

                pr_p = stream_dir / "premium_1m" / sh_name
                pr_map = (
                    {str(k): g for k, g in pd.read_parquet(pr_p).groupby("symbol", observed=True)}
                    if pr_p.exists()
                    else {}
                )
                mt_p = stream_dir / "metrics_5m" / sh_name
                mt_map = (
                    {
                        str(k): g
                        for k, g in pd.read_parquet(mt_p, columns=["symbol", "timestamp", "sum_open_interest"]).groupby(
                            "symbol", observed=True
                        )
                    }
                    if mt_p.exists()
                    else {}
                )
                fn_p = stream_dir / "funding" / sh_name
                fn_map = (
                    {str(k): g for k, g in pd.read_parquet(fn_p).groupby("symbol", observed=True)}
                    if fn_p.exists()
                    else {}
                )
                bk_p = stream_dir / "book_depth_events" / sh_name
                bk_map = (
                    {str(k): g for k, g in pd.read_parquet(bk_p).groupby("symbol", observed=True)}
                    if bk_p.exists() and bk_p.stat().st_size > 512
                    else {}
                )

                for mod_d in ("bars_1m", "premium_1m", "metrics_5m", "funding", "book_depth_events"):
                    (stream_dir / mod_d / sh_name).unlink(missing_ok=True)

                sh_recs: list[dict[str, object]] = []
                for sym_s, p_df in perp_map.items():
                    st = prepare_symbol_p3_p4_state(
                        symbol=sym_s,
                        perp_1m=p_df,
                        prem_1m=pr_map.get(sym_s),
                        metrics_5m=mt_map.get(sym_s),
                        funding_df=fn_map.get(sym_s),
                        hl_funding_df=hl_by_sym.get(sym_s),
                        peer_cex_df=peer_by_sym.get(sym_s),
                    )
                    if st is None:
                        continue
                    bk_sym = bk_map.get(sym_s)
                    need_d = set(st["needed_dates"])
                    if bk_sym is not None and not bk_sym.empty:
                        have_d = {
                            datetime.fromtimestamp(int(x), tz=timezone.utc).strftime("%Y-%m-%d")
                            for x in bk_sym["timestamp"].iloc[::20].to_numpy(dtype=np.int64)
                        }
                        miss_d = need_d - have_d
                    else:
                        miss_d = need_d
                    if miss_d:
                        extra_bk = fetch_shard_event_book_depth(stream_dir, f"p3_{sym_s}", {sym_s: miss_d}, workers=8)
                        (stream_dir / "book_depth_events" / f"shard_p3_{sym_s}.parquet").unlink(missing_ok=True)
                        if extra_bk is not None and not extra_bk.empty:
                            bk_sym = (
                                pd.concat([bk_sym, extra_bk], ignore_index=True) if bk_sym is not None else extra_bk
                            )
                    sh_recs.extend(evaluate_symbol_p3_p4_mechanisms(st, bk_sym))
                    del st, bk_sym

                del perp_map, pr_map, mt_map, fn_map, bk_map
                sh_tdf = pd.DataFrame(sh_recs) if sh_recs else pd.DataFrame({"_empty": pd.Series([1], dtype=np.int8)})
                sh_tdf.to_parquet(sh_ckpt, index=False, compression="zstd")
                subprocess.run(["rclone", "copy", str(sh_ckpt), f"{rb}/p3_p4_trades_ckpt_v1"], check=False)
                if sh_recs:
                    shard_dfs.append(sh_tdf)
                gc.collect()
                cum_t = sum(len(x) for x in shard_dfs)
                print(
                    f"  [{idx_s + 1:02d}/20] {sh_name}: +{len(sh_recs):,} P3/P4 trades (cum={cum_t:,}, dt={time.time() - t_sh:.1f}s)",
                    flush=True,
                )

            trades_df = pd.concat(shard_dfs, ignore_index=True) if shard_dfs else pd.DataFrame()
            if not trades_df.empty:
                all_tr_p = stream_dir / "p3_p4_all_trades.parquet"
                trades_df.to_parquet(all_tr_p, index=False, compression="zstd")
                subprocess.run(["rclone", "copy", str(all_tr_p), f"{rb}/p3_p4_cross_venue"], check=False)

            lb_df = evaluate_p3_p4_leaderboard(trades_df)
            output_dir.mkdir(parents=True, exist_ok=True)
            lb_df.to_csv(output_dir / "p3_p4_cross_venue_leaderboard.csv", index=False)
            treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
            summary = {
                "benchmark": "p3_p4_cross_venue_onchain_lab",
                "elapsed_sec": round(time.time() - t0, 1),
                "cross_venue_symbols_evaluated": len(eval_syms),
                "hl_matched_coins": len(hl_by_sym),
                "peer_cex_matched_coins": len(peer_by_sym),
                "total_simulated_trades": int(len(trades_df)),
                "treasure_count": len(treasures),
                "treasures": treasures,
                "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
            }
            (output_dir / "p3_p4_cross_venue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            md = [
                "# Phase P3/P4 Cross-Exchange (`Bybit/OKX`) & Full-Universe On-Chain (`Hyperliquid L1`) Lab (`2023–2026`)",
                "",
                f"- **Cross-Venue Universe Evaluated:** `{len(eval_syms)}` liquid perps (`Hyperliquid L1`: `{len(hl_by_sym)}` coins, `Peer CEX`: `{len(peer_by_sym)}` coins, `{len(trades_df):,}` simulated `1m` linear trades, `{summary['elapsed_sec']}s`).",
                f"- **Strict 4-Gate Universal Treasure Survivors:** **`{len(treasures)}`**",
                "",
                "| Mechanism | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** | Treasure |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            for _, r in lb_df.iterrows():
                md.append(
                    f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
                    f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
                    f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['med_net_bps_full']:+.1f}` | `{r['win_rate_pct']:.1f}%` | "
                    f"`{r['fee_share_of_gross_pct']:.1f}%` | `{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | "
                    f"**`{r['net_bps_2026']:+.1f}`** | `{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
                    f"**`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
                )
            (output_dir / "p3_p4_cross_venue_report.md").write_text("\n".join(md) + "\n")
            print("P3_P4_CROSS_VENUE_COMPLETE!", flush=True)
            return summary
        finally:
            if stream_dir.exists():
                shutil.rmtree(stream_dir, ignore_errors=True)
                print("[*] Wiped /tmp/p3_p4_rclone_stream -> 0 bytes local residue.", flush=True)
    meta_p = m16_data_dir / "meta.json"
    if meta_p.exists():
        meta_j = json.loads(meta_p.read_text())
        cand_syms = list(meta_j["liq_symbols"])
        max_ts = int(meta_j["max_1h_ts"])
    else:
        f_tmp = pd.concat(
            [
                pd.read_parquet(p, columns=["symbol", "timestamp"])
                for p in sorted((m16_data_dir / "funding").glob("shard_*.parquet"))
            ],
            ignore_index=True,
        )
        cand_syms = sorted(f_tmp["symbol"].astype(str).unique().tolist())
        max_ts = int(f_tmp["timestamp"].max())

    hl_path = ensure_full_hyperliquid_onchain_parquet(m16_data_dir, cand_syms, workers=4)
    peer_path = ensure_cex_peer_funding_parquet(m16_data_dir, cand_syms, workers=8)

    hl_df = pd.read_parquet(hl_path)
    peer_df = pd.read_parquet(peer_path)
    hl_by_sym = (
        {str(k): g.reset_index(drop=True) for k, g in hl_df.groupby("symbol", observed=True)} if not hl_df.empty else {}
    )
    peer_by_sym = (
        {str(k): g.reset_index(drop=True) for k, g in peer_df.groupby("symbol", observed=True)}
        if not peer_df.empty
        else {}
    )
    del hl_df, peer_df
    gc.collect()

    fnd_sym_dir = m16_data_dir / "funding_by_sym"
    met_sym_dir = m16_data_dir / "metrics_by_sym"
    fnd_fallback_map: dict[str, pd.DataFrame] = {}
    met_fallback_map: dict[str, pd.DataFrame] = {}
    if not fnd_sym_dir.exists():
        f_sh = sorted((m16_data_dir / "funding").glob("shard_*.parquet"))
        if f_sh:
            f_cat = pd.concat([pd.read_parquet(p) for p in f_sh], ignore_index=True)
            fnd_fallback_map = {str(k): g.reset_index(drop=True) for k, g in f_cat.groupby("symbol", observed=True)}
            del f_cat
    if not met_sym_dir.exists():
        m_sh = sorted((m16_data_dir / "metrics_5m").glob("shard_*.parquet"))
        if m_sh:
            m_cat = pd.concat([pd.read_parquet(p) for p in m_sh], ignore_index=True)
            met_fallback_map = {str(k): g.reset_index(drop=True) for k, g in m_cat.groupby("symbol", observed=True)}
            del m_cat

    eval_syms = [s for s in cand_syms if (s in hl_by_sym or s in peer_by_sym)]
    print(
        f"🌐 Phase P3/P4 Cross-Venue Lab | Evaluating {len(eval_syms)} liquid cross-venue perps "
        f"(HL={len(hl_by_sym)}, PeerCEX={len(peer_by_sym)}) in Strict O(1) RAM (<0.6 GB peak)...",
        flush=True,
    )

    end_dt = pd.Timestamp(max_ts, unit="s", tz="UTC") + timedelta(days=1)
    all_yms = [
        (y, m) for y in range(2023, end_dt.year + 1) for m in range(1, 13) if (y, m) < (end_dt.year, end_dt.month)
    ]
    recent_days = [date(end_dt.year, end_dt.month, d).isoformat() for d in range(1, end_dt.day + 1)]
    sess = _make_pooled_session(64)

    def _fetch_sym_perp_and_prem_1m(
        sym_c: str,
        min_needed_ts: int,
    ) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
        tks = PERP_TICKER_ALIASES.get(sym_c, (sym_c,))
        st_dt = pd.Timestamp(max(1_672_531_200, min_needed_ts), unit="s", tz="UTC")
        min_ym = (st_dt.year, st_dt.month)
        sym_yms = [ym for ym in all_yms if ym >= min_ym]
        p1m = fetch_gap_free_1m_klines("futures/um", tks, sym_c, sym_yms, recent_days, session=sess)
        if p1m is None or p1m.empty:
            return sym_c, None, None
        p1m = p1m[(p1m["timestamp"] >= min_needed_ts) & (p1m["timestamp"] <= max_ts + 3600)][
            ["timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
        ].reset_index(drop=True)

        pr_parts: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=16) as pr_pool:
            futs_p = [
                pr_pool.submit(_fetch_single_premium_archive, sess, sym_c, tk, y, m)
                for tk in tks
                for y, m in sym_yms + [(end_dt.year, end_dt.month)]
            ]
            for fp in as_completed(futs_p):
                pr_parts.extend(fp.result())
        prem_df = pd.concat(pr_parts, ignore_index=True) if pr_parts else None
        return sym_c, p1m, prem_df

    out_p3_dir = m16_data_dir / "p3_p4_out"
    out_p3_dir.mkdir(parents=True, exist_ok=True)
    all_trades: list[dict[str, object]] = []
    for done_s, s_c in enumerate(eval_syms, start=1):
        hl_s = hl_by_sym.get(s_c)
        pr_s = peer_by_sym.get(s_c)
        min_cv_list = []
        if hl_s is not None and not hl_s.empty:
            min_cv_list.append(int(hl_s["timestamp"].min()))
        if pr_s is not None and not pr_s.empty:
            min_cv_list.append(int(pr_s["timestamp"].min()))
        min_needed_ts = (min(min_cv_list) - 21 * 86400) if min_cv_list else 1_672_531_200

        _, p1m, prem_df = _fetch_sym_perp_and_prem_1m(s_c, min_needed_ts)
        if p1m is not None and not p1m.empty:
            fp_s = fnd_sym_dir / f"{s_c}.parquet"
            mp_s = met_sym_dir / f"{s_c}.parquet"
            fnd_s = pd.read_parquet(fp_s) if fp_s.exists() else fnd_fallback_map.get(s_c)
            met_s = pd.read_parquet(mp_s) if mp_s.exists() else met_fallback_map.get(s_c)
            st = prepare_symbol_p3_p4_state(
                symbol=s_c,
                perp_1m=p1m,
                prem_1m=prem_df,
                metrics_5m=met_s,
                funding_df=fnd_s,
                hl_funding_df=hl_s,
                peer_cex_df=pr_s,
            )
            del fnd_s, met_s
            if st is not None:
                need_d = set(st["needed_dates"])
                bk_sym = None
                if need_d:
                    bk_sym = fetch_shard_event_book_depth(m16_data_dir, f"p3_{s_c}", {s_c: need_d}, workers=8)
                    tmp_bk_p = m16_data_dir / "book_depth_events" / f"shard_p3_{s_c}.parquet"
                    tmp_bk_p.unlink(missing_ok=True)
                all_trades.extend(evaluate_symbol_p3_p4_mechanisms(st, bk_sym))
                del bk_sym
            del st
        del p1m, prem_df
        if done_s % 15 == 0 or done_s == len(eval_syms):
            gc.collect()
            if all_trades:
                pd.DataFrame(all_trades).to_parquet(
                    out_p3_dir / "p3_p4_all_trades.parquet", index=False, compression="zstd"
                )
            print(
                f"  • Evaluated {done_s}/{len(eval_syms)} cross-venue perps | trades={len(all_trades):,} ({time.time() - t0:.1f}s)...",
                flush=True,
            )

    trades_df = pd.DataFrame(all_trades)
    out_p3_dir = m16_data_dir / "p3_p4_out"
    out_p3_dir.mkdir(parents=True, exist_ok=True)
    if not trades_df.empty:
        trades_df.to_parquet(out_p3_dir / "p3_p4_all_trades.parquet", index=False, compression="zstd")

    lb_df = evaluate_p3_p4_leaderboard(trades_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "p3_p4_cross_venue_leaderboard.csv", index=False)

    treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
    summary = {
        "benchmark": "p3_p4_cross_venue_onchain_lab",
        "elapsed_sec": round(time.time() - t0, 1),
        "cross_venue_symbols_evaluated": len(eval_syms),
        "hl_matched_coins": len(hl_by_sym),
        "peer_cex_matched_coins": len(peer_by_sym),
        "total_simulated_trades": int(len(trades_df)),
        "treasure_count": len(treasures),
        "treasures": treasures,
        "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
    }
    (output_dir / "p3_p4_cross_venue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Phase P3/P4 Cross-Exchange (`Bybit/OKX`) & Full-Universe On-Chain (`Hyperliquid L1`) Lab (`2023–2026`)",
        "",
        f"- **Cross-Venue Universe Evaluated:** `{len(eval_syms)}` liquid perps (`Hyperliquid L1`: `{len(hl_by_sym)}` coins, `Peer CEX`: `{len(peer_by_sym)}` coins, `{len(trades_df):,}` simulated `1m` linear trades, `{summary['elapsed_sec']}s`).",
        f"- **Strict 4-Gate Universal Treasure Survivors:** **`{len(treasures)}`**",
        "",
        "| Mechanism | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Med | Win`%` | FeeSh`%` | `23-24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** | Treasure |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in lb_df.iterrows():
        md.append(
            f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
            f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
            f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['med_net_bps_full']:+.1f}` | `{r['win_rate_pct']:.1f}%` | "
            f"`{r['fee_share_of_gross_pct']:.1f}%` | `{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | "
            f"**`{r['net_bps_2026']:+.1f}`** | `{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
            f"**`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
        )
    (output_dir / "p3_p4_cross_venue_report.md").write_text("\n".join(md) + "\n")

    print("P3_P4_CROSS_VENUE_COMPLETE!", flush=True)
    return summary


def run_core_rebalance_barrier_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict[str, object]:
    """
    Executes Step 1: Unified `1-Minute` Core P1 Rebalance Barrier, Intracycle Drift &
    Turnover-Hysteresis Ablation (`2023–2026` + `$2,700` `2026` Cushion + `15`-split CPCV)
    with `Core P1 + Manga P2 + M09/M23/M04 + M26_L2SweepAirPocket` (`M25=OFF`).
    """
    import json
    import shutil
    import subprocess
    import tarfile
    import time

    if colab_session:
        from src.backtest.p2_portfolio_integration import PROD_TOP_15
        from src.cloud.colab_client import ColabClient
        from src.strategies.event_snipers import SNIPER_UNIVERSE_82

        stage_cb = Path("/tmp/core_barrier_bridge")
        if stage_cb.exists():
            shutil.rmtree(stage_cb, ignore_errors=True)
        raw_cb = stage_cb / "raw"
        for sub_p in ("bars_1h", "funding", "p2_aggtrades"):
            (raw_cb / sub_p).mkdir(parents=True, exist_ok=True)

        client = ColabClient(session_name=colab_session)
        try:
            if not rclone_stream:
                raise ValueError("--rclone-stream is required for zero-click Colab Core Barrier bridge.")
            rb = rclone_stream.rstrip("/")
            print("[*] Staging compact 1h + funding + M26 trades via rclone (auto-cleaned on exit)...", flush=True)
            for sub_p in ("bars_1h", "funding"):
                subprocess.run(
                    ["rclone", "copy", "--multi-thread-streams=8", f"{rb}/{sub_p}", str(raw_cb / sub_p)],
                    check=True,
                )
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    f"{rb}/p2_aggtrades/p2_all_variants_trades.parquet",
                    str(raw_cb / "p2_aggtrades"),
                ],
                check=True,
            )

            keep_syms = set(PROD_TOP_15) | set(SNIPER_UNIVERSE_82)
            b1h_cols = [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_quote_volume",
            ]
            b1h_parts = [
                pd.read_parquet(bf, columns=b1h_cols)[lambda d: d["symbol"].isin(keep_syms)]
                for bf in sorted((raw_cb / "bars_1h").glob("shard_*.parquet"))
            ]
            b1h_all = pd.concat([x for x in b1h_parts if not x.empty], ignore_index=True)

            fnd_parts = [
                pd.read_parquet(ff, columns=["timestamp", "symbol", "funding_rate"])[
                    lambda d: d["symbol"].isin(keep_syms)
                ]
                for ff in sorted((raw_cb / "funding").glob("shard_*.parquet"))
            ]
            fnd_all = pd.concat([x for x in fnd_parts if not x.empty], ignore_index=True)

            slim_cb = stage_cb / "slim"
            for sub_p in ("bars_1h", "funding", "p2_aggtrades"):
                (slim_cb / sub_p).mkdir(parents=True, exist_ok=True)
            b1h_all.to_parquet(slim_cb / "bars_1h" / "shard_00.parquet", index=False, compression="zstd")
            fnd_all.to_parquet(slim_cb / "funding" / "shard_00.parquet", index=False, compression="zstd")
            shutil.copy2(
                raw_cb / "p2_aggtrades" / "p2_all_variants_trades.parquet",
                slim_cb / "p2_aggtrades" / "p2_all_variants_trades.parquet",
            )

            tar_cb = stage_cb / "cb_bundle.tar"
            with tarfile.open(tar_cb, "w") as tf:
                for p_it in sorted(slim_cb.rglob("*")):
                    if p_it.is_file():
                        tf.add(p_it, arcname=str(p_it.relative_to(slim_cb)))

            print(f"[*] Provisioning headless Colab CPU VM '{colab_session}' (0 OAuth prompts)...", flush=True)
            if not client.ensure_session(gpu=None):
                raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
            assert client.sync_codebase(REPO_ROOT), "Failed syncing codebase to Colab"
            assert client.upload(tar_cb, "/content/_cb_bundle.tar"), "Failed uploading Core Barrier bundle"
            rc_u, out_u, err_u = client.exec_inline(
                "import os, tarfile; "
                "os.makedirs('/content/cb_data', exist_ok=True); "
                "os.makedirs('/content/reports', exist_ok=True); "
                "tarfile.open('/content/_cb_bundle.tar', 'r').extractall('/content/cb_data'); "
                "os.remove('/content/_cb_bundle.tar'); "
                "print('CB_BUNDLE_READY')",
                timeout=120,
            )
            if rc_u != 0 or "CB_BUNDLE_READY" not in out_u:
                raise RuntimeError(f"Failed unpacking Core Barrier bundle: {(err_u or out_u)[:160]}")

            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/cb.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                "'--suite', 'core-rebalance-barrier-lab', '--m16-data-dir', '/content/cb_data', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/cb.pid', 'w').write(str(p.pid)); "
                "print(f'CB_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=90)
            print(f"[+] Launched remote Core Rebalance Barrier Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(10.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/cb.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/cb.log').read().splitlines() if os.path.exists('/content/cb.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=90)
                if rc_p != 0:
                    consec_err += 1
                    if consec_err >= 4:
                        raise RuntimeError(f"Colab session lost during Core Barrier Lab: {(err_p or out_p)[:160]}")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "CORE_REBALANCE_BARRIER_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote Core Rebalance Barrier Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "core_rebalance_barrier_leaderboard.csv",
                "core_rebalance_barrier_summary.json",
                "core_rebalance_barrier_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            if stage_cb.exists():
                shutil.rmtree(stage_cb, ignore_errors=True)
                print("[*] Wiped /tmp/core_barrier_bridge -> 0 bytes local residue.", flush=True)

    from datetime import date, timedelta

    from src.backtest.core_rebalance_barrier_lab import run_core_rebalance_barrier_ablation
    from src.backtest.p2_portfolio_integration import (
        PROD_TOP_15,
        build_core_1m_tensors_and_base_snipers,
        identify_required_1m_symbols_for_portfolio,
    )
    from src.data.multimodal_vision_ingestor import (
        PERP_TICKER_ALIASES,
        _make_pooled_session,
        fetch_gap_free_1m_klines,
    )

    t0 = time.time()
    b1h_files = sorted((m16_data_dir / "bars_1h").glob("shard_*.parquet"))
    fnd_files = sorted((m16_data_dir / "funding").glob("shard_*.parquet"))
    p2_trades_path = m16_data_dir / "p2_aggtrades" / "p2_all_variants_trades.parquet"
    if not b1h_files:
        raise FileNotFoundError(f"Missing bars_1h in {m16_data_dir}")

    bars_1h_by_sym: dict[str, pd.DataFrame] = {}
    for bf in b1h_files:
        bdf = pd.read_parquet(bf)
        for s_k, g_df in bdf.groupby("symbol", sort=False):
            bars_1h_by_sym[str(s_k)] = g_df.reset_index(drop=True)

    funding_by_sym: dict[str, pd.DataFrame] = {}
    for ff in fnd_files:
        fdf = pd.read_parquet(ff)
        for s_k, g_df in fdf.groupby("symbol", sort=False):
            funding_by_sym[str(s_k)] = g_df.reset_index(drop=True)

    bars_1h_top15: dict[str, pd.DataFrame] = {}
    fnd_1h_top15: dict[str, pd.Series] = {}
    for s_c in PROD_TOP_15:
        if s_c not in bars_1h_by_sym:
            continue
        b = bars_1h_by_sym[s_c].sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        if "volume" not in b.columns:
            b["volume"] = b["quote_volume"] / np.maximum(b["close"], 1e-12)
        b.index = pd.DatetimeIndex(pd.to_datetime(b["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        b.drop(columns=["timestamp"], inplace=True)
        bars_1h_top15[s_c] = b
        if s_c in funding_by_sym and not funding_by_sym[s_c].empty:
            f = funding_by_sym[s_c].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_idx = pd.DatetimeIndex(
                pd.to_datetime(((f["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600, unit="s", utc=True)
            )
            f_s = pd.Series(f["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            fnd_1h_top15[s_c] = f_s[~f_s.index.duplicated(keep="last")]

    needed_1m_syms = identify_required_1m_symbols_for_portfolio(
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=funding_by_sym,
        eval_start="2023-03-01",
    )
    print(
        f"🛡️ Core P1 Rebalance Barrier Lab | Streaming {len(needed_1m_syms)} required 1m symbols in O(1) RAM...",
        flush=True,
    )

    btc_max_ts = int(bars_1h_by_sym["BTCUSDT"]["timestamp"].max())
    end_dt = pd.Timestamp(btc_max_ts, unit="s", tz="UTC") + timedelta(days=1)
    all_yms = [
        (y, m) for y in range(2023, end_dt.year + 1) for m in range(1, 13) if (y, m) < (end_dt.year, end_dt.month)
    ]
    recent_days = [date(end_dt.year, end_dt.month, d).isoformat() for d in range(1, end_dt.day + 1)]
    sess = _make_pooled_session(64)
    req_cols_1m = ["timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]

    def _stream_sym_1m(sym_c: str) -> pd.DataFrame | None:
        tks = PERP_TICKER_ALIASES.get(sym_c, (sym_c,))
        df_c = fetch_gap_free_1m_klines("futures/um", tks, sym_c, all_yms, recent_days, session=sess)
        if df_c is not None and not df_c.empty:
            df_c = df_c[df_c["timestamp"] <= btc_max_ts + 3600][req_cols_1m].reset_index(drop=True)
        return df_c

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=funding_by_sym,
        eval_start="2023-03-01",
        symbol_1m_loader=_stream_sym_1m,
        symbols_to_stream=needed_1m_syms,
        loader_workers=2,
    )
    m26_df = pd.read_parquet(p2_trades_path) if p2_trades_path.exists() else None
    lb_df = run_core_rebalance_barrier_ablation(
        bundle=bundle,
        bars_1h_top15=bars_1h_top15,
        funding_1h_top15=fnd_1h_top15,
        m26_trades_df=m26_df,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "core_rebalance_barrier_leaderboard.csv", index=False)
    summary = {
        "benchmark": "core_rebalance_barrier_1m_ablation",
        "elapsed_sec": round(time.time() - t0, 1),
        "results": lb_df.to_dict(orient="records"),
    }
    (output_dir / "core_rebalance_barrier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Core P1 Rebalance Barrier, Intracycle Drift & Turnover-Hysteresis `1m` Ablation (`2023–2026`)",
        "",
        f"- **Full Live Stack Simulated (`1m` Linear Physics):** `Core P1 + Manga P2 + M09/M23/M04 + M26_L2SweepAirPocket` (`M25=OFF`, `{summary['elapsed_sec']}s`).",
        "- **Account Physics:** `$2,700` Two-Phase SafeRamp, `$2,500` Cushion-Safe CPPI Floor, One-Way non-collision.",
        "",
        "| Config | `4Y SR` | `4Y CAGR` | `4Y DD` | `23-24 / 25 / 26 SR` | `CPCV SR (Pos%)` | `Intra` | `RebalClose` | `Lock / TP` | `FeeDrag%` | **`2026` `$2,700 -> Final` (`MinEq`)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in lb_df.iterrows():
        fl = "✅" if bool(r["fresh_2026_floor_2500_ok"]) else "⚠️"
        md.append(
            f"| `{r['config_id']}` | **`{r['sharpe_4y']:.2f}`** | `{r['cagr_4y_pct']:+.1f}%` | `{r['max_dd_4y_pct']:.2f}%` | "
            f"`{r['sharpe_23_24']:.2f} / {r['sharpe_2025']:.2f} / {r['sharpe_2026']:.2f}` | "
            f"`{r['cpcv_mean_sharpe']:.2f}` (`{r['cpcv_pos_splits_pct']:.0f}%`) | "
            f"`{int(r['intracycle_reranks'])}` | `{int(r['rebal_full_closes'])}` (`{r['rebal_exit_share_pct']:.0f}%`) | "
            f"`{int(r['profit_lock_exits'])} / {int(r['tp_exits'])}` | `{r['core_fee_drag_pct']:.1f}%` | "
            f"**`${r['fresh_2026_final_eq']:,.2f}`** (`${r['fresh_2026_min_eq']:,.2f}` {fl}) |"
        )
    (output_dir / "core_rebalance_barrier_report.md").write_text("\n".join(md) + "\n")
    print("CORE_REBALANCE_BARRIER_COMPLETE!", flush=True)
    return summary


def run_p4_m4_liquidation_aggtrades_lab_benchmark(
    m16_data_dir: Path,
    output_dir: Path,
    colab_session: str | None = None,
    keep_colab: bool = False,
    rclone_stream: str | None = None,
) -> dict[str, object]:
    """
    Executes Step 2: `1-Millisecond` Matching-Engine Forced-Liquidation Climax, Exhaustion &
    Institutional Absorption (`aggTrades`) on `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG` (`281` events,
    `0.000 ms` lookahead, True In-Fold CPCV `2023–2026`).
    """
    import json
    import shutil
    import subprocess
    import time

    if colab_session:
        from src.cloud.colab_client import ColabClient

        stage_p4 = Path("/tmp/p4_m4_agg_bridge")
        if stage_p4.exists():
            shutil.rmtree(stage_p4, ignore_errors=True)
        stage_p4.mkdir(parents=True, exist_ok=True)
        client = ColabClient(session_name=colab_session)
        try:
            if not rclone_stream:
                raise ValueError("--rclone-stream is required for zero-click Colab P4-M4 aggTrades bridge.")
            rb = rclone_stream.rstrip("/")
            print("[*] Staging P3/P4 trade events via rclone (auto-cleaned on exit)...", flush=True)
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    f"{rb}/p3_p4_cross_venue/p3_p4_all_trades.parquet",
                    str(stage_p4),
                ],
                check=True,
            )
            tr_loc = stage_p4 / "p3_p4_all_trades.parquet"
            print(f"[*] Provisioning headless Colab CPU VM '{colab_session}' (0 OAuth prompts)...", flush=True)
            if not client.ensure_session(gpu=None):
                raise RuntimeError(f"Failed to provision Colab session '{colab_session}'")
            assert client.sync_codebase(REPO_ROOT), "Failed syncing codebase to Colab"
            client.exec_inline(
                "import os; os.makedirs('/content/p4_data', exist_ok=True); os.makedirs('/content/reports', exist_ok=True)",
                timeout=90,
            )
            assert client.upload(tr_loc, "/content/p4_data/p3_p4_all_trades.parquet")

            launch_py = (
                "import subprocess, sys; "
                "lf = open('/content/p4_agg.log', 'w', buffering=1); "
                "p = subprocess.Popen([sys.executable, '-u', '/content/scripts/run_benchmarks.py', "
                "'--suite', 'p4-m4-liquidation-aggtrades-lab', '--m16-data-dir', '/content/p4_data', "
                "'--output-dir', '/content/reports'], "
                "stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); "
                "open('/content/p4_agg.pid', 'w').write(str(p.pid)); "
                "print(f'P4_AGG_PID={p.pid}')"
            )
            rc_l, out_l, err_l = client.exec_inline(launch_py, timeout=90)
            print(f"[+] Launched remote P4-M4 1ms aggTrades Lab ({rc_l}): {(out_l or err_l).strip()}", flush=True)

            seen = 0
            consec_err = 0
            while True:
                time.sleep(8.0)
                poll_py = (
                    "import os; "
                    "pid = int(open('/content/p4_agg.pid').read().strip()); "
                    "alive = os.path.exists(f'/proc/{pid}'); "
                    "lines = open('/content/p4_agg.log').read().splitlines() if os.path.exists('/content/p4_agg.log') else []; "
                    f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                    f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
                )
                rc_p, out_p, err_p = client.exec_inline(poll_py, timeout=90)
                if rc_p != 0:
                    consec_err += 1
                    if consec_err >= 4:
                        raise RuntimeError(f"Colab session lost during P4-M4 aggTrades Lab: {(err_p or out_p)[:160]}")
                    continue
                consec_err = 0
                alive, finished = True, False
                for ln in out_p.splitlines():
                    if ln.startswith("ALIVE="):
                        parts = ln.split()
                        alive = parts[0].split("=")[1] == "True"
                        seen = int(parts[1].split("=")[1])
                    elif ln.startswith("LOG|"):
                        msg = ln[4:]
                        print(msg, flush=True)
                        if "P4_M4_AGGTRADES_COMPLETE!" in msg:
                            finished = True
                if finished or not alive:
                    break

            if not finished:
                raise RuntimeError("Remote P4-M4 aggTrades Lab terminated early; inspect logs above.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "p4_m4_liquidation_aggtrades_leaderboard.csv",
                "p4_m4_liquidation_aggtrades_summary.json",
                "p4_m4_liquidation_aggtrades_report.md",
            ):
                client.download(f"/content/reports/{fname}", output_dir / fname)

            loc_enr = stage_p4 / "p4_m4_all_variants_trades.parquet"
            if client.download("/content/p4_data/p4_m4_all_variants_trades.parquet", loc_enr) and loc_enr.exists():
                subprocess.run(["rclone", "copy", str(loc_enr), f"{rb}/p3_p4_cross_venue"], check=False)
            return {}
        finally:
            if not keep_colab:
                client.stop()
                print(f"[*] Stopped Colab session '{colab_session}' cleanly.", flush=True)
            if stage_p4.exists():
                shutil.rmtree(stage_p4, ignore_errors=True)
                print("[*] Wiped /tmp/p4_m4_agg_bridge -> 0 bytes local residue.", flush=True)

    from src.models.liquidation_absorption_p4_lab import (
        enrich_p4_m4_events_with_1ms_aggtrades,
        evaluate_p4_m4_absorption_variants,
    )

    t0 = time.time()
    tr_path = m16_data_dir / "p3_p4_all_trades.parquet"
    if not tr_path.exists():
        tr_path = m16_data_dir / "p3_p4_cross_venue" / "p3_p4_all_trades.parquet"
    all_tr = pd.read_parquet(tr_path)
    p4_m4 = all_tr[all_tr["mech_id"] == "P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG"].copy()
    n_uniq = int(p4_m4[["symbol", "entry_ts"]].drop_duplicates().shape[0])
    print(
        f"🔬 Step 2 P4-M4 1ms Liquidation Absorption Lab | {n_uniq} unique flush events ({len(p4_m4)} trades)...",
        flush=True,
    )

    met_df = enrich_p4_m4_events_with_1ms_aggtrades(p4_m4, workers=10)
    cov_pct = round(100.0 * float((met_df["has_agg"] > 0.5).mean()), 1) if not met_df.empty else 0.0
    lb_df, all_v_df = evaluate_p4_m4_absorption_variants(p4_m4, met_df)
    if not all_v_df.empty:
        all_v_df.to_parquet(m16_data_dir / "p4_m4_all_variants_trades.parquet", index=False, compression="zstd")

    output_dir.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_dir / "p4_m4_liquidation_aggtrades_leaderboard.csv", index=False)
    treasures = lb_df[lb_df["is_treasure"]].to_dict(orient="records") if not lb_df.empty else []
    summary = {
        "benchmark": "p4_m4_liquidation_aggtrades_1ms_lab",
        "elapsed_sec": round(time.time() - t0, 1),
        "unique_events": n_uniq,
        "aggtrades_coverage_pct": cov_pct,
        "treasure_count": len(treasures),
        "treasures": treasures,
        "leaderboard": lb_df.to_dict(orient="records") if not lb_df.empty else [],
    }
    (output_dir / "p4_m4_liquidation_aggtrades_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Step 2: `P4_M4` Cross-Venue Anchored Liquidation Snap-Back + `1-ms aggTrades` Climax/Exhaustion/Absorption (`2023–2026`)",
        "",
        f"- **Events Audited (`0.000 ms` Lookahead):** `{n_uniq}` unique liquidation flushes (`aggTrades` coverage: `{cov_pct}%`, `{summary['elapsed_sec']}s`).",
        f"- **Strict 4-Gate Universal Treasure Survivors:** **`{len(treasures)}`**",
        "",
        "| Variant | Monetization | `N` | Gross | Fnd | Fee | **Net Full** | Win`%` | FeeSh`%` | `23-24` | **`2025`** | **`2026`** | **CPCV OOS** | **CPCV`%`** | Treasure |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in lb_df.iterrows():
        md.append(
            f"| `{r['mech_id']}` | `{r['exec_mode']}` | `{int(r['trades_full'])}` | "
            f"`{r['mean_gross_bps']:+.1f}` | `{r['mean_fnd_bps']:+.1f}` | `{r['mean_fee_bps']:.1f}` | "
            f"**`{r['mean_net_bps_full']:+.1f}`** | `{r['win_rate_pct']:.1f}%` | `{r['fee_share_of_gross_pct']:.1f}%` | "
            f"`{r['net_bps_23_24']:+.1f}` | **`{r['net_bps_2025']:+.1f}`** | **`{r['net_bps_2026']:+.1f}`** | "
            f"`{r['cpcv_infold_oos_bps']:+.1f}` | `{r['cpcv_pos_splits_pct']:.0f}%` | "
            f"**`{'TREASURE 🏆' if r['is_treasure'] else '—'}`** |"
        )
    (output_dir / "p4_m4_liquidation_aggtrades_report.md").write_text("\n".join(md) + "\n")
    print("P4_M4_AGGTRADES_COMPLETE!", flush=True)
    return summary


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
            "counterparty-mechanics-lab",
            "p1-microstructure-lab",
            "p2-aggtrades-sweep",
            "p2-unified-portfolio-1m",
            "p3-p4-cross-venue-lab",
            "core-rebalance-barrier-lab",
            "p4-m4-liquidation-aggtrades-lab",
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
            elif s_name == "counterparty-mechanics-lab":
                run_counterparty_mechanics_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    max_shards=args.max_shards,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                )
            elif s_name == "p1-microstructure-lab":
                run_p1_microstructure_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    max_shards=args.max_shards,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "p2-aggtrades-sweep":
                run_p2_aggtrades_sweep_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "p2-unified-portfolio-1m":
                run_p2_unified_portfolio_1m_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "p3-p4-cross-venue-lab":
                run_p3_p4_cross_venue_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "core-rebalance-barrier-lab":
                run_core_rebalance_barrier_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
            elif s_name == "p4-m4-liquidation-aggtrades-lab":
                run_p4_m4_liquidation_aggtrades_lab_benchmark(
                    m16_data_dir=Path(args.m16_data_dir),
                    output_dir=out_dir,
                    colab_session=args.colab_session,
                    keep_colab=args.keep_colab,
                    rclone_stream=args.rclone_stream,
                )
    finally:
        if pipe_stage_dir is not None and pipe_stage_dir.exists():
            import shutil

            shutil.rmtree(pipe_stage_dir, ignore_errors=True)
            print("[*] Cleaned temporary staging directory (/tmp/m16_pipe_stage -> 0 bytes).", flush=True)


if __name__ == "__main__":
    main()
