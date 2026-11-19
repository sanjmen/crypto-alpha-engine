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
    day = pd.to_datetime(df["decision_ts"], unit="s", utc=True).dt.floor("D")
    # Pillar 1 proxy: Cross-sectional 24h momentum minus funding carry
    p1_score = df.groupby("decision_ts")["ret_24h"].transform(
        lambda s: (s - s.mean()) / max(float(s.std()), 1e-8)
    ) - 0.35 * np.sign(df["funding_rate_bps"])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Quantitative & Microstructure Benchmark Suite CLI")
    parser.add_argument(
        "--suite",
        choices=["all", "hrp", "info-bars", "market-impact", "anomalies", "cointegration", "m16-feature-atlas"],
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

    suites = (
        ["hrp", "market-impact", "cointegration", "anomalies", "info-bars"] if args.suite == "all" else [args.suite]
    )

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


if __name__ == "__main__":
    main()
