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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Quantitative & Microstructure Benchmark Suite CLI")
    parser.add_argument(
        "--suite",
        choices=["all", "hrp", "info-bars", "market-impact", "anomalies", "cointegration"],
        default="hrp",
        help="Benchmark suite to execute (default: hrp).",
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


if __name__ == "__main__":
    main()
