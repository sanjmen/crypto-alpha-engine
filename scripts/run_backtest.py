#!/usr/bin/env python3
"""
Unified Institutional Multi-Strategy & Multi-Regime Backtest Engine CLI (Issue #66).

Consolidates and generalizes all historical backtest and ablation runners:
- Strategies (`--strategy`):
    * `seven_factor` (default): 7-Factor Causal Online Engine + Stepped Leverage + Chop Range-Arb + Holistic Regime Shields
    * `meta`: Legacy 3-strategy Meta-Allocator (+ optional Cointegrated Pairs & Anomaly Overlays)
    * `momentum`: Cross-Sectional Momentum
    * `stat_arb`: Statistical Arbitrage Mean-Reversion
    * `carry`: Perpetual Funding Carry Arbitrage
    * `anomalies`: Meta-Strategy + Extreme Anomaly & Illiquid Tail Overlays ($15 vs $100 sizing comparison)
- Modes (`--mode`):
    * `standard` (default): Single-window simulation with fee/funding decomposition, ASCII equity curve, CSV & Plotly HTML tearsheet
    * `6m`: 6-month daily/monthly granular breakdown with Cointegrated Pairs (RENDER/FET) and Markdown report
    * `anomalies`: 3-way comparison (Baseline vs $15 Test Anomaly Sizing vs $100 Prod Anomaly Sizing)
    * `multi-period`: Multi-strategy & 4-regime cycle stress-testing across 2023–2026 (Accumulation, Bull, Chop, Continuation)
    * `ablation`: Component ablation of the 7-Factor Engine (Baseline vs Chop Range-Arb vs Holistic Shields)
- Universes (`--preset`):
    * `top5`, `top10`, `top15` (Production 15-symbol basket), `top20`, `anomalies23`, `all82` (or custom `--symbols`)

Usage Examples:
    python scripts/run_backtest.py --strategy seven_factor --preset top15 --source cache
    python scripts/run_backtest.py --mode 6m --preset top15 --capital 1000 --start 2026-04-01 --end 2026-09-30
    python scripts/run_backtest.py --mode anomalies --preset anomalies23 --capital 1000
    python scripts/run_backtest.py --mode multi-period --preset all82 --source drive
    python scripts/run_backtest.py --mode ablation --preset top15 --source cache
"""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.engine import BacktestResult, VectorizedBacktester
from src.domain.entities import Portfolio
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.cointegrated_pairs import CointegratedPairsStrategy
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine
from src.strategies.event_snipers import (
    SNIPER_BRACKET_PROFILES,
    SNIPER_M04_STRAT,
    SNIPER_M25_STRAT,
    EventDrivenSniperEngine,
)
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
)
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_82_SYMBOLS = [
    "1000FLOKIUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "1INCHUSDT",
    "AAVEUSDT",
    "ADAUSDT",
    "ALGOUSDT",
    "APTUSDT",
    "ARBUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "AXSUSDT",
    "BATUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "CHZUSDT",
    "COMPUSDT",
    "CRVUSDT",
    "CTSIUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "DYDXUSDT",
    "EGLDUSDT",
    "ENAUSDT",
    "ENJUSDT",
    "ETCUSDT",
    "ETHUSDT",
    "FETUSDT",
    "FILUSDT",
    "FLOWUSDT",
    "FTMUSDT",
    "GALAUSDT",
    "GRTUSDT",
    "GTCUSDT",
    "ICPUSDT",
    "INJUSDT",
    "IOTAUSDT",
    "JUPUSDT",
    "KAVAUSDT",
    "KSMUSDT",
    "LDOUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "MAGICUSDT",
    "MANAUSDT",
    "MATICUSDT",
    "MEMEUSDT",
    "METUSDT",
    "MINAUSDT",
    "MKRUSDT",
    "NEARUSDT",
    "OGNUSDT",
    "ONDOUSDT",
    "OPUSDT",
    "PENDLEUSDT",
    "PYTHUSDT",
    "QNTUSDT",
    "RENDERUSDT",
    "RLCUSDT",
    "RUNEUSDT",
    "SANDUSDT",
    "SEIUSDT",
    "SKLUSDT",
    "SNXUSDT",
    "SOLUSDT",
    "STRKUSDT",
    "SUIUSDT",
    "SUSHIUSDT",
    "TAOUSDT",
    "THETAUSDT",
    "TIAUSDT",
    "UNIUSDT",
    "WIFUSDT",
    "WLDUSDT",
    "WUSDT",
    "XMRUSDT",
    "XRPUSDT",
    "ZECUSDT",
    "ZILUSDT",
    "ZKUSDT",
    "ZROUSDT",
]

TOP_15_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "RENDERUSDT",
    "FETUSDT",
]

ANOMALIES_23_SYMBOLS = [
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

PRESETS = {
    "top5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    "top10": [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "NEARUSDT",
    ],
    "top15": TOP_15_SYMBOLS,
    "top20": [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "NEARUSDT",
        "SUIUSDT",
        "APTUSDT",
        "DOTUSDT",
        "ATOMUSDT",
        "FTMUSDT",
        "ALGOUSDT",
        "EGLDUSDT",
        "KAVAUSDT",
        "ICPUSDT",
        "ARBUSDT",
    ],
    "anomalies23": ANOMALIES_23_SYMBOLS,
    "all82": ALL_82_SYMBOLS,
    "all": ALL_82_SYMBOLS,
}

MARKET_PERIODS = {
    "Full Horizon (2023–2026)": ("2023-01-01", "2026-10-01"),
    "Period 1: 2023 Accumulation & Low Vol": ("2023-01-01", "2023-10-01"),
    "Period 2: 2023-24 Bull Expansion": ("2023-10-01", "2024-03-31"),
    "Period 3: 2024 Volatile Chop & Correction": ("2024-04-01", "2024-11-01"),
    "Period 4: 2024-2026 Cycle Continuation": ("2024-11-01", "2026-10-01"),
}


def load_drive_market_data(
    symbols: list[str],
    timeframe: str = "1h",
    cache_dir: Path = Path("/tmp/crypto_alpha_backtest_cache"),
    sync_drive: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Loads symbol Parquet files from local cache (`cache_dir`), optionally syncing missing files
    from Google Drive via `rclone` when `sync_drive=True`.
    """
    bars_dir = cache_dir / "bars" / timeframe
    funding_dir = cache_dir / "funding"
    bars_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)

    if sync_drive:
        missing_bars = [s for s in symbols if not (bars_dir / f"{s}.parquet").exists()]
        missing_funding = [s for s in symbols if not (funding_dir / f"{s}.parquet").exists()]

        if missing_bars:
            print(f"📥 Downloading {len(missing_bars)} missing bar files from Google Drive via rclone...")
            pattern = "{" + ",".join([f"{s}.parquet" for s in missing_bars]) + "}"
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    f"gdrive:trading/crypto-alpha-engine/data/bars/{timeframe}/",
                    str(bars_dir),
                    "--include",
                    pattern,
                    "--transfers",
                    "16",
                    "--checkers",
                    "16",
                ],
                check=False,
            )

        if missing_funding:
            print(f"📥 Downloading {len(missing_funding)} missing funding files from Google Drive via rclone...")
            pattern = "{" + ",".join([f"{s}.parquet" for s in missing_funding]) + "}"
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    "gdrive:trading/crypto-alpha-engine/data/funding/",
                    str(funding_dir),
                    "--include",
                    pattern,
                    "--transfers",
                    "16",
                    "--checkers",
                    "16",
                ],
                check=False,
            )

    bars_dict: dict[str, pd.DataFrame] = {}
    funding_dict: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        bar_path = bars_dir / f"{sym}.parquet"
        if bar_path.exists():
            df = pd.read_parquet(bar_path)
            if "timestamp" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            bars_dict[sym] = df

        funding_path = funding_dir / f"{sym}.parquet"
        if funding_path.exists():
            f_df = pd.read_parquet(funding_path)
            if "timestamp" in f_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(f_df["timestamp"]):
                    ts_col = "fundingTime" if "fundingTime" in f_df.columns else "timestamp"
                    unit_val = "s" if float(f_df[ts_col].iloc[0]) < 1e11 else "ms"
                    f_df["timestamp"] = pd.to_datetime(f_df[ts_col], unit=unit_val, utc=True)
                f_df.set_index("timestamp", inplace=True)
            if "funding_rate" not in f_df.columns:
                for alt_c in ("fundingRate", "rate"):
                    if alt_c in f_df.columns:
                        f_df["funding_rate"] = f_df[alt_c].astype(float)
                        break
            f_df.sort_index(inplace=True)
            funding_dict[sym] = f_df

    return bars_dict, funding_dict


def generate_synthetic_universe(
    symbols: list[str],
    n_bars: int = 2000,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Generates synthetic multi-asset price and funding data for deterministic testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    bars_dict: dict[str, pd.DataFrame] = {}
    funding_dict: dict[str, pd.DataFrame] = {}
    base_prices = {"BTCUSDT": 45000.0, "ETHUSDT": 2500.0, "SOLUSDT": 100.0, "BNBUSDT": 300.0, "DOGEUSDT": 0.12}

    for sym in symbols:
        p0 = base_prices.get(sym, 50.0)
        daily_drift = np.random.uniform(-0.0001, 0.0004)
        vol = np.random.uniform(0.015, 0.035)
        returns = np.random.normal(daily_drift, vol, n_bars)
        price_series = p0 * np.exp(np.cumsum(returns))

        bars_dict[sym] = pd.DataFrame(
            {
                "open": price_series * (1.0 - np.random.uniform(0, 0.002, n_bars)),
                "high": price_series * (1.0 + np.random.uniform(0, 0.005, n_bars)),
                "low": price_series * (1.0 - np.random.uniform(0, 0.005, n_bars)),
                "close": price_series,
                "volume": np.random.uniform(500, 5000, n_bars) * (p0 / 100.0),
            },
            index=timestamps,
        )
        funding_dict[sym] = pd.DataFrame(
            {"funding_rate": np.random.normal(0.0001, 0.0002, n_bars)},
            index=timestamps,
        )

    return bars_dict, funding_dict


def run_simulation(
    strategy_name: str,
    bars_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
    rebalance_freq_bars: int = 8,
    execution_mode: str = "maker",
    turnover_deadband: float = 0.04,
    start_date: str | None = None,
    end_date: str | None = None,
    enable_pairs: bool = False,
    enable_anomalies: bool = False,
    anomaly_size: float = 15.0,
    enable_chop_arb: bool = True,
    enable_holistic_shields: bool = True,
    start_leverage: float = 2.5,
    max_leverage: float = 5.0,
    bhs_enter_top_k: int | None = 3,
    bhs_hold_top_k: int | None = 6,
    deadband_mult: float = 1.25,
    enable_btc_lead_lag: bool = False,
    enable_dollar_bar_tsmom: bool = False,
    dollar_bar_max_gross: float = 0.95,
    core_p1_risk_scale: float = 0.85,
    enable_event_snipers: bool = False,
    enable_post_settle_sniper: bool = True,
    sniper_slot_leverage: float = 0.24,
    sniper_max_positions: int = 3,
    cushion_floor_usd: float | None = None,
    cushion_buffer_usd: float = 240.0,
    recovery_boost_equity: float | None = None,
    recovery_boost_leverage: float | None = None,
    bars_1m_dict: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """
    Executes historical portfolio simulation across any supported strategy (`seven_factor`,
    `meta`, `momentum`, `stat_arb`, `carry`, `anomalies`) with frequency gating and window slicing.
    """
    valid_symbols = [s for s in bars_dict if not bars_dict[s].empty]
    if not valid_symbols:
        raise ValueError("No valid price data available for backtest.")

    anchor_sym = "BTCUSDT" if "BTCUSDT" in valid_symbols else valid_symbols[0]
    master_idx = bars_dict[anchor_sym].index

    if start_date:
        master_idx = master_idx[master_idx >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        if end_ts.hour == 0 and end_ts.minute == 0 and len(str(end_date)) <= 10:
            end_ts = end_ts + pd.Timedelta(hours=23)
        master_idx = master_idx[master_idx <= end_ts]

    if len(master_idx) < 100:
        raise ValueError(f"Insufficient aligned bars ({len(master_idx)}) to evaluate backtest.")

    close_dfs = {sym: bars_dict[sym]["close"].reindex(master_idx).ffill() for sym in valid_symbols}
    close_df = pd.DataFrame(close_dfs)
    returns_df = close_df.pct_change().fillna(0.0)

    funding_dfs = {}
    for sym in valid_symbols:
        if sym in funding_dict and not funding_dict[sym].empty:
            f_series = funding_dict[sym]["funding_rate"]
            f_aligned = f_series.reindex(master_idx, method="ffill").fillna(0.0)
            is_settlement = master_idx.hour.isin([0, 8, 16])
            funding_dfs[sym] = f_aligned * is_settlement
        else:
            funding_dfs[sym] = pd.Series(0.0, index=master_idx)
    funding_df = pd.DataFrame(funding_dfs)

    if strategy_name == "seven_factor":
        full_anchor_idx = bars_dict[anchor_sym].index
        warmup_start_ts = (
            (pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(hours=960)) if start_date else full_anchor_idx.min()
        )
        end_bound_ts = master_idx.max()
        calc_idx = full_anchor_idx[(full_anchor_idx >= warmup_start_ts) & (full_anchor_idx <= end_bound_ts)]

        bars_1h_in: dict[str, pd.DataFrame] = {}
        fnd_in: dict[str, pd.DataFrame] = {}
        tf_in: dict[str, pd.DataFrame] = {}
        present_mask = pd.DataFrame(False, index=calc_idx, columns=valid_symbols)
        open_mat = pd.DataFrame(index=calc_idx, columns=valid_symbols, dtype=float)
        high_mat = pd.DataFrame(index=calc_idx, columns=valid_symbols, dtype=float)
        low_mat = pd.DataFrame(index=calc_idx, columns=valid_symbols, dtype=float)

        for sym in valid_symbols:
            d_s = bars_dict[sym]
            d_s = d_s[~d_s.index.duplicated(keep="last")].sort_index()
            sub_s = d_s.loc[(d_s.index >= warmup_start_ts) & (d_s.index <= end_bound_ts)].copy()
            if sub_s.empty:
                continue
            s_close_re = sub_s["close"].reindex(calc_idx)
            min_list_bars = min(168, max(12, len(calc_idx) // 4))
            present_mask[sym] = (
                s_close_re.notna().rolling(min_list_bars, min_periods=min_list_bars).sum() >= min_list_bars
            ).fillna(False)
            open_mat[sym] = sub_s["open"].reindex(calc_idx).ffill().bfill()
            high_mat[sym] = sub_s["high"].reindex(calc_idx).ffill().bfill()
            low_mat[sym] = sub_s["low"].reindex(calc_idx).ffill().bfill()

            sub_df = sub_s.reset_index().rename(columns={sub_s.index.name or "index": "timestamp"})
            bars_1h_in[sym] = sub_df
            if "taker_buy_vol" in sub_s.columns and sub_s["taker_buy_vol"].notna().mean() > 0.80:
                tf_in[sym] = pd.DataFrame(
                    {
                        "timestamp": sub_s.index,
                        "quote_volume": (sub_s["close"].astype(float) * sub_s["volume"].astype(float)).values,
                        "taker_buy_quote_volume": (
                            sub_s["close"].astype(float) * sub_s["taker_buy_vol"].astype(float)
                        ).values,
                    }
                )
            if sym in funding_dict and not funding_dict[sym].empty:
                fd_s = funding_dict[sym]
                fd_s = fd_s[~fd_s.index.duplicated(keep="last")].sort_index()
                fd_sub = fd_s.loc[(fd_s.index >= warmup_start_ts) & (fd_s.index <= end_bound_ts)]
                if not fd_sub.empty:
                    fnd_in[sym] = pd.DataFrame(
                        {"timestamp": fd_sub.index, "fundingRate": fd_sub["funding_rate"].values}
                    )

        engine = SevenFactorOnlineEngine(
            priors=None,
            f6_min_tstat=0.75,
            f5_min_tstat=1.25 if enable_chop_arb else None,
            enable_chop_range_arb=enable_chop_arb,
            enable_holistic_regime_shields=enable_holistic_shields,
            enable_btc_lead_lag=enable_btc_lead_lag,
        )
        mats = engine.compute_matrices(
            bars_1h=bars_1h_in,
            funding_rates=fnd_in if fnd_in else None,
            taker_flow=tf_in if tf_in else None,
        )
        w_raw = mats["w_base_1x"].reindex(index=calc_idx, columns=valid_symbols).fillna(0.0).where(present_mask, 0.0)
        og_s = mats["w_base_1x"].reindex(calc_idx).abs().sum(axis=1).fillna(0.0)
        pos_w = w_raw.clip(lower=0.0)
        neg_w = (-w_raw).clip(lower=0.0)
        ps_w = pos_w.sum(axis=1)
        ns_w = neg_w.sum(axis=1)
        vld_w = (ps_w > 1e-10) & (ns_w > 1e-10) & (og_s > 1e-10)
        w_clean = (
            pos_w.div(ps_w.replace(0, 1.0), axis=0).mul(og_s / 2.0, axis=0)
            - neg_w.div(ns_w.replace(0, 1.0), axis=0).mul(og_s / 2.0, axis=0)
        ).mul(vld_w.astype(float), axis=0)

        eval_idx = master_idx if start_date else master_idx[min(72, len(master_idx) // 5) :]
        w1x_sh = w_clean.shift(1).fillna(0.0).reindex(eval_idx).values.astype(float)
        vol_sh = mats["vol_scalar"].shift(1).fillna(1.0).reindex(eval_idx).values.astype(float)
        chop_sh = mats["alpha_chop_overlay"].shift(1).fillna(0.0).reindex(eval_idx).values.astype(float)
        c_arr = mats["close"].reindex(index=eval_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
        o_arr = open_mat.reindex(index=eval_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
        h_arr = high_mat.reindex(index=eval_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
        l_arr = low_mat.reindex(index=eval_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
        f_arr = funding_df.reindex(index=eval_idx, columns=valid_symbols).fillna(0.0).values.astype(float)

        ctrl = SteppedLeverageController(
            initial_capital=initial_capital,
            start_leverage=start_leverage,
            max_leverage=max_leverage,
            step_leverage=0.5,
            gain_per_step=0.15,
            dd_bracket=0.06,
            base_deadband_ratio=(0.04 / 2.2) * max(0.5, float(deadband_mult)),
            bhs_enter_top_k=bhs_enter_top_k,
            bhs_hold_top_k=bhs_hold_top_k,
            recovery_boost_equity=recovery_boost_equity,
            recovery_boost_leverage=recovery_boost_leverage,
            cushion_floor_usd=cushion_floor_usd,
            cushion_buffer_usd=cushion_buffer_usd,
        )

        p2_w_mat: np.ndarray | None = None
        if enable_dollar_bar_tsmom:
            db_eng = CausalDollarBarTSMOMEngine(max_gross_leverage=dollar_bar_max_gross)
            src_p2 = bars_1m_dict if bars_1m_dict else {s: bars_1h_in[s] for s in db_eng.symbols if s in bars_1h_in}
            p2_hist_df = db_eng.compute_historical_weights(src_p2, eval_idx)
            p2_w_mat = p2_hist_df.reindex(index=eval_idx, columns=valid_symbols, fill_value=0.0).values.astype(float)

        events_by_t: dict[int, list[tuple[int, int, str, float, int]]] = {}
        if enable_event_snipers:
            sniper_eng = EventDrivenSniperEngine()
            t_grid_1h = (calc_idx.view("int64") // 10**9).astype(np.int64)
            c_1h_m = mats["close"].reindex(index=calc_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
            h_1h_m = high_mat.reindex(index=calc_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
            l_1h_m = low_mat.reindex(index=calc_idx, columns=valid_symbols).ffill().bfill().values.astype(float)
            qv_1h_df = pd.DataFrame(0.0, index=calc_idx, columns=valid_symbols)
            tb_1h_df = pd.DataFrame(0.0, index=calc_idx, columns=valid_symbols)
            for sym in valid_symbols:
                if sym in bars_1h_in and not bars_1h_in[sym].empty:
                    b_df = bars_1h_in[sym].set_index("timestamp")
                    b_df = b_df[~b_df.index.duplicated(keep="last")]
                    vol_s = b_df["volume"].reindex(calc_idx).fillna(0.0).astype(float)
                    cls_s = b_df["close"].reindex(calc_idx).ffill().fillna(0.0).astype(float)
                    qv_s = (
                        b_df["quote_volume"].reindex(calc_idx).fillna(0.0).astype(float)
                        if "quote_volume" in b_df.columns
                        else (cls_s * vol_s)
                    )
                    if "taker_buy_quote_volume" in b_df.columns:
                        tb_s = b_df["taker_buy_quote_volume"].reindex(calc_idx).fillna(0.0).astype(float)
                    elif "taker_buy_vol" in b_df.columns:
                        tb_s = cls_s * b_df["taker_buy_vol"].reindex(calc_idx).fillna(0.0).astype(float)
                    else:
                        tb_s = qv_s * 0.50
                    qv_1h_df[sym] = qv_s
                    tb_1h_df[sym] = tb_s
            adv_30d = qv_1h_df.shift(1).rolling(24 * 30, min_periods=24).mean().fillna(0.0)
            rk_1h_m = (adv_30d.rank(axis=1, ascending=False, method="first") - 1.0).fillna(99.0).values.astype(int)
            qv_1h_m = qv_1h_df.values.astype(float)
            tb_1h_m = tb_1h_df.values.astype(float)

            raw_evs = sniper_eng.scan_historical_1h_events(
                t_grid_1h=t_grid_1h,
                syms=valid_symbols,
                c_1h=c_1h_m,
                h_1h=h_1h_m,
                l_1h=l_1h_m,
                qv_1h=qv_1h_m,
                tb_1h=tb_1h_m,
                rank_1h=rk_1h_m,
            )
            # Hourly M04 Negative-Funding Short Squeeze scan when 15m bars are not passed
            f_calc_df = pd.DataFrame(0.0, index=calc_idx, columns=valid_symbols)
            for sym in valid_symbols:
                if sym in funding_dict and not funding_dict[sym].empty:
                    f_calc_df[sym] = funding_dict[sym]["funding_rate"].reindex(calc_idx, method="ffill").fillna(0.0)
            raw_evs.extend(
                sniper_eng.scan_historical_15m_events(
                    t_grid_15m=t_grid_1h,
                    c_15m=c_1h_m,
                    h_15m=h_1h_m,
                    qv_15m=qv_1h_m,
                    tb_15m=tb_1h_m,
                    fr_15m=f_calc_df.values.astype(float),
                    rank_15m=rk_1h_m,
                )
            )
            if enable_post_settle_sniper:
                f_settle_mat = pd.DataFrame(0.0, index=calc_idx, columns=valid_symbols)
                for sym in valid_symbols:
                    if sym in funding_dict and not funding_dict[sym].empty:
                        f_raw_s = funding_dict[sym]["funding_rate"]
                        f_raw_s = f_raw_s[~f_raw_s.index.duplicated(keep="last")]
                        f_settle_mat[sym] = f_raw_s.reindex(calc_idx).fillna(0.0)
                f_settle_arr = f_settle_mat.values.astype(float)
                adv_24h_usd = adv_30d.values.astype(float) * 24.0
                settle_t, settle_j = np.where(
                    (f_settle_arr <= sniper_eng.m25_max_funding_rate)
                    & (adv_24h_usd >= sniper_eng.m25_min_volume_24h_usd)
                )
                for t_m25, j_m25 in zip(settle_t.tolist(), settle_j.tolist(), strict=False):
                    if 24 <= t_m25 < len(t_grid_1h) - 1:
                        sc_m25 = float(abs(f_settle_arr[t_m25, j_m25]) * 10_000.0)
                        raw_evs.append(
                            (
                                int(t_grid_1h[t_m25]),
                                int(j_m25),
                                -1,
                                SNIPER_M25_STRAT,
                                sc_m25,
                                int(rk_1h_m[t_m25, j_m25]),
                            )
                        )
            ts_to_eval_t = {int(ts_v): idx_t for idx_t, ts_v in enumerate((eval_idx.view("int64") // 10**9).tolist())}
            for entry_ts_sec, j_ev, side_ev, strat_ev, sc_ev, rk_ev in raw_evs:
                eval_t = ts_to_eval_t.get(int(entry_ts_sec)) or ts_to_eval_t.get(int(entry_ts_sec) + 2700)
                if eval_t is not None:
                    events_by_t.setdefault(eval_t, []).append((j_ev, side_ev, strat_ev, sc_ev, rk_ev))

        n_b, n_s = len(eval_idx), len(valid_symbols)
        curr_w = np.zeros(n_s, dtype=float)
        prev_p2_w = np.zeros(n_s, dtype=float)
        entry_px = np.zeros(n_s, dtype=float)
        lock_armed = np.zeros(n_s, dtype=bool)
        cooldown_until = np.zeros((n_s, 2), dtype=int)
        active_snipers: dict[int, dict[str, float | int | bool | str]] = {}
        sniper_cooldown_until = np.zeros(n_s, dtype=int)
        last_ramp_sc = 1.0
        net_rets = np.zeros(n_b, dtype=float)
        gross_rets = np.zeros(n_b, dtype=float)
        fee_rets = np.zeros(n_b, dtype=float)
        fnd_rets = np.zeros(n_b, dtype=float)
        eq_arr = np.zeros(n_b, dtype=float)
        turn_bars = np.zeros(n_b, dtype=float)
        sim_eq = initial_capital
        hrs_utc = eval_idx.hour.values

        base_maker_cost = 0.0002 if execution_mode == "maker" else (0.0006 if execution_mode == "taker" else 0.0001)
        fallback_cost = 0.0002 if execution_mode == "maker" else (0.0006 if execution_mode == "taker" else 0.00085)
        use_p1_scale = bool(enable_dollar_bar_tsmom or enable_event_snipers)

        for t in range(n_b):
            prev_c = c_arr[t - 1] if t > 0 else o_arr[0]
            cur_o, cur_h, cur_l, cur_c = o_arr[t], h_arr[t], l_arr[t], c_arr[t]
            gap_ret = np.where(prev_c > 0, (cur_o - prev_c) / prev_c, 0.0)
            sniper_w_vec = np.zeros(n_s, dtype=float)
            for j_sn, sn_pos in active_snipers.items():
                sniper_w_vec[j_sn] = float(sn_pos["w"])
            bar_gross = float(np.sum((curr_w + sniper_w_vec) * gap_ret))
            bar_fee = 0.0
            bar_turn = 0.0

            is_reb = (hrs_utc[t] % 8 == 0) if rebalance_freq_bars == 8 else (t % max(1, rebalance_freq_bars) == 0)
            if is_reb:
                raw_u = w1x_sh[t].copy()
                for s_i in range(n_s):
                    if (
                        s_i in active_snipers
                        or (raw_u[s_i] > 0 and t < cooldown_until[s_i, 0])
                        or (raw_u[s_i] < 0 and t < cooldown_until[s_i, 1])
                    ):
                        raw_u[s_i] = 0.0
                pl = float(np.maximum(raw_u, 0.0).sum())
                nl = float(np.maximum(-raw_u, 0.0).sum())
                og = float(np.abs(w1x_sh[t]).sum())
                if pl > 1e-9 and nl > 1e-9 and og > 1e-9:
                    raw_u = np.where(
                        raw_u > 0, raw_u * ((og / 2.0) / pl), np.where(raw_u < 0, raw_u * ((og / 2.0) / nl), 0.0)
                    )
                else:
                    raw_u[:] = 0.0
                u_dict = {valid_symbols[s_i]: float(raw_u[s_i]) for s_i in range(n_s)}
                if use_p1_scale and np.any(np.abs(curr_w) > 1e-8):
                    inv_sc = 1.0 / max(core_p1_risk_scale, 0.25)
                    p1_est_arr = (curr_w - (prev_p2_w if p2_w_mat is not None else 0.0)) * inv_sc
                    prev_dict = (
                        {valid_symbols[s_i]: float(p1_est_arr[s_i]) for s_i in range(n_s)}
                        if np.any(np.abs(p1_est_arr) > 1e-6)
                        else None
                    )
                else:
                    prev_dict = (
                        {valid_symbols[s_i]: float(curr_w[s_i]) for s_i in range(n_s)}
                        if np.any(np.abs(curr_w) > 1e-8)
                        else None
                    )
                scaled_d, snap_t = ctrl.scale_and_deadband_weights(
                    unit_weights=u_dict,
                    current_weights=prev_dict,
                    current_equity=sim_eq,
                    vol_scalar=float(vol_sh[t]),
                )
                ramp_m = min(1.65, max(0.50, snap_t.active_base_leverage / max(start_leverage, 1.0)))
                last_ramp_sc = ramp_m * snap_t.cushion_scalar
                desired_p1 = np.array([scaled_d.get(sym, 0.0) for sym in valid_symbols], dtype=float)
                if p2_w_mat is not None:
                    p2_sc = last_ramp_sc
                    raw_p2 = np.where(
                        np.isin(np.arange(n_s), list(active_snipers.keys())),
                        0.0,
                        p2_w_mat[t] * p2_sc,
                    )
                    eff_p2 = np.where(
                        (np.abs(raw_p2) >= 1e-6)
                        & ((raw_p2 * prev_p2_w) > 0.0)
                        & (np.abs(raw_p2 - prev_p2_w) < 0.030 * p2_sc),
                        prev_p2_w,
                        raw_p2,
                    )
                    desired_w = desired_p1 * core_p1_risk_scale + eff_p2
                    if float(np.sum(np.abs(desired_w)) * sim_eq) >= 2000.0:
                        stub_m = (
                            (np.abs(desired_w) > 0.0) & (np.abs(desired_w * sim_eq) < 155.0) & (np.abs(eff_p2) > 0.0)
                        )
                        eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                        desired_w = np.where(stub_m, 0.0, desired_w)
                    prev_p2_w = eff_p2
                else:
                    desired_w = desired_p1 * core_p1_risk_scale if use_p1_scale else desired_p1
                for s_i in range(n_s):
                    if s_i in active_snipers:
                        continue
                    dw = desired_w[s_i] - curr_w[s_i]
                    if abs(dw) < 1e-6:
                        continue
                    touched = (
                        (cur_l[s_i] <= cur_o[s_i] * (1.0 - 0.0002))
                        if dw > 0
                        else (cur_h[s_i] >= cur_o[s_i] * (1.0 + 0.0002))
                    )
                    bar_turn += abs(dw)
                    if touched:
                        bar_fee += abs(dw) * base_maker_cost
                    else:
                        reducing = (curr_w[s_i] * desired_w[s_i] < -1e-9) or (abs(desired_w[s_i]) < abs(curr_w[s_i]))
                        bar_fee += abs(dw) * (0.00060 if (reducing and execution_mode == "hybrid") else fallback_cost)
                    if abs(desired_w[s_i]) < 1e-8:
                        entry_px[s_i] = 0.0
                        lock_armed[s_i] = False
                    elif abs(curr_w[s_i]) < 1e-8 or (curr_w[s_i] * desired_w[s_i] < 0):
                        entry_px[s_i] = cur_o[s_i]
                        lock_armed[s_i] = False
                    elif abs(desired_w[s_i]) > abs(curr_w[s_i]) + 1e-6:
                        add_w = abs(desired_w[s_i]) - abs(curr_w[s_i])
                        entry_px[s_i] = (abs(curr_w[s_i]) * entry_px[s_i] + add_w * cur_o[s_i]) / abs(desired_w[s_i])
                curr_w = np.where(np.isin(np.arange(n_s), list(active_snipers.keys())), 0.0, desired_w)

            if enable_event_snipers and t in events_by_t and len(active_snipers) < sniper_max_positions:
                cands_t = sorted(events_by_t[t], key=lambda x: x[3], reverse=True)
                slot_lev = float(sniper_slot_leverage) * last_ramp_sc
                for j_ev, side_ev, strat_ev, _sc_ev, _rk_ev in cands_t:
                    if len(active_snipers) >= sniper_max_positions:
                        break
                    if (
                        j_ev in active_snipers
                        or abs(curr_w[j_ev]) > 1e-8
                        or t < sniper_cooldown_until[j_ev]
                        or cur_o[j_ev] <= 0.0
                        or slot_lev * sim_eq < 20.0
                    ):
                        continue
                    w_sn = float(side_ev) * slot_lev
                    active_snipers[j_ev] = {
                        "w": w_sn,
                        "side": int(side_ev),
                        "ep": float(cur_o[j_ev]),
                        "peak": float(cur_o[j_ev]),
                        "lock": False,
                        "strat": strat_ev,
                        "open_t": t,
                    }
                    bar_fee += abs(w_sn) * base_maker_cost
                    bar_turn += abs(w_sn)

            is_chop_b = chop_sh[t] >= 0.25
            for s_i in range(n_s):
                w_i = curr_w[s_i]
                if abs(w_i) < 1e-8 or entry_px[s_i] <= 0:
                    continue
                aligned_p2 = (p2_w_mat is not None) and (abs(prev_p2_w[s_i]) >= 0.15) and ((prev_p2_w[s_i] * w_i) > 0.0)
                use_chop_p = is_chop_b and not aligned_p2
                tp_tgt = 0.032 if use_chop_p else 0.050
                lk_trig_b = 0.024 if use_chop_p else 0.035
                lk_lvl_b = 0.012 if use_chop_p else 0.015
                sgn = 1.0 if w_i > 0 else -1.0
                ep = entry_px[s_i]
                pnl_open = sgn * (cur_o[s_i] - ep) / ep
                fav_pnl = sgn * ((cur_h[s_i] if sgn > 0 else cur_l[s_i]) - ep) / ep
                adv_pnl = sgn * ((cur_l[s_i] if sgn > 0 else cur_h[s_i]) - ep) / ep
                exited = False
                exit_px = cur_c[s_i]
                exit_fee = 0.0002
                if lock_armed[s_i]:
                    if pnl_open <= lk_lvl_b:
                        exit_px, exit_fee, exited = cur_o[s_i], 0.00055, True
                    elif adv_pnl <= lk_lvl_b:
                        exit_px, exit_fee, exited = ep * (1.0 + sgn * lk_lvl_b), 0.00055, True
                    elif fav_pnl >= tp_tgt:
                        exit_px = cur_o[s_i] if pnl_open >= tp_tgt else ep * (1.0 + sgn * tp_tgt)
                        exit_fee, exited = 0.00020, True
                else:
                    if pnl_open >= tp_tgt:
                        exit_px, exit_fee, exited = cur_o[s_i], 0.00020, True
                    elif fav_pnl >= tp_tgt:
                        exit_px, exit_fee, exited = ep * (1.0 + sgn * tp_tgt), 0.00020, True
                    elif fav_pnl >= lk_trig_b:
                        lock_armed[s_i] = True
                if exited:
                    bar_gross += w_i * ((exit_px - cur_o[s_i]) / cur_o[s_i])
                    bar_fee += abs(w_i) * exit_fee
                    bar_turn += abs(w_i)
                    if sgn * (exit_px - ep) < 0:
                        cooldown_until[s_i, 0 if sgn > 0 else 1] = t + 4
                    curr_w[s_i] = 0.0
                    entry_px[s_i] = 0.0
                    lock_armed[s_i] = False
                else:
                    bar_gross += w_i * ((cur_c[s_i] - cur_o[s_i]) / cur_o[s_i])

            for j_sn in list(active_snipers.keys()):
                sn = active_snipers[j_sn]
                w_sn = float(sn["w"])
                sgn_sn = float(sn["side"])
                ep_sn = float(sn["ep"])
                prof = SNIPER_BRACKET_PROFILES.get(str(sn["strat"]), SNIPER_BRACKET_PROFILES[SNIPER_M04_STRAT])
                fav_px = cur_h[j_sn] if sgn_sn > 0 else cur_l[j_sn]
                adv_px = cur_l[j_sn] if sgn_sn > 0 else cur_h[j_sn]
                peak_sn = max(float(sn["peak"]), fav_px) if sgn_sn > 0 else min(float(sn["peak"]), fav_px)
                sn["peak"] = peak_sn
                fav_r = sgn_sn * (fav_px - ep_sn) / ep_sn
                adv_r = sgn_sn * (adv_px - ep_sn) / ep_sn
                stop_r = -prof.stop_loss_pct
                if bool(sn["lock"]):
                    stop_r = max(stop_r, prof.lock_level_pct)
                peak_r = sgn_sn * (peak_sn - ep_sn) / ep_sn
                if peak_r >= prof.trail_trigger_pct:
                    stop_r = max(stop_r, peak_r - prof.trail_distance_pct)
                ex_sn = False
                ex_px_sn = cur_c[j_sn]
                ex_fee_sn = 0.00055
                max_hold_bars = max(1, int(round(prof.max_hold_hours)))
                if adv_r <= stop_r:
                    ex_px_sn, ex_fee_sn, ex_sn = ep_sn * (1.0 + sgn_sn * stop_r), 0.00055, True
                elif fav_r >= prof.take_profit_pct:
                    ex_px_sn, ex_fee_sn, ex_sn = ep_sn * (1.0 + sgn_sn * prof.take_profit_pct), 0.00020, True
                elif (t - int(sn["open_t"]) + 1) >= max_hold_bars:
                    ex_px_sn, ex_fee_sn, ex_sn = cur_c[j_sn], 0.00055, True
                else:
                    if fav_r >= prof.lock_trigger_pct:
                        sn["lock"] = True
                if ex_sn:
                    bar_gross += w_sn * ((ex_px_sn - cur_o[j_sn]) / cur_o[j_sn])
                    bar_fee += abs(w_sn) * ex_fee_sn
                    bar_turn += abs(w_sn)
                    was_loss_sn = (sgn_sn * (ex_px_sn - ep_sn)) < 0.0
                    sniper_cooldown_until[j_sn] = t + (4 if was_loss_sn else 1)
                    del active_snipers[j_sn]
                else:
                    bar_gross += w_sn * ((cur_c[j_sn] - cur_o[j_sn]) / cur_o[j_sn])

            active_sn_fnd_w = np.zeros(n_s, dtype=float)
            for j_sn, sn_pos in active_snipers.items():
                active_sn_fnd_w[j_sn] = float(sn_pos["w"])
            bar_fnd = -float(np.sum((curr_w + active_sn_fnd_w) * f_arr[t]))
            nr_t = bar_gross - bar_fee + bar_fnd
            sim_eq = max(0.0, sim_eq * (1.0 + nr_t))
            eq_arr[t] = sim_eq
            net_rets[t] = nr_t
            gross_rets[t] = bar_gross
            fee_rets[t] = bar_fee
            fnd_rets[t] = bar_fnd
            turn_bars[t] = bar_turn

        eq_s = pd.Series(eq_arr, index=eval_idx)
        nr_s = pd.Series(net_rets, index=eval_idx)
        dd_s = ((eq_s - eq_s.cummax().replace(0.0, np.nan)) / eq_s.cummax().replace(0.0, np.nan)).fillna(0.0)
        tot_ret_pct = float((eq_s.iloc[-1] / initial_capital - 1.0) * 100.0) if len(eq_s) > 0 else 0.0
        cagr_v = (
            float(((eq_s.iloc[-1] / initial_capital) ** (8760.0 / max(1, len(nr_s))) - 1.0) * 100.0)
            if len(eq_s) > 0 and eq_s.iloc[-1] > 0
            else -100.0
        )
        mu_r = float(nr_s.mean())
        sd_r = max(float(nr_s.std()), 1e-8)
        neg_r = nr_s[nr_s < 0]
        dsd_r = max(float(neg_r.std()) if len(neg_r) > 1 else 1e-8, 1e-8)
        max_dd_v = float(abs(dd_s.min()) * 100.0)
        act_cnt = int((nr_s != 0).sum())
        win_pct = float((nr_s > 0).sum() / act_cnt * 100.0) if act_cnt > 0 else 0.0
        pf_v = float(nr_s[nr_s > 0].sum() / max(abs(float(neg_r.sum())), 1e-8))
        return BacktestResult(
            total_return_pct=tot_ret_pct,
            cagr_pct=cagr_v,
            annualized_vol_pct=float(sd_r * np.sqrt(8760.0) * 100.0),
            sharpe_ratio=float((mu_r / sd_r) * np.sqrt(8760.0)),
            sortino_ratio=float((mu_r / dsd_r) * np.sqrt(8760.0)),
            max_drawdown_pct=max_dd_v,
            calmar_ratio=float(cagr_v / max_dd_v) if max_dd_v > 0 else 0.0,
            total_trades=int((turn_bars > 0.01).sum()),
            win_rate_pct=win_pct,
            profit_factor=pf_v,
            gross_return_pct=float(gross_rets.sum() * 100.0),
            total_fees_pct=float(fee_rets.sum() * 100.0),
            total_funding_pnl_pct=float(fnd_rets.sum() * 100.0),
            total_turnover=float(turn_bars.sum()),
            equity_curve=eq_s,
            drawdown_curve=dd_s,
        )

    # Legacy & Satellite Strategies (`meta`, `momentum`, `stat_arb`, `carry`, `anomalies`)
    use_anoms = enable_anomalies or (strategy_name == "anomalies")
    pairs_strat = CointegratedPairsStrategy(lookback_bars=240) if (enable_pairs or use_anoms) else None
    oversold_strat = OversoldBounceStrategy(max_position_usd=anomaly_size) if use_anoms else None
    breakout_strat = EarlyBreakoutStrategy(max_position_usd=anomaly_size) if use_anoms else None
    fade_strat = PumpFundingFadeStrategy(max_position_usd=anomaly_size) if use_anoms else None

    if strategy_name == "momentum":
        strat = CrossSectionalMomentumStrategy()
    elif strategy_name == "stat_arb":
        strat = StatArbMeanReversionStrategy()
    elif strategy_name == "carry":
        strat = FundingCarryArbitrageStrategy()
    else:
        strat = MetaStrategyAllocator(
            pairs_strategy=pairs_strat,
            oversold_bounce_strategy=oversold_strat,
            early_breakout_strategy=breakout_strat,
            funding_fade_strategy=fade_strat,
        )

    weights_history = []
    portfolio = Portfolio(cash=initial_capital, initial_cash=initial_capital)
    lookback = 72
    current_weights = {sym: 0.0 for sym in valid_symbols}

    for i in range(lookback, len(master_idx)):
        ts = master_idx[i]
        if (i - lookback) % rebalance_freq_bars == 0:
            window_start = max(0, i - 250)
            sub_idx = master_idx[window_start : i + 1]
            sub_market = {}
            for sym in valid_symbols:
                common_sub = bars_dict[sym].index.intersection(sub_idx)
                if len(common_sub) >= 30:
                    sub_market[sym] = bars_dict[sym].loc[common_sub]

            sub_funding = {
                sym: funding_dict[sym].loc[funding_dict[sym].index <= ts]
                for sym in valid_symbols
                if sym in funding_dict and not funding_dict[sym].empty
            }
            volatilities = {sym: float(returns_df[sym].iloc[max(0, i - 24) : i].std()) or 0.02 for sym in sub_market}

            if strategy_name in ("meta", "anomalies"):
                target_dollars, _, _ = strat.generate_portfolio_allocations(
                    market_data=sub_market,
                    portfolio=portfolio,
                    volatilities=volatilities,
                    funding_data=sub_funding,
                )
            elif strategy_name == "carry":
                signals = strat.generate_signals(sub_market, funding_data=sub_funding)
                target_dollars = strat.allocate_weights(signals, portfolio, volatilities)
            else:
                signals = strat.generate_signals(sub_market)
                target_dollars = strat.allocate_weights(signals, portfolio, volatilities)

            total_eq = portfolio.total_equity
            current_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in valid_symbols}

        weights_history.append((ts, current_weights))

    weight_records = {ts: w for ts, w in weights_history}
    weights_df = pd.DataFrame.from_dict(weight_records, orient="index").reindex(master_idx).fillna(0.0)

    backtester = VectorizedBacktester(
        initial_capital=initial_capital,
        taker_fee=0.0004,
        maker_fee=0.0002,
        slippage_bps=1.0,
        execution_mode=execution_mode,
        turnover_deadband=turnover_deadband,
    )
    return backtester.run(returns_df=returns_df, weights_df=weights_df, funding_rates_df=funding_df)


def format_ascii_chart(equity_series: pd.Series, width: int = 50, height: int = 10) -> str:
    """Generates an ASCII sparkline graph of the portfolio equity curve."""
    values = equity_series.values
    if len(values) < 2:
        return ""
    indices = np.linspace(0, len(values) - 1, width).astype(int)
    sampled = values[indices]
    min_v = float(np.min(sampled))
    max_v = float(np.max(sampled))
    rng = max_v - min_v or 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, v in enumerate(sampled):
        row = int((v - min_v) / rng * (height - 1))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][col] = "█"

    lines = [f"  ${max_v:,.2f} ┐"]
    for r in range(height):
        row_str = "".join(grid[r])
        if r == height // 2:
            mid_val = min_v + rng / 2
            lines.append(f"  ${mid_val:,.2f} ┼{row_str}")
        else:
            lines.append(f"          │{row_str}")
    lines.append(f"  ${min_v:,.2f} ┴" + "─" * width)
    return "\n".join(lines)


def generate_plotly_tearsheet(
    result: BacktestResult,
    strategy_name: str,
    preset: str,
    output_path: Path,
) -> None:
    """Generates an institutional interactive HTML report with Plotly."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"Institutional Cumulative Equity Curve — {strategy_name.upper()} ({preset.upper()})",
            "Underwater Drawdown Profile (%)",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name="Portfolio Equity ($)",
            line=dict(color="#00E676", width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=result.drawdown_curve.index,
            y=result.drawdown_curve.values * 100.0,
            mode="lines",
            fill="tozeroy",
            name="Drawdown (%)",
            line=dict(color="#FF1744", width=1.5),
            fillcolor="rgba(255, 23, 68, 0.2)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title_text=(
            f"<b>Crypto Alpha Engine: {strategy_name.upper()} Tearsheet</b> | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Net Return: {result.total_return_pct:+.2f}% | "
            f"Max DD: {result.max_drawdown_pct:.2f}%"
        ),
        height=800,
        showlegend=True,
        margin=dict(l=60, r=40, t=90, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"📊 Interactive Plotly tear sheet exported to: {output_path}")


def print_performance_table(
    result: BacktestResult,
    strategy_name: str,
    preset: str,
    execution_mode: str,
    deadband: float,
) -> None:
    """Prints institutional summary table with fee breakdown."""
    print("\n" + "=" * 70)
    print(f"  INSTITUTIONAL PERFORMANCE REPORT: {strategy_name.upper()} ({preset.upper()})")
    print(f"  Execution: {execution_mode.upper()} | Turnover Deadband Buffer: {deadband * 100:.1f}%")
    print("=" * 70)
    print(f"  Gross Alpha Return:  {result.gross_return_pct:+8.2f}%")
    print(f"  Total Turnover:      {result.total_turnover:8.2f}x portfolio")
    print(f"  Total Fees/Slippage: {result.total_fees_pct:8.2f}%")
    print(f"  Funding Cashflow PnL:{result.total_funding_pnl_pct:+8.2f}%")
    print("-" * 70)
    print(f"  NET TOTAL RETURN:    {result.total_return_pct:+8.2f}%")
    print(f"  CAGR (Annualized):   {result.cagr_pct:+8.2f}%")
    print(f"  Annualized Vol:      {result.annualized_vol_pct:8.2f}%")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:8.2f}")
    print(f"  Sortino Ratio:       {result.sortino_ratio:8.2f}")
    print(f"  Calmar Ratio:        {result.calmar_ratio:8.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:8.2f}%")
    print(f"  Profit Factor:       {result.profit_factor:8.2f}")
    print(f"  Win Rate:            {result.win_rate_pct:8.2f}%")
    print(f"  Active Trades:       {result.total_trades:8d}")
    print("=" * 70 + "\n")


def run_ablation_suite(
    bars_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Runs a 3-way component ablation on the 7-Factor Causal Online Engine."""
    print("\n" + "=" * 95)
    print("  7-FACTOR CAUSAL ENGINE: COMPONENT ABLATION SUITE")
    print("=" * 95)
    variants = [
        ("1. Baseline 7-Factor (No Chop Arb / No Shields)", False, False),
        ("2. + Chop Range-Arb (F5 Compression + Tighter Gates)", True, False),
        ("3. + Holistic Regime Shields (Shield 1A + Shield 1B)", True, True),
    ]
    print(f"  {'Variant':<54} | {'Net Ret':>9} | {'Sharpe':>7} | {'Max DD':>8} | {'Calmar':>7}")
    print("-" * 95)
    for label, chop_on, shields_on in variants:
        res = run_simulation(
            strategy_name="seven_factor",
            bars_dict=bars_dict,
            funding_dict=funding_dict,
            initial_capital=args.capital,
            rebalance_freq_bars=args.rebalance_freq,
            execution_mode=args.execution_mode,
            turnover_deadband=args.turnover_deadband,
            start_date=args.start,
            end_date=args.end,
            enable_chop_arb=chop_on,
            enable_holistic_shields=shields_on,
            start_leverage=args.start_leverage,
            max_leverage=args.max_leverage,
        )
        print(
            f"  {label:<54} | {res.total_return_pct:+8.2f}% | {res.sharpe_ratio:7.2f} | "
            f"{res.max_drawdown_pct:7.2f}% | {res.calmar_ratio:7.2f}"
        )
    print("=" * 95 + "\n")


def run_multi_period_suite(
    bars_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Runs multi-strategy & 4-regime cycle stress-testing across 2023–2026."""
    print("\n" + "=" * 95)
    print("  MULTI-PERIOD REGIME STRESS-TEST SUITE")
    print("=" * 95)
    print(f"  {'Market Cycle / Period':<44} | {'Window':<23} | {'Net Ret':>9} | {'Sharpe':>7} | {'Max DD':>8}")
    print("-" * 95)
    for period_name, (start_d, end_d) in MARKET_PERIODS.items():
        try:
            p_res = run_simulation(
                strategy_name=args.strategy,
                bars_dict=bars_dict,
                funding_dict=funding_dict,
                initial_capital=args.capital,
                rebalance_freq_bars=args.rebalance_freq,
                execution_mode=args.execution_mode,
                turnover_deadband=args.turnover_deadband,
                start_date=start_d,
                end_date=end_d,
                enable_pairs=args.enable_pairs,
                enable_chop_arb=args.enable_chop_arb,
                enable_holistic_shields=args.enable_holistic_shields,
                start_leverage=args.start_leverage,
                max_leverage=args.max_leverage,
            )
            win_str = f"{start_d}..{end_d}"
            print(
                f"  {period_name:<44} | {win_str:<23} | {p_res.total_return_pct:+8.2f}% | "
                f"{p_res.sharpe_ratio:7.2f} | {p_res.max_drawdown_pct:7.2f}%"
            )
        except ValueError as exc:
            print(f"  {period_name:<44} | {start_d}..{end_d:<11} | SKIPPED ({exc})")
    print("=" * 95 + "\n")


def run_anomalies_comparison_suite(
    bars_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Runs 3-way comparison: Baseline Meta vs $15 Test Anomaly Sizing vs $100 Prod Sizing."""
    start_d = args.start or "2026-04-14"
    end_d = args.end or "2026-10-14"
    print("\n" + "=" * 95)
    print(f"  EXTREME ANOMALY & ILLIQUID TAIL BACKTEST COMPARISON ({start_d} to {end_d})")
    print("=" * 95)
    configs = [
        ("1. Baseline (Core Meta + Cointegrated Pairs)", False, 0.0),
        ("2. Anomaly Test Mode ($15 Max Notional)", True, 15.0),
        ("3. Anomaly Prod Mode ($100 Max Notional)", True, 100.0),
    ]
    print(f"  {'Configuration':<48} | {'Net Ret':>9} | {'Sharpe':>7} | {'Max DD':>8} | {'Final Equity':>12}")
    print("-" * 95)
    for label, anom_on, anom_sz in configs:
        res = run_simulation(
            strategy_name="meta",
            bars_dict=bars_dict,
            funding_dict=funding_dict,
            initial_capital=args.capital,
            rebalance_freq_bars=args.rebalance_freq,
            execution_mode=args.execution_mode,
            turnover_deadband=args.turnover_deadband,
            start_date=start_d,
            end_date=end_d,
            enable_pairs=True,
            enable_anomalies=anom_on,
            anomaly_size=anom_sz,
        )
        final_eq = float(res.equity_curve.iloc[-1])
        print(
            f"  {label:<48} | {res.total_return_pct:+8.2f}% | {res.sharpe_ratio:7.2f} | "
            f"{res.max_drawdown_pct:7.2f}% | ${final_eq:>11,.2f}"
        )
    print("=" * 95 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Institutional Multi-Strategy & Multi-Regime Crypto Backtester"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "6m", "anomalies", "multi-period", "ablation"],
        default="standard",
        help="Execution mode: standard (single run), 6m (6-month report), anomalies (3-way sizing), multi-period (4-cycle stress test), or ablation (7-Factor shields).",
    )
    parser.add_argument(
        "--strategy",
        choices=["seven_factor", "meta", "momentum", "stat_arb", "carry", "anomalies"],
        default="seven_factor",
        help="Strategy to evaluate (default: seven_factor).",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="top15",
        help="Asset universe preset (default: top15).",
    )
    parser.add_argument(
        "--symbols", type=str, default=None, help="Comma-separated symbol override (e.g. BTCUSDT,ETHUSDT)."
    )
    parser.add_argument(
        "--source",
        choices=["cache", "drive", "synthetic"],
        default="cache",
        help="Data source: cache (local /tmp parquet cache), drive (sync via rclone), or synthetic.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/crypto_alpha_backtest_cache", help="Path to local Parquet cache."
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=8,
        help="Rebalancing frequency in 1h bars (default: 8h funding-aligned UTC).",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["taker", "maker", "hybrid"],
        default="hybrid",
        help="Execution mode (maker=0.02%% passive, taker=0.05%% market+slip, hybrid=75%% maker / 25%% taker).",
    )
    parser.add_argument(
        "--turnover-deadband", type=float, default=0.04, help="Turnover deadband threshold (default: 0.04)."
    )
    parser.add_argument(
        "--deadband-mult",
        type=float,
        default=1.25,
        help="Proportional deadband multiplier for seven_factor (default: 1.25).",
    )
    parser.add_argument(
        "--bhs-enter-top-k",
        type=int,
        default=3,
        help="Novy-Marx Buy/Hold Spread entry rank cutoff (default: 3).",
    )
    parser.add_argument(
        "--bhs-hold-top-k",
        type=int,
        default=6,
        help="Novy-Marx Buy/Hold Spread hold rank buffer (default: 6).",
    )
    parser.add_argument(
        "--enable-btc-lead-lag",
        action="store_true",
        default=False,
        help="Enable optional 6h BTC -> Alt lead-lag catch-up tilt inside seven_factor.",
    )
    parser.add_argument(
        "--capital", type=float, default=3000.0, help="Initial portfolio capital in USDT (default: 3000.0)."
    )
    parser.add_argument("--start", type=str, default=None, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--start-leverage",
        type=float,
        default=2.5,
        help="Initial gross leverage tier for seven_factor (default: 2.5x).",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=5.0,
        help="Maximum stepped leverage tier for seven_factor (default: 5.0x).",
    )
    parser.add_argument("--enable-pairs", action="store_true", help="Include Cointegrated Pairs in Meta-Strategy.")
    parser.add_argument(
        "--enable-anomalies", action="store_true", help="Include Extreme Anomaly overlays in Meta-Strategy."
    )
    parser.add_argument(
        "--anomaly-size", type=float, default=15.0, help="Max notional USD per anomaly trade (default: 15.0)."
    )
    parser.add_argument(
        "--disable-chop-arb",
        dest="enable_chop_arb",
        action="store_false",
        default=True,
        help="Disable Chop Range-Arb F5 compression.",
    )
    parser.add_argument(
        "--disable-holistic-shields",
        dest="enable_holistic_shields",
        action="store_false",
        default=True,
        help="Disable Holistic Regime Shields 1A & 1B.",
    )
    parser.add_argument(
        "--enable-dollar-bar-tsmom",
        action="store_true",
        default=False,
        help="Enable Phase 1 Manga P2 Causal Dollar-Bar TSMOM + Taker-CVD sleeve.",
    )
    parser.add_argument(
        "--enable-event-snipers",
        action="store_true",
        default=False,
        help="Enable Phase 2 Event-Driven Microstructure Snipers (M09, M23, M04, M25).",
    )
    parser.add_argument(
        "--enable-m16-post-settle",
        dest="enable_post_settle_sniper",
        action="store_true",
        default=True,
        help="Enable M16-6 Post-Settlement Negative-Funding Unwind Sniper M25 when Event Snipers are active (default: True).",
    )
    parser.add_argument(
        "--disable-m16-post-settle",
        dest="enable_post_settle_sniper",
        action="store_false",
        help="Disable M16-6 Post-Settlement Negative-Funding Unwind Sniper M25.",
    )
    parser.add_argument(
        "--sniper-slot-leverage",
        type=float,
        default=0.24,
        help="Base equity fraction per Phase 2 Sniper slot (default: 0.24).",
    )
    parser.add_argument(
        "--sniper-max-positions",
        type=int,
        default=3,
        help="Maximum concurrent Phase 2 Sniper positions (default: 3).",
    )
    parser.add_argument(
        "--cushion-floor-usd",
        type=float,
        default=None,
        help="Enable Phase 1 Cushion-Safe CPPI equity floor protection (e.g. 2500.0).",
    )
    parser.add_argument(
        "--cushion-buffer-usd",
        type=float,
        default=240.0,
        help="Buffer zone above cushion-floor-usd for smooth CPPI de-risking (default: 240.0).",
    )
    parser.add_argument(
        "--recovery-boost-equity",
        type=float,
        default=None,
        help="Two-Phase SafeRamp HWM threshold (e.g. 3050.0).",
    )
    parser.add_argument(
        "--recovery-boost-leverage",
        type=float,
        default=None,
        help="Two-Phase SafeRamp unlocked base leverage (e.g. 3.2).",
    )
    parser.add_argument("--report", default=None, help="Output path for HTML tearsheet report.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "6m":
        if not args.start:
            args.start = "2026-04-01"
        if not args.end:
            args.end = "2026-09-30"
        args.enable_pairs = True
    elif args.mode == "anomalies":
        if args.preset == "top15" and not args.symbols:
            args.preset = "anomalies23"

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else PRESETS[args.preset]
    report_file = (
        Path(args.report)
        if args.report
        else REPORTS_DIR / f"tearsheet_{args.strategy}_{args.preset}_{args.execution_mode}.html"
    )

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: UNIFIED INSTITUTIONAL BACKTEST RUNNER")
    print("=" * 70)
    print(f"  Mode / Strategy:  {args.mode.upper()} / {args.strategy.upper()}")
    print(f"  Universe:         {args.preset.upper()} ({len(symbols)} symbols: {', '.join(symbols[:5])}...)")
    print(f"  Source:           {args.source.upper()} ({args.cache_dir})")
    print(
        f"  Execution Mode:   {args.execution_mode.upper()} ({'0.02% passive' if args.execution_mode == 'maker' else '0.04% + 1bp taker'})"
    )
    print(f"  Initial Capital:  ${args.capital:,.2f} | Rebalance: Every {args.rebalance_freq}h")
    print("-" * 70)

    if args.source in ("cache", "drive"):
        bars_dict, funding_dict = load_drive_market_data(
            symbols=symbols,
            cache_dir=Path(args.cache_dir),
            sync_drive=(args.source == "drive"),
        )
        if not bars_dict:
            print("⚠️  No cached/Drive data found. Falling back to synthetic simulation.")
            bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)
    else:
        bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)

    if args.mode == "ablation":
        run_ablation_suite(bars_dict, funding_dict, args)
        return
    if args.mode == "multi-period":
        run_multi_period_suite(bars_dict, funding_dict, args)
        return
    if args.mode == "anomalies":
        run_anomalies_comparison_suite(bars_dict, funding_dict, args)
        return

    result = run_simulation(
        strategy_name=args.strategy,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        initial_capital=args.capital,
        rebalance_freq_bars=args.rebalance_freq,
        execution_mode=args.execution_mode,
        turnover_deadband=args.turnover_deadband,
        start_date=args.start,
        end_date=args.end,
        enable_pairs=args.enable_pairs,
        enable_anomalies=args.enable_anomalies,
        anomaly_size=args.anomaly_size,
        enable_chop_arb=args.enable_chop_arb,
        enable_holistic_shields=args.enable_holistic_shields,
        start_leverage=args.start_leverage,
        max_leverage=args.max_leverage,
        bhs_enter_top_k=args.bhs_enter_top_k,
        bhs_hold_top_k=args.bhs_hold_top_k,
        deadband_mult=args.deadband_mult,
        enable_btc_lead_lag=args.enable_btc_lead_lag,
        enable_dollar_bar_tsmom=args.enable_dollar_bar_tsmom,
        enable_event_snipers=args.enable_event_snipers,
        enable_post_settle_sniper=args.enable_post_settle_sniper,
        sniper_slot_leverage=args.sniper_slot_leverage,
        sniper_max_positions=args.sniper_max_positions,
        cushion_floor_usd=args.cushion_floor_usd,
        cushion_buffer_usd=args.cushion_buffer_usd,
        recovery_boost_equity=args.recovery_boost_equity,
        recovery_boost_leverage=args.recovery_boost_leverage,
    )

    print("\n📈 Portfolio Equity Curve:")
    print(format_ascii_chart(result.equity_curve))
    print_performance_table(result, args.strategy, args.preset, args.execution_mode, args.turnover_deadband)
    generate_plotly_tearsheet(result, args.strategy, args.preset, report_file)


if __name__ == "__main__":
    main()
