#!/usr/bin/env python3
"""
Pure Event-Driven Anomaly Strategies Backtest (6 Months).
Simulates:
1. Oversold Bounce Strategy (LONG on liquidations > 15% with RSI < 28, TP: +6%, SL: -5%)
2. Early Breakout Strategy (LONG on volume explosions > 2.5x with 4h return > 8%, TP: +15%, SL: -4%)
3. Pump Funding Fade Strategy (SHORT on blow-off tops > 25% with funding > 0.05%, TP: +8%, SL: -8%)

Evaluates both individual strategy performance and the combined pure anomaly portfolio on the 23-asset universe.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
)

cache_dir = Path("/tmp/crypto_alpha_backtest_cache")
bars_dir = cache_dir / "bars" / "1h"
funding_dir = cache_dir / "funding"

symbols = [
    "BTCUSDT", "ETHUSDT", "RENDERUSDT", "FETUSDT",
    "CTSIUSDT", "MAGICUSDT", "OGNUSDT", "RLCUSDT",
    "GTCUSDT", "COMPUSDT", "CRVUSDT", "DYDXUSDT",
    "WALUSDT", "BSPUSDT", "TRUTHUSDT", "ZEREBROUSDT",
    "BTWUSDT", "QUSDT", "AVAUSDT", "EPICUSDT", "MUBARAKUSDT", "CRWVUSDT", "CAPUSDT",
]

# Load 1h bars
bars = {}
funding = {}
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

# Common evaluation index: 2026-04-14 to 2026-10-14
eval_start = pd.Timestamp("2026-04-14", tz="UTC")
eval_end = pd.Timestamp("2026-10-14 23:00:00", tz="UTC")
anchor_idx = bars["BTCUSDT"].index
sim_idx = anchor_idx[(anchor_idx >= eval_start) & (anchor_idx <= eval_end)]

print(f"Loaded {len(bars)} symbols. Simulating {len(sim_idx)} hours ({len(sim_idx)/24:.1f} days)...")

# Instantiate strategies
strat_bounce = OversoldBounceStrategy(dump_threshold_24h=-0.15, rsi_threshold=28.0)
strat_breakout = EarlyBreakoutStrategy(pump_threshold_4h=0.08, volume_multiplier=2.5)
strat_fade = PumpFundingFadeStrategy(pump_threshold_24h=0.25, min_funding_rate=0.0005)

trades = []

# Simulate trade events hourly
for pos in range(30, len(sim_idx)):
    ts = sim_idx[pos]
    window_start = sim_idx[max(0, pos - 30)]

    sub_market = {}
    sub_funding = {}
    for s, df in bars.items():
        sub = df.loc[(df.index >= window_start) & (df.index <= ts)]
        if len(sub) >= 26:
            sub_market[s] = sub
        if s in funding:
            sub_funding[s] = funding[s].loc[funding[s].index <= ts]

    # 1. Bounce signals
    sigs_bounce = strat_bounce.generate_signals(sub_market)
    for sig in sigs_bounce:
        meta = sig.metadata
        entry_p = meta["entry_price"]
        sl_p = meta["stop_loss"]
        tp_p = meta["take_profit"]

        # Forward simulate up to 24h
        sym_df = bars[sig.symbol]
        future_bars = sym_df.loc[(sym_df.index > ts) & (sym_df.index <= ts + pd.Timedelta(hours=24))]
        if future_bars.empty:
            continue

        exit_p = future_bars["close"].iloc[-1]
        outcome = "TIME_EXIT"
        for _, bar in future_bars.iterrows():
            if bar["low"] <= sl_p:
                exit_p = sl_p
                outcome = "STOP_LOSS"
                break
            elif bar["high"] >= tp_p:
                exit_p = tp_p
                outcome = "TAKE_PROFIT"
                break

        pnl_pct = (exit_p / entry_p - 1.0)
        trades.append({
            "timestamp": ts,
            "strategy": "OversoldBounce",
            "symbol": sig.symbol,
            "side": "BUY",
            "entry_price": entry_p,
            "exit_price": exit_p,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        })

    # 2. Breakout signals
    sigs_break = strat_breakout.generate_signals(sub_market)
    for sig in sigs_break:
        meta = sig.metadata
        entry_p = meta["entry_price"]
        sl_p = meta["stop_loss"]
        tp_p = meta["take_profit"]

        sym_df = bars[sig.symbol]
        future_bars = sym_df.loc[(sym_df.index > ts) & (sym_df.index <= ts + pd.Timedelta(hours=12))]
        if future_bars.empty:
            continue

        exit_p = future_bars["close"].iloc[-1]
        outcome = "TIME_EXIT"
        for _, bar in future_bars.iterrows():
            if bar["low"] <= sl_p:
                exit_p = sl_p
                outcome = "STOP_LOSS"
                break
            elif bar["high"] >= tp_p:
                exit_p = tp_p
                outcome = "TAKE_PROFIT"
                break

        pnl_pct = (exit_p / entry_p - 1.0)
        trades.append({
            "timestamp": ts,
            "strategy": "EarlyBreakout",
            "symbol": sig.symbol,
            "side": "BUY",
            "entry_price": entry_p,
            "exit_price": exit_p,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        })

    # 3. Fade signals
    sigs_fade = strat_fade.generate_signals(sub_market, funding_data=sub_funding)
    for sig in sigs_fade:
        meta = sig.metadata
        entry_p = meta["entry_price"]
        sl_p = meta["stop_loss"]
        tp_p = meta["take_profit"]

        sym_df = bars[sig.symbol]
        future_bars = sym_df.loc[(sym_df.index > ts) & (sym_df.index <= ts + pd.Timedelta(hours=24))]
        if future_bars.empty:
            continue

        exit_p = future_bars["close"].iloc[-1]
        outcome = "TIME_EXIT"
        for _, bar in future_bars.iterrows():
            if bar["high"] >= sl_p:
                exit_p = sl_p
                outcome = "STOP_LOSS"
                break
            elif bar["low"] <= tp_p:
                exit_p = tp_p
                outcome = "TAKE_PROFIT"
                break

        # Short PnL
        pnl_pct = (entry_p / exit_p - 1.0)
        trades.append({
            "timestamp": ts,
            "strategy": "PumpFundingFade",
            "symbol": sig.symbol,
            "side": "SELL",
            "entry_price": entry_p,
            "exit_price": exit_p,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        })

df_trades = pd.DataFrame(trades)
print(f"\nTotal trade events executed: {len(df_trades)}")

for strat_name, grp in df_trades.groupby("strategy"):
    n_t = len(grp)
    win_t = (grp["pnl_pct"] > 0).sum()
    loss_t = (grp["pnl_pct"] < 0).sum()
    win_rate = win_t / n_t * 100.0 if n_t > 0 else 0.0
    mean_ret = grp["pnl_pct"].mean() * 100.0
    total_cum_ret = grp["pnl_pct"].sum() * 100.0
    tp_count = (grp["outcome"] == "TAKE_PROFIT").sum()
    sl_count = (grp["outcome"] == "STOP_LOSS").sum()
    time_count = (grp["outcome"] == "TIME_EXIT").sum()

    print(f"\n--- Strategy: {strat_name} ---")
    print(f"  Trades: {n_t} | Win Rate: {win_rate:.1f}% ({win_t}W / {loss_t}L)")
    print(f"  Avg Return per Trade: {mean_ret:+.2f}% | Cumulative Return: {total_cum_ret:+.2f}%")
    print(f"  Outcomes: TP={tp_count} ({tp_count/n_t*100:.1f}%), SL={sl_count} ({sl_count/n_t*100:.1f}%), TimeExit={time_count}")

# Portfolio Sizing Simulation ($15 vs $100 notional)
initial_capital = 1000.0
for notional in [15.0, 100.0]:
    df_trades[f"pnl_usd_{int(notional)}"] = df_trades["pnl_pct"] * notional - (notional * 0.0004) # 0.04% round-trip maker fee
    total_dollar_pnl = df_trades[f"pnl_usd_{int(notional)}"].sum()
    return_on_capital = (total_dollar_pnl / initial_capital) * 100.0
    print(f"\n=== Pure Anomaly Portfolio (${int(notional)} Sizing on $1,000 Capital) ===")
    print(f"  Total Dollar PnL: ${total_dollar_pnl:+,.2f} USDT")
    print(f"  Return on $1,000 Capital: {return_on_capital:+.2f}%")
