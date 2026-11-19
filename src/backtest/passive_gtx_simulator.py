"""
Conservative 1-Minute Passive Post-Only (GTX) Wick-Sniper & Queue-Penetration Simulator (Issue #69 / M16-3).

Eliminates optimistic passive-limit backtest illusions ("touch = 100% fill", "same-bar TP before SL",
"Maker rebate on marketable cross") by enforcing 4 strict institutional anti-optimism guarantees:

  1. Strict Price-Through Queue Penetration (`queue_penetration_bps`, e.g. 1.0 - 3.0 bps):
     Exact touch (`low == limit_price` for BUY, `high == limit_price` for SELL) NEVER fills.
     Price must trade strictly through `limit_price` by `queue_penetration_bps`, and bar quote volume
     must support `order_notional_usd <= max_bar_participation * quote_volume`.
  2. Post-Only (`GTX`) Book-Crossing Check at Submission (`open[t + 1]`):
     If `limit_price` would cross `open[t + 1]`, `GTX` either strictly rejects (`gtx_cross_policy='reject'`)
     or re-anchors `gtx_reprice_offset_bps` passive of `open[t + 1]` (`gtx_cross_policy='reprice'`).
  3. Deterministic 100% Worst-Case Intrabar Stop Priority (Adverse Selection Haircut):
     - On the entry fill bar `m_fill`: if `stop_loss` is also breached on `m_fill`, the trade ALWAYS
       fills and immediately stops out (`STOP_LOSS_SAME_BAR`), ignoring any high wick to `take_profit`.
     - On holding bars `m > m_fill`: if both `stop_loss` and `take_profit` are touched on bar `m`,
       `stop_loss` ALWAYS triggers first (`100%` pessimistic ordering).
  4. Short Order TTL (`ttl_minutes` in 1..5m) & Maker/Taker Fee Asymmetry:
     - Unfilled orders auto-cancel after `ttl_minutes`.
     - Passive `GTX` entries & passive `LIMIT` Take-Profits pay `maker_fee_bps` (`2.0 bps` default).
     - `STOP_LOSS` and `TIME_STOP` exits pay `taker_stop_fee_bps + stop_slippage_bps` (`5.0 + 1.0 = 6.0 bps`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PassiveGTXConfig:
    """Configuration for adverse-selection-aware 1m Post-Only (`GTX`) simulation."""

    queue_penetration_bps: float = 2.0
    tp_penetration_bps: float = 1.0
    maker_fee_bps: float = 2.0
    taker_stop_fee_bps: float = 5.0
    stop_slippage_bps: float = 1.0
    ttl_minutes: int = 3
    max_hold_minutes: int = 45
    gtx_cross_policy: Literal["reject", "reprice"] = "reject"
    gtx_reprice_offset_bps: float = 1.5
    order_notional_usd: float = 600.0
    max_bar_participation: float = 0.02
    profit_lock_trigger_pct: float | None = None
    profit_lock_level_pct: float | None = None


@dataclass(frozen=True)
class GTXSignalEvent:
    """Represents a causal signal emitted at close of bar index `signal_idx`."""

    signal_idx: int
    side: int  # +1 for LONG (BUY LIMIT), -1 for SHORT (SELL LIMIT)
    limit_offset_bps: float  # passive offset from `close[signal_idx]` (>0 places inside book below/above ref)
    take_profit_pct: float
    stop_loss_pct: float
    symbol: str = "UNKNOWN"
    signal_score: float = 1.0


class PassiveGTXExecutionSimulator:
    """
    Fast NumPy-accelerated 1-minute Post-Only (`GTX`) execution & bracket lifecycle simulator.
    """

    def __init__(self, config: PassiveGTXConfig | None = None) -> None:
        self.config = config or PassiveGTXConfig()

    def simulate_signals_on_1m_bars(
        self,
        bars_1m: pd.DataFrame,
        events: list[GTXSignalEvent],
        min_cooldown_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Simulates a sequence of causal `GTXSignalEvent`s on a single symbol's `1m` OHLCV DataFrame.
        Enforces strict non-overlapping execution per symbol (`signal_idx >= next_free_idx`).
        """
        if bars_1m.empty or not events:
            return pd.DataFrame()

        cfg = self.config
        ts_arr = bars_1m["timestamp"].to_numpy(dtype=np.int64)
        open_arr = bars_1m["open"].to_numpy(dtype=np.float64)
        high_arr = bars_1m["high"].to_numpy(dtype=np.float64)
        low_arr = bars_1m["low"].to_numpy(dtype=np.float64)
        close_arr = bars_1m["close"].to_numpy(dtype=np.float64)
        qvol_arr = (
            bars_1m["quote_volume"].to_numpy(dtype=np.float64)
            if "quote_volume" in bars_1m.columns
            else np.full(len(bars_1m), 1e9, dtype=np.float64)
        )
        n_bars = len(bars_1m)

        q_pen = cfg.queue_penetration_bps / 10_000.0
        tp_pen = cfg.tp_penetration_bps / 10_000.0
        maker_fee = cfg.maker_fee_bps / 10_000.0
        taker_exit_cost = (cfg.taker_stop_fee_bps + cfg.stop_slippage_bps) / 10_000.0
        reprice_off = cfg.gtx_reprice_offset_bps / 10_000.0
        min_req_qvol = cfg.order_notional_usd / max(1e-6, cfg.max_bar_participation)

        sorted_events = sorted(events, key=lambda e: e.signal_idx)
        next_free_idx = 0
        records: list[dict[str, Any]] = []

        for ev in sorted_events:
            sig_i = int(ev.signal_idx)
            if sig_i < next_free_idx or sig_i + 1 >= n_bars:
                continue

            side = 1 if ev.side > 0 else -1
            ref_close = close_arr[sig_i]
            if ref_close <= 0.0 or not np.isfinite(ref_close):
                continue

            off_frac = ev.limit_offset_bps / 10_000.0
            raw_limit = ref_close * (1.0 - off_frac) if side == 1 else ref_close * (1.0 + off_frac)

            sub_i = sig_i + 1
            sub_open = open_arr[sub_i]
            crosses_book = (side == 1 and raw_limit >= sub_open) or (side == -1 and raw_limit <= sub_open)

            gtx_repriced = False
            if crosses_book:
                if cfg.gtx_cross_policy == "reject":
                    records.append(
                        {
                            "symbol": ev.symbol,
                            "signal_ts": int(ts_arr[sig_i]),
                            "entry_ts": 0,
                            "exit_ts": 0,
                            "side": side,
                            "status": "GTX_REJECTED",
                            "exit_reason": "GTX_REJECTED",
                            "filled": False,
                            "gtx_repriced": False,
                            "limit_price": raw_limit,
                            "entry_price": 0.0,
                            "exit_price": 0.0,
                            "hold_minutes": 0,
                            "gross_ret_bps": 0.0,
                            "fee_bps": 0.0,
                            "net_ret_bps": 0.0,
                            "signal_score": ev.signal_score,
                        }
                    )
                    next_free_idx = sub_i + 1
                    continue
                gtx_repriced = True
                limit_px = sub_open * (1.0 - reprice_off) if side == 1 else sub_open * (1.0 + reprice_off)
            else:
                limit_px = raw_limit

            # Scan TTL window `[sub_i .. min(n_bars - 1, sub_i + ttl_minutes - 1)]` for strict price-through fill
            ttl_end = min(n_bars, sub_i + max(1, int(cfg.ttl_minutes)))
            fill_i = -1
            req_px = limit_px * (1.0 - q_pen) if side == 1 else limit_px * (1.0 + q_pen)

            for m in range(sub_i, ttl_end):
                if qvol_arr[m] < min_req_qvol:
                    continue
                if (side == 1 and low_arr[m] <= req_px) or (side == -1 and high_arr[m] >= req_px):
                    fill_i = m
                    break

            if fill_i < 0:
                records.append(
                    {
                        "symbol": ev.symbol,
                        "signal_ts": int(ts_arr[sig_i]),
                        "entry_ts": 0,
                        "exit_ts": int(ts_arr[ttl_end - 1]),
                        "side": side,
                        "status": "TTL_EXPIRED",
                        "exit_reason": "TTL_EXPIRED",
                        "filled": False,
                        "gtx_repriced": gtx_repriced,
                        "limit_price": limit_px,
                        "entry_price": 0.0,
                        "exit_price": 0.0,
                        "hold_minutes": 0,
                        "gross_ret_bps": 0.0,
                        "fee_bps": 0.0,
                        "net_ret_bps": 0.0,
                        "signal_score": ev.signal_score,
                    }
                )
                next_free_idx = ttl_end
                continue

            entry_px = limit_px
            tp_px = entry_px * (1.0 + ev.take_profit_pct) if side == 1 else entry_px * (1.0 - ev.take_profit_pct)
            sl_px = entry_px * (1.0 - ev.stop_loss_pct) if side == 1 else entry_px * (1.0 + ev.stop_loss_pct)
            tp_req_px = tp_px * (1.0 + tp_pen) if side == 1 else tp_px * (1.0 - tp_pen)

            exit_i = fill_i
            exit_px = close_arr[fill_i]
            exit_reason = "TIME_STOP"
            exit_cost = taker_exit_cost

            # Check #3A: Same-bar worst-case adverse selection check on `fill_i` (NEVER allow same-bar TP!)
            same_bar_sl = (side == 1 and low_arr[fill_i] <= sl_px) or (side == -1 and high_arr[fill_i] >= sl_px)
            if same_bar_sl:
                exit_i = fill_i
                exit_px = sl_px
                exit_reason = "STOP_LOSS_SAME_BAR"
                exit_cost = taker_exit_cost
            else:
                hold_end = min(n_bars - 1, fill_i + max(1, int(cfg.max_hold_minutes)))
                cur_sl = sl_px
                peak_fav = close_arr[fill_i]

                for h in range(fill_i + 1, hold_end + 1):
                    # 1. Always evaluate Stop-Loss FIRST on bar `h` (100% pessimistic same-bar ordering)
                    hit_sl = (side == 1 and low_arr[h] <= cur_sl) or (side == -1 and high_arr[h] >= cur_sl)
                    if hit_sl:
                        exit_i = h
                        exit_px = cur_sl
                        exit_reason = (
                            "PROFIT_LOCK_SL"
                            if ((side == 1 and cur_sl > entry_px) or (side == -1 and cur_sl < entry_px))
                            else "STOP_LOSS"
                        )
                        exit_cost = taker_exit_cost
                        break

                    # 2. Evaluate passive Maker Take-Profit with strict price-through
                    hit_tp = (side == 1 and high_arr[h] >= tp_req_px) or (side == -1 and low_arr[h] <= tp_req_px)
                    if hit_tp:
                        exit_i = h
                        exit_px = tp_px
                        exit_reason = "TAKE_PROFIT_MAKER"
                        exit_cost = maker_fee
                        break

                    # 3. End-of-bar causal Profit-Lock ratchet for subsequent bars `h + 1..`
                    if cfg.profit_lock_trigger_pct and cfg.profit_lock_level_pct:
                        if side == 1:
                            peak_fav = max(peak_fav, high_arr[h])
                            if (peak_fav - entry_px) / entry_px >= cfg.profit_lock_trigger_pct:
                                lock_px = entry_px * (1.0 + cfg.profit_lock_level_pct)
                                if close_arr[h] > lock_px and lock_px > cur_sl:
                                    cur_sl = lock_px
                        else:
                            peak_fav = min(peak_fav, low_arr[h])
                            if (entry_px - peak_fav) / entry_px >= cfg.profit_lock_trigger_pct:
                                lock_px = entry_px * (1.0 - cfg.profit_lock_level_pct)
                                if close_arr[h] < lock_px and lock_px < cur_sl:
                                    cur_sl = lock_px

                    if h == hold_end:
                        exit_i = h
                        exit_px = close_arr[h]
                        exit_reason = "TIME_STOP"
                        exit_cost = taker_exit_cost

            gross_ret = ((exit_px - entry_px) / entry_px) if side == 1 else ((entry_px - exit_px) / entry_px)
            total_fee = maker_fee + exit_cost
            net_ret = gross_ret - total_fee

            records.append(
                {
                    "symbol": ev.symbol,
                    "signal_ts": int(ts_arr[sig_i]),
                    "entry_ts": int(ts_arr[fill_i]),
                    "exit_ts": int(ts_arr[exit_i]),
                    "side": side,
                    "status": "FILLED",
                    "exit_reason": exit_reason,
                    "filled": True,
                    "gtx_repriced": gtx_repriced,
                    "limit_price": limit_px,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "hold_minutes": int(exit_i - fill_i),
                    "gross_ret_bps": round(gross_ret * 1e4, 3),
                    "fee_bps": round(total_fee * 1e4, 3),
                    "net_ret_bps": round(net_ret * 1e4, 3),
                    "signal_score": ev.signal_score,
                }
            )
            next_free_idx = exit_i + max(1, int(min_cooldown_minutes))

        return pd.DataFrame(records)

    @staticmethod
    def summarize_simulation(trades_df: pd.DataFrame) -> dict[str, Any]:
        """Computes fill rate, GTX rejection rate, same-bar adverse stop rate, net bps, and Sharpe."""
        if trades_df is None or trades_df.empty:
            return {
                "total_signals": 0,
                "filled_trades": 0,
                "fill_rate_pct": 0.0,
                "gtx_reject_rate_pct": 0.0,
                "same_bar_sl_rate_pct": 0.0,
                "win_rate_pct": 0.0,
                "mean_gross_bps": 0.0,
                "mean_fee_bps": 0.0,
                "mean_net_bps": 0.0,
                "cum_net_bps": 0.0,
                "trade_tstat": 0.0,
                "annualized_sharpe": 0.0,
            }

        n_sig = len(trades_df)
        filled = trades_df[trades_df["filled"]].copy()
        n_fill = len(filled)
        n_rej = int((trades_df["status"] == "GTX_REJECTED").sum())

        if n_fill == 0:
            return {
                "total_signals": n_sig,
                "filled_trades": 0,
                "fill_rate_pct": 0.0,
                "gtx_reject_rate_pct": round(100.0 * n_rej / n_sig, 2),
                "same_bar_sl_rate_pct": 0.0,
                "win_rate_pct": 0.0,
                "mean_gross_bps": 0.0,
                "mean_fee_bps": 0.0,
                "mean_net_bps": 0.0,
                "cum_net_bps": 0.0,
                "trade_tstat": 0.0,
                "annualized_sharpe": 0.0,
            }

        same_bar_sl = int((filled["exit_reason"] == "STOP_LOSS_SAME_BAR").sum())
        net_bps = filled["net_ret_bps"].to_numpy(dtype=np.float64)
        gross_bps = filled["gross_ret_bps"].to_numpy(dtype=np.float64)
        fee_bps = filled["fee_bps"].to_numpy(dtype=np.float64)

        mean_net = float(net_bps.mean())
        std_net = float(net_bps.std(ddof=1)) if n_fill > 2 else 1.0
        tstat = (mean_net / (std_net / np.sqrt(n_fill))) if (n_fill > 2 and std_net > 1e-9) else 0.0

        # Daily aggregated Sharpe across active trading span
        filled["_day"] = pd.to_datetime(filled["entry_ts"], unit="s", utc=True).dt.floor("D")
        min_d = filled["_day"].min()
        max_d = filled["_day"].max()
        all_days = pd.date_range(min_d, max_d, freq="D", tz="UTC") if pd.notna(min_d) else pd.DatetimeIndex([])
        if len(all_days) >= 5:
            daily_bps = filled.groupby("_day")["net_ret_bps"].sum().reindex(all_days, fill_value=0.0)
            d_std = float(daily_bps.std(ddof=1))
            ann_sharpe = float((daily_bps.mean() / d_std) * np.sqrt(365.25)) if d_std > 1e-9 else 0.0
        else:
            ann_sharpe = 0.0

        return {
            "total_signals": n_sig,
            "filled_trades": n_fill,
            "fill_rate_pct": round(100.0 * n_fill / n_sig, 2),
            "gtx_reject_rate_pct": round(100.0 * n_rej / n_sig, 2),
            "same_bar_sl_rate_pct": round(100.0 * same_bar_sl / n_fill, 2),
            "win_rate_pct": round(100.0 * float((net_bps > 0.0).mean()), 2),
            "mean_gross_bps": round(float(gross_bps.mean()), 2),
            "mean_fee_bps": round(float(fee_bps.mean()), 2),
            "mean_net_bps": round(mean_net, 2),
            "cum_net_bps": round(float(net_bps.sum()), 1),
            "trade_tstat": round(float(tstat), 3),
            "annualized_sharpe": round(ann_sharpe, 3),
        }

    @staticmethod
    def run_penetration_sensitivity_matrix(
        bars_by_symbol: dict[str, pd.DataFrame],
        events_by_symbol: dict[str, list[GTXSignalEvent]],
        penetration_bps_grid: tuple[float, ...] = (1.0, 2.0, 2.5, 3.0),
        ttl_grid: tuple[int, ...] = (2, 4),
        base_config: PassiveGTXConfig | None = None,
    ) -> pd.DataFrame:
        """
        Evaluates execution survival across a grid of `queue_penetration_bps` (1.0..3.0 bps)
        and `ttl_minutes` under 100% same-bar adverse stop ordering.
        """
        base = base_config or PassiveGTXConfig()
        rows: list[dict[str, Any]] = []
        for pen in penetration_bps_grid:
            for ttl in ttl_grid:
                cfg = PassiveGTXConfig(
                    queue_penetration_bps=pen,
                    tp_penetration_bps=base.tp_penetration_bps,
                    maker_fee_bps=base.maker_fee_bps,
                    taker_stop_fee_bps=base.taker_stop_fee_bps,
                    stop_slippage_bps=base.stop_slippage_bps,
                    ttl_minutes=ttl,
                    max_hold_minutes=base.max_hold_minutes,
                    gtx_cross_policy=base.gtx_cross_policy,
                    gtx_reprice_offset_bps=base.gtx_reprice_offset_bps,
                    order_notional_usd=base.order_notional_usd,
                    max_bar_participation=base.max_bar_participation,
                    profit_lock_trigger_pct=base.profit_lock_trigger_pct,
                    profit_lock_level_pct=base.profit_lock_level_pct,
                )
                sim = PassiveGTXExecutionSimulator(cfg)
                parts = [
                    sim.simulate_signals_on_1m_bars(bars_by_symbol[s], evs)
                    for s, evs in events_by_symbol.items()
                    if s in bars_by_symbol and evs
                ]
                non_empty = [p for p in parts if not p.empty]
                all_trades = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
                stats = PassiveGTXExecutionSimulator.summarize_simulation(all_trades)
                stats["queue_penetration_bps"] = pen
                stats["ttl_minutes"] = ttl
                rows.append(stats)
        return pd.DataFrame(rows)
