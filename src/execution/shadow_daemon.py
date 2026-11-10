"""
Autonomous Live Shadow Trading Daemon.
Connects to Binance USDS-M Futures Mainnet via free public read-only endpoints,
reconciles live mark-to-market prices, enforces passive Maker execution alpha,
triggers 8h multi-strategy rebalancing, and settles real funding cashflows in SQLite.
"""

import logging
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import ccxt
import numpy as np
import pandas as pd

from src.domain.entities import Order, Portfolio
from src.domain.enums import (
    OrderSide,
    OrderType,
    PositionSide,
)
from src.execution.execution_router import ExecutionUrgency, SmartExecutionRouter
from src.execution.rate_limiter import ExchangeRateLimiter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.ws_manager import PriceTick, ResilientWebSocketClient
from src.features.volatility import parkinson_volatility
from src.strategies.extreme_anomaly import compute_anomaly_confluence_size
from src.strategies.meta_allocator import MetaStrategyAllocator

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "SUI/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "RENDER/USDT",
    "FET/USDT",
]

ANOMALY_STRATEGIES: set[str] = {
    "Extreme Drop Bounce",
    "Early Breakout",
    "Pump & Funding Fade",
    "Extreme Funding Harvester",
    "Funding Harvester Hedge",
}

NON_CRYPTO_TRADFI_BASES: set[str] = {
    "AAOI", "AAPL", "AMD", "AMZN", "ANTHROPIC", "APP", "AXTI", "BE", "BILL",
    "BMNR", "BZ", "CBRS", "CHIP", "CL", "COHR", "COIN", "CONL", "CRCL", "CRWV",
    "CSOPSKHYNIX2L", "CXMT", "DELL", "EWZ", "FLNC", "GOOGL", "HOOD", "INTC",
    "INTW", "IREN", "KORU", "KSTR", "LITE", "META", "MINIMAX", "MRNA", "MRVL",
    "MSFT", "MSTR", "MSTU", "MSTZ", "MUU", "MVLL", "NATGAS", "NBIS", "NFLX",
    "NOW", "NVDA", "NVDL", "NVDS", "ORCL", "PLTR", "QCOM", "QQQ", "RKLB",
    "SAMSUNG", "SKDD", "SKHY", "SKHYNIX", "SKUU", "SMCI", "SNDK", "SNXX",
    "SOXL", "SOXS", "SPY", "SQQQ", "STXX", "TQQQ", "TSLA", "TSLL", "TSLS",
    "UNITREE", "USDBRL", "UVXY", "WDC", "XAG", "XAU", "ZHIPU", "ZHONGJI",
}


class ShadowTradingDaemon:
    """
    Multi-frequency shadow trading daemon operating on live mainnet data.
    """

    def __init__(
        self,
        broker: ShadowExecutionBroker,
        allocator: MetaStrategyAllocator | None = None,
        router: SmartExecutionRouter | None = None,
        exchange_client: ccxt.Exchange | None = None,
        symbols: list[str] | None = None,
        fast_interval_sec: float = 10.0,
        rebalance_interval_sec: float = 28_800.0,  # 8 hours
        min_rebalance_notional: float = 50.0,
        turnover_deadband: float = 0.04,          # 4% deadband to eliminate fee drag
        rate_limiter: ExchangeRateLimiter | None = None,
        ws_client: ResilientWebSocketClient | None = None,
        min_volume_24h_usd: float = 20_000_000.0,
        enable_market_scan: bool = True,
        enable_anomalies: bool = True,
        anomaly_size: float = 15.0,
        max_anomaly_positions: int = 3,
        profit_lock_trigger_pct: float = 0.035,
        profit_lock_level_pct: float = 0.015,
        trailing_trigger_pct: float = 0.045,
        trailing_distance_pct: float = 0.020,
        anomaly_profit_lock_trigger_pct: float = 0.045,
        anomaly_profit_lock_level_pct: float = 0.025,
        anomaly_trailing_trigger_pct: float = 0.060,
        anomaly_trailing_distance_pct: float = 0.020,
        anomaly_cooldown_win_sec: float = 12.0 * 3600.0,
        anomaly_cooldown_loss_sec: float = 24.0 * 3600.0,
        anomaly_max_hold_hours: float = 48.0,
        anomaly_breakout_min_24h_pct: float = 12.0,
        anomaly_min_notional: float = 10.0,
        anomaly_max_notional: float = 45.0,
        anomaly_pct_equity: float | None = 0.045,
        anomaly_prefer_maker: bool = True,
        enable_dynamic_anomaly_sizing: bool = True,
        harvester_min_abs_funding: float = 0.0015,
        enable_harvester_beta_hedge: bool = True,
        intracycle_drift_threshold_pct: float = 0.25,
        intracycle_min_rebalance_age_sec: float = 1800.0,
        intracycle_post_stop_settle_sec: float = 900.0,
        core_bracket_cooldown_sec: float = 4.0 * 3600.0,
        core_take_profit_pct: float = 0.080,
        cache_dir: str = "data/cache",
        enable_warm_start: bool = True,
        enable_subhourly_pullback: bool = True,
    ):
        self.broker = broker
        self.allocator = allocator or MetaStrategyAllocator()
        self.router = router or SmartExecutionRouter(broker=broker)
        self.symbols = symbols or DEFAULT_SHADOW_SYMBOLS
        self.fast_interval_sec = fast_interval_sec
        self.rebalance_interval_sec = rebalance_interval_sec
        self.min_rebalance_notional = min_rebalance_notional
        self.turnover_deadband = turnover_deadband
        self.rate_limiter = rate_limiter or ExchangeRateLimiter()
        self.ws_client = ws_client
        self.min_volume_24h_usd = min_volume_24h_usd
        self.enable_market_scan = enable_market_scan
        self.enable_anomalies = enable_anomalies
        self.anomaly_size = anomaly_size
        self.max_anomaly_positions: int = max_anomaly_positions
        self.profit_lock_trigger_pct = profit_lock_trigger_pct
        self.profit_lock_level_pct = profit_lock_level_pct
        self.trailing_trigger_pct = trailing_trigger_pct
        self.trailing_distance_pct = trailing_distance_pct
        self.anomaly_profit_lock_trigger_pct = anomaly_profit_lock_trigger_pct
        self.anomaly_profit_lock_level_pct = anomaly_profit_lock_level_pct
        self.anomaly_trailing_trigger_pct = anomaly_trailing_trigger_pct
        self.anomaly_trailing_distance_pct = anomaly_trailing_distance_pct
        self.anomaly_cooldown_win_sec = anomaly_cooldown_win_sec
        self.anomaly_cooldown_loss_sec = anomaly_cooldown_loss_sec
        self.anomaly_max_hold_hours = anomaly_max_hold_hours
        self.anomaly_breakout_min_24h_pct = anomaly_breakout_min_24h_pct
        self.anomaly_min_notional = anomaly_min_notional
        self.anomaly_max_notional = anomaly_max_notional
        self.anomaly_pct_equity = anomaly_pct_equity
        self.anomaly_prefer_maker = anomaly_prefer_maker
        self.enable_dynamic_anomaly_sizing = enable_dynamic_anomaly_sizing
        self.harvester_min_abs_funding = harvester_min_abs_funding
        self.enable_harvester_beta_hedge = enable_harvester_beta_hedge
        self._cached_universe_funding: dict[str, float] = {}
        self._last_universe_funding_ts: float = 0.0
        self.anomaly_last_exit: dict[str, tuple[datetime, bool]] = {}
        self.intracycle_drift_threshold_pct = intracycle_drift_threshold_pct
        self.intracycle_min_rebalance_age_sec = intracycle_min_rebalance_age_sec
        self.intracycle_post_stop_settle_sec = intracycle_post_stop_settle_sec
        self.core_bracket_cooldown_sec = core_bracket_cooldown_sec
        self.core_bracket_last_exit: dict[str, tuple[datetime, PositionSide]] = {}
        self.last_core_bracket_fill_time: datetime | None = None
        self.last_rest_tickers_ok: bool = True
        self._last_kline_fetch_ok: bool = True
        self.core_take_profit_pct = core_take_profit_pct
        self.cache_dir = Path(cache_dir)
        self.enable_warm_start = enable_warm_start
        self.enable_subhourly_pullback = enable_subhourly_pullback

        if self.ws_client:
            self.ws_client.on_tick_callback = self._on_ws_tick
            self.ws_client.rest_poll_callback = self.fetch_live_tickers

        # Free public read-only client for Binance Futures
        if exchange_client:
            self.exchange = exchange_client
        else:
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 20000,
                "options": {"defaultType": "future"},
            })

        self.is_running = False
        self._thread: threading.Thread | None = None
        self.last_rebalance_time: datetime | None = None
        db_inst = getattr(self.broker, "db", None)
        if db_inst:
            latest_reb = db_inst.get_latest_rebalance_snapshot()
            if latest_reb and latest_reb.get("timestamp"):
                try:
                    self.last_rebalance_time = datetime.fromisoformat(str(latest_reb["timestamp"]))
                except Exception:
                    self.last_rebalance_time = None
            if hasattr(db_inst, "get_latest_seven_factor_telemetry"):
                sf_tel = db_inst.get_latest_seven_factor_telemetry()
                lev_ctrl = getattr(self.allocator, "stepped_leverage_controller", None)
                if sf_tel and lev_ctrl and sf_tel.get("high_water_mark"):
                    tel_init = float(sf_tel.get("initial_capital") or lev_ctrl.initial_capital)
                    saved_hwm = float(sf_tel["high_water_mark"])
                    if abs(lev_ctrl.initial_capital - tel_init) > 1.0 and tel_init > 0:
                        saved_hwm = saved_hwm * (lev_ctrl.initial_capital / tel_init)
                        self.last_rebalance_time = None
                    lev_ctrl.high_water_mark = max(
                        lev_ctrl.high_water_mark,
                        saved_hwm,
                    )
            if getattr(self.broker, "capital_rescaled_ratio", None) is not None:
                self.last_rebalance_time = None
        self._refresh_anomaly_cooldowns()
        self.last_funding_check_time: datetime | None = None
        self.anomaly_scan_interval_sec: float = 30.0
        self.last_anomaly_scan_time: datetime | None = None
        self.latest_tickers: dict[str, dict[str, float]] = {}
        self.start_time = datetime.now(timezone.utc)
        self.total_ticks_processed = 0
        self.last_tick_time: datetime | None = None

    def _refresh_anomaly_cooldowns(self, filled_orders: list[Order] | None = None) -> None:
        """
        Hydrates and updates per-symbol anomaly exit timestamps and win/loss state,
        as well as Core bracket exit timestamps and exited position side (`core_bracket_last_exit`),
        from filled bracket orders and/or the SQLite trades table so cooldowns
        persist across fast cycles and daemon restarts.
        """
        now = datetime.now(timezone.utc)
        if filled_orders:
            for o in filled_orders:
                if o.order_type in (OrderType.TAKE_PROFIT, OrderType.STOP_LOSS):
                    clean_sym = o.symbol.replace("/", "")
                    if getattr(o, "strategy", None) in ANOMALY_STRATEGIES:
                        self.anomaly_last_exit[clean_sym] = (now, o.order_type == OrderType.STOP_LOSS)
                    else:
                        self.last_core_bracket_fill_time = now
                        exited_side = PositionSide.LONG if o.side == OrderSide.SELL else PositionSide.SHORT
                        self.core_bracket_last_exit[clean_sym] = (now, exited_side)

        db_inst = getattr(self.broker, "db", None)
        if db_inst and hasattr(db_inst, "get_trades"):
            try:
                recent_trades = db_inst.get_trades(limit=200)
                if isinstance(recent_trades, list):
                    for r in reversed(recent_trades):
                        if not isinstance(r, dict):
                            continue
                        strat = r.get("strategy")
                        rpnl = float(r.get("realized_pnl") or 0.0)
                        oid = str(r.get("order_id") or "")
                        ts_raw = r.get("timestamp")
                        if not ts_raw:
                            continue
                        clean_sym = str(r.get("symbol") or "").replace("/", "")
                        if not clean_sym:
                            continue
                        if strat in ANOMALY_STRATEGIES:
                            if abs(rpnl) > 1e-8 or oid.startswith(("shd_tp_", "shd_sl_")):
                                ts = datetime.fromisoformat(str(ts_raw))
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                was_loss = rpnl < -1e-8 if abs(rpnl) > 1e-8 else oid.startswith("shd_sl_")
                                self.anomaly_last_exit[clean_sym] = (ts, was_loss)
                        elif oid.startswith(("shd_tp_", "shd_sl_")):
                            ts = datetime.fromisoformat(str(ts_raw))
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            side_str = str(r.get("side") or "").upper()
                            exited_side = PositionSide.LONG if side_str == "SELL" else PositionSide.SHORT
                            self.core_bracket_last_exit[clean_sym] = (ts, exited_side)
                            if self.last_core_bracket_fill_time is None or ts > self.last_core_bracket_fill_time:
                                self.last_core_bracket_fill_time = ts
            except Exception:
                pass

    def _apply_core_bracket_cooldowns(
        self,
        target_dollars: dict[str, float],
        now: datetime | None = None,
    ) -> dict[str, float]:
        """
        Enforces a per-symbol same-direction cooldown (`core_bracket_cooldown_sec`, default 4h)
        on Core symbols that exited mid-cycle via Take-Profit or Trailing-Stop/Profit-Lock,
        and re-centers the remaining Long and Short legs so strict Dollar Neutrality
        (`sum(Longs) == sum(Shorts)`) is preserved without immediately re-entering a reversing asset.
        """
        if not target_dollars or self.core_bracket_cooldown_sec <= 0 or not self.core_bracket_last_exit:
            return target_dollars

        now_dt = now or datetime.now(timezone.utc)
        filtered = dict(target_dollars)
        blocked_any = False

        for sym_key, target_val in list(filtered.items()):
            if abs(target_val) <= 1e-9:
                continue
            clean_sym = sym_key.replace("/", "")
            if clean_sym not in self.core_bracket_last_exit:
                continue
            exit_ts, exited_side = self.core_bracket_last_exit[clean_sym]
            elapsed_sec = (now_dt - exit_ts).total_seconds()
            if 0.0 <= elapsed_sec < self.core_bracket_cooldown_sec:
                same_dir = (
                    (target_val > 0 and exited_side == PositionSide.LONG)
                    or (target_val < 0 and exited_side == PositionSide.SHORT)
                )
                if same_dir:
                    filtered[sym_key] = 0.0
                    blocked_any = True
                    logger.info(
                        f"⏳ Core Bracket Cooldown active for {clean_sym} "
                        f"({exited_side.value} exited {elapsed_sec / 60.0:.1f}m ago < "
                        f"{self.core_bracket_cooldown_sec / 3600.0:.1f}h): blocking same-direction "
                        f"re-entry (${target_val:+,.2f} -> $0.00)"
                    )

        if not blocked_any:
            return filtered

        long_sum = sum(v for v in filtered.values() if v > 0)
        short_sum = sum(abs(v) for v in filtered.values() if v < 0)
        if long_sum <= 1e-6 or short_sum <= 1e-6:
            return {k: 0.0 for k in filtered}

        balanced_side = min(long_sum, short_sum)
        long_scale = balanced_side / long_sum if long_sum > balanced_side else 1.0
        short_scale = balanced_side / short_sum if short_sum > balanced_side else 1.0

        for sym_key, val in list(filtered.items()):
            if val > 0 and long_scale < 1.0:
                filtered[sym_key] = val * long_scale
            elif val < 0 and short_scale < 1.0:
                filtered[sym_key] = val * short_scale

        return filtered

    def _on_ws_tick(self, tick: PriceTick) -> None:
        """Processes real-time WebSocket tick updates."""
        self.total_ticks_processed += 1
        self.last_tick_time = datetime.now(timezone.utc)
        # Normalize symbol formatting e.g. BTCUSDT -> BTC/USDT
        sym_slash = f"{tick.symbol[:-4]}/{tick.symbol[-4:]}" if tick.symbol.endswith("USDT") else tick.symbol
        prev = self.latest_tickers.get(sym_slash, {})
        self.latest_tickers[sym_slash] = {
            "price": tick.price,
            "bid": tick.bid,
            "ask": tick.ask,
            "high": max(tick.price, prev.get("high", tick.price)),
            "low": min(tick.price, prev.get("low", tick.price)),
            "change_24h": prev.get("change_24h", 0.0),
            "volume_24h": prev.get("volume_24h", 0.0),
        }
        self.broker.set_market_price(sym_slash, price=tick.price, bid=tick.bid, ask=tick.ask)

    def fetch_live_tickers(self) -> dict[str, dict[str, float]]:
        """
        Fetches current top-of-book L1 prices and 24h metrics from Binance Futures public API.
        If enable_market_scan is True, scans the entire Binance Futures market, filters by
        min_volume_24h_usd, and populates the Top Gainers, Top Losers, and monitored assets.
        Returns mapping: symbol -> {price, bid, ask, high, low, change_24h, volume_24h}
        """
        ticks: dict[str, dict[str, float]] = {}
        try:
            if self.rate_limiter:
                self.rate_limiter.acquire(weight=20 if self.enable_market_scan else 2, priority="LOW", wait=True)

            open_pos_syms: set[str] = set()
            open_pos_clean: set[str] = set()
            try:
                portfolio = self.broker.get_portfolio()
                for s, p in portfolio.positions.items():
                    if p.quantity > 0 and p.side != PositionSide.FLAT:
                        open_pos_syms.add(s)
                        open_pos_clean.add(s.replace("/", ""))
            except Exception:
                pass
            base_clean = {s.replace("/", "") for s in self.symbols}
            fallback_syms = list(dict.fromkeys(list(self.symbols) + list(open_pos_syms)))

            raw_tickers = {}
            if self.enable_market_scan:
                try:
                    raw_tickers = self.exchange.fetch_tickers()
                except Exception as e:
                    logger.debug(f"Global ticker fetch failed, falling back to symbols list: {e}")
                    raw_tickers = self.exchange.fetch_tickers(fallback_syms)
            else:
                raw_tickers = self.exchange.fetch_tickers(fallback_syms)

            if self.rate_limiter and hasattr(self.exchange, "last_response_headers"):
                self.rate_limiter.on_http_response(200, self.exchange.last_response_headers)

            parsed_universe: list[tuple[str, dict[str, float]]] = []

            for sym, t in raw_tickers.items():
                if "-" in sym:
                    continue
                norm_sym = sym.split(":")[0]
                if not (norm_sym.endswith("/USDT") or norm_sym.endswith("USDT")):
                    continue

                last = float(t.get("last") or t.get("close") or 0.0)
                bid = float(t.get("bid") or last)
                ask = float(t.get("ask") or last)
                high = float(t.get("high") or last)
                low = float(t.get("low") or last)
                pct = float(t.get("percentage") or 0.0)
                vol = float(t.get("quoteVolume") or 0.0)

                if last > 0:
                    data_dict = {
                        "price": last,
                        "bid": bid,
                        "ask": ask,
                        "low": low,
                        "high": high,
                        "high_24h": high,
                        "low_24h": low,
                        "change_24h": pct,
                        "volume_24h": vol,
                    }
                    parsed_universe.append((norm_sym, data_dict))

                    # Always track symbols in portfolio or base symbols list
                    clean_sym = norm_sym.replace("/", "")
                    if (
                        norm_sym in self.symbols
                        or clean_sym in base_clean
                        or norm_sym in open_pos_syms
                        or clean_sym in open_pos_clean
                    ):
                        ticks[norm_sym] = data_dict

            # Apply Volume Filter and extract Top Gainers & Losers across the whole exchange
            filtered = [x for x in parsed_universe if x[1]["volume_24h"] >= self.min_volume_24h_usd]
            filtered.sort(key=lambda x: x[1]["change_24h"], reverse=True)

            top_gainers = filtered[:10]
            top_losers = filtered[-10:] if len(filtered) >= 10 else []

            for norm_sym, d in top_gainers + top_losers:
                ticks[norm_sym] = d

            self.last_rest_tickers_ok = bool(raw_tickers)
        except Exception as e:  # noqa: BLE001
            self.last_rest_tickers_ok = False
            logger.warning(f"Error fetching live tickers from Binance: {e}")

        if ticks:
            fresh_clean = {k.replace("/", "") for k in ticks}
            keep_clean = fresh_clean | base_clean | open_pos_clean
            self.latest_tickers = {
                k: v for k, v in self.latest_tickers.items()
                if k.replace("/", "") in keep_clean
            }
            self.latest_tickers.update(ticks)
        return self.latest_tickers

    def fetch_symbol_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetches recent OHLCV bars for alpha calculation."""
        try:
            if self.rate_limiter:
                self.rate_limiter.acquire(weight=5, priority="LOW", wait=True)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if self.rate_limiter and hasattr(self.exchange, "last_response_headers"):
                self.rate_limiter.on_http_response(200, self.exchange.last_response_headers)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            self._last_kline_fetch_ok = not df.empty
            return df
        except Exception as e:  # noqa: BLE001
            self._last_kline_fetch_ok = False
            logger.warning(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normalize_timestamp_series(s: pd.Series) -> pd.Series:
        """Converts numeric (sec/ms) or string timestamps to UTC DatetimeIndex series."""
        if pd.api.types.is_numeric_dtype(s):
            max_val = float(s.dropna().max()) if len(s.dropna()) > 0 else 0.0
            unit = "ms" if max_val > 1e11 else "s"
            return pd.to_datetime(s, unit=unit, utc=True)
        return pd.to_datetime(s, utc=True)

    def fetch_warm_klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        target_bars: int = 1000,
    ) -> pd.DataFrame:
        """
        Loads local Parquet cache (`data/cache/bars/{timeframe}/{CLEAN_SYM}.parquet`) for
        instant 1000-bar warm-start, tops up missing live bars from Binance Futures API,
        and persists the updated cache atomically.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        parquet_path = self.cache_dir / "bars" / timeframe / f"{clean_sym}.parquet"
        cached_df = pd.DataFrame()

        if self.enable_warm_start and parquet_path.exists():
            try:
                raw_cache = pd.read_parquet(parquet_path)
                if not raw_cache.empty and "timestamp" in raw_cache.columns:
                    cached_df = raw_cache[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                    cached_df["timestamp"] = self._normalize_timestamp_series(cached_df["timestamp"])
                    cached_df = (
                        cached_df.drop_duplicates(subset=["timestamp"], keep="last")
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not read warm cache {parquet_path}: {e}")

        # Determine how many live bars to fetch from exchange
        tf_sec = 3600.0 if timeframe == "1h" else (900.0 if timeframe == "15m" else 3600.0)
        if not cached_df.empty:
            last_ts = cached_df["timestamp"].iloc[-1]
            now_utc = datetime.now(timezone.utc)
            gap_bars = int(max(2.0, (now_utc - last_ts).total_seconds() / tf_sec + 3.0))
            fetch_limit = min(target_bars, max(10, gap_bars))
        else:
            fetch_limit = target_bars

        live_df = self.fetch_symbol_klines(symbol, timeframe=timeframe, limit=fetch_limit)

        if not cached_df.empty and not live_df.empty:
            merged = pd.concat([cached_df, live_df], ignore_index=True)
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
            merged = (
                merged.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        elif not live_df.empty:
            merged = live_df.sort_values("timestamp").reset_index(drop=True)
        else:
            merged = cached_df

        # Persist updated cache atomically if not using a mock exchange
        if (
            self.enable_warm_start
            and not merged.empty
            and not isinstance(self.exchange, MagicMock)
            and self.cache_dir.exists()
        ):
            try:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                to_save = merged.tail(max(target_bars, 2000)).copy()
                to_save.to_parquet(parquet_path, index=False)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not persist warm cache {parquet_path}: {e}")

        return merged.tail(target_bars).reset_index(drop=True)

    def fetch_funding_history_72h(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the trailing 72h (9 settlements) of perpetual funding rates for `symbol`,
        combining local cache (`data/cache/FUNDING/{CLEAN_SYM}.parquet`) with live API data.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        records: list[dict[str, Any]] = []

        fund_path = self.cache_dir / "FUNDING" / f"{clean_sym}.parquet"
        if self.enable_warm_start and fund_path.exists():
            try:
                f_cache = pd.read_parquet(fund_path).tail(15)
                if not f_cache.empty and "funding_rate" in f_cache.columns:
                    ts_s = self._normalize_timestamp_series(f_cache["timestamp"])
                    for ts_val, fr_val in zip(ts_s, f_cache["funding_rate"]):
                        records.append({
                            "timestamp": ts_val,
                            "fundingRate": float(fr_val),
                            "funding_rate": float(fr_val),
                        })
            except Exception:
                pass

        # Fetch live funding rate history or current funding rate
        try:
            if hasattr(self.exchange, "fetch_funding_rate_history") and not isinstance(
                getattr(self.exchange, "fetch_funding_rate_history", None), MagicMock
            ):
                hist = self.exchange.fetch_funding_rate_history(symbol, limit=9)
                if isinstance(hist, list):
                    for item in hist:
                        if item and item.get("fundingRate") is not None:
                            ts_ms = item.get("timestamp")
                            ts_dt = (
                                pd.to_datetime(ts_ms, unit="ms", utc=True)
                                if ts_ms
                                else datetime.now(timezone.utc)
                            )
                            f_r = float(item["fundingRate"])
                            records.append({
                                "timestamp": ts_dt,
                                "fundingRate": f_r,
                                "funding_rate": f_r,
                            })
        except Exception:
            pass

        try:
            fr = self.exchange.fetch_funding_rate(symbol)
            if isinstance(fr, dict) and fr.get("fundingRate") is not None:
                f_r = float(fr["fundingRate"])
                records.append({
                    "timestamp": datetime.now(timezone.utc),
                    "fundingRate": f_r,
                    "funding_rate": f_r,
                })
        except Exception:
            pass

        if not records:
            return pd.DataFrame()

        df_f = pd.DataFrame(records)
        df_f["timestamp"] = pd.to_datetime(df_f["timestamp"], utc=True)
        df_f = (
            df_f.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df_f

    def fetch_open_interest_history_24h(self, symbol: str, limit: int = 36) -> pd.DataFrame:
        """
        Fetches trailing hourly Open Interest history (`sumOpenInterest`) for `symbol`
        (Issue #46), combining local cache (`data/cache/open_interest/1h/{CLEAN_SYM}.parquet`)
        with live Binance Futures `/futures/data/openInterestHist`.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        oi_path = self.cache_dir / "open_interest" / "1h" / f"{clean_sym}.parquet"
        cached_df = pd.DataFrame()

        if self.enable_warm_start and oi_path.exists():
            try:
                raw_oi = pd.read_parquet(oi_path).tail(max(limit, 72))
                if not raw_oi.empty and "timestamp" in raw_oi.columns and "sumOpenInterest" in raw_oi.columns:
                    cached_df = raw_oi[["timestamp", "sumOpenInterest"]].copy()
                    cached_df["timestamp"] = self._normalize_timestamp_series(cached_df["timestamp"])
                    cached_df["sumOpenInterest"] = cached_df["sumOpenInterest"].astype(float)
            except Exception:
                pass

        live_records: list[dict[str, Any]] = []
        try:
            fn = getattr(self.exchange, "fapiDataGetOpenInterestHist", None)
            if callable(fn) and not isinstance(fn, MagicMock):
                rows = fn({"symbol": clean_sym, "period": "1h", "limit": min(limit, 48)})
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict) and r.get("sumOpenInterest") is not None:
                            ts_ms = int(r.get("timestamp") or 0)
                            if ts_ms > 0:
                                live_records.append({
                                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                                    "sumOpenInterest": float(r["sumOpenInterest"]),
                                })
        except Exception:
            pass

        if live_records:
            live_df = pd.DataFrame(live_records)
            merged = pd.concat([cached_df, live_df], ignore_index=True) if not cached_df.empty else live_df
        else:
            merged = cached_df

        if merged.empty:
            return pd.DataFrame()

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = (
            merged.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return merged

    def fetch_taker_flow_history(self, symbol: str, limit: int = 192) -> pd.DataFrame:
        """
        Fetches trailing hourly Taker Buy/Sell Volume (`quote_volume`, `taker_buy_quote_volume`)
        for `symbol` (Issue #48), combining local cache (`data/cache/taker_flow/1h/{CLEAN_SYM}.parquet`)
        with live Binance Futures `/fapi/v1/klines`.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        tk_path = self.cache_dir / "taker_flow" / "1h" / f"{clean_sym}.parquet"
        cached_df = pd.DataFrame()

        if self.enable_warm_start and tk_path.exists():
            try:
                raw_tk = pd.read_parquet(tk_path).tail(max(limit, 240))
                req_cols = {"timestamp", "quote_volume", "taker_buy_quote_volume"}
                if not raw_tk.empty and req_cols.issubset(raw_tk.columns):
                    cached_df = raw_tk[list(req_cols)].copy()
                    cached_df["timestamp"] = self._normalize_timestamp_series(cached_df["timestamp"])
                    cached_df["quote_volume"] = cached_df["quote_volume"].astype(float)
                    cached_df["taker_buy_quote_volume"] = cached_df["taker_buy_quote_volume"].astype(float)
            except Exception:
                pass

        live_records: list[dict[str, Any]] = []
        try:
            fn = getattr(self.exchange, "fapiPublicGetKlines", None)
            if callable(fn) and not isinstance(fn, MagicMock):
                rows = fn({"symbol": clean_sym, "interval": "1h", "limit": min(limit, 36)})
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, (list, tuple)) and len(r) >= 11:
                            ts_ms = int(r[0])
                            if ts_ms > 0:
                                live_records.append({
                                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                                    "quote_volume": float(r[7]),
                                    "taker_buy_quote_volume": float(r[10]),
                                })
        except Exception:
            pass

        if live_records:
            live_df = pd.DataFrame(live_records)
            merged = pd.concat([cached_df, live_df], ignore_index=True) if not cached_df.empty else live_df
        else:
            merged = cached_df

        if merged.empty:
            return pd.DataFrame()

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = (
            merged.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return merged.tail(limit).reset_index(drop=True)

    def fetch_premium_index_history(self, symbol: str, limit: int = 192) -> pd.DataFrame:
        """
        Fetches trailing hourly Spot-Perp Premium Index (`close`) for `symbol`
        (Issue #49), combining local cache (`data/cache/premium_index/1h/{CLEAN_SYM}.parquet`)
        with live Binance Futures `/fapi/v1/premiumIndexKlines`.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        pr_path = self.cache_dir / "premium_index" / "1h" / f"{clean_sym}.parquet"
        cached_df = pd.DataFrame()

        if self.enable_warm_start and pr_path.exists():
            try:
                raw_pr = pd.read_parquet(pr_path).tail(max(limit, 240))
                if not raw_pr.empty and "timestamp" in raw_pr.columns and "close" in raw_pr.columns:
                    cached_df = raw_pr[["timestamp", "close"]].copy()
                    cached_df["timestamp"] = self._normalize_timestamp_series(cached_df["timestamp"])
                    cached_df["close"] = cached_df["close"].astype(float)
            except Exception:
                pass

        live_records: list[dict[str, Any]] = []
        try:
            fn = getattr(self.exchange, "fapiPublicGetPremiumIndexKlines", None)
            if callable(fn) and not isinstance(fn, MagicMock):
                rows = fn({"symbol": clean_sym, "interval": "1h", "limit": min(limit, 36)})
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, (list, tuple)) and len(r) >= 5:
                            ts_ms = int(r[0])
                            if ts_ms > 0:
                                live_records.append({
                                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                                    "close": float(r[4]),
                                })
        except Exception:
            pass

        if live_records:
            live_df = pd.DataFrame(live_records)
            merged = pd.concat([cached_df, live_df], ignore_index=True) if not cached_df.empty else live_df
        else:
            merged = cached_df

        if merged.empty:
            return pd.DataFrame()

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = (
            merged.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return merged.tail(limit).reset_index(drop=True)

    def fetch_smart_money_history(self, symbol: str, limit: int = 192) -> pd.DataFrame:
        """
        Fetches trailing hourly Binance Institutional Smart-Money Positioning (`top_pos_ls_ratio`,
        `global_acc_ls_ratio`, and `smart_money_div = ln(top_pos_ls_ratio / global_acc_ls_ratio)`)
        for `symbol` (Issue #55 / Tier 1A), combining local cache
        (`data/cache/smart_money/1h/{CLEAN_SYM}.parquet`) with live Binance Futures
        `/futures/data/topLongShortPositionRatio` and `/futures/data/globalLongShortAccountRatio`.
        """
        clean_sym = symbol.split(":")[0].replace("/", "")
        sm_path = self.cache_dir / "smart_money" / "1h" / f"{clean_sym}.parquet"
        cached_df = pd.DataFrame()

        if self.enable_warm_start and sm_path.exists():
            try:
                raw_sm = pd.read_parquet(sm_path).tail(max(limit, 240))
                if not raw_sm.empty and "timestamp" in raw_sm.columns:
                    cached_df = raw_sm.copy()
                    cached_df["timestamp"] = self._normalize_timestamp_series(cached_df["timestamp"])
                    if "smart_money_div" not in cached_df.columns and (
                        "top_pos_ls_ratio" in cached_df.columns and "global_acc_ls_ratio" in cached_df.columns
                    ):
                        w_r = cached_df["top_pos_ls_ratio"].astype(float).clip(0.05, 20.0)
                        r_r = cached_df["global_acc_ls_ratio"].astype(float).clip(0.05, 20.0)
                        cached_df["smart_money_div"] = np.log(w_r / r_r)
            except Exception:
                pass

        live_records: list[dict[str, Any]] = []
        try:
            fn_top = getattr(self.exchange, "fapiDataGetTopLongShortPositionRatio", None)
            fn_glob = getattr(self.exchange, "fapiDataGetGlobalLongShortAccountRatio", None)
            if (
                callable(fn_top)
                and callable(fn_glob)
                and not isinstance(fn_top, MagicMock)
                and not isinstance(fn_glob, MagicMock)
            ):
                top_rows = fn_top({"symbol": clean_sym, "period": "1h", "limit": min(limit, 36)})
                glob_rows = fn_glob({"symbol": clean_sym, "period": "1h", "limit": min(limit, 36)})
                top_map: dict[int, float] = {}
                if isinstance(top_rows, list):
                    for r in top_rows:
                        if isinstance(r, dict):
                            ts_ms = int(r.get("timestamp") or 0)
                            ratio_val = float(r.get("longShortRatio") or 0.0)
                            if ts_ms > 0 and ratio_val > 0:
                                top_map[ts_ms] = ratio_val
                if isinstance(glob_rows, list):
                    for r in glob_rows:
                        if isinstance(r, dict):
                            ts_ms = int(r.get("timestamp") or 0)
                            g_val = float(r.get("longShortRatio") or 0.0)
                            w_val = top_map.get(ts_ms)
                            if ts_ms > 0 and g_val > 0 and w_val and w_val > 0:
                                w_clip = float(np.clip(w_val, 0.05, 20.0))
                                g_clip = float(np.clip(g_val, 0.05, 20.0))
                                live_records.append({
                                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                                    "top_pos_ls_ratio": w_clip,
                                    "global_acc_ls_ratio": g_clip,
                                    "smart_money_div": float(np.log(w_clip / g_clip)),
                                })
        except Exception:
            pass

        if live_records:
            live_df = pd.DataFrame(live_records)
            merged = pd.concat([cached_df, live_df], ignore_index=True) if not cached_df.empty else live_df
        else:
            merged = cached_df

        if merged.empty:
            return pd.DataFrame()

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = (
            merged.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return merged.tail(limit).reset_index(drop=True)

    def fetch_symbol_obi(self, symbol: str, levels: int = 5) -> float | None:
        """
        Fetches Top-N Level-2 Order Book depth from Binance Futures and returns
        Order Book Imbalance `OBI_N` in [-1.0, +1.0] for Adaptive Maker Routing (Issue #47).
        """
        try:
            fn = getattr(self.exchange, "fetch_order_book", None)
            if not callable(fn):
                return None
            ob = fn(symbol, limit=levels)
            if not isinstance(ob, dict):
                return None
            return SmartExecutionRouter.compute_order_book_imbalance(ob, levels=levels)
        except Exception:
            return None

    def run_fast_cycle(self) -> tuple[Portfolio, list[Order]]:
        """
        Executes fast loop (every 5-10s):
        1. Fetch fresh L1 tickers
        2. Check & fill pending Maker limit orders
        3. Revalue portfolio Mark-to-Market
        """
        ticks = self.fetch_live_tickers()
        if not ticks:
            return self.broker.get_portfolio(), []

        # 1. Update broker prices
        for sym, t in ticks.items():
            self.broker.set_market_price(
                symbol=sym,
                price=t["price"],
                bid=t["bid"],
                ask=t["ask"],
            )

        # 2. Check pending limit orders against fresh ticks
        filled_orders = self.broker.check_pending_orders(ticks)
        if filled_orders:
            self._refresh_anomaly_cooldowns(filled_orders)

        # 3. Ensure open positions have active TP and SL bracket orders
        self.ensure_position_brackets()

        # 4. Manage Dynamic Trailing Stop & Profit-Lock Ratchet
        self.manage_trailing_and_profit_locks(ticks)

        # 5. Mark to market
        mark_prices = {sym: t["price"] for sym, t in ticks.items()}
        portfolio = self.broker.mark_to_market(mark_prices)

        # 6. Record Heartbeat and Market Pulse in database
        db = getattr(self.broker, "db", None) or getattr(self.broker, "database", None)
        if db:
            ws_state = self.ws_client.state.value if self.ws_client else "POLLING_REST"
            elapsed_sec = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            tps = self.total_ticks_processed / max(1.0, elapsed_sec)
            last_tick_iso = self.last_tick_time.isoformat() if self.last_tick_time else ""
            last_reb_iso = self.last_rebalance_time.isoformat() if self.last_rebalance_time else None

            if self.last_rebalance_time:
                time_since_reb = (datetime.now(timezone.utc) - self.last_rebalance_time).total_seconds()
                next_reb = max(0.0, self.rebalance_interval_sec - time_since_reb)
            else:
                next_reb = 0.0

            db.update_market_pulse(self.latest_tickers)
            db.update_daemon_heartbeat(
                ws_state=ws_state,
                symbols_count=len(self.symbols),
                total_ticks=self.total_ticks_processed,
                ticks_per_sec=round(tps, 2),
                last_tick_time=last_tick_iso,
                last_rebalance_time=last_reb_iso,
                next_rebalance_in_sec=round(next_reb, 1),
                status="ONLINE",
            )

            lev_ctrl = getattr(self.allocator, "stepped_leverage_controller", None)
            sf_res = getattr(self.allocator, "last_seven_factor_result", None)
            if lev_ctrl and hasattr(db, "save_seven_factor_telemetry"):
                prev_tel = (
                    db.get_latest_seven_factor_telemetry()
                    if (sf_res is None and hasattr(db, "get_latest_seven_factor_telemetry"))
                    else None
                ) or {}
                v_sc = sf_res.vol_scalar if sf_res else float(prev_tel.get("vol_scalar", 1.0))
                snap = lev_ctrl.update_state(current_equity=portfolio.total_equity, vol_scalar=v_sc)
                self.allocator.last_leverage_snapshot = snap
                db.save_seven_factor_telemetry(
                    initial_capital=snap.initial_capital,
                    current_equity=snap.current_equity,
                    high_water_mark=snap.high_water_mark,
                    drawdown_pct=snap.drawdown_pct,
                    unlocked_tier=snap.unlocked_tier,
                    active_base_leverage=snap.active_base_leverage,
                    vol_scalar=snap.vol_scalar,
                    effective_leverage=snap.effective_leverage,
                    next_tier_leverage=snap.next_tier_leverage,
                    next_tier_hwm_target=snap.next_tier_hwm_target,
                    progress_to_next_tier_pct=snap.progress_to_next_tier_pct,
                    portfolio_beta=sf_res.portfolio_beta if sf_res else float(prev_tel.get("portfolio_beta", 0.0)),
                    conviction=sf_res.conviction if sf_res else float(prev_tel.get("conviction", 1.0)),
                    factor_weights=sf_res.factor_weights if sf_res else dict(prev_tel.get("factor_weights") or {}),
                    factor_tstats=sf_res.factor_tstats if sf_res else dict(prev_tel.get("factor_tstats") or {}),
                )

        return portfolio, filled_orders

    def ensure_position_brackets(self) -> None:
        """
        Ensures active positions have appropriate bracket orders:
        - Anomaly strategies: both Take-Profit and initial Stop-Loss brackets.
        - Core / 7-Factor strategies: passive Maker Take-Profit (+8.0% default) with
          dynamic Stop-Profit / Trailing Stop armed via manage_trailing_and_profit_locks().
        """
        portfolio = self.broker.get_portfolio()
        anomaly_strats = ANOMALY_STRATEGIES
        for sym, pos in portfolio.positions.items():
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue

            entry = pos.entry_price
            if entry <= 0:
                continue

            if pos.strategy in anomaly_strats:
                if pos.take_profit is None or pos.stop_loss is None:
                    if pos.strategy == "Extreme Drop Bounce":
                        tp = round(entry * 1.06, 6 if entry < 1.0 else 2)
                        sl = round(entry * 0.95, 6 if entry < 1.0 else 2)
                    elif pos.side == PositionSide.LONG:
                        tp = round(entry * 1.15, 6 if entry < 1.0 else 2)
                        sl = round(entry * 0.96, 6 if entry < 1.0 else 2)
                    else:  # SHORT
                        tp = round(entry * 0.92, 6 if entry < 1.0 else 2)
                        sl = round(entry * 1.08, 6 if entry < 1.0 else 2)

                    self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=sl)
            elif self.core_take_profit_pct and self.core_take_profit_pct > 0:
                if pos.take_profit is None:
                    if pos.side == PositionSide.LONG:
                        tp = round(entry * (1.0 + self.core_take_profit_pct), 6 if entry < 1.0 else 2)
                    else:  # SHORT
                        tp = round(entry * (1.0 - self.core_take_profit_pct), 6 if entry < 1.0 else 2)
                    self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=pos.stop_loss)

    def manage_trailing_and_profit_locks(
        self,
        ticks: dict[str, dict[str, float]] | None = None,
    ) -> list[tuple[str, float, float]]:
        """
        Monitors all active positions on each fast cycle to enforce:
        1. Peak Price High-Water Mark tracking across all open positions
        2. Phase 2: Profit-Lock (Break-even + 0.8% fee buffer) once peak uPnL >= +2.5%
        3. Phase 3: Continuous Trailing Stop (Chandelier ratchet behind peak) once peak uPnL >= +4.0%

        Returns list of (symbol, old_sl, new_sl) for ratcheted positions.
        """
        portfolio = self.broker.get_portfolio()
        ratcheted: list[tuple[str, float, float]] = []

        for sym, pos in list(portfolio.positions.items()):
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue

            entry = pos.entry_price
            if entry <= 0:
                continue

            # Determine current price
            t_data = (ticks.get(sym) or ticks.get(sym.replace("/", "")) or {}) if ticks else {}
            curr_p = float(t_data.get("price") or pos.current_price or entry)
            if curr_p <= 0:
                continue

            # 1. Update Peak Price (anchored to at least entry price)
            if pos.side == PositionSide.LONG:
                if pos.peak_price <= 0:
                    pos.peak_price = max(entry, curr_p)
                else:
                    pos.peak_price = max(entry, pos.peak_price, curr_p)
                peak_ret = (pos.peak_price - entry) / entry
            else:  # SHORT
                if pos.peak_price <= 0:
                    pos.peak_price = min(entry, curr_p)
                else:
                    pos.peak_price = min(entry, pos.peak_price, curr_p)
                peak_ret = (entry - pos.peak_price) / entry

            # Persist updated peak_price
            self.broker.db.save_position(pos)

            # Determine candidate ratcheted stop-loss
            candidate_sl: float | None = None
            is_anomaly = pos.strategy in ANOMALY_STRATEGIES

            # Enforce Time-Stop (48h default) on open Anomaly positions so stagnant slots never zombie-lock
            if is_anomaly and self.anomaly_max_hold_hours and self.anomaly_max_hold_hours > 0:
                opened_dt = getattr(pos, "opened_at", None)
                if opened_dt is None and hasattr(self.broker, "db") and hasattr(self.broker.db, "get_position"):
                    try:
                        db_row = self.broker.db.get_position(sym)
                        if isinstance(db_row, dict) and db_row.get("opened_at"):
                            opened_dt = datetime.fromisoformat(str(db_row["opened_at"]))
                            if opened_dt.tzinfo is None:
                                opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                            pos.opened_at = opened_dt
                    except Exception:
                        opened_dt = None

                if isinstance(opened_dt, datetime):
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    age_hours = (now_utc - opened_dt).total_seconds() / 3600.0
                    if age_hours >= self.anomaly_max_hold_hours:
                        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
                        qty_to_close = pos.quantity
                        strat_name = pos.strategy
                        try:
                            self.broker.set_market_price(sym, price=curr_p)
                            self.broker.create_order(
                                symbol=sym,
                                side=close_side,
                                order_type=OrderType.MARKET,
                                quantity=qty_to_close,
                                strategy=strat_name,
                            )
                            was_loss = (curr_p < entry) if close_side == OrderSide.SELL else (curr_p > entry)
                            self.anomaly_last_exit[sym.replace("/", "")] = (now_utc, was_loss)
                            logger.info(
                                f"⏰ Time-Stop ({age_hours:.1f}h >= {self.anomaly_max_hold_hours:.0f}h) "
                                f"closed stagnant anomaly position {sym} @ ${curr_p:,.4f}"
                            )
                            continue
                        except Exception as e:
                            logger.warning(f"Error closing time-stopped anomaly position {sym}: {e}")

            trail_trig = self.anomaly_trailing_trigger_pct if is_anomaly else self.trailing_trigger_pct
            trail_dist = self.anomaly_trailing_distance_pct if is_anomaly else self.trailing_distance_pct
            lock_trig = self.anomaly_profit_lock_trigger_pct if is_anomaly else self.profit_lock_trigger_pct
            lock_lvl = self.anomaly_profit_lock_level_pct if is_anomaly else self.profit_lock_level_pct

            # Phase 3: Trailing Stop (Peak gain >= 4.0% Core / >= 5.5% Anomaly)
            if peak_ret >= trail_trig:
                if pos.side == PositionSide.LONG:
                    trail_target = pos.peak_price * (1.0 - trail_dist)
                    trail_target = round(trail_target, 6 if trail_target < 1.0 else 2)
                    if (pos.stop_loss is None or trail_target > pos.stop_loss) and curr_p > trail_target:
                        candidate_sl = trail_target
                else:  # SHORT
                    trail_target = pos.peak_price * (1.0 + trail_dist)
                    trail_target = round(trail_target, 6 if trail_target < 1.0 else 2)
                    if (pos.stop_loss is None or trail_target < pos.stop_loss) and curr_p < trail_target:
                        candidate_sl = trail_target

            # Phase 2: Profit-Lock (Peak gain >= 2.5% -> +0.8% Core / >= 4.0% -> +2.0% Anomaly)
            elif peak_ret >= lock_trig:
                if pos.side == PositionSide.LONG:
                    lock_target = entry * (1.0 + lock_lvl)
                    lock_target = round(lock_target, 6 if lock_target < 1.0 else 2)
                    if (pos.stop_loss is None or lock_target > pos.stop_loss) and curr_p > lock_target:
                        candidate_sl = lock_target
                else:  # SHORT
                    lock_target = entry * (1.0 - lock_lvl)
                    lock_target = round(lock_target, 6 if lock_target < 1.0 else 2)
                    if (pos.stop_loss is None or lock_target < pos.stop_loss) and curr_p < lock_target:
                        candidate_sl = lock_target

            # Apply ratchet if candidate_sl is an improvement
            if candidate_sl is not None:
                old_sl = pos.stop_loss or 0.0
                updated_order = self.broker.update_stop_loss(symbol=sym, new_stop_loss=candidate_sl)
                if updated_order:
                    ratcheted.append((sym, old_sl, candidate_sl))
                    logger.info(
                        f"🔒 Ratcheted Stop-Loss for {sym} ({pos.side.value}): "
                        f"${old_sl:,.4f} -> ${candidate_sl:,.4f} "
                        f"(Peak: ${pos.peak_price:,.4f} [{peak_ret*100:+.1f}%], "
                        f"Guaranteed SL PnL: {pos.sl_pnl_pct:+.1f}%)"
                    )

        return ratcheted

    def run_rebalance_cycle(self) -> list[Order]:
        """
        Executes multi-strategy portfolio rebalance:
        1. Download klines for universe
        2. Detect regime & compute target allocations
        3. Net orders through turnover deadband
        4. Route passive Maker limit orders at the touch
        """
        if getattr(self.broker, "is_live", False) and hasattr(self.broker, "sync_from_live_exchange"):
            self.broker.sync_from_live_exchange(sync_positions=True)
        portfolio = self.broker.get_portfolio()
        market_data: dict[str, pd.DataFrame] = {}
        volatilities: dict[str, float] = {}
        current_prices: dict[str, float] = {}

        # Collect symbols to evaluate: curated universe + any open core positions
        # (Exclude isolated satellite anomaly positions so 4h rebalance doesn't flatten their TP/SL brackets)
        open_pos_symbols = [
            s for s, p in portfolio.positions.items()
            if p.side != PositionSide.FLAT
            and p.quantity > 0
            and p.strategy not in ANOMALY_STRATEGIES
        ]
        base_clean_set = {s.replace("/", "") for s in self.symbols}
        scan_symbols = list(dict.fromkeys(list(self.symbols) + open_pos_symbols))
        use_7f = bool(getattr(self.allocator, "use_seven_factor_engine", False))
        bars_15m: dict[str, pd.DataFrame] = {}
        is_mock_kline_fn = isinstance(
            getattr(self, "fetch_warm_klines", None), MagicMock
        ) or isinstance(
            getattr(self, "fetch_symbol_klines", None), MagicMock
        )
        live_kline_ok_count = 0

        for sym in scan_symbols:
            if use_7f:
                df = self.fetch_warm_klines(sym, timeframe="1h", target_bars=1000)
            else:
                df = self.fetch_symbol_klines(sym, timeframe="1h", limit=100)

            if not df.empty and len(df) >= 30:
                if is_mock_kline_fn or self._last_kline_fetch_ok:
                    live_kline_ok_count += 1
                clean_sym = sym.replace("/", "")
                current_prices[sym] = float(df["close"].iloc[-1])

                # Only feed curated universe symbols (canonical clean_sym) into MetaStrategyAllocator
                if clean_sym in base_clean_set:
                    market_data[clean_sym] = df
                    vol_val = parkinson_volatility(df["high"].values[-30:], df["low"].values[-30:])
                    volatilities[clean_sym] = float(vol_val) if vol_val > 0 else 0.02
                    if use_7f and self.enable_subhourly_pullback:
                        df15 = self.fetch_warm_klines(sym, timeframe="15m", target_bars=64)
                        if not df15.empty:
                            bars_15m[clean_sym] = df15

        if not market_data or live_kline_ok_count == 0:
            logger.warning(
                "No live market data fetched (network offline or API unreachable). "
                "Deferring rebalance cycle without advancing last_rebalance_time."
            )
            return []

        # Fetch live funding rates (or 72h history for 7-Factor Engine), Open Interest (#46), Taker Flow (#48), Premium Index (#49) & Smart Money (#55)
        funding_data: dict[str, pd.DataFrame] = {}
        open_interest_data: dict[str, pd.DataFrame] = {}
        taker_flow_data: dict[str, pd.DataFrame] = {}
        premium_index_data: dict[str, pd.DataFrame] = {}
        smart_money_data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            clean_sym = sym.replace("/", "")
            if clean_sym not in market_data:
                continue
            if use_7f:
                df_f72 = self.fetch_funding_history_72h(sym)
                if not df_f72.empty:
                    funding_data[clean_sym] = df_f72
                df_oi = self.fetch_open_interest_history_24h(sym)
                if not df_oi.empty:
                    open_interest_data[clean_sym] = df_oi
                df_tk = self.fetch_taker_flow_history(sym)
                if not df_tk.empty:
                    taker_flow_data[clean_sym] = df_tk
                df_pr = self.fetch_premium_index_history(sym)
                if not df_pr.empty:
                    premium_index_data[clean_sym] = df_pr
                df_sm = self.fetch_smart_money_history(sym)
                if not df_sm.empty:
                    smart_money_data[clean_sym] = df_sm
            else:
                try:
                    fr = self.exchange.fetch_funding_rate(sym)
                    if fr and "fundingRate" in fr and fr["fundingRate"] is not None:
                        f_rate = float(fr["fundingRate"])
                        df_f = pd.DataFrame([{"funding_rate": f_rate, "fundingRate": f_rate}])
                        funding_data[clean_sym] = df_f
                except Exception:
                    pass

        total_equity = max(portfolio.total_equity, 1.0)
        current_weights_map: dict[str, float] = {}
        for sym, pos in portfolio.positions.items():
            if (
                pos
                and pos.side != PositionSide.FLAT
                and pos.quantity > 0
                and pos.strategy not in ANOMALY_STRATEGIES
            ):
                px = current_prices.get(sym) or pos.current_price or pos.entry_price
                signed_val = (-pos.quantity * px) if pos.side == PositionSide.SHORT else (pos.quantity * px)
                current_weights_map[sym.replace("/", "")] = signed_val / total_equity

        if use_7f:
            target_dollars, regime, weights = self.allocator.generate_portfolio_allocations(
                market_data=market_data,
                portfolio=portfolio,
                volatilities=volatilities,
                funding_data=funding_data,
                bars_15m=bars_15m if bars_15m else None,
                current_weights=current_weights_map if current_weights_map else None,
                open_interest_data=open_interest_data if open_interest_data else None,
                taker_flow_data=taker_flow_data if taker_flow_data else None,
                premium_index_data=premium_index_data if premium_index_data else None,
                smart_money_data=smart_money_data if smart_money_data else None,
            )
        else:
            target_dollars, regime, weights = self.allocator.generate_portfolio_allocations(
                market_data=market_data,
                portfolio=portfolio,
                volatilities=volatilities,
                funding_data=funding_data,
            )

        target_dollars = self._apply_core_bracket_cooldowns(target_dollars)

        if getattr(self.broker, "is_live", False) and getattr(self.broker, "live_broker", None) is not None:
            lev_snap_live = getattr(self.allocator, "last_leverage_snapshot", None)
            eff_lev_live = float(lev_snap_live.effective_leverage) if lev_snap_live else 2.5
            self.broker._current_live_leverage = max(2, min(5, int(eff_lev_live + 0.999)))
            try:
                self.broker.live_broker.kill_switch.sync_margin_and_leverage(
                    exchange_client=self.broker.live_broker.client,
                    symbols=list(self.symbols),
                    effective_leverage=max(2.5, eff_lev_live),
                    dry_run=False,
                )
                self.broker._configured_live_symbols.update(self.symbols)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Live margin/leverage sync note: {e}")

        executed_orders: list[Order] = []
        executed_deltas: dict[str, float] = {}
        dominant_strats = getattr(self.allocator, "last_dominant_strategy", {})

        for sym in scan_symbols:
            price = current_prices.get(sym) or self.latest_tickers.get(sym, {}).get("price", 0.0)
            if price <= 0:
                continue

            clean_sym = sym.replace("/", "")
            target_val = target_dollars.get(clean_sym, target_dollars.get(sym, 0.0))

            pos = portfolio.positions.get(sym)
            current_qty = pos.quantity if (pos and pos.side != PositionSide.FLAT) else 0.0
            if pos and pos.side == PositionSide.SHORT:
                current_val = -current_qty * price
            else:
                current_val = current_qty * price

            dollar_delta = target_val - current_val

            # Always cancel any stale pending LIMIT order from a previous rebalance cycle first
            db = getattr(self.broker, "db", None)
            if db:
                for pending_ord in db.get_pending_orders(symbol=sym):
                    if pending_ord.order_type == OrderType.LIMIT:
                        self.broker.cancel_order(pending_ord.id, sym)

            # 1. Minimum notional gate (relaxed for anomaly test sizing)
            min_notional = min(self.min_rebalance_notional, self.anomaly_size) if self.enable_anomalies else self.min_rebalance_notional
            if abs(dollar_delta) < min_notional:
                continue

            # 2. Turnover deadband gate:
            # When 7-Factor Engine + SteppedLeverageController is active, scale_and_deadband_weights()
            # has ALREADY applied the proportional deadband and neutrality re-centering. Do not apply
            # a conflicting second deadband filter that would discard neutrality re-centering legs.
            lev_snap = getattr(self.allocator, "last_leverage_snapshot", None)
            if lev_snap and hasattr(self.broker, "max_gross_leverage"):
                self.broker.max_gross_leverage = max(
                    float(self.broker.max_gross_leverage),
                    float(lev_snap.effective_leverage) * 1.20,
                )
            if not (use_7f and lev_snap is not None):
                effective_deadband = self.turnover_deadband
                is_new_position = (current_qty == 0.0 and abs(target_val) > 0)
                delta_turnover = abs(dollar_delta) / total_equity
                if not is_new_position and delta_turnover < effective_deadband and abs(target_val) > 0:
                    continue

            order_qty = abs(dollar_delta) / price
            side = OrderSide.BUY if dollar_delta > 0 else OrderSide.SELL
            strat_label = dominant_strats.get(clean_sym) or (pos.strategy if pos and pos.strategy else "Cross-Sectional Momentum")

            # Route as passive Maker limit order with Level-2 OBI Adaptive Routing (Issue #47)
            try:
                obi_val = self.fetch_symbol_obi(sym, levels=5)
                order = self.router.route_order(
                    symbol=sym,
                    side=side,
                    quantity=order_qty,
                    current_price=price,
                    urgency=ExecutionUrgency.LOW,
                    strategy=strat_label,
                    obi=obi_val,
                )
                if pos and pos.side != PositionSide.FLAT and pos.quantity > 0 and abs(target_val) > 0:
                    pos.strategy = strat_label
                    if db:
                        db.save_position(pos)
                executed_orders.append(order)
                executed_deltas[sym] = dollar_delta
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to submit rebalance order for {sym}: {e}")

        # Persist snapshot
        mom_w = weights.get(
            "momentum",
            weights.get("macro", 0.0) + weights.get("med", 0.0) + weights.get("fast", 0.0),
        )
        arb_w = weights.get("stat_arb", weights.get("rev", 0.0))
        cry_w = weights.get(
            "carry",
            weights.get("carry", 0.0) + weights.get("liq", 0.0) + weights.get("size", 0.0),
        )
        self.broker.db.save_rebalance_snapshot(
            regime_detected=regime.value if hasattr(regime, "value") else str(regime),
            momentum_weight=float(mom_w),
            stat_arb_weight=float(arb_w),
            carry_weight=float(cry_w),
            target_allocations=target_dollars,
            executed_delta=executed_deltas,
        )

        sf_res = getattr(self.allocator, "last_seven_factor_result", None)
        lev_snap = getattr(self.allocator, "last_leverage_snapshot", None)
        if sf_res and lev_snap and hasattr(self.broker.db, "save_seven_factor_telemetry"):
            self.broker.db.save_seven_factor_telemetry(
                initial_capital=lev_snap.initial_capital,
                current_equity=lev_snap.current_equity,
                high_water_mark=lev_snap.high_water_mark,
                drawdown_pct=lev_snap.drawdown_pct,
                unlocked_tier=lev_snap.unlocked_tier,
                active_base_leverage=lev_snap.active_base_leverage,
                vol_scalar=lev_snap.vol_scalar,
                effective_leverage=lev_snap.effective_leverage,
                next_tier_leverage=lev_snap.next_tier_leverage,
                next_tier_hwm_target=lev_snap.next_tier_hwm_target,
                progress_to_next_tier_pct=lev_snap.progress_to_next_tier_pct,
                portfolio_beta=sf_res.portfolio_beta,
                conviction=sf_res.conviction,
                factor_weights=sf_res.factor_weights,
                factor_tstats=sf_res.factor_tstats,
            )

        if getattr(self.broker, "is_live", False) and hasattr(self.broker, "sync_from_live_exchange"):
            self.broker.sync_from_live_exchange(sync_positions=True)
        self.ensure_position_brackets()

        self.last_rebalance_time = datetime.now(timezone.utc)
        if hasattr(self.broker, "db") and self.broker.db:
            ws_state = self.ws_client.state.value if self.ws_client else "POLLING_REST"
            elapsed_sec = max(1.0, (self.last_rebalance_time - self.start_time).total_seconds())
            tps = self.total_ticks_processed / elapsed_sec
            last_tick_iso = self.last_tick_time.isoformat() if self.last_tick_time else self.last_rebalance_time.isoformat()
            self.broker.db.update_daemon_heartbeat(
                ws_state=ws_state,
                symbols_count=len(self.symbols),
                total_ticks=self.total_ticks_processed,
                ticks_per_sec=round(tps, 2),
                last_tick_time=last_tick_iso,
                last_rebalance_time=self.last_rebalance_time.isoformat(),
                next_rebalance_in_sec=round(self.rebalance_interval_sec, 1),
                status="ONLINE" if self.is_running else "IDLE_ONCE",
            )
        return executed_orders

    def run_funding_cycle(self) -> dict[str, float]:
        """
        Polls official Binance funding rates and settles cashflows for open positions.
        """
        settled_cashflows: dict[str, float] = {}
        open_positions = self.broker.portfolio.positions

        for sym, pos in list(open_positions.items()):
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue
            try:
                funding_info = self.exchange.fetch_funding_rate(sym)
                rate = float(funding_info.get("fundingRate") or 0.0)
                cf = self.broker.settle_funding(sym, rate)
                settled_cashflows[sym] = cf
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error settling funding for {sym}: {e}")

        if getattr(self.broker, "enable_yield_collateral", False) and hasattr(
            self.broker, "settle_collateral_yield"
        ):
            try:
                y_cf = self.broker.settle_collateral_yield()
                if y_cf > 0:
                    settled_cashflows["BFUSD/USDT"] = y_cf
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error settling yield-bearing collateral (BFUSD/LDUSDT): {e}")

        self.last_funding_check_time = datetime.now(timezone.utc)
        return settled_cashflows

    @staticmethod
    def _is_tradfi_or_gapped_symbol(symbol: str, df: pd.DataFrame | None = None) -> bool:
        """
        Returns True if `symbol` is a non-crypto TradFi equity/ETF/FX/commodity perpetual
        or exhibits non-24/7 session gaps / flat market-closed bars.
        """
        base = symbol.split(":")[0].replace("/USDT", "").replace("USDT", "").upper()
        if base in NON_CRYPTO_TRADFI_BASES:
            return True
        if df is not None and not df.empty and len(df) >= 20:
            zero_vol_bars = int((df["volume"] <= 0).sum())
            flat_range_bars = int((df["high"] <= df["low"]).sum())
            if zero_vol_bars >= 2 or flat_range_bars >= 3:
                return True
        return False

    def is_intracycle_rebalance_needed(self, now: datetime | None = None) -> bool:
        """
        Checks whether mid-cycle Take-Profit / Trailing-Stop fills have caused the
        Core 7-Factor book to drift beyond intracycle_drift_threshold_pct (default 25% of Equity)
        so an automatic neutrality re-centering rebalance should be triggered.
        """
        if not getattr(self.allocator, "use_seven_factor_engine", False):
            return False
        now_dt = now or datetime.now(timezone.utc)
        if self.last_rebalance_time is None:
            return False
        if (now_dt - self.last_rebalance_time).total_seconds() < self.intracycle_min_rebalance_age_sec:
            return False
        if (
            self.last_core_bracket_fill_time is not None
            and (now_dt - self.last_core_bracket_fill_time).total_seconds() < self.intracycle_post_stop_settle_sec
        ):
            return False

        portfolio = self.broker.get_portfolio()
        core_long_usd = 0.0
        core_short_usd = 0.0
        for pos in portfolio.positions.values():
            if (
                pos
                and pos.side != PositionSide.FLAT
                and pos.quantity > 0
                and pos.strategy not in ANOMALY_STRATEGIES
            ):
                if pos.side == PositionSide.LONG:
                    core_long_usd += pos.market_value
                elif pos.side == PositionSide.SHORT:
                    core_short_usd += pos.market_value

        if (core_long_usd + core_short_usd) <= 0.0:
            return False

        equity = max(portfolio.total_equity, 1.0)
        net_drift_pct = abs(core_long_usd - core_short_usd) / equity
        if net_drift_pct >= self.intracycle_drift_threshold_pct:
            logger.info(
                f"⚖️ Intra-cycle Core neutrality drift detected: "
                f"Long=${core_long_usd:,.1f} vs Short=${core_short_usd:,.1f} "
                f"(Net Drift={net_drift_pct*100:.1f}% >= {self.intracycle_drift_threshold_pct*100:.1f}%). "
                f"Triggering automatic neutrality rebalance."
            )
            return True
        return False

    def fetch_universe_funding_rates(self, ttl_sec: float = 60.0) -> dict[str, float]:
        """
        Fetches real-time funding rates across all 300+ USDS-M perpetuals in a single
        batch REST call (`/fapi/v1/premiumIndex`), cached for `ttl_sec` seconds.
        """
        now_ts = time.time()
        if self._cached_universe_funding and (now_ts - self._last_universe_funding_ts) < ttl_sec:
            return self._cached_universe_funding

        f_map: dict[str, float] = {}
        try:
            fn = getattr(self.exchange, "fapiPublicGetPremiumIndex", None)
            if callable(fn) and not isinstance(fn, MagicMock):
                rows = fn()
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict):
                            raw_sym = str(r.get("symbol") or "")
                            raw_rate = r.get("lastFundingRate")
                            if raw_sym.endswith("USDT") and raw_rate is not None:
                                rate_f = float(raw_rate)
                                f_map[raw_sym] = rate_f
                                f_map[f"{raw_sym[:-4]}/USDT"] = rate_f
        except Exception:
            pass

        if f_map:
            self._cached_universe_funding = f_map
            self._last_universe_funding_ts = now_ts
        return self._cached_universe_funding

    def run_anomaly_scan(self) -> list[Order]:
        """
        Continuously evaluates extreme gainers and losers in real-time for anomaly alpha:
        1. Oversold Bounce (severe drop <= -15%, RSI <= 28) -> LONG
        2. Early Breakout (+8% to +22% in 4h with volume surge >= 2.5x) -> LONG
        3. Pump & Funding Fade (+25% blow-off top, funding >= +0.05%, RSI >= 72) -> SHORT
        """
        if not self.enable_anomalies or not self.latest_tickers:
            self.last_anomaly_scan_time = datetime.now(timezone.utc)
            return []

        portfolio = self.broker.get_portfolio()
        all_open_positions = {
            s: p for s, p in portfolio.positions.items()
            if p.quantity > 0 and p.side != PositionSide.FLAT
        }
        active_anomaly_positions = {
            s: p for s, p in all_open_positions.items()
            if p.strategy in ANOMALY_STRATEGIES
        }

        if len(active_anomaly_positions) >= self.max_anomaly_positions:
            self.last_anomaly_scan_time = datetime.now(timezone.utc)
            return []

        min_req_cash = self.anomaly_min_notional if self.enable_dynamic_anomaly_sizing else self.anomaly_size
        if portfolio.cash < min_req_cash:
            self.last_anomaly_scan_time = datetime.now(timezone.utc)
            return []

        executed_orders: list[Order] = []
        now = datetime.now(timezone.utc)
        self._refresh_anomaly_cooldowns()

        # Compute live 24h cross-sectional dispersion across tracked core symbols for Dynamic Confluence Sizing (Issue #50)
        core_changes_dec = [
            float(self.latest_tickers[s].get("change_24h", 0.0)) / 100.0
            for s in self.symbols
            if s in self.latest_tickers and "change_24h" in self.latest_tickers[s]
        ]
        if len(core_changes_dec) >= 4:
            mkt_disp_24h = float(pd.Series(core_changes_dec).std())
        else:
            mkt_disp_24h = 0.030

        def _resolve_notional(strat_label: str, sig_meta: dict[str, Any]) -> tuple[float, float]:
            if not self.enable_dynamic_anomaly_sizing:
                return min(float(self.anomaly_size), float(portfolio.cash)), 0.50
            eff_max_notional = float(self.anomaly_max_notional)
            if self.anomaly_pct_equity is not None and self.anomaly_max_notional >= 45.0:
                eff_max_notional = max(eff_max_notional, float(portfolio.total_equity) * float(self.anomaly_pct_equity))
            elif self.anomaly_pct_equity is not None and self.anomaly_pct_equity != 0.045:
                eff_max_notional = float(portfolio.total_equity) * float(self.anomaly_pct_equity)
            rec_not, conf_sc = compute_anomaly_confluence_size(
                strategy_type=strat_label,
                metadata=sig_meta or {},
                mkt_disp_24h=mkt_disp_24h,
                min_notional=self.anomaly_min_notional,
                max_notional=max(self.anomaly_min_notional, eff_max_notional),
            )
            return min(rec_not, float(portfolio.cash)), conf_sc

        def _submit_anomaly_entry(
            sym_str: str,
            side: OrderSide,
            qty_val: float,
            ref_px: float,
            bid_px: float,
            ask_px: float,
            strat_name: str,
        ) -> Order:
            if self.anomaly_prefer_maker:
                touch_px = ask_px if side == OrderSide.BUY else bid_px
                return self.broker.create_order(
                    symbol=sym_str,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=qty_val,
                    price=touch_px,
                    role="MAKER",
                    strategy=strat_name,
                )
            return self.broker.create_order(
                symbol=sym_str,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty_val,
                strategy=strat_name,
            )

        # Scan latest tickers for anomaly & extreme funding harvester candidates
        universe_funding: dict[str, float] = {}
        if getattr(self.allocator, "funding_harvester_strategy", None):
            universe_funding = self.fetch_universe_funding_rates()

        candidates = []
        open_clean_syms = {s.replace("/", "") for s in all_open_positions}
        for sym, t in self.latest_tickers.items():
            clean_sym = sym.replace("/", "")
            # Skip if any position (core or anomaly) is already open on this symbol
            if sym in all_open_positions or clean_sym in open_clean_syms:
                continue

            # Skip non-crypto TradFi equities/ETFs/FX perpetuals
            if self._is_tradfi_or_gapped_symbol(sym):
                continue

            # Asymmetric Per-Symbol Cooldown (12h after Win / 24h after Stop-Loss)
            if clean_sym in self.anomaly_last_exit:
                last_ts, was_loss = self.anomaly_last_exit[clean_sym]
                cd_sec = self.anomaly_cooldown_loss_sec if was_loss else self.anomaly_cooldown_win_sec
                if (now - last_ts).total_seconds() < cd_sec:
                    continue

            vol = float(t.get("volume_24h", 0.0))
            if vol < self.min_volume_24h_usd:
                continue

            pct = float(t.get("change_24h", 0.0))
            f_uni = float(universe_funding.get(clean_sym, universe_funding.get(sym, 0.0)))
            is_price_anomaly = (
                pct <= -15.0
                or (self.anomaly_breakout_min_24h_pct <= pct <= 22.0)
                or pct >= 25.0
            )
            is_extreme_funding = abs(f_uni) >= self.harvester_min_abs_funding
            if is_price_anomaly or is_extreme_funding:
                candidates.append((sym, pct, float(t.get("price", 0.0))))

        for sym, pct, price in candidates:
            if len(active_anomaly_positions) + len(executed_orders) >= self.max_anomaly_positions:
                break
            if price <= 0:
                continue

            df = self.fetch_symbol_klines(sym, timeframe="1h", limit=30)
            if df.empty or len(df) < 26 or self._is_tradfi_or_gapped_symbol(sym, df):
                continue

            # Synchronize execution price and broker L1 quote with fresh kline close to prevent stale-ticker fills
            bid_p = price
            ask_p = price
            kline_close = float(df["close"].iloc[-1])
            if kline_close > 0:
                if abs(kline_close - price) / kline_close > 0.015:
                    price = kline_close
                t_info = self.latest_tickers.get(sym, {})
                bid_p = float(t_info.get("bid") or price)
                ask_p = float(t_info.get("ask") or price)
                if abs(bid_p - price) / price > 0.015:
                    bid_p = price
                if abs(ask_p - price) / price > 0.015:
                    ask_p = price
                self.broker.set_market_price(sym, price=price, bid=bid_p, ask=ask_p)
                if sym in self.latest_tickers:
                    self.latest_tickers[sym]["price"] = price
                    self.latest_tickers[sym]["bid"] = bid_p
                    self.latest_tickers[sym]["ask"] = ask_p

            # 1. Oversold Bounce (drop <= -15%, RSI <= 28)
            if pct <= -15.0 and self.allocator.oversold_bounce_strategy:
                sigs = self.allocator.oversold_bounce_strategy.generate_signals({sym: df})
                if sigs:
                    notional_usd, conf_sc = _resolve_notional("Extreme Drop Bounce", sigs[0].metadata or {})
                    qty = round(notional_usd / price, 6 if price < 1.0 else 4)
                    if qty > 0:
                        try:
                            order = _submit_anomaly_entry(
                                sym_str=sym,
                                side=OrderSide.BUY,
                                qty_val=qty,
                                ref_px=price,
                                bid_px=bid_p,
                                ask_px=ask_p,
                                strat_name="Extreme Drop Bounce",
                            )
                            fill_p = order.average_fill_price or price
                            tp = round(fill_p * 1.06, 6 if fill_p < 1.0 else 2)
                            sl = round(fill_p * 0.95, 6 if fill_p < 1.0 else 2)
                            self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=sl)
                            executed_orders.append(order)
                            logger.info(
                                f"Anomaly Executed: Oversold Bounce LONG on {sym} @ ${fill_p:,.4f} "
                                f"[Size: ${notional_usd:.2f} | Conf: {conf_sc:.2f}] (TP: ${tp}, SL: ${sl})"
                            )
                            continue
                        except Exception as e:
                            logger.warning(f"Error executing Oversold Bounce for {sym}: {e}")

            # 2. Early Breakout (+12% to +22% with volume surge)
            if self.anomaly_breakout_min_24h_pct <= pct <= 22.0 and self.allocator.early_breakout_strategy:
                sigs = self.allocator.early_breakout_strategy.generate_signals({sym: df})
                if sigs:
                    notional_usd, conf_sc = _resolve_notional("Early Breakout", sigs[0].metadata or {})
                    qty = round(notional_usd / price, 6 if price < 1.0 else 4)
                    if qty > 0:
                        try:
                            order = _submit_anomaly_entry(
                                sym_str=sym,
                                side=OrderSide.BUY,
                                qty_val=qty,
                                ref_px=price,
                                bid_px=bid_p,
                                ask_px=ask_p,
                                strat_name="Early Breakout",
                            )
                            fill_p = order.average_fill_price or price
                            tp = round(fill_p * 1.15, 6 if fill_p < 1.0 else 2)
                            sl = round(fill_p * 0.96, 6 if fill_p < 1.0 else 2)
                            self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=sl)
                            executed_orders.append(order)
                            logger.info(
                                f"Anomaly Executed: Early Breakout LONG on {sym} @ ${fill_p:,.4f} "
                                f"[Size: ${notional_usd:.2f} | Conf: {conf_sc:.2f}] (TP: ${tp}, SL: ${sl})"
                            )
                            continue
                        except Exception as e:
                            logger.warning(f"Error executing Early Breakout for {sym}: {e}")

            # 3. Pump & Funding Fade (>= +25% with overheated funding)
            if pct >= 25.0 and self.allocator.funding_fade_strategy:
                try:
                    fr = self.exchange.fetch_funding_rate(sym)
                    f_rate = float(fr.get("fundingRate") or 0.0) if fr else 0.0
                except Exception:
                    f_rate = 0.0

                df_f = pd.DataFrame([{"funding_rate": f_rate}])
                sigs = self.allocator.funding_fade_strategy.generate_signals({sym: df}, funding_data={sym: df_f})
                if sigs:
                    notional_usd, conf_sc = _resolve_notional("Pump & Funding Fade", sigs[0].metadata or {})
                    qty = round(notional_usd / price, 6 if price < 1.0 else 4)
                    if qty > 0:
                        try:
                            order = _submit_anomaly_entry(
                                sym_str=sym,
                                side=OrderSide.SELL,
                                qty_val=qty,
                                ref_px=price,
                                bid_px=bid_p,
                                ask_px=ask_p,
                                strat_name="Pump & Funding Fade",
                            )
                            fill_p = order.average_fill_price or price
                            tp = round(fill_p * 0.92, 6 if fill_p < 1.0 else 2)
                            sl = round(fill_p * 1.05, 6 if fill_p < 1.0 else 2)
                            self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=sl)
                            executed_orders.append(order)
                            logger.info(
                                f"Anomaly Executed: Pump & Funding Fade SHORT on {sym} @ ${fill_p:,.4f} "
                                f"[Size: ${notional_usd:.2f} | Conf: {conf_sc:.2f}] (TP: ${tp}, SL: ${sl})"
                            )
                            continue
                        except Exception as e:
                            logger.warning(f"Error executing Pump & Funding Fade for {sym}: {e}")

            # 4. Extreme Funding Rate Harvester with Beta-Neutral Hedge (Issue #56 / Tier 1B)
            harvester_strat = getattr(self.allocator, "funding_harvester_strategy", None)
            if harvester_strat:
                clean_s = sym.replace("/", "")
                f_harv = float(universe_funding.get(clean_s, universe_funding.get(sym, 0.0)))
                if abs(f_harv) >= self.harvester_min_abs_funding:
                    df_fh = pd.DataFrame([{"funding_rate": f_harv}])
                    sigs = harvester_strat.generate_signals({sym: df}, funding_data={sym: df_fh})
                    if sigs:
                        sig_h = sigs[0]
                        notional_usd, conf_sc = _resolve_notional(
                            "Extreme Funding Harvester", sig_h.metadata or {}
                        )
                        qty = round(notional_usd / price, 6 if price < 1.0 else 4)
                        if qty > 0:
                            side_h = OrderSide.BUY if sig_h.score >= 0 else OrderSide.SELL
                            try:
                                order = _submit_anomaly_entry(
                                    sym_str=sym,
                                    side=side_h,
                                    qty_val=qty,
                                    ref_px=price,
                                    bid_px=bid_p,
                                    ask_px=ask_p,
                                    strat_name="Extreme Funding Harvester",
                                )
                                fill_p = order.average_fill_price or price
                                if side_h == OrderSide.BUY:
                                    tp = round(fill_p * 1.075, 6 if fill_p < 1.0 else 2)
                                    sl = round(fill_p * 0.960, 6 if fill_p < 1.0 else 2)
                                else:
                                    tp = round(fill_p * 0.925, 6 if fill_p < 1.0 else 2)
                                    sl = round(fill_p * 1.040, 6 if fill_p < 1.0 else 2)
                                self.broker.create_bracket_orders(symbol=sym, take_profit=tp, stop_loss=sl)
                                executed_orders.append(order)
                                open_clean_syms.add(clean_s)

                                # Optional Beta-Neutral Hedge Leg on an unheld liquid major
                                if self.enable_harvester_beta_hedge:
                                    beta_est = float((sig_h.metadata or {}).get("beta_est", 1.0))
                                    hedge_notional = notional_usd * float(np.clip(beta_est, 0.50, 1.25))
                                    for h_sym in ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"):
                                        h_clean = h_sym.replace("/", "")
                                        if (
                                            h_sym != sym
                                            and h_sym not in all_open_positions
                                            and h_clean not in open_clean_syms
                                            and h_sym in self.latest_tickers
                                        ):
                                            h_t = self.latest_tickers[h_sym]
                                            h_px = float(h_t.get("price") or 0.0)
                                            h_bid = float(h_t.get("bid") or h_px)
                                            h_ask = float(h_t.get("ask") or h_px)
                                            if h_px > 0:
                                                self.broker.set_market_price(h_sym, price=h_px, bid=h_bid, ask=h_ask)
                                                h_qty = round(hedge_notional / h_px, 6 if h_px < 1.0 else 4)
                                                h_side = OrderSide.SELL if side_h == OrderSide.BUY else OrderSide.BUY
                                                if h_qty > 0:
                                                    h_ord = _submit_anomaly_entry(
                                                        sym_str=h_sym,
                                                        side=h_side,
                                                        qty_val=h_qty,
                                                        ref_px=h_px,
                                                        bid_px=h_bid,
                                                        ask_px=h_ask,
                                                        strat_name="Funding Harvester Hedge",
                                                    )
                                                    h_fill = h_ord.average_fill_price or h_px
                                                    if h_side == OrderSide.BUY:
                                                        h_tp = round(h_fill * 1.075, 6 if h_fill < 1.0 else 2)
                                                        h_sl = round(h_fill * 0.960, 6 if h_fill < 1.0 else 2)
                                                    else:
                                                        h_tp = round(h_fill * 0.925, 6 if h_fill < 1.0 else 2)
                                                        h_sl = round(h_fill * 1.040, 6 if h_fill < 1.0 else 2)
                                                    self.broker.create_bracket_orders(
                                                        symbol=h_sym, take_profit=h_tp, stop_loss=h_sl
                                                    )
                                                    executed_orders.append(h_ord)
                                                    open_clean_syms.add(h_clean)
                                                    break
                                logger.info(
                                    f"Harvester Executed: {side_h.value} on {sym} @ ${fill_p:,.4f} "
                                    f"[Funding: {f_harv*100:+.3f}% | Size: ${notional_usd:.2f}] (TP: ${tp}, SL: ${sl})"
                                )
                                continue
                            except Exception as e:
                                logger.warning(f"Error executing Extreme Funding Harvester for {sym}: {e}")

        self.last_anomaly_scan_time = now
        return executed_orders

    def step(self) -> dict[str, Any]:
        """
        Performs a single synchronous step across fast, rebalance, and funding cycles.
        Useful for scheduled cron executions and unit tests.
        """
        now = datetime.now(timezone.utc)

        # 1. Fast cycle (L1 quote update, pending limit/bracket fill check, MTM)
        portfolio, filled_orders = self.run_fast_cycle()

        # 2. Event-driven Anomaly Scan (every 30s)
        anomaly_orders = []
        if self.enable_anomalies:
            anomaly_due = False
            if self.last_anomaly_scan_time is None or (now - self.last_anomaly_scan_time).total_seconds() >= self.anomaly_scan_interval_sec:
                anomaly_due = True
            if anomaly_due:
                anomaly_orders = self.run_anomaly_scan()

        # 3. Check if rebalance is due (scheduled 4h interval or intra-cycle neutrality drift)
        rebalance_due = False
        if self.last_rest_tickers_ok:
            if (
                self.last_rebalance_time is None
                or (now - self.last_rebalance_time).total_seconds() >= self.rebalance_interval_sec
                or self.is_intracycle_rebalance_needed(now)
            ):
                rebalance_due = True

        rebalance_orders = []
        if rebalance_due:
            rebalance_orders = self.run_rebalance_cycle()

        # 4. Check if funding settlement is due (00:00, 08:00, 16:00 UTC)
        funding_due = False
        if (
            now.minute in (0, 1)
            and now.hour in (0, 8, 16)
            and (
                self.last_funding_check_time is None
                or (now - self.last_funding_check_time).total_seconds() > 300
            )
        ):
            funding_due = True

        funding_cashflows = {}
        if funding_due:
            funding_cashflows = self.run_funding_cycle()

        return {
            "timestamp": now.isoformat(),
            "total_equity": portfolio.total_equity,
            "cash": portfolio.cash,
            "filled_orders": len(filled_orders),
            "anomaly_orders": len(anomaly_orders),
            "rebalance_orders": len(rebalance_orders),
            "funding_cashflows": funding_cashflows,
        }

    def start(self) -> None:
        """Starts the daemon in a background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        if self.ws_client:
            self.ws_client.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("ShadowTradingDaemon started successfully.")

    def stop(self) -> None:
        """Stops the daemon background thread."""
        self.is_running = False
        if self.ws_client:
            self.ws_client.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        db = getattr(self.broker, "db", None)
        if db and hasattr(db, "update_daemon_heartbeat"):
            try:
                last_reb_iso = self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
                last_tick_iso = self.last_tick_time.isoformat() if self.last_tick_time else ""
                db.update_daemon_heartbeat(
                    ws_state="STOPPED",
                    symbols_count=len(self.symbols),
                    total_ticks=self.total_ticks_processed,
                    ticks_per_sec=0.0,
                    last_tick_time=last_tick_iso,
                    last_rebalance_time=last_reb_iso,
                    next_rebalance_in_sec=0.0,
                    status="STOPPED",
                )
            except Exception:
                pass
        logger.info("ShadowTradingDaemon stopped.")

    def _run_loop(self) -> None:
        """Continuous execution loop."""
        while self.is_running:
            try:
                self.step()
            except Exception:
                logger.exception("Error in shadow daemon loop")
            time.sleep(self.fast_interval_sec)
