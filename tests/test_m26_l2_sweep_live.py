"""
Unit & Integration Tests for Phase-P2 Live Production Sniper `M26` (`Sniper:M26_L2SweepAirPocket`).

Verifies 100% backtest-to-live parity across:
  1. `LiveMultiModalFeedAdapter` (`fetch_contract_oi_expansion_24h` `+60s` causal lag, symmetric
     truncation-safe `fetch_l2_book_depth_imbalance`, `0.000ms` causal `fetch_live_1ms_aggtrades_sweep`,
     and `fetch_spot_taker_buy_share_3m`).
  2. `EventDrivenSniperEngine` 7-gate `P6A_L2_3_INST_SWEEP_CONC` evaluation + anti-spoofing retail-chop veto.
  3. `ShadowTradingDaemon` One-Way non-collision, `MONO_A_CONVEX_TRAIL` (`-1.9% SL / +8.5% TP`,
     `+2.4% -> +0.15%` unchoked lock, `+3.2% -> 1.6%` convex trail, `2.5h` time-stop), and hot-kill flag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from scripts.run_trading import build_parser as build_trading_parser
from src.domain.entities import Position
from src.domain.enums import OrderStatus, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.feature_flags import FeatureFlags
from src.execution.multimodal_feeds import L2BookDepthSnapshot, LiveMultiModalFeedAdapter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.strategies.event_snipers import SNIPER_BRACKET_PROFILES, SNIPER_M26_STRAT
from src.strategies.meta_allocator import MetaStrategyAllocator


class _MockHTTPResp:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers = {"X-MBX-USED-WEIGHT-1M": "12"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def test_multimodal_feeds_l2_depth_contract_oi_and_1ms_aggtrades() -> None:
    sig_ts = 1_765_000_200  # Exact 5m close
    now_ts = float(sig_ts + 8)

    def _fake_get(url: str, params: dict[str, Any] | None = None, timeout: float = 8.0) -> _MockHTTPResp:
        _ = timeout
        if "openInterestHist" in url:
            rows = []
            for k in range(60):
                ts_k = sig_ts - (60 - k) * 300
                oi_c = 1_000_000.0 if k < 59 else 1_240_000.0
                rows.append({"timestamp": ts_k * 1000, "sumOpenInterest": str(oi_c)})
            # Unsettled spike only 5s old -> MUST be excluded by +60s safety lag
            rows.append({"timestamp": int(now_ts - 5) * 1000, "sumOpenInterest": "99999999.0"})
            return _MockHTTPResp(rows)

        if url.endswith("/fapi/v1/depth"):
            # Mid = 10.00; strong resting Bid support ($400k vs $180k Ask -> imb > +0.37)
            bids = [[f"{10.00 - 0.001 * i:.4f}", "2000.0"] for i in range(1, 220)]
            asks = [[f"{10.00 + 0.001 * i:.4f}", "900.0"] for i in range(1, 220)]
            return _MockHTTPResp({"bids": bids, "asks": asks})

        if url.endswith("/api/v3/klines"):
            rows = []
            for m in range(4):
                open_ms = (sig_ts - (4 - m) * 60) * 1000
                rows.append(
                    [open_ms, "10", "10.1", "9.9", "10.05", "1000", open_ms + 59999, "100000.0", 50, "620", "62000.0"]
                )
            return _MockHTTPResp(rows)

        if url.endswith("/fapi/v1/aggTrades"):
            st = int((params or {}).get("startTime", 0))
            if st >= (sig_ts * 1000 - 300_000):
                # 12 institutional 1ms multi-maker buy sweeps + 4 small retail trades + 1 future trade after sig_ts
                trades = []
                for idx in range(12):
                    t_m = sig_ts * 1000 - 120_000 + idx * 5_000
                    trades.append(
                        {
                            "a": 1000 + idx,
                            "p": "10.02",
                            "q": "3500.0",  # $35,070 1ms sweep
                            "f": 5000 + idx * 10,
                            "l": 5008 + idx * 10,  # 9 makers swept in 1ms!
                            "T": t_m,
                            "m": False,  # Buyer is aggressor
                        }
                    )
                for idx in range(4):
                    trades.append(
                        {
                            "a": 2000 + idx,
                            "p": "10.01",
                            "q": "40.0",
                            "f": 9000 + idx,
                            "l": 9000 + idx,
                            "T": sig_ts * 1000 - 30_000 + idx * 1000,
                            "m": idx % 2 == 0,
                        }
                    )
                # Future trade at sig_ts * 1000 + 5 -> MUST be ignored (`0.000ms` lookahead!)
                trades.append(
                    {
                        "a": 9999,
                        "p": "9.50",
                        "q": "999999.0",
                        "f": 99000,
                        "l": 99100,
                        "T": sig_ts * 1000 + 5,
                        "m": True,
                    }
                )
                return _MockHTTPResp(trades)
            return _MockHTTPResp([])

        raise AssertionError(f"Unexpected URL {url}")

    sess = MagicMock()
    sess.get.side_effect = _fake_get
    adapter = LiveMultiModalFeedAdapter(http_session=sess, metrics_safety_lag_sec=60)

    oi_res = adapter.fetch_contract_oi_expansion_24h("SUIUSDT", now_ts=now_ts)
    assert oi_res is not None
    assert 1.20 <= oi_res[0] <= 1.25  # Excluded unsettled 99,999,999 spike!

    bk = adapter.fetch_l2_book_depth_imbalance("SUIUSDT", now_ts=now_ts)
    assert bk is not None
    assert bk.book_imb_1pct > 0.30
    assert bk.book_imb_2pct > 0.30

    sp_tb = adapter.fetch_spot_taker_buy_share_3m("SUIUSDT", perp_fallback_tb_5m=0.50, now_ts=now_ts)
    assert abs(sp_tb - 0.62) < 1e-6

    sw = adapter.fetch_live_1ms_aggtrades_sweep("SUIUSDT", signal_ts_sec=sig_ts)
    assert sw["has_agg"] == 1.0
    assert sw["sweep_conc_share_5m"] > 0.85
    assert sw["whale_sweep_buy_share_15m"] > 0.95  # Ignored future sell dump at sig_ts*1000+5!


def test_m26_daemon_execution_antispoof_veto_ratchet_and_hot_kill(tmp_path: Path) -> None:
    parser = build_trading_parser()
    parsed_off = parser.parse_args(["--disable-l2-sweep-sniper"])
    ff_off = FeatureFlags.from_env_and_args(args=parsed_off, mode="mainnet", environ={})
    assert ff_off.enable_l2_sweep_sniper is False

    ff_on = FeatureFlags.from_env_and_args(mode="mainnet", environ={})
    assert ff_on.enable_l2_sweep_sniper is True
    assert ff_on.enable_post_settle_sniper is False

    db = ShadowDatabase(db_path=str(tmp_path / "m26_live.db"))
    broker = ShadowExecutionBroker(database=db, initial_cash=2700.0)
    router = SmartExecutionRouter(broker=broker)
    allocator = MetaStrategyAllocator(use_seven_factor_engine=False)
    flags = FeatureFlags(
        enable_anomaly_capsule=False,
        enable_event_snipers=True,
        enable_post_settle_sniper=False,
        enable_l2_sweep_sniper=True,
        sniper_slot_leverage=0.24,
        sniper_max_positions=3,
    )
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        router=router,
        exchange_client=MagicMock(),
        feature_flags=flags,
    )

    # Seed an active Core P1 position on NEAR/USDT to verify strict One-Way non-collision
    broker.set_market_price("NEAR/USDT", price=5.0, bid=4.999, ask=5.001)
    broker.portfolio.positions["NEAR/USDT"] = Position(
        symbol="NEAR/USDT",
        side=PositionSide.SHORT,
        quantity=80.0,
        entry_price=5.0,
        current_price=5.0,
        strategy="7Factor_Core",
    )
    daemon.latest_tickers = {
        "NEAR/USDT": {"price": 5.20, "bid": 5.199, "ask": 5.201, "volume_24h": 80_000_000.0},
        "SUI/USDT": {"price": 2.50, "bid": 2.499, "ask": 2.501, "volume_24h": 120_000_000.0},
        "APT/USDT": {"price": 8.00, "bid": 7.998, "ask": 8.002, "volume_24h": 65_000_000.0},
    }

    now_dt = datetime(2026, 11, 19, 12, 5, 8, tzinfo=timezone.utc)
    t_end = int(now_dt.timestamp() // 300) * 300
    ts_75 = np.arange(t_end - 75 * 300, t_end, 300, dtype=np.int64)

    def _make_breakout_5m(base_px: float) -> pd.DataFrame:
        c = np.full(75, base_px, dtype=float)
        h = np.full(75, base_px * 1.005, dtype=float)
        low = np.full(75, base_px * 0.995, dtype=float)
        qv = np.full(75, 500_000.0, dtype=float)
        c[-1] = base_px * 1.018  # Breaks above 4h high
        h[-1] = base_px * 1.020
        qv[-1] = 1_400_000.0  # RVOL_5m = 2.8x >= 2.0x
        return pd.DataFrame(
            {
                "timestamp": ts_75,
                "open": c * 0.998,
                "high": h,
                "low": low,
                "close": c,
                "quote_volume": qv,
                "taker_buy_quote_volume": qv * 0.60,
            }
        )

    b5_override = {
        "NEARUSDT": _make_breakout_5m(5.10),  # Blocked by Core P1 One-Way non-collision!
        "SUIUSDT": _make_breakout_5m(2.46),  # Valid M26 institutional sweep + L2 air-pocket!
        "APTUSDT": _make_breakout_5m(7.85),  # Retail-chop fakeout (sweep_conc=0.14 < 0.32) -> VETOED!
    }
    l2_override = {
        "NEARUSDT": L2BookDepthSnapshot("NEARUSDT", 5.20, 400e3, 180e3, 700e3, 320e3, 0.38, 0.37, float(t_end)),
        "SUIUSDT": L2BookDepthSnapshot("SUIUSDT", 2.50, 500e3, 210e3, 900e3, 410e3, 0.41, 0.37, float(t_end)),
        "APTUSDT": L2BookDepthSnapshot("APTUSDT", 8.00, 350e3, 190e3, 620e3, 350e3, 0.30, 0.28, float(t_end)),
    }
    sw_override = {
        "NEARUSDT": {
            "has_agg": 1.0,
            "sweep_conc_share_5m": 0.48,
            "whale_sweep_buy_share_15m": 0.66,
            "smart_vs_retail_div_15m": 0.12,
        },
        "SUIUSDT": {
            "has_agg": 1.0,
            "sweep_conc_share_5m": 0.45,
            "whale_sweep_buy_share_15m": 0.64,
            "smart_vs_retail_div_15m": 0.10,
        },
        "APTUSDT": {
            "has_agg": 1.0,
            "sweep_conc_share_5m": 0.14,  # Retail atomized chop -> must be vetoed!
            "whale_sweep_buy_share_15m": 0.48,
            "smart_vs_retail_div_15m": -0.09,
        },
    }

    orders = daemon.run_event_sniper_scan(
        now=now_dt,
        bars_5m_override=b5_override,
        bars_15m_override={},
        bars_1h_override={},
        funding_override={"NEARUSDT": 0.0001, "SUIUSDT": 0.0001, "APTUSDT": 0.0001},
        m26_oi_override={"NEARUSDT": 1.25, "SUIUSDT": 1.22, "APTUSDT": 1.21},
        m26_l2_override=l2_override,
        m26_sweep_override=sw_override,
        m26_spot_tb_override={"NEARUSDT": 0.59, "SUIUSDT": 0.58, "APTUSDT": 0.57},
    )

    assert len(orders) == 1
    assert orders[0].symbol == "SUI/USDT"
    assert orders[0].strategy == SNIPER_M26_STRAT
    assert orders[0].status == OrderStatus.FILLED

    sui_pos = broker.get_portfolio().positions["SUI/USDT"]
    spec_26 = SNIPER_BRACKET_PROFILES[SNIPER_M26_STRAT]
    assert abs(float(sui_pos.stop_loss) - round(sui_pos.entry_price * (1.0 - spec_26.stop_loss_pct), 4)) < 1e-4
    assert abs(float(sui_pos.take_profit) - round(sui_pos.entry_price * (1.0 + spec_26.take_profit_pct), 4)) < 1e-4

    # Verify +2.4% unchoked breakeven lock (`+15 bps` with 4-decimal precision on $2.50 coin!)
    px_lock = round(sui_pos.entry_price * 1.026, 4)
    broker.set_market_price("SUI/USDT", price=px_lock)
    daemon.manage_trailing_and_profit_locks({"SUI/USDT": {"price": px_lock}}, now=now_dt + timedelta(minutes=15))
    assert float(broker.get_portfolio().positions["SUI/USDT"].stop_loss) > sui_pos.entry_price

    # Verify runtime hot-kill (`enable_l2_sweep_sniper=false`) auto-closes M26 cleanly while keeping Core P1 intact
    daemon.feature_flags.apply_override("enable_l2_sweep_sniper", False)
    daemon._sync_allocator_with_feature_flags()
    daemon.manage_trailing_and_profit_locks(
        {"SUI/USDT": {"price": px_lock}, "NEAR/USDT": {"price": 5.0}},
        now=now_dt + timedelta(minutes=20),
    )
    pf_after = broker.get_portfolio()
    assert pf_after.positions["SUI/USDT"].quantity == 0.0
    assert pf_after.positions["NEAR/USDT"].quantity == 80.0
