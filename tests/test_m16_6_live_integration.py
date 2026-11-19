"""
Unit & Integration Tests for M16-6 Live Production Capsule Integration (Issue #72).

Verifies:
1. `LiveMultiModalFeedAdapter` post-settlement `fundingRate` filtering (`T+30s..600s`, `<= -20 bps`),
   `1x` & `1000x` Spot-Perp basis parity, `+60s` causal `5m` OI lag, and `< 2%` API weight budget.
2. `Sniper:M25_PostSettleUnwind` signal generation, OI blow-off veto, One-Way non-collision,
   and `(symbol, settlement_hour)` deduplication across restarts.
3. Non-blocking Passive `Post-Only (GTX)` Maker limit placement, Maker fill + immediate bracket
   attachment, and `ttl_sec` auto-cancellation in both Shadow and Live (`BinanceLiveBroker`) modes.
4. Strict `< 54m` & `:56` UTC pre-settlement exit guard (`0.0 bps` negative funding paid),
   `enable_post_settle_sniper` hot-kill isolation, and CLI parity (`run_trading.py` & `run_backtest.py`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from scripts.run_backtest import build_parser as build_backtest_parser
from scripts.run_trading import build_parser as build_trading_parser
from src.domain.entities import Position
from src.domain.enums import OrderSide, OrderStatus, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.feature_flags import FeatureFlags
from src.execution.live_broker import BinanceLiveBroker
from src.execution.multimodal_feeds import LiveMultiModalFeedAdapter, SettledFundingEvent
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.strategies.event_snipers import (
    SNIPER_M09_STRAT,
    SNIPER_M25_STRAT,
    EventDrivenSniperEngine,
)
from src.strategies.meta_allocator import MetaStrategyAllocator


class _DummyResp:
    def __init__(self, payload: Any, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers = headers or {"X-MBX-USED-WEIGHT-1M": "8"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def test_multimodal_feed_adapter_post_settle_spot_basis_and_causal_lag() -> None:
    settle_ts = 1_760_000_000 - (1_760_000_000 % 3600)  # Exact :00:00 UTC boundary
    now_ts = float(settle_ts + 120)  # T + 2m strictly AFTER settlement

    def _mock_get(url: str, params: dict[str, Any] | None = None, timeout: float = 8.0) -> _DummyResp:
        if url.endswith("/fapi/v1/fundingRate"):
            return _DummyResp(
                [
                    # Valid M25 candidate (-34 bps settled 2m ago)
                    {
                        "symbol": "ENAUSDT",
                        "fundingTime": settle_ts * 1000,
                        "fundingRate": "-0.0034",
                        "markPrice": "0.8500",
                    },
                    # Too mild (-8 bps > -20 bps)
                    {
                        "symbol": "SOLUSDT",
                        "fundingTime": settle_ts * 1000,
                        "fundingRate": "-0.0008",
                        "markPrice": "180.0",
                    },
                    # TradFi perp (must be filtered out)
                    {
                        "symbol": "TSLAUSDT",
                        "fundingTime": settle_ts * 1000,
                        "fundingRate": "-0.0050",
                        "markPrice": "250.0",
                    },
                    # Old settlement (8h ago -> outside 10m window)
                    {
                        "symbol": "INJUSDT",
                        "fundingTime": (settle_ts - 28800) * 1000,
                        "fundingRate": "-0.0040",
                        "markPrice": "22.0",
                    },
                ]
            )
        if url.endswith("/fapi/v1/premiumIndex"):
            return _DummyResp(
                [
                    {
                        "symbol": "ENAUSDT",
                        "markPrice": "0.8510",
                        "nextFundingTime": (settle_ts + 14400) * 1000,  # 4h clock -> 238m away
                        "lastFundingRate": "-0.0034",
                    },
                ]
            )
        if url.endswith("/api/v3/ticker/bookTicker"):
            return _DummyResp(
                [
                    {"symbol": "ENAUSDT", "bidPrice": "0.8515", "askPrice": "0.8525"},  # mid = 0.8520
                    {"symbol": "PEPEUSDT", "bidPrice": "0.00001000", "askPrice": "0.00001002"},  # mid = 0.00001001
                ]
            )
        if "openInterestHist" in url:
            rows = []
            for k in range(15):
                ts_k = int(now_ts) - (15 - k) * 300
                rows.append({"timestamp": ts_k * 1000, "sumOpenInterestValue": str(50_000_000.0 - k * 400_000.0)})
            # Unsettled bar only 15s old -> MUST be excluded by +60s safety lag!
            rows.append({"timestamp": int(now_ts - 15) * 1000, "sumOpenInterestValue": "999999999.0"})
            return _DummyResp(rows)
        if "LongShort" in url or "takerlongshortRatio" in url:
            key = "buySellRatio" if "takerlongshortRatio" in url else "longShortRatio"
            return _DummyResp([{"timestamp": int(now_ts - 120) * 1000, key: "0.92"}])
        raise AssertionError(f"Unexpected URL: {url}")

    sess = MagicMock()
    sess.get.side_effect = _mock_get

    adapter = LiveMultiModalFeedAdapter(http_session=sess, metrics_safety_lag_sec=60)
    matched, recent_cnt = adapter.fetch_recent_settled_funding(now_ts=now_ts)
    assert recent_cnt == 2  # ENAUSDT + SOLUSDT settled at settle_ts (TSLA excluded)
    assert len(matched) == 1
    assert matched[0].symbol == "ENAUSDT"
    assert abs(matched[0].funding_rate - (-0.0034)) < 1e-9
    assert matched[0].mins_to_next_settlement > 230.0

    # Verify 1000x Spot-Perp multiplier parity on 1000PEPEUSDT
    pepe_basis = adapter.fetch_spot_perp_basis_snapshot("1000PEPEUSDT", perp_mid=0.01000, now_ts=now_ts)
    assert pepe_basis is not None
    assert pepe_basis.spot_symbol == "PEPEUSDT"
    assert pepe_basis.multiplier == 1000.0
    assert -15.0 < pepe_basis.basis_bps < -5.0

    # Verify +60s causal lag excluded the 999,999,999 unsettled OI spike
    pos = adapter.fetch_derivatives_positioning_5m("ENAUSDT", now_ts=now_ts)
    assert pos is not None
    assert pos.oi_usd < 60_000_000.0
    assert pos.oi_vel_15m < 0.0  # Unwinding OI

    # Total 1m weight across all calls must stay strictly < 2% of 2,400/min (< 48 weight)
    assert adapter.rolling_1m_weight < 48
    assert adapter.budget_utilization_pct < 2.0


def test_m25_post_settle_sniper_non_collision_gtx_fill_ttl_and_pre_settle_exit(tmp_path: Path) -> None:
    db = ShadowDatabase(db_path=str(tmp_path / "m16_6_shadow.db"))
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0)
    router = SmartExecutionRouter(broker=broker)
    allocator = MetaStrategyAllocator(use_seven_factor_engine=False)
    flags = FeatureFlags(
        enable_anomaly_capsule=False,
        enable_event_snipers=True,
        enable_post_settle_sniper=True,
        sniper_slot_leverage=0.20,
        sniper_max_positions=3,
        post_settle_max_funding_bps=-20.0,
        post_settle_gtx_offset_bps=2.0,
        post_settle_gtx_ttl_sec=90.0,
    )
    mock_ex = MagicMock()
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        router=router,
        exchange_client=mock_ex,
        feature_flags=flags,
    )

    # Seed an active Core P1 position on SOL/USDT to verify strict One-Way non-collision
    broker.set_market_price("SOL/USDT", price=180.0, bid=179.95, ask=180.05)
    broker.portfolio.positions["SOL/USDT"] = Position(
        symbol="SOL/USDT",
        side=PositionSide.LONG,
        quantity=3.0,
        entry_price=180.0,
        current_price=180.0,
        strategy="7-Factor Core",
        opened_at=datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc),
    )
    db.save_position(broker.portfolio.positions["SOL/USDT"])
    saved_reb_ts = datetime(2026, 11, 19, 8, 1, 6, tzinfo=timezone.utc)
    daemon.last_rebalance_time = saved_reb_ts

    # Live tickers for ENA/USDT ($45M vol) and WLD/USDT ($30M vol) and SOL/USDT ($500M vol, blocked by Core)
    daemon.latest_tickers = {
        "SOL/USDT": {"price": 180.0, "bid": 179.95, "ask": 180.05, "volume_24h": 500_000_000.0},
        "ENA/USDT": {"price": 1.0000, "bid": 0.9998, "ask": 1.0000, "volume_24h": 45_000_000.0},
        "WLD/USDT": {"price": 2.0000, "bid": 1.9996, "ask": 2.0000, "volume_24h": 30_000_000.0},
    }

    t_scan = datetime(2026, 11, 19, 12, 2, 0, tzinfo=timezone.utc)
    settle_sec = int(datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    settled_evs = [
        # SOLUSDT is more negative (-45 bps), MUST be skipped due to active Core P1 position!
        SettledFundingEvent("SOLUSDT", settle_sec, -0.0045, 180.0, 120.0, 58.0, 8.0),
        SettledFundingEvent("ENAUSDT", settle_sec, -0.0032, 1.0000, 120.0, 58.0, 4.0),
        SettledFundingEvent("WLDUSDT", settle_sec, -0.0025, 2.0000, 120.0, 58.0, 4.0),
    ]

    orders = daemon.run_event_sniper_scan(now=t_scan, settled_funding_override=settled_evs)
    assert len(orders) == 2
    syms_posted = {o.symbol for o in orders}
    assert syms_posted == {"ENA/USDT", "WLD/USDT"}  # SOL/USDT untouched!
    for o in orders:
        assert o.status == OrderStatus.PENDING
        assert o.id.startswith("shd_gtx_")
        assert o.side == OrderSide.SELL
        assert o.strategy == SNIPER_M25_STRAT

    # Re-running scan in the same settlement hour must produce 0 duplicate orders
    dup_orders = daemon.run_event_sniper_scan(now=t_scan + timedelta(seconds=30), settled_funding_override=settled_evs)
    assert dup_orders == []

    # Case A: ENA/USDT touches passive GTX ask + 2 bps (bid rises to 1.0003) -> Maker FILL + immediate brackets!
    # Case B: WLD/USDT drifts lower without touching passive ask -> auto-canceled after 90s TTL!
    ena_ord = next(o for o in orders if o.symbol == "ENA/USDT")
    wld_ord = next(o for o in orders if o.symbol == "WLD/USDT")

    ticks_fill = {
        "ENA/USDT": {"price": 1.0004, "bid": float(ena_ord.price) + 0.0001, "ask": float(ena_ord.price) + 0.0003},
        "WLD/USDT": {"price": 1.9850, "bid": 1.9845, "ask": 1.9855},
    }
    filled = broker.check_pending_orders(ticks_fill)
    assert len(filled) == 1
    assert filled[0].id == ena_ord.id
    daemon.ensure_position_brackets()

    ena_pos = broker.portfolio.positions["ENA/USDT"]
    assert ena_pos.side == PositionSide.SHORT
    assert ena_pos.strategy == SNIPER_M25_STRAT
    assert ena_pos.take_profit is not None and ena_pos.take_profit < ena_pos.entry_price
    assert ena_pos.stop_loss is not None and ena_pos.stop_loss > ena_pos.entry_price

    # Advance past 90s TTL -> WLD/USDT unfilled GTX order must auto-cancel!
    gtx_filled, gtx_expired = broker.reconcile_pending_gtx_orders(
        now=datetime.now(timezone.utc) + timedelta(seconds=120),
        default_ttl_sec=90.0,
    )
    assert gtx_filled == []
    assert [o.id for o in gtx_expired] == [wld_ord.id]
    assert db.get_order(wld_ord.id).status == OrderStatus.CANCELED

    # Verify Pre-Settlement :56 UTC Guard closes ENA/USDT BEFORE 13:00:00 UTC -> 0.0 bps funding paid!
    ena_pos.opened_at = datetime(2026, 11, 19, 12, 10, 0, tzinfo=timezone.utc)
    db.save_position(ena_pos)
    t_pre_settle = datetime(2026, 11, 19, 12, 56, 10, tzinfo=timezone.utc)  # Only 46m old, but minute == 56!
    daemon.manage_trailing_and_profit_locks(
        ticks={"ENA/USDT": {"price": 0.9820}, "SOL/USDT": {"price": 180.0}},
        now=t_pre_settle,
    )
    assert broker.portfolio.positions["ENA/USDT"].side == PositionSide.FLAT
    # Core P1 SOL/USDT and last_rebalance_time must remain 100% untouched!
    assert broker.portfolio.positions["SOL/USDT"].side == PositionSide.LONG
    assert broker.portfolio.positions["SOL/USDT"].quantity == 3.0
    assert daemon.last_rebalance_time == saved_reb_ts
    db.close()


def test_live_broker_gtx_post_only_reprice_and_ttl_cancel(tmp_path: Path) -> None:
    live = BinanceLiveBroker(dry_run=True, initial_capital=3000.0)
    live.dry_run = False
    mock_client = MagicMock()
    mock_client.markets = {
        "ENA/USDT": {
            "limits": {"amount": {"min": 1.0}, "cost": {"min": 5.0}},
            "info": {"filters": [{"filterType": "MIN_NOTIONAL", "notional": "5.0"}]},
        }
    }
    mock_client.amount_to_precision.side_effect = lambda _s, a: f"{float(a):.1f}"
    mock_client.price_to_precision.side_effect = lambda _s, p: f"{float(p):.4f}"
    mock_client.fetch_order_book.return_value = {"bids": [[0.9998, 1000.0]], "asks": [[1.0000, 1000.0]]}

    # Simulate first GTX attempt rejected (-5022 post-only cross), second attempt accepted
    call_count = {"n": 0}

    def _create_order(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        assert kwargs["type"] == "limit"
        assert kwargs["params"]["timeInForce"] == "GTX"
        if call_count["n"] == 1:
            raise RuntimeError("binance -5022 Post Only order rejected")
        return {"id": "bin_gtx_777", "status": "open", "filled": 0.0, "price": kwargs["price"]}

    mock_client.create_order.side_effect = _create_order
    mock_client.fetch_order.return_value = {"id": "bin_gtx_777", "status": "open", "filled": 0.0, "price": 1.0003}
    mock_client.cancel_order.return_value = {"id": "bin_gtx_777", "status": "canceled", "filled": 0.0}
    live.client = mock_client

    db = ShadowDatabase(db_path=str(tmp_path / "m16_6_live_mock.db"))
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0, live_broker=live)
    broker.set_market_price("ENA/USDT", price=1.0000, bid=0.9998, ask=1.0000)

    t0 = datetime(2026, 11, 19, 12, 2, 0, tzinfo=timezone.utc)
    ord_gtx = broker.create_passive_gtx_order(
        symbol="ENA/USDT",
        side=OrderSide.SELL,
        quantity=150.0,
        limit_price=1.0002,
        ttl_sec=60.0,
        strategy=SNIPER_M25_STRAT,
        now=t0,
    )
    assert call_count["n"] == 2
    assert ord_gtx.status == OrderStatus.PENDING

    # Before TTL: stays open, zero market fallback orders sent
    f1, e1 = broker.reconcile_pending_gtx_orders(now=t0 + timedelta(seconds=20), default_ttl_sec=60.0)
    assert f1 == [] and e1 == []
    mock_client.cancel_order.assert_not_called()

    # After TTL: cancels cleanly on Binance without ever sending a market order
    f2, e2 = broker.reconcile_pending_gtx_orders(now=t0 + timedelta(seconds=65), default_ttl_sec=60.0)
    assert f2 == [] and len(e2) == 1
    mock_client.cancel_order.assert_called_once_with("bin_gtx_777", "ENA/USDT")
    assert db.get_order(ord_gtx.id).status == OrderStatus.CANCELED
    db.close()


def test_feature_flag_hot_kill_isolates_m25_from_m09_and_cli_parity(tmp_path: Path) -> None:
    db = ShadowDatabase(db_path=str(tmp_path / "m16_6_ff.db"))
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=MetaStrategyAllocator(use_seven_factor_engine=False),
        exchange_client=MagicMock(),
    )

    # Open one M25 position and one M09 position
    now_dt = datetime(2026, 11, 19, 12, 15, 0, tzinfo=timezone.utc)
    broker.set_market_price("ENA/USDT", price=1.0)
    broker.set_market_price("LINK/USDT", price=15.0)
    broker.portfolio.positions["ENA/USDT"] = Position(
        symbol="ENA/USDT",
        side=PositionSide.SHORT,
        quantity=200.0,
        entry_price=1.0,
        current_price=1.0,
        strategy=SNIPER_M25_STRAT,
        opened_at=now_dt - timedelta(minutes=10),
    )
    broker.portfolio.positions["LINK/USDT"] = Position(
        symbol="LINK/USDT",
        side=PositionSide.LONG,
        quantity=20.0,
        entry_price=15.0,
        current_price=15.0,
        strategy=SNIPER_M09_STRAT,
        opened_at=now_dt - timedelta(minutes=30),
    )
    db.save_position(broker.portfolio.positions["ENA/USDT"])
    db.save_position(broker.portfolio.positions["LINK/USDT"])

    # Hot-disable ONLY `enable_post_settle_sniper` in SQLite
    ff = daemon.feature_flags.to_dict()
    ff["enable_post_settle_sniper"] = False
    db.save_feature_flags(ff)
    daemon.sync_feature_flags_from_db()
    assert daemon.enable_post_settle_sniper is False
    assert daemon.enable_event_snipers is True

    daemon.manage_trailing_and_profit_locks(
        ticks={"ENA/USDT": {"price": 1.0}, "LINK/USDT": {"price": 15.0}},
        now=now_dt,
    )
    # M25 auto-closed, while M09 remains untouched!
    assert broker.portfolio.positions["ENA/USDT"].side == PositionSide.FLAT
    assert broker.portfolio.positions["LINK/USDT"].side == PositionSide.LONG

    # Verify CLI parity across run_trading.py and run_backtest.py
    tp = build_trading_parser()
    args_off = tp.parse_args(["--shadow", "--disable-m16-post-settle"])
    ff_off = FeatureFlags.from_env_and_args(args=args_off, mode="shadow", environ={})
    assert ff_off.enable_post_settle_sniper is False

    bp = build_backtest_parser()
    b_args = bp.parse_args(["--disable-m16-post-settle"])
    assert b_args.enable_post_settle_sniper is False

    eng = EventDrivenSniperEngine()
    # Verify OI velocity > +8% blow-off veto
    vetoed = eng.evaluate_post_settlement_unwind_snipers(
        settled_events=[SettledFundingEvent("ENAUSDT", 1_760_000_000, -0.0035, 1.0, 90.0, 55.0)],
        ticker_volumes_24h={"ENAUSDT": 50_000_000.0},
        positioning_snapshots={"ENAUSDT": {"oi_vel_15m": 0.12}},
    )
    assert vetoed == []
    db.close()
