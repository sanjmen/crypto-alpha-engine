"""
Unit tests for PositionReconciliationEngine, Dual-Entry Shadow Accounting,
Orphan Order Cancellation, and Drift Resolution Protocol.
"""

from unittest.mock import MagicMock

from src.alerts.dispatcher import (
    AlertDispatcher,
    AlertMessage,
    AlertPriority,
    WebhookConfig,
)
from src.domain.entities import Order, Portfolio, Position
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.risk.reconciliation import (
    PositionReconciliationEngine,
    ReconciliationStatus,
)


def test_position_reconciliation_in_sync():
    """Verifies that matching internal and exchange positions results in IN_SYNC status."""
    db = ShadowDatabase(":memory:")
    mock_exchange = MagicMock()

    # Local portfolio: 0.5 BTC, -10.0 ETH
    portfolio = Portfolio(
        cash=50_000.0,
        initial_cash=50_000.0,
        positions={
            "BTC/USDT": Position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=0.5,
                entry_price=60_000.0,
                current_price=60_000.0,
            ),
            "ETH/USDT": Position(
                symbol="ETH/USDT",
                side=PositionSide.SHORT,
                quantity=-10.0,
                entry_price=3_000.0,
                current_price=3_000.0,
            ),
        },
    )

    # Exchange reports identical positions
    exchange_positions = [
        {"symbol": "BTCUSDT", "contracts": 0.5, "side": "long"},
        {"symbol": "ETHUSDT", "contracts": 10.0, "side": "short"},
    ]
    exchange_balance = {"USDT": {"total": 50_000.0}}

    engine = PositionReconciliationEngine(
        exchange_client=mock_exchange,
        database=db,
    )

    report = engine.run_reconciliation(
        local_portfolio=portfolio,
        exchange_positions=exchange_positions,
        exchange_balance=exchange_balance,
        exchange_open_orders=[],
    )

    assert report.status == ReconciliationStatus.IN_SYNC
    assert not report.trading_halted
    assert len(report.drifts) == 2
    assert all(not d.exceeds_threshold for d in report.drifts)


def test_position_reconciliation_drift_triggers_halt_and_alert():
    """Verifies that a >0.1% position discrepancy trips the halt breaker and emits a critical alert."""
    db = ShadowDatabase(":memory:")
    mock_exchange = MagicMock()

    dispatched_alerts: list[AlertMessage] = []

    def on_alert(msg: AlertMessage, _ch: str):
        dispatched_alerts.append(msg)

    dispatcher = AlertDispatcher(
        config=WebhookConfig(mock_mode=True),
        on_delivered=on_alert,
    )
    dispatcher.start()

    engine = PositionReconciliationEngine(
        exchange_client=mock_exchange,
        database=db,
        alert_dispatcher=dispatcher,
        max_relative_drift=0.001,  # 0.1%
        auto_halt_trading=True,
    )

    # Local says 1.0 BTC, Exchange says 0.8 BTC (20% drift!)
    portfolio = Portfolio(
        cash=10_000.0,
        initial_cash=10_000.0,
        positions={
            "BTCUSDT": Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=1.0,
                entry_price=60000.0,
                current_price=60000.0,
            )
        },
    )

    exchange_positions = [
        {"symbol": "BTCUSDT", "contracts": 0.8, "side": "long"},
    ]

    report = engine.run_reconciliation(
        local_portfolio=portfolio,
        exchange_positions=exchange_positions,
        exchange_balance={"USDT": {"total": 10_000.0}},
    )

    assert report.status == ReconciliationStatus.DRIFT_DETECTED
    assert report.trading_halted
    assert engine.is_trading_halted()

    # Verify drift details
    btc_drift = next(d for d in report.drifts if d.symbol == "BTCUSDT")
    assert btc_drift.exceeds_threshold
    assert abs(btc_drift.absolute_drift - 0.2) < 1e-6
    assert abs(btc_drift.relative_drift - 0.2) < 1e-4

    # Stop dispatcher and inspect alert
    dispatcher.stop(timeout=1.0)
    assert len(dispatched_alerts) >= 1
    crit_alert = dispatched_alerts[0]
    assert crit_alert.priority == AlertPriority.CRITICAL
    assert "POSITION RECONCILIATION DRIFT" in crit_alert.title

    # Resume trading manual test
    engine.resume_trading("Manual resolution of BTC drift")
    assert not engine.is_trading_halted()


def test_position_reconciliation_orphan_orders_cancellation():
    """Verifies orphan exchange order detection and automated cancellation."""
    mock_exchange = MagicMock()

    engine = PositionReconciliationEngine(
        exchange_client=mock_exchange,
        cancel_orphan_orders=True,
    )

    portfolio = Portfolio(cash=1000.0, initial_cash=1000.0, positions={})

    # Exchange has 2 open orders, but locally only order 1 is known
    exchange_open_orders = [
        {"id": "order-1", "symbol": "BTC/USDT", "side": "buy", "amount": 0.1},
        {"id": "orphan-order-99", "symbol": "ETH/USDT", "side": "sell", "amount": 2.0},
    ]

    local_open_orders = [
        Order(
            id="order-1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
            status=OrderStatus.PENDING,
        )
    ]

    report = engine.run_reconciliation(
        local_portfolio=portfolio,
        local_open_orders=local_open_orders,
        exchange_positions=[],
        exchange_balance={"USDT": 1000.0},
        exchange_open_orders=exchange_open_orders,
    )

    assert report.status == ReconciliationStatus.ORPHAN_ORDERS_DETECTED
    assert len(report.orphan_orders) == 1
    assert report.orphan_orders[0].order_id == "orphan-order-99"
    assert report.orphan_orders[0].canceled_automatically

    # Verify mock_exchange.cancel_order was called for the orphan order
    mock_exchange.cancel_order.assert_called_once_with("orphan-order-99", "ETH/USDT")


def test_reconciliation_database_audit_logging():
    """Verifies that reconciliation audit history is saved and queryable from SQLite."""
    db = ShadowDatabase(":memory:")
    mock_exchange = MagicMock()

    engine = PositionReconciliationEngine(
        exchange_client=mock_exchange,
        database=db,
    )

    portfolio = Portfolio(cash=10_000.0, initial_cash=10_000.0, positions={})
    report = engine.run_reconciliation(
        local_portfolio=portfolio,
        exchange_positions=[],
        exchange_balance={"USDT": 10_000.0},
    )

    recent = db.get_recent_reconciliations(limit=10)
    assert len(recent) == 1
    assert recent[0]["status"] == report.status.value
    assert "details" in recent[0]
    assert recent[0]["details"]["local_cash"] == 10_000.0
