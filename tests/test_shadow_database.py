"""
Unit Tests for SQLite WAL Shadow Trading Database.
"""

from datetime import datetime, timezone
import os
import tempfile
import pytest

from src.domain.entities import Order, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase


@pytest.fixture
def db():
    """Provides an in-memory SQLite database instance for tests."""
    database = ShadowDatabase(db_path=":memory:")
    yield database
    database.close()


def test_schema_initialization(db: ShadowDatabase):
    """Verifies that all tables and indices are created successfully."""
    cursor = db._conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "accounts" in tables
    assert "positions" in tables
    assert "orders" in tables
    assert "trades" in tables
    assert "funding_settlements" in tables
    assert "rebalance_snapshots" in tables


def test_account_snapshot_lifecycle(db: ShadowDatabase):
    """Verifies saving and querying account balance snapshots."""
    row_id = db.save_account_snapshot(
        initial_capital=100_000.0,
        cash_balance=95_000.0,
        unrealized_pnl=2_500.0,
        realized_pnl=1_000.0,
        total_equity=98_500.0,
        margin_used=15_000.0,
        free_margin=80_000.0,
        leverage=0.985,
    )
    assert row_id == 1

    latest = db.get_latest_account()
    assert latest is not None
    assert latest["cash_balance"] == 95_000.0
    assert latest["total_equity"] == 98_500.0
    assert latest["leverage"] == 0.985

    # Insert second snapshot
    db.save_account_snapshot(
        initial_capital=100_000.0,
        cash_balance=96_000.0,
        unrealized_pnl=3_000.0,
        realized_pnl=1_000.0,
        total_equity=100_000.0,
        margin_used=15_000.0,
        free_margin=81_000.0,
        leverage=1.0,
    )
    history = db.get_account_history(limit=10)
    assert len(history) == 2
    assert history[0]["id"] == 1
    assert history[1]["id"] == 2
    assert history[1]["total_equity"] == 100_000.0


def test_position_lifecycle(db: ShadowDatabase):
    """Verifies upserting positions, mark-to-market updates, and filtering open positions."""
    pos_btc = Position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=0.5,
        entry_price=90_000.0,
        current_price=92_000.0,
    )
    db.save_position(pos_btc, realized_pnl=0.0, liquidation_price=75_000.0)

    # Query single position
    retrieved = db.get_position("BTC/USDT")
    assert retrieved is not None
    assert retrieved["symbol"] == "BTC/USDT"
    assert retrieved["side"] == "LONG"
    assert retrieved["quantity"] == 0.5
    assert retrieved["unrealized_pnl"] == 1_000.0
    assert retrieved["liquidation_price"] == 75_000.0

    # Query open positions
    open_pos = db.get_open_positions()
    assert "BTC/USDT" in open_pos
    assert open_pos["BTC/USDT"].side == PositionSide.LONG

    # Update position to FLAT
    pos_flat = Position(
        symbol="BTC/USDT",
        side=PositionSide.FLAT,
        quantity=0.0,
        entry_price=0.0,
        current_price=92_000.0,
    )
    db.save_position(pos_flat, realized_pnl=1_000.0)

    open_pos_after = db.get_open_positions()
    assert "BTC/USDT" not in open_pos_after

    retrieved_flat = db.get_position("BTC/USDT")
    assert retrieved_flat["side"] == "FLAT"
    assert retrieved_flat["realized_pnl"] == 1_000.0


def test_order_and_trade_lifecycle(db: ShadowDatabase):
    """Verifies order persistence, status transitions, and executed trades."""
    order = Order(
        id="ord_test_001",
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2.0,
        price=3_500.0,
        status=OrderStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )
    db.save_order(order, role="MAKER", client_order_id="cl_001")

    # Verify pending
    pending = db.get_pending_orders()
    assert len(pending) == 1
    assert pending[0].id == "ord_test_001"
    assert pending[0].status == OrderStatus.PENDING

    # Fill order
    updated = db.update_order_fill(
        order_id="ord_test_001",
        status=OrderStatus.FILLED,
        filled_quantity=2.0,
        average_fill_price=3_500.0,
        fee=1.40,
        role="MAKER",
    )
    assert updated is True

    # Check pending list is now empty
    assert len(db.get_pending_orders()) == 0

    retrieved_order = db.get_order("ord_test_001")
    assert retrieved_order is not None
    assert retrieved_order.status == OrderStatus.FILLED
    assert retrieved_order.filled_quantity == 2.0
    assert retrieved_order.average_fill_price == 3_500.0

    # Record trade
    trade = Trade(
        id="trd_test_001",
        order_id="ord_test_001",
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        quantity=2.0,
        price=3_500.0,
        fee=1.40,
        timestamp=datetime.now(timezone.utc),
    )
    db.save_trade(trade, is_maker=True, realized_pnl=0.0)

    trades = db.get_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]["id"] == "trd_test_001"
    assert trades[0]["order_id"] == "ord_test_001"
    assert trades[0]["is_maker"] == 1
    assert trades[0]["fee_paid"] == 1.40
    assert trades[0]["notional"] == 7_000.0


def test_funding_settlement(db: ShadowDatabase):
    """Verifies recording of 8h perpetual funding settlements."""
    settlement_id = db.record_funding_settlement(
        symbol="BTC/USDT",
        funding_rate=0.0001,  # 1 bp
        position_side="LONG",
        position_notional=50_000.0,
        cashflow_credited=-5.0,  # Long pays positive funding
    )
    assert settlement_id == 1

    settlements = db.get_funding_settlements(limit=5)
    assert len(settlements) == 1
    assert settlements[0]["symbol"] == "BTC/USDT"
    assert settlements[0]["funding_rate"] == 0.0001
    assert settlements[0]["cashflow_credited"] == -5.0


def test_rebalance_snapshot(db: ShadowDatabase):
    """Verifies storing and retrieving meta-strategy rebalance decisions."""
    target_allocs = {"BTC/USDT": 40_000.0, "ETH/USDT": 30_000.0}
    executed_deltas = {"BTC/USDT": 5_000.0, "ETH/USDT": -2_000.0}

    snapshot_id = db.save_rebalance_snapshot(
        regime_detected="TRENDING",
        momentum_weight=0.5,
        stat_arb_weight=0.3,
        carry_weight=0.2,
        target_allocations=target_allocs,
        executed_delta=executed_deltas,
    )
    assert snapshot_id == 1

    latest = db.get_latest_rebalance_snapshot()
    assert latest is not None
    assert latest["regime_detected"] == "TRENDING"
    assert latest["momentum_weight"] == 0.5
    assert latest["target_allocations"]["BTC/USDT"] == 40_000.0
    assert latest["executed_delta"]["ETH/USDT"] == -2_000.0


def test_reset_all_data(db: ShadowDatabase):
    """Verifies table cleanup on reset."""
    db.save_account_snapshot(
        initial_capital=100_000.0,
        cash_balance=100_000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        total_equity=100_000.0,
        margin_used=0.0,
        free_margin=100_000.0,
        leverage=0.0,
    )
    assert len(db.get_account_history()) == 1

    db.reset_all_data()
    assert len(db.get_account_history()) == 0


def test_file_based_wal_mode():
    """Verifies that a disk-backed SQLite database enables WAL mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_shadow.db")
        disk_db = ShadowDatabase(db_path=db_path)

        cursor = disk_db._conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

        disk_db.close()
