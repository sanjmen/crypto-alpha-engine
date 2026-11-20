from pathlib import Path

from scripts.audit_live_execution_microstructure import audit_live_execution_db
from src.domain.entities import Order, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType
from src.execution.database import ShadowDatabase


def test_audit_live_execution_db_slippage_and_maker_split(tmp_path: Path):
    db_file = tmp_path / "test_live.db"
    db = ShadowDatabase(str(db_file))

    from datetime import datetime, timezone

    t0 = datetime(2026, 11, 20, 8, 0, 10, tzinfo=timezone.utc)
    o1 = Order(
        id="ord_m1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=90000.0,
        status=OrderStatus.PENDING,
        created_at=t0,
        strategy="7-Factor Core",
    )
    db.save_order(o1, role="MAKER")
    db.update_order_fill("ord_m1", OrderStatus.FILLED, 0.01, 89991.0, fee=0.18, role="MAKER")
    db.save_trade(
        Trade(
            id="trd_1",
            order_id="ord_m1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.01,
            price=89991.0,
            fee=0.18,
            timestamp=t0,
            strategy="7-Factor Core",
        ),
        is_maker=True,
    )

    o2 = Order(
        id="ord_t2",
        symbol="FET/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=500.0,
        price=1.0000,
        status=OrderStatus.PENDING,
        created_at=t0,
        strategy="7-Factor Core",
    )
    db.save_order(o2, role="MAKER")
    db.update_order_fill("ord_t2", OrderStatus.FILLED, 500.0, 1.0012, fee=0.2503, role="TAKER")
    db.save_trade(
        Trade(
            id="trd_2",
            order_id="ord_t2",
            symbol="FET/USDT",
            side=OrderSide.BUY,
            quantity=500.0,
            price=1.0012,
            fee=0.2503,
            timestamp=t0,
            strategy="7-Factor Core",
        ),
        is_maker=False,
    )

    rep = audit_live_execution_db(db_file)
    assert rep["total_filled_orders"] == 2
    assert rep["by_execution_path"]["LIMIT_FILLED_AS_MAKER"]["n_orders"] == 1
    assert rep["by_execution_path"]["LIMIT_FELL_BACK_TO_TAKER_15S"]["n_orders"] == 1
    # FET BUY at 1.0012 vs 1.0000 = +12.0 bps slippage + 5.0 bps fee = +17.0 bps all-in!
    assert abs(rep["by_execution_path"]["LIMIT_FELL_BACK_TO_TAKER_15S"]["mean_slippage_bps"] - 12.0) < 0.05
