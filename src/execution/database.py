"""
ACID-Compliant SQLite Persistence Layer for Live Shadow Trading.
Stores accounts, positions, orders, trades, funding settlements, and rebalance snapshots.
Uses SQLite WAL (Write-Ahead Logging) mode for high-concurrency non-blocking reads.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities import Order, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/shadow_trading.db"


class ShadowDatabase:
    """
    Thread-safe, WAL-enabled relational storage for local shadow trading.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()

        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, manual transactions via BEGIN/COMMIT
        )
        self._conn.row_factory = sqlite3.Row

        self._initialize_pragmas_and_schema()

    def _initialize_pragmas_and_schema(self) -> None:
        """Configures performance pragmas and creates tables if they don't exist."""
        with self._lock:
            cursor = self._conn.cursor()
            if self.db_path != ":memory:":
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA foreign_keys = ON;")

            # 1. Accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_equity REAL NOT NULL,
                    margin_used REAL NOT NULL,
                    free_margin REAL NOT NULL,
                    leverage REAL NOT NULL
                );
            """)

            # 2. Positions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL CHECK(side IN ('LONG', 'SHORT', 'FLAT')),
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    notional REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    liquidation_price REAL,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 3. Orders
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    client_order_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    order_type TEXT NOT NULL CHECK(order_type IN ('LIMIT', 'MARKET', 'STOP_LOSS', 'TAKE_PROFIT')),
                    price REAL,
                    quantity REAL NOT NULL,
                    filled_quantity REAL NOT NULL DEFAULT 0.0,
                    average_fill_price REAL DEFAULT 0.0,
                    status TEXT NOT NULL CHECK(status IN ('PENDING', 'FILLED', 'CANCELED', 'CANCELLED', 'REJECTED', 'PARTIALLY_FILLED')),
                    role TEXT CHECK(role IN ('MAKER', 'TAKER', NULL)),
                    fee REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 4. Trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notional REAL NOT NULL,
                    fee_paid REAL NOT NULL,
                    is_maker INTEGER NOT NULL CHECK(is_maker IN (0, 1)),
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES orders(id)
                );
            """)

            # 5. Funding settlements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_settlements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL NOT NULL,
                    position_side TEXT NOT NULL CHECK(position_side IN ('LONG', 'SHORT')),
                    position_notional REAL NOT NULL,
                    cashflow_credited REAL NOT NULL
                );
            """)

            # 6. Rebalance snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rebalance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime_detected TEXT NOT NULL,
                    momentum_weight REAL NOT NULL,
                    stat_arb_weight REAL NOT NULL,
                    carry_weight REAL NOT NULL,
                    target_allocations_json TEXT NOT NULL,
                    executed_delta_json TEXT NOT NULL
                );
            """)

            # Indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_accounts_ts ON accounts(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_settlements(timestamp);")

    def close(self) -> None:
        """Closes the underlying database connection."""
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------------------------
    # Account & Portfolio Operations
    # -------------------------------------------------------------------------

    def save_account_snapshot(
        self,
        initial_capital: float,
        cash_balance: float,
        unrealized_pnl: float,
        realized_pnl: float,
        total_equity: float,
        margin_used: float,
        free_margin: float,
        leverage: float,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Inserts an account equity snapshot."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO accounts (
                    timestamp, initial_capital, cash_balance, unrealized_pnl,
                    realized_pnl, total_equity, margin_used, free_margin, leverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    float(initial_capital),
                    float(cash_balance),
                    float(unrealized_pnl),
                    float(realized_pnl),
                    float(total_equity),
                    float(margin_used),
                    float(free_margin),
                    float(leverage),
                ),
            )
            return cursor.lastrowid

    def get_latest_account(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent account snapshot or None."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM accounts ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_account_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns recent account equity snapshots in ascending order."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT * FROM (
                    SELECT * FROM accounts ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Position Operations
    # -------------------------------------------------------------------------

    def save_position(
        self,
        position: Position,
        realized_pnl: float = 0.0,
        liquidation_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Upserts an active or flat position record."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO positions (
                    symbol, side, quantity, entry_price, current_price,
                    notional, unrealized_pnl, realized_pnl, liquidation_price,
                    opened_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side = excluded.side,
                    quantity = excluded.quantity,
                    entry_price = excluded.entry_price,
                    current_price = excluded.current_price,
                    notional = excluded.notional,
                    unrealized_pnl = excluded.unrealized_pnl,
                    realized_pnl = excluded.realized_pnl,
                    liquidation_price = excluded.liquidation_price,
                    updated_at = excluded.updated_at
                """,
                (
                    position.symbol,
                    position.side.value if hasattr(position.side, "value") else str(position.side),
                    float(position.quantity),
                    float(position.entry_price),
                    float(position.current_price),
                    float(position.market_value),
                    float(position.unrealized_pnl),
                    float(realized_pnl),
                    float(liquidation_price) if liquidation_price is not None else None,
                    ts,
                    ts,
                ),
            )

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Queries single position by symbol."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_open_positions(self) -> Dict[str, Position]:
        """Returns all non-flat positions mapped to domain Position objects."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM positions WHERE side != 'FLAT' AND quantity > 0"
            )
            rows = cursor.fetchall()
            positions: Dict[str, Position] = {}
            for r in rows:
                side_enum = PositionSide(r["side"])
                positions[r["symbol"]] = Position(
                    symbol=r["symbol"],
                    side=side_enum,
                    quantity=float(r["quantity"]),
                    entry_price=float(r["entry_price"]),
                    current_price=float(r["current_price"]),
                )
            return positions

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------

    def save_order(
        self,
        order: Order,
        role: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> None:
        """Inserts or updates an order in the database."""
        created_str = (
            order.created_at.isoformat()
            if isinstance(order.created_at, datetime)
            else str(order.created_at)
        )
        updated_str = datetime.now(timezone.utc).isoformat()
        side_val = order.side.value if hasattr(order.side, "value") else str(order.side)
        type_val = order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type)
        status_val = order.status.value if hasattr(order.status, "value") else str(order.status)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO orders (
                    id, client_order_id, symbol, side, order_type,
                    price, quantity, filled_quantity, average_fill_price,
                    status, role, fee, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filled_quantity = excluded.filled_quantity,
                    average_fill_price = excluded.average_fill_price,
                    status = excluded.status,
                    role = COALESCE(excluded.role, orders.role),
                    fee = excluded.fee,
                    updated_at = excluded.updated_at
                """,
                (
                    order.id,
                    client_order_id,
                    order.symbol,
                    side_val,
                    type_val,
                    float(order.price) if order.price is not None else None,
                    float(order.quantity),
                    float(order.filled_quantity),
                    float(order.average_fill_price),
                    status_val,
                    role,
                    0.0,
                    created_str,
                    updated_str,
                ),
            )

    def update_order_fill(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: float,
        average_fill_price: float,
        fee: float,
        role: Optional[str] = None,
    ) -> bool:
        """Updates fill state and status of an order."""
        status_val = status.value if hasattr(status, "value") else str(status)
        updated_str = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE orders
                SET status = ?,
                    filled_quantity = ?,
                    average_fill_price = ?,
                    fee = ?,
                    role = COALESCE(?, role),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status_val,
                    float(filled_quantity),
                    float(average_fill_price),
                    float(fee),
                    role,
                    updated_str,
                    order_id,
                ),
            )
            return cursor.rowcount > 0

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieves domain Order entity by id."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
            row = cursor.fetchone()
            if not row:
                return None

            status_str = row["status"]
            # Handle normalized status string
            if status_str == "CANCELLED":
                status_enum = OrderStatus.CANCELED
            else:
                status_enum = OrderStatus(status_str)

            return Order(
                id=row["id"],
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
                order_type=OrderType(row["order_type"]),
                quantity=float(row["quantity"]),
                price=float(row["price"]) if row["price"] is not None else None,
                status=status_enum,
                filled_quantity=float(row["filled_quantity"]),
                average_fill_price=float(row["average_fill_price"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Retrieves all open or partially filled orders."""
        with self._lock:
            cursor = self._conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIALLY_FILLED') AND symbol = ?",
                    (symbol,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIALLY_FILLED')"
                )
            rows = cursor.fetchall()
            orders = []
            for r in rows:
                status_str = r["status"]
                status_enum = (
                    OrderStatus.CANCELED if status_str == "CANCELLED" else OrderStatus(status_str)
                )
                orders.append(
                    Order(
                        id=r["id"],
                        symbol=r["symbol"],
                        side=OrderSide(r["side"]),
                        order_type=OrderType(r["order_type"]),
                        quantity=float(r["quantity"]),
                        price=float(r["price"]) if r["price"] is not None else None,
                        status=status_enum,
                        filled_quantity=float(r["filled_quantity"]),
                        average_fill_price=float(r["average_fill_price"]),
                        created_at=datetime.fromisoformat(r["created_at"]),
                    )
                )
            return orders

    # -------------------------------------------------------------------------
    # Trade Operations
    # -------------------------------------------------------------------------

    def save_trade(
        self,
        trade: Trade,
        is_maker: bool = False,
        realized_pnl: float = 0.0,
    ) -> None:
        """Records an executed trade in the database."""
        ts = (
            trade.timestamp.isoformat()
            if isinstance(trade.timestamp, datetime)
            else str(trade.timestamp)
        )
        notional = float(trade.price * trade.quantity)
        side_val = trade.side.value if hasattr(trade.side, "value") else str(trade.side)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    id, order_id, symbol, side, price, quantity,
                    notional, fee_paid, is_maker, realized_pnl, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    trade.order_id,
                    trade.symbol,
                    side_val,
                    float(trade.price),
                    float(trade.quantity),
                    notional,
                    float(trade.fee),
                    1 if is_maker else 0,
                    float(realized_pnl),
                    ts,
                ),
            )

    def get_trades(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns recent trade executions."""
        with self._lock:
            cursor = self._conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Funding Settlements Operations
    # -------------------------------------------------------------------------

    def record_funding_settlement(
        self,
        symbol: str,
        funding_rate: float,
        position_side: str,
        position_notional: float,
        cashflow_credited: float,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Records an official 8h perpetual funding settlement."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO funding_settlements (
                    timestamp, symbol, funding_rate, position_side,
                    position_notional, cashflow_credited
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol,
                    float(funding_rate),
                    position_side,
                    float(position_notional),
                    float(cashflow_credited),
                ),
            )
            return cursor.lastrowid

    def get_funding_settlements(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent funding settlements."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM funding_settlements ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Rebalance Snapshots Operations
    # -------------------------------------------------------------------------

    def save_rebalance_snapshot(
        self,
        regime_detected: str,
        momentum_weight: float,
        stat_arb_weight: float,
        carry_weight: float,
        target_allocations: Dict[str, float],
        executed_delta: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Saves a meta-strategy allocation snapshot."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO rebalance_snapshots (
                    timestamp, regime_detected, momentum_weight,
                    stat_arb_weight, carry_weight, target_allocations_json,
                    executed_delta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    regime_detected,
                    float(momentum_weight),
                    float(stat_arb_weight),
                    float(carry_weight),
                    json.dumps(target_allocations),
                    json.dumps(executed_delta),
                ),
            )
            return cursor.lastrowid

    def get_latest_rebalance_snapshot(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent rebalance snapshot."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM rebalance_snapshots ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["target_allocations"] = json.loads(res["target_allocations_json"])
            res["executed_delta"] = json.loads(res["executed_delta_json"])
            return res

    def reset_all_data(self) -> None:
        """Clears all records from all tables (used for testing and fresh start)."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM trades;")
            cursor.execute("DELETE FROM orders;")
            cursor.execute("DELETE FROM positions;")
            cursor.execute("DELETE FROM accounts;")
            cursor.execute("DELETE FROM funding_settlements;")
            cursor.execute("DELETE FROM rebalance_snapshots;")
