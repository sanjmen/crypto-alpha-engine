"""
ACID-Compliant SQLite Persistence Layer for Live Shadow Trading.
Stores accounts, positions, orders, trades, funding settlements, and rebalance snapshots.
Uses SQLite WAL (Write-Ahead Logging) mode for high-concurrency non-blocking reads.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any

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
        self._last_account_row_id: int | None = None
        self._last_account_insert_mono: float = 0.0
        self._last_account_cash: float | None = None
        self._last_account_init_capital: float | None = None

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
                    take_profit REAL,
                    stop_loss REAL,
                    peak_price REAL DEFAULT 0.0,
                    strategy TEXT,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # Schema migrations for existing databases
            try:
                cursor.execute("ALTER TABLE positions ADD COLUMN take_profit REAL;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE positions ADD COLUMN stop_loss REAL;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE positions ADD COLUMN peak_price REAL DEFAULT 0.0;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE positions ADD COLUMN strategy TEXT;")
            except Exception:
                pass

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
                    strategy TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            try:
                cursor.execute("ALTER TABLE orders ADD COLUMN strategy TEXT;")
            except Exception:
                pass

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
                    strategy TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES orders(id)
                );
            """)

            try:
                cursor.execute("ALTER TABLE trades ADD COLUMN strategy TEXT;")
            except Exception:
                pass

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

            # 7. Reconciliation audit logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reconciliation_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    drift_count INTEGER NOT NULL,
                    orphan_count INTEGER NOT NULL,
                    cash_drift REAL NOT NULL,
                    trading_halted INTEGER NOT NULL,
                    details_json TEXT NOT NULL
                );
            """)

            # 8. Daemon Heartbeats & WebSocket Telemetry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daemon_heartbeat (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    timestamp TEXT NOT NULL,
                    ws_state TEXT NOT NULL,
                    symbols_count INTEGER NOT NULL,
                    total_ticks INTEGER NOT NULL,
                    ticks_per_sec REAL NOT NULL,
                    last_tick_time TEXT,
                    last_rebalance_time TEXT,
                    next_rebalance_in_sec REAL,
                    status TEXT NOT NULL
                );
            """)

            # 9. Market Pulse & 24h Live Rankings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_pulse (
                    symbol TEXT PRIMARY KEY,
                    price REAL NOT NULL,
                    change_24h REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    high_24h REAL,
                    low_24h REAL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 10. 7-Factor Online Engine & Stepped Leverage Telemetry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seven_factor_telemetry (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    timestamp TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    current_equity REAL NOT NULL,
                    high_water_mark REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    unlocked_tier REAL NOT NULL,
                    active_base_leverage REAL NOT NULL,
                    vol_scalar REAL NOT NULL,
                    effective_leverage REAL NOT NULL,
                    next_tier_leverage REAL NOT NULL,
                    next_tier_hwm_target REAL NOT NULL,
                    progress_to_next_tier_pct REAL NOT NULL,
                    portfolio_beta REAL NOT NULL,
                    conviction REAL NOT NULL,
                    factor_weights_json TEXT NOT NULL,
                    factor_tstats_json TEXT NOT NULL
                );
            """)

            # Indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_accounts_ts ON accounts(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_settlements(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reconciliation_ts ON reconciliation_audit_logs(timestamp);")

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
        timestamp: datetime | None = None,
        min_interval_sec: float = 0.0,
    ) -> int:
        """
        Inserts an account equity snapshot, or updates the latest snapshot in-place
        when `min_interval_sec > 0` and neither cash_balance nor initial_capital changed
        within the throttling window (Issue #59 / M14 Option A).
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        now_mono = time.monotonic()
        init_cap_f = float(initial_capital)
        cash_f = float(cash_balance)
        upnl_f = float(unrealized_pnl)
        rpnl_f = float(realized_pnl)
        eq_f = float(total_equity)
        margin_f = float(margin_used)
        free_f = float(free_margin)
        lev_f = float(leverage)

        with self._lock:
            cursor = self._conn.cursor()
            can_update_in_place = (
                min_interval_sec > 0.0
                and self._last_account_row_id is not None
                and (now_mono - self._last_account_insert_mono) < min_interval_sec
                and self._last_account_cash is not None
                and abs(cash_f - self._last_account_cash) < 1e-4
                and self._last_account_init_capital is not None
                and abs(init_cap_f - self._last_account_init_capital) < 1e-4
            )
            if can_update_in_place:
                cursor.execute(
                    """
                    UPDATE accounts
                    SET timestamp = ?,
                        unrealized_pnl = ?,
                        realized_pnl = ?,
                        total_equity = ?,
                        margin_used = ?,
                        free_margin = ?,
                        leverage = ?
                    WHERE id = ?
                    """,
                    (
                        ts,
                        upnl_f,
                        rpnl_f,
                        eq_f,
                        margin_f,
                        free_f,
                        lev_f,
                        self._last_account_row_id,
                    ),
                )
                if cursor.rowcount > 0:
                    return int(self._last_account_row_id)

            cursor.execute(
                """
                INSERT INTO accounts (
                    timestamp, initial_capital, cash_balance, unrealized_pnl,
                    realized_pnl, total_equity, margin_used, free_margin, leverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    init_cap_f,
                    cash_f,
                    upnl_f,
                    rpnl_f,
                    eq_f,
                    margin_f,
                    free_f,
                    lev_f,
                ),
            )
            row_id = int(cursor.lastrowid)
            self._last_account_row_id = row_id
            self._last_account_insert_mono = now_mono
            self._last_account_cash = cash_f
            self._last_account_init_capital = init_cap_f
            return row_id

    def get_latest_account(self) -> dict[str, Any] | None:
        """Returns the most recent account snapshot or None."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM accounts ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_account_history(self, limit: int = 100) -> list[dict[str, Any]]:
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
        liquidation_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Upserts an active or flat position record."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        opened_dt = getattr(position, "opened_at", None)
        opened_ts = opened_dt.isoformat() if isinstance(opened_dt, datetime) else ts
        tp_val = take_profit if take_profit is not None else getattr(position, "take_profit", None)
        sl_val = stop_loss if stop_loss is not None else getattr(position, "stop_loss", None)
        peak_val = getattr(position, "peak_price", 0.0) or 0.0
        strat_val = getattr(position, "strategy", None)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO positions (
                    symbol, side, quantity, entry_price, current_price,
                    notional, unrealized_pnl, realized_pnl, liquidation_price,
                    take_profit, stop_loss, peak_price, strategy, opened_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    opened_at = CASE
                        WHEN positions.side = 'FLAT' OR positions.quantity <= 0 THEN excluded.opened_at
                        ELSE positions.opened_at
                    END,
                    side = excluded.side,
                    quantity = excluded.quantity,
                    entry_price = excluded.entry_price,
                    current_price = excluded.current_price,
                    notional = excluded.notional,
                    unrealized_pnl = excluded.unrealized_pnl,
                    realized_pnl = excluded.realized_pnl,
                    liquidation_price = excluded.liquidation_price,
                    take_profit = excluded.take_profit,
                    stop_loss = excluded.stop_loss,
                    peak_price = excluded.peak_price,
                    strategy = excluded.strategy,
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
                    float(tp_val) if tp_val is not None else None,
                    float(sl_val) if sl_val is not None else None,
                    float(peak_val),
                    str(strat_val) if strat_val is not None else None,
                    opened_ts,
                    ts,
                ),
            )

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Queries single position by symbol."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_open_positions(self) -> dict[str, Position]:
        """Returns all non-flat positions mapped to domain Position objects."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM positions WHERE side != 'FLAT' AND quantity > 0"
            )
            rows = cursor.fetchall()
            positions: dict[str, Position] = {}
            for r in rows:
                side_enum = PositionSide(r["side"])
                col_keys = r.keys() if hasattr(r, "keys") else []
                tp = float(r["take_profit"]) if ("take_profit" in col_keys and r["take_profit"] is not None) else None
                sl = float(r["stop_loss"]) if ("stop_loss" in col_keys and r["stop_loss"] is not None) else None
                peak_p = float(r["peak_price"]) if ("peak_price" in col_keys and r["peak_price"] is not None) else 0.0
                strat = str(r["strategy"]) if ("strategy" in col_keys and r["strategy"] is not None) else None
                opened_dt: datetime | None = None
                if "opened_at" in col_keys and r["opened_at"]:
                    try:
                        opened_dt = datetime.fromisoformat(str(r["opened_at"]))
                        if opened_dt.tzinfo is None:
                            opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        opened_dt = None
                positions[r["symbol"]] = Position(
                    symbol=r["symbol"],
                    side=side_enum,
                    quantity=float(r["quantity"]),
                    entry_price=float(r["entry_price"]),
                    current_price=float(r["current_price"]),
                    peak_price=peak_p,
                    take_profit=tp,
                    stop_loss=sl,
                    strategy=strat,
                    opened_at=opened_dt,
                )
            return positions

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------

    def save_order(
        self,
        order: Order,
        role: str | None = None,
        client_order_id: str | None = None,
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
        strat_val = getattr(order, "strategy", None)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO orders (
                    id, client_order_id, symbol, side, order_type,
                    price, quantity, filled_quantity, average_fill_price,
                    status, role, fee, strategy, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filled_quantity = excluded.filled_quantity,
                    average_fill_price = excluded.average_fill_price,
                    status = excluded.status,
                    role = COALESCE(excluded.role, orders.role),
                    fee = excluded.fee,
                    strategy = COALESCE(excluded.strategy, orders.strategy),
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
                    str(strat_val) if strat_val is not None else None,
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
        role: str | None = None,
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

    def update_order_status(self, order_id: str, status: OrderStatus | str) -> bool:
        """Updates the status of an existing order."""
        status_val = status.value if hasattr(status, "value") else str(status)
        updated_str = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE orders
                SET status = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (status_val, updated_str, order_id),
            )
            return cursor.rowcount > 0

    def get_order(self, order_id: str) -> Order | None:
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

            col_keys = row.keys() if hasattr(row, "keys") else []
            strat = str(row["strategy"]) if ("strategy" in col_keys and row["strategy"] is not None) else None

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
                strategy=strat,
            )

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
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
                col_keys = r.keys() if hasattr(r, "keys") else []
                strat = str(r["strategy"]) if ("strategy" in col_keys and r["strategy"] is not None) else None
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
                        strategy=strat,
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
        strategy: str | None = None,
    ) -> None:
        """Records an executed trade in the database."""
        ts = (
            trade.timestamp.isoformat()
            if isinstance(trade.timestamp, datetime)
            else str(trade.timestamp)
        )
        notional = float(trade.price * trade.quantity)
        side_val = trade.side.value if hasattr(trade.side, "value") else str(trade.side)
        strat_val = strategy or getattr(trade, "strategy", None)

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    id, order_id, symbol, side, price, quantity,
                    notional, fee_paid, is_maker, realized_pnl, strategy, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(strat_val) if strat_val is not None else None,
                    ts,
                ),
            )

    def get_trades(self, limit: int = 50, symbol: str | None = None) -> list[dict[str, Any]]:
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

    def get_trades_24h(self) -> list[dict[str, Any]]:
        """Returns all trades executed in the last 24 hours."""
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff_iso,),
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
        timestamp: datetime | None = None,
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

    def get_funding_settlements(self, limit: int = 50) -> list[dict[str, Any]]:
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
        target_allocations: dict[str, float],
        executed_delta: dict[str, float],
        timestamp: datetime | None = None,
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

    def get_latest_rebalance_snapshot(self) -> dict[str, Any] | None:
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

    def save_reconciliation_audit(
        self,
        status: str,
        drift_count: int,
        orphan_count: int,
        cash_drift: float,
        trading_halted: bool,
        details: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> int:
        """Records an audit log entry for a portfolio reconciliation cycle."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO reconciliation_audit_logs (
                    timestamp, status, drift_count, orphan_count,
                    cash_drift, trading_halted, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    status,
                    int(drift_count),
                    int(orphan_count),
                    float(cash_drift),
                    1 if trading_halted else 0,
                    json.dumps(details),
                ),
            )
            return cursor.lastrowid

    def get_recent_reconciliations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns recent reconciliation audit entries."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM reconciliation_audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["details"] = json.loads(item["details_json"])
                item["trading_halted"] = bool(item["trading_halted"])
                results.append(item)
            return results

    def update_daemon_heartbeat(
        self,
        ws_state: str,
        symbols_count: int,
        total_ticks: int,
        ticks_per_sec: float,
        last_tick_time: str,
        last_rebalance_time: str | None,
        next_rebalance_in_sec: float,
        status: str = "ONLINE",
    ) -> None:
        """Upserts the single-row daemon heartbeat."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.cursor().execute(
                """
                INSERT INTO daemon_heartbeat (
                    id, timestamp, ws_state, symbols_count, total_ticks,
                    ticks_per_sec, last_tick_time, last_rebalance_time,
                    next_rebalance_in_sec, status
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    ws_state = excluded.ws_state,
                    symbols_count = excluded.symbols_count,
                    total_ticks = excluded.total_ticks,
                    ticks_per_sec = excluded.ticks_per_sec,
                    last_tick_time = excluded.last_tick_time,
                    last_rebalance_time = excluded.last_rebalance_time,
                    next_rebalance_in_sec = excluded.next_rebalance_in_sec,
                    status = excluded.status;
                """,
                (
                    now, ws_state, symbols_count, total_ticks,
                    ticks_per_sec, last_tick_time, last_rebalance_time,
                    next_rebalance_in_sec, status
                ),
            )

    def get_daemon_heartbeat(self) -> dict[str, Any] | None:
        """Retrieves latest daemon heartbeat."""
        with self._lock:
            row = self._conn.cursor().execute("SELECT * FROM daemon_heartbeat WHERE id = 1;").fetchone()
            return dict(row) if row else None

    def update_market_pulse(self, tickers: dict[str, dict[str, float]]) -> None:
        """Upserts current market pulse for all symbols."""
        now = datetime.now(timezone.utc).isoformat()
        records = [
            (
                sym,
                float(t.get("price", 0.0)),
                float(t.get("change_24h", 0.0)),
                float(t.get("volume_24h", 0.0)),
                float(t.get("high", 0.0)),
                float(t.get("low", 0.0)),
                now,
            )
            for sym, t in tickers.items()
            if t.get("price", 0.0) > 0
        ]
        if not records:
            return
        with self._lock:
            self._conn.cursor().executemany(
                """
                INSERT INTO market_pulse (symbol, price, change_24h, volume_24h, high_24h, low_24h, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    price = excluded.price,
                    change_24h = excluded.change_24h,
                    volume_24h = excluded.volume_24h,
                    high_24h = excluded.high_24h,
                    low_24h = excluded.low_24h,
                    updated_at = excluded.updated_at;
                """,
                records,
            )

    def get_market_pulse(self) -> list[dict[str, Any]]:
        """Retrieves market pulse for all monitored symbols ordered by change_24h descending."""
        with self._lock:
            rows = self._conn.cursor().execute(
                "SELECT * FROM market_pulse ORDER BY change_24h DESC;"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_seven_factor_telemetry(
        self,
        initial_capital: float,
        current_equity: float,
        high_water_mark: float,
        drawdown_pct: float,
        unlocked_tier: float,
        active_base_leverage: float,
        vol_scalar: float,
        effective_leverage: float,
        next_tier_leverage: float,
        next_tier_hwm_target: float,
        progress_to_next_tier_pct: float,
        portfolio_beta: float,
        conviction: float,
        factor_weights: dict[str, float],
        factor_tstats: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Upserts the latest 7-Factor Online Engine and Stepped Leverage telemetry snapshot."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            self._conn.cursor().execute(
                """
                INSERT INTO seven_factor_telemetry (
                    id, timestamp, initial_capital, current_equity, high_water_mark,
                    drawdown_pct, unlocked_tier, active_base_leverage, vol_scalar,
                    effective_leverage, next_tier_leverage, next_tier_hwm_target,
                    progress_to_next_tier_pct, portfolio_beta, conviction,
                    factor_weights_json, factor_tstats_json
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    initial_capital = excluded.initial_capital,
                    current_equity = excluded.current_equity,
                    high_water_mark = excluded.high_water_mark,
                    drawdown_pct = excluded.drawdown_pct,
                    unlocked_tier = excluded.unlocked_tier,
                    active_base_leverage = excluded.active_base_leverage,
                    vol_scalar = excluded.vol_scalar,
                    effective_leverage = excluded.effective_leverage,
                    next_tier_leverage = excluded.next_tier_leverage,
                    next_tier_hwm_target = excluded.next_tier_hwm_target,
                    progress_to_next_tier_pct = excluded.progress_to_next_tier_pct,
                    portfolio_beta = CASE WHEN excluded.factor_weights_json = '{}' THEN seven_factor_telemetry.portfolio_beta ELSE excluded.portfolio_beta END,
                    conviction = CASE WHEN excluded.factor_weights_json = '{}' THEN seven_factor_telemetry.conviction ELSE excluded.conviction END,
                    factor_weights_json = CASE WHEN excluded.factor_weights_json = '{}' THEN seven_factor_telemetry.factor_weights_json ELSE excluded.factor_weights_json END,
                    factor_tstats_json = CASE WHEN excluded.factor_tstats_json = '{}' THEN seven_factor_telemetry.factor_tstats_json ELSE excluded.factor_tstats_json END;
                """,
                (
                    ts,
                    float(initial_capital),
                    float(current_equity),
                    float(high_water_mark),
                    float(drawdown_pct),
                    float(unlocked_tier),
                    float(active_base_leverage),
                    float(vol_scalar),
                    float(effective_leverage),
                    float(next_tier_leverage),
                    float(next_tier_hwm_target),
                    float(progress_to_next_tier_pct),
                    float(portfolio_beta),
                    float(conviction),
                    json.dumps(factor_weights),
                    json.dumps(factor_tstats),
                ),
            )

    def get_latest_seven_factor_telemetry(self) -> dict[str, Any] | None:
        """Retrieves the latest 7-Factor & Stepped Leverage telemetry snapshot."""
        with self._lock:
            row = self._conn.cursor().execute(
                "SELECT * FROM seven_factor_telemetry WHERE id = 1;"
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            res["factor_weights"] = json.loads(res.get("factor_weights_json") or "{}")
            res["factor_tstats"] = json.loads(res.get("factor_tstats_json") or "{}")
            return res

    def prune_old_records(
        self,
        accounts_minute_res_days: float = 7.0,
        canceled_orders_retention_hours: float = 24.0,
        audit_retention_days: float = 14.0,
        keep_recent_accounts: int = 500,
        run_vacuum: bool = False,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """
        Issue #59 (M14 Option A): Native TTL pruning, historical account snapshot
        downsampling, and WAL auto-compaction.
        - Retains the latest `keep_recent_accounts` rows in `accounts` at full resolution.
        - Downsamples `accounts` rows within `accounts_minute_res_days` (default 7d) to
          1 snapshot per minute (`YYYY-MM-DDTHH:MM`), and older rows to 1 snapshot per hour
          (`YYYY-MM-DDTHH`).
        - Deletes terminal zero-fill `CANCELED` / `REJECTED` orders older than
          `canceled_orders_retention_hours` that are not referenced in `trades`.
        - Deletes `reconciliation_audit_logs` older than `audit_retention_days` (preserving
          the latest 200 entries).
        - Runs `PRAGMA wal_checkpoint(TRUNCATE)` and optional `VACUUM`.
        """
        ref_now = now or datetime.now(timezone.utc)
        cutoff_7d_iso = (ref_now - timedelta(days=float(accounts_minute_res_days))).isoformat()
        cutoff_orders_iso = (ref_now - timedelta(hours=float(canceled_orders_retention_hours))).isoformat()
        cutoff_audit_iso = (ref_now - timedelta(days=float(audit_retention_days))).isoformat()
        keep_n = max(1, int(keep_recent_accounts))

        with self._lock:
            cursor = self._conn.cursor()

            # Find boundary id for the most recent `keep_n` account rows
            row_bound = cursor.execute(
                "SELECT MIN(id) FROM (SELECT id FROM accounts ORDER BY id DESC LIMIT ?)",
                (keep_n,),
            ).fetchone()
            min_recent_id = int(row_bound[0]) if (row_bound and row_bound[0] is not None) else None

            accounts_deleted = 0
            if min_recent_id is not None:
                # 1a. Downsample recent history (< 7d, id < min_recent_id) to 1 row per minute
                cursor.execute(
                    """
                    DELETE FROM accounts
                    WHERE id < ?
                      AND timestamp >= ?
                      AND id NOT IN (
                          SELECT MAX(id)
                          FROM accounts
                          WHERE id < ? AND timestamp >= ?
                          GROUP BY substr(timestamp, 1, 16)
                      )
                    """,
                    (min_recent_id, cutoff_7d_iso, min_recent_id, cutoff_7d_iso),
                )
                accounts_deleted += max(0, int(cursor.rowcount))

                # 1b. Downsample older history (>= 7d, id < min_recent_id) to 1 row per hour
                cursor.execute(
                    """
                    DELETE FROM accounts
                    WHERE id < ?
                      AND timestamp < ?
                      AND id NOT IN (
                          SELECT MAX(id)
                          FROM accounts
                          WHERE id < ? AND timestamp < ?
                          GROUP BY substr(timestamp, 1, 13)
                      )
                    """,
                    (min_recent_id, cutoff_7d_iso, min_recent_id, cutoff_7d_iso),
                )
                accounts_deleted += max(0, int(cursor.rowcount))

            # 2. Prune terminal zero-fill CANCELED / REJECTED orders unreferenced by trades
            cursor.execute(
                """
                DELETE FROM orders
                WHERE status IN ('CANCELED', 'CANCELLED', 'REJECTED')
                  AND COALESCE(filled_quantity, 0.0) <= 0.0
                  AND updated_at < ?
                  AND id NOT IN (SELECT DISTINCT order_id FROM trades)
                """,
                (cutoff_orders_iso,),
            )
            orders_deleted = max(0, int(cursor.rowcount))

            # 3. Prune old reconciliation_audit_logs (keeping latest 200 rows)
            cursor.execute(
                """
                DELETE FROM reconciliation_audit_logs
                WHERE timestamp < ?
                  AND id NOT IN (
                      SELECT id FROM reconciliation_audit_logs ORDER BY id DESC LIMIT 200
                  )
                """,
                (cutoff_audit_iso,),
            )
            audit_deleted = max(0, int(cursor.rowcount))

            total_deleted = accounts_deleted + orders_deleted + audit_deleted
            vacuumed = 0
            if self.db_path != ":memory:":
                try:
                    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    cursor.execute("PRAGMA optimize;")
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"SQLite WAL checkpoint note: {exc}")

                if run_vacuum or total_deleted >= 2000:
                    try:
                        cursor.execute("VACUUM;")
                        vacuumed = 1
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"SQLite VACUUM note: {exc}")

            return {
                "accounts_deleted": accounts_deleted,
                "orders_deleted": orders_deleted,
                "audit_deleted": audit_deleted,
                "total_deleted": total_deleted,
                "vacuumed": vacuumed,
            }

    def reset_all_data(self) -> None:
        """Clears all records from all tables (used for testing and fresh start)."""
        with self._lock:
            self._last_account_row_id = None
            self._last_account_insert_mono = 0.0
            self._last_account_cash = None
            self._last_account_init_capital = None
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM trades;")
            cursor.execute("DELETE FROM orders;")
            cursor.execute("DELETE FROM positions;")
            cursor.execute("DELETE FROM accounts;")
            cursor.execute("DELETE FROM funding_settlements;")
            cursor.execute("DELETE FROM rebalance_snapshots;")
            cursor.execute("DELETE FROM reconciliation_audit_logs;")
            cursor.execute("DELETE FROM daemon_heartbeat;")
            cursor.execute("DELETE FROM market_pulse;")
            cursor.execute("DELETE FROM seven_factor_telemetry;")
