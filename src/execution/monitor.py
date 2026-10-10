"""
Real-Time Terminal Telemetry Dashboard for Shadow Trading.
Queries local SQLite database without locking (WAL mode) and renders
live equity, open positions, active regime, execution alpha (maker savings), and countdowns.
"""

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.execution.database import DEFAULT_DB_PATH, ShadowDatabase


class ShadowMonitor:
    """
    Renders live telemetry and risk dashboards from the local SQLite shadow database.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.console = Console() if HAS_RICH else None

    def _get_database(self) -> ShadowDatabase:
        return ShadowDatabase(db_path=self.db_path)

    def render_snapshot(self) -> Any:
        """Builds the complete terminal telemetry layout."""
        if not os.path.exists(self.db_path):
            if HAS_RICH:
                return Panel(
                    f"[yellow]Waiting for shadow database to initialize at: {self.db_path}[/yellow]",
                    title="Crypto Alpha Engine: Shadow Monitor",
                )
            else:
                print(f"Waiting for shadow database at {self.db_path}...")
                return None

        db = self._get_database()
        latest_acc = db.get_latest_account()
        open_positions = db.get_open_positions()
        recent_orders = db._conn.cursor().execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 6").fetchall()
        recent_trades = db.get_trades(limit=6)
        recent_funding = db.get_funding_settlements(limit=4)
        latest_rebalance = db.get_latest_rebalance_snapshot()
        db.close()

        if not latest_acc:
            if HAS_RICH:
                return Panel("[yellow]Database connected, waiting for initial account snapshot...[/yellow]")
            else:
                print("Waiting for initial account snapshot...")
                return None

        if HAS_RICH:
            return self._build_rich_layout(
                latest_acc, open_positions, recent_orders, recent_trades, recent_funding, latest_rebalance
            )
        else:
            self._build_plain_layout(
                latest_acc, open_positions, recent_orders, recent_trades, recent_funding, latest_rebalance
            )
            return None

    def _build_rich_layout(
        self,
        account: Dict,
        positions: Dict,
        orders: List,
        trades: List,
        funding: List,
        rebalance: Optional[Dict],
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="positions", size=10),
            Layout(name="orders_and_funding", size=10),
            Layout(name="footer", size=3),
        )
        layout["orders_and_funding"].split_row(
            Layout(name="orders", ratio=2),
            Layout(name="funding", ratio=1),
        )

        # 1. Header Panel
        equity = float(account["total_equity"])
        cash = float(account["cash_balance"])
        init_cap = float(account["initial_capital"])
        ret_pct = ((equity - init_cap) / init_cap) * 100.0 if init_cap > 0 else 0.0
        ret_color = "green" if ret_pct >= 0 else "red"
        unrealized = float(account["unrealized_pnl"])
        unreal_color = "green" if unrealized >= 0 else "red"
        realized = float(account["realized_pnl"])
        real_color = "green" if realized >= 0 else "red"
        leverage = float(account["leverage"])

        regime_str = rebalance["regime_detected"] if rebalance else "CALIBRATING"
        regime_color = "green" if regime_str == "TRENDING" else ("yellow" if regime_str == "RANGING" else "red")

        header_text = Text()
        header_text.append(" MERCADO: ", style="bold white on blue")
        header_text.append(" Binance Futures Mainnet (Solo Lectura - 0 Riesgo de Capital)\n", style="cyan")
        header_text.append(f" Equity: ${equity:,.2f}  |  ", style="bold white")
        header_text.append(f"Cash: ${cash:,.2f}  |  ", style="white")
        header_text.append(f"Retorno: {ret_pct:+.2f}%  |  ", style=f"bold {ret_color}")
        header_text.append(f"PnL No Real.: ${unrealized:+,.2f}  |  ", style=unreal_color)
        header_text.append(f"PnL Real.: ${realized:+,.2f}  |  ", style=real_color)
        header_text.append(f"Apalancamiento: {leverage:.2f}x\n", style="white")
        header_text.append(" Régimen de Mercado: ", style="bold white")
        header_text.append(f"[{regime_str}]  ", style=f"bold {regime_color}")
        if rebalance:
            header_text.append(
                f"(Momentum: {rebalance['momentum_weight']*100:.0f}%, "
                f"Stat-Arb: {rebalance['stat_arb_weight']*100:.0f}%, "
                f"Carry: {rebalance['carry_weight']*100:.0f}%)",
                style="dim",
            )

        layout["header"].update(Panel(header_text, title="[bold cyan]CRYPTO-ALPHA-ENGINE: LIVE SHADOW TRADING[/bold cyan]"))

        # 2. Positions Table
        pos_table = Table(title="Posiciones Activas en Inventario", expand=True, header_style="bold magenta")
        pos_table.add_column("Símbolo", style="white")
        pos_table.add_column("Lado", justify="center")
        pos_table.add_column("Cantidad", justify="right")
        pos_table.add_column("Precio Entrada", justify="right")
        pos_table.add_column("Precio Actual", justify="right")
        pos_table.add_column("Nocional (USD)", justify="right")
        pos_table.add_column("PnL ($)", justify="right")
        pos_table.add_column("PnL (%)", justify="right")

        if positions:
            for sym, p in positions.items():
                side_style = "bold green" if p.side.value == "LONG" else "bold red"
                pnl = p.unrealized_pnl
                pnl_color = "green" if pnl >= 0 else "red"
                val = p.market_value
                pnl_pct = (pnl / (p.quantity * p.entry_price)) * 100.0 if (p.quantity * p.entry_price) > 0 else 0.0

                pos_table.add_row(
                    sym,
                    Text(p.side.value, style=side_style),
                    f"{p.quantity:,.4f}",
                    f"${p.entry_price:,.2f}",
                    f"${p.current_price:,.2f}",
                    f"${val:,.2f}",
                    Text(f"${pnl:+,.2f}", style=pnl_color),
                    Text(f"{pnl_pct:+.2f}%", style=pnl_color),
                )
        else:
            pos_table.add_row("-", "FLAT", "0.00", "$0.00", "$0.00", "$0.00", "$0.00", "0.00%")

        layout["positions"].update(Panel(pos_table))

        # 3. Orders & Execution Alpha Table
        ord_table = Table(title="Auditoría de Órdenes & Execution Alpha", expand=True, header_style="bold blue")
        ord_table.add_column("ID", style="dim")
        ord_table.add_column("Símbolo")
        ord_table.add_column("Lado", justify="center")
        ord_table.add_column("Tipo", justify="center")
        ord_table.add_column("Precio", justify="right")
        ord_table.add_column("Estado", justify="center")
        ord_table.add_column("Rol", justify="center")
        ord_table.add_column("Fee Pagada", justify="right")
        ord_table.add_column("Ahorro Maker", justify="right")

        for r in orders:
            side_color = "green" if r["side"] == "BUY" else "red"
            status_color = "green" if r["status"] == "FILLED" else ("yellow" if r["status"] == "PENDING" else "dim")
            role_str = r["role"] or "-"
            role_style = "bold cyan" if role_str == "MAKER" else "bold magenta"
            fee = float(r["fee"])
            # Taker fee is 4 bps, Maker fee is 2 bps -> Maker saves 50% of fee
            savings = fee if role_str == "MAKER" else 0.0

            ord_table.add_row(
                r["id"][:12],
                r["symbol"],
                Text(r["side"], style=side_color),
                r["order_type"],
                f"${float(r['price'] or 0):,.2f}",
                Text(r["status"], style=status_color),
                Text(role_str, style=role_style),
                f"${fee:,.2f}",
                Text(f"+${savings:,.2f}" if savings > 0 else "-", style="green" if savings > 0 else "dim"),
            )
        layout["orders_and_funding"]["orders"].update(Panel(ord_table))

        # 4. Funding Table
        fnd_table = Table(title="Liquidaciones 8h Funding", expand=True, header_style="bold green")
        fnd_table.add_column("Símbolo")
        fnd_table.add_column("Tasa", justify="right")
        fnd_table.add_column("Lado", justify="center")
        fnd_table.add_column("Cashflow", justify="right")

        if funding:
            for f in funding:
                cf = float(f["cashflow_credited"])
                cf_color = "green" if cf >= 0 else "red"
                fnd_table.add_row(
                    f["symbol"],
                    f"{float(f['funding_rate'])*100:+.4f}%",
                    f["position_side"],
                    Text(f"${cf:+,.2f}", style=cf_color),
                )
        else:
            fnd_table.add_row("-", "0.0000%", "-", "$0.00")

        layout["orders_and_funding"]["funding"].update(Panel(fnd_table))

        # 5. Footer / Countdowns
        now = datetime.now(timezone.utc)
        # Next funding hours: 00, 08, 16
        next_funding_hour = (now.hour // 8 + 1) * 8
        if next_funding_hour >= 24:
            next_funding_hour = 0
            mins_left = (24 - now.hour) * 60 - now.minute
        else:
            mins_left = (next_funding_hour - now.hour) * 60 - now.minute

        hrs = mins_left // 60
        mins = mins_left % 60

        footer_text = Text()
        footer_text.append(f" Próxima Liquidación de Funding: en {hrs:02d}h {mins:02d}m  |  ", style="bold yellow")
        footer_text.append(f"Hora Local UTC: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  ", style="dim")
        footer_text.append("Presiona Ctrl+C para salir", style="dim italic")

        layout["footer"].update(Panel(footer_text))
        return layout

    def _build_plain_layout(self, account, positions, orders, trades, funding, rebalance):
        """Plain ANSI fallback for terminals without rich."""
        equity = float(account["total_equity"])
        cash = float(account["cash_balance"])
        print("\n" + "=" * 80)
        print(" CRYPTO-ALPHA-ENGINE: LIVE SHADOW TRADING (Binance Mainnet Read-Only)")
        print(f" Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | PnL No Real.: ${account['unrealized_pnl']:+,.2f}")
        print(" Posiciones Abiertas:")
        for sym, p in positions.items():
            print(f"  - {sym}: {p.side.value} {p.quantity} @ ${p.entry_price:,.2f} (PnL: ${p.unrealized_pnl:+,.2f})")
        print("=" * 80 + "\n")

    def run_live(self, refresh_rate: float = 2.0) -> None:
        """Starts live dashboard loop."""
        if not HAS_RICH:
            print("Running in non-rich mode...")
            while True:
                self.render_snapshot()
                time.sleep(refresh_rate)

        with Live(self.render_snapshot(), refresh_per_second=int(1.0 / refresh_rate), console=self.console) as live:
            while True:
                try:
                    time.sleep(refresh_rate)
                    live.update(self.render_snapshot())
                except KeyboardInterrupt:
                    break


def main():
    parser = argparse.ArgumentParser(description="Real-Time Shadow Trading Telemetry Dashboard")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to shadow SQLite database")
    parser.add_argument("--once", action="store_true", help="Print single snapshot and exit")
    parser.add_argument("--refresh-rate", type=float, default=2.0, help="Refresh interval in seconds (default: 2.0)")

    args = parser.parse_args()
    monitor = ShadowMonitor(db_path=args.db)

    if args.once:
        layout = monitor.render_snapshot()
        if HAS_RICH and monitor.console:
            monitor.console.print(layout)
    else:
        monitor.run_live(refresh_rate=args.refresh_rate)


if __name__ == "__main__":
    main()
