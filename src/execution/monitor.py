"""
Real-Time Terminal Telemetry Dashboard for Shadow Trading.
Queries local SQLite database without locking (WAL mode) and renders
live equity, open positions, active regime, execution alpha (maker savings),
live WebSocket heartbeat, and 24h market pulse / anomaly radar.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

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
    Renders live telemetry, WebSocket pulse, and risk dashboards from the local SQLite shadow database.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, console: Optional[Console] = None):
        self.db_path = db_path
        self.console = console or (Console() if HAS_RICH else None)

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
        recent_orders = db._conn.cursor().execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 10").fetchall()
        recent_trades = db.get_trades(limit=8)
        trades_24h = db.get_trades_24h()
        recent_funding = db.get_funding_settlements(limit=4)
        latest_rebalance = db.get_latest_rebalance_snapshot()
        heartbeat = db.get_daemon_heartbeat()
        market_pulse = db.get_market_pulse()
        sf_telemetry = (
            db.get_latest_seven_factor_telemetry()
            if hasattr(db, "get_latest_seven_factor_telemetry")
            else None
        )
        db.close()

        if not latest_acc:
            if HAS_RICH:
                return Panel("[yellow]Database connected, waiting for initial account snapshot...[/yellow]")
            else:
                print("Waiting for initial account snapshot...")
                return None

        if HAS_RICH:
            return self._build_rich_layout(
                latest_acc,
                open_positions,
                recent_orders,
                recent_trades,
                trades_24h,
                recent_funding,
                latest_rebalance,
                heartbeat,
                market_pulse,
                sf_telemetry,
            )
        else:
            self._build_plain_layout(
                latest_acc,
                open_positions,
                recent_orders,
                recent_trades,
                trades_24h,
                recent_funding,
                latest_rebalance,
                heartbeat,
                market_pulse,
                sf_telemetry,
            )
            return None

    def _build_rich_layout(
        self,
        account: Dict,
        positions: Dict,
        orders: List,
        trades: List,
        trades_24h: List,
        funding: List,
        rebalance: Optional[Dict],
        heartbeat: Optional[Dict],
        market_pulse: List[Dict],
        sf_telemetry: Optional[Dict] = None,
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8 if sf_telemetry else 7),
            Layout(name="main_section", ratio=3),
            Layout(name="strategies_and_funding", ratio=2),
            Layout(name="orders_and_trades", ratio=3),
            Layout(name="footer", size=3),
        )
        layout["main_section"].split_row(
            Layout(name="positions", ratio=7),
            Layout(name="market_pulse", ratio=3),
        )
        layout["strategies_and_funding"].split_row(
            Layout(name="strategies", ratio=3),
            Layout(name="funding", ratio=2),
        )
        layout["orders_and_trades"].split_row(
            Layout(name="orders", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # 1. Header Panel with WebSocket Telemetry & Health
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
        is_live_prod = "live_trading" in str(getattr(self, "db_path", ""))
        # Row 1: Market & Engine Status
        header_text.append(" MERCADO: ", style="bold white on green" if is_live_prod else "bold white on blue")
        if is_live_prod:
            header_text.append(" 🟢 PRODUCCIÓN EN VIVO — Binance USDS-M Futures (Capital Real)  |  ", style="bold green")
        else:
            header_text.append(" Binance Futures Mainnet (Solo Lectura - 0 Riesgo de Capital)  |  ", style="cyan")

        if heartbeat:
            ws_state = heartbeat.get("ws_state", "CONNECTED")
            hb_status = heartbeat.get("status", "ONLINE")
            total_ticks = heartbeat.get("total_ticks", 0)
            tps = heartbeat.get("ticks_per_sec", 0.0)
            symbols_count = heartbeat.get("symbols_count", 15)
            next_reb_sec = heartbeat.get("next_rebalance_in_sec", 0.0)

            # Compute seconds since last heartbeat & last tick
            hb_ts_str = heartbeat.get("timestamp")
            hb_age_sec = 0.0
            if hb_ts_str:
                try:
                    dt_hb = datetime.fromisoformat(hb_ts_str)
                    hb_age_sec = max(0.0, (datetime.now(timezone.utc) - dt_hb).total_seconds())
                except Exception:
                    hb_age_sec = 0.0

            last_tick_str = heartbeat.get("last_tick_time")
            if last_tick_str:
                try:
                    dt_last = datetime.fromisoformat(last_tick_str)
                    age_sec = max(0.0, (datetime.now(timezone.utc) - dt_last).total_seconds())
                    age_text = f"{age_sec:.1f}s" if age_sec < 120 else f"{int(age_sec // 60)}m"
                except Exception:
                    age_text = "<1s"
            else:
                age_text = f"{hb_age_sec:.1f}s" if hb_age_sec < 120 else f"{int(hb_age_sec // 60)}m"

            reb_m = int(next_reb_sec // 60)
            reb_h = reb_m // 60
            reb_rem_m = reb_m % 60
            reb_text = f"{reb_h:02d}h {reb_rem_m:02d}m" if reb_h > 0 else f"{reb_rem_m:02d}m"

            header_text.append("ESTADO: ", style="bold white")
            if hb_status == "ONLINE" and hb_age_sec <= 45.0:
                header_text.append("● ONLINE", style="bold green")
                header_text.append(f" (WS {ws_state})\n", style="green")
            else:
                idle_min = int(hb_age_sec // 60)
                header_text.append("○ DETENIDO / EN ESPERA", style="bold yellow")
                header_text.append(f" (Último pulso hace {idle_min}m - Inicia --daemon)\n", style="yellow")

            header_text.append(f" TELEMETRÍA EN VIVO: ", style="bold white on dark_green")
            header_text.append(f" Streams Activos: {symbols_count}  |  ", style="bold white")
            header_text.append(f"Ticks Recibidos: {total_ticks:,} (~{tps:.1f} ticks/s)  |  ", style="white")
            header_text.append(f"Último Tick: hace {age_text}  |  ", style="cyan")
            header_text.append(f"Próx. Rebalanceo: en {reb_text}\n", style="yellow")
        else:
            header_text.append("ESTADO: ", style="bold white")
            header_text.append("● INICIANDO TELEMETRÍA...\n", style="yellow")

        # Row 2: Financials
        header_text.append(f" Equity: ${equity:,.2f}  |  ", style="bold white")
        header_text.append(f"Cash: ${cash:,.2f}  |  ", style="white")
        header_text.append(f"Retorno: {ret_pct:+.2f}%  |  ", style=f"bold {ret_color}")
        header_text.append(f"uPnL: ${unrealized:+,.2f}  |  ", style=f"bold {unreal_color}")
        header_text.append(f"PnL: ${realized:+,.2f}  |  ", style=f"bold {real_color}")
        header_text.append(f"Apalancamiento: {leverage:.2f}x  |  ", style="white")
        header_text.append(f"Régimen: [{regime_str}]", style=f"bold {regime_color}")

        # Row 3: 7-Factor Stepped Leverage & Neutrality Telemetry
        if sf_telemetry:
            u_tier = float(sf_telemetry.get("unlocked_tier", 2.5))
            act_lev = float(sf_telemetry.get("active_base_leverage", 2.5))
            v_sc = float(sf_telemetry.get("vol_scalar", 1.0))
            eff_lev = float(sf_telemetry.get("effective_leverage", 2.5))
            hwm_val = float(sf_telemetry.get("high_water_mark", equity))
            nxt_tier = float(sf_telemetry.get("next_tier_leverage", 3.0))
            nxt_hwm = float(sf_telemetry.get("next_tier_hwm_target", 1150.0))
            prog_pct = float(sf_telemetry.get("progress_to_next_tier_pct", 0.0))
            p_beta = float(sf_telemetry.get("portfolio_beta", 0.0))

            long_usd = sum(p.quantity * p.current_price for p in positions.values() if p.side.value == "LONG")
            short_usd = sum(p.quantity * p.current_price for p in positions.values() if p.side.value == "SHORT")
            net_usd = long_usd - short_usd

            header_text.append("\n ESCALADO 2.5x→5.0x: ", style="bold black on cyan")
            header_text.append(f" Tier: {u_tier:.1f}x (Activo: {act_lev:.1f}x × Vol72h: {v_sc:.2f}x = {eff_lev:.2f}x)  |  ", style="bold cyan")
            header_text.append(f"HWM: ${hwm_val:,.2f} → Meta {nxt_tier:.1f}x: ${nxt_hwm:,.2f} [{prog_pct:.1f}%]  |  ", style="bold yellow")
            header_text.append(f"Net $: ${net_usd:+,.2f}  |  Beta BTC: {p_beta:+.3f}", style="bold green")

        panel_title = (
            "[bold green]CRYPTO-ALPHA-ENGINE: LIVE PRODUCTION TRADING (7-FACTOR CAUSAL ENGINE)[/bold green]"
            if is_live_prod
            else "[bold cyan]CRYPTO-ALPHA-ENGINE: LIVE SHADOW TRADING (7-FACTOR CAUSAL ENGINE)[/bold cyan]"
        )
        layout["header"].update(Panel(header_text, title=panel_title))

        # 2. Left Column: Positions Table
        pos_table = Table(title="Inventario de Posiciones Abiertas", expand=True, header_style="bold magenta", pad_edge=False)
        pos_table.add_column("Símbolo", style="bold white", min_width=11, no_wrap=True)
        pos_table.add_column("Factor", style="cyan", min_width=8, no_wrap=True)
        pos_table.add_column("Lado", justify="center", min_width=5, no_wrap=True)
        pos_table.add_column("Entrada", justify="right", no_wrap=True)
        pos_table.add_column("Actual", justify="right", no_wrap=True)
        pos_table.add_column("uPnL", justify="right", no_wrap=True)
        pos_table.add_column("Pico", justify="right", no_wrap=True)
        pos_table.add_column("🎯 Take-Profit", justify="right", no_wrap=True)
        pos_table.add_column("🛑 Stop-Profit / Trail", justify="right", no_wrap=True)

        if positions:
            for sym, p in positions.items():
                side_style = "bold green" if p.side.value == "LONG" else "bold red"
                pnl = p.unrealized_pnl
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_pct = (pnl / (p.quantity * p.entry_price)) * 100.0 if (p.quantity * p.entry_price) > 0 else 0.0

                p_entry_fmt = f"${p.entry_price:,.4f}" if p.entry_price < 1.0 else f"${p.entry_price:,.2f}"
                p_curr_fmt = f"${p.current_price:,.4f}" if p.current_price < 1.0 else f"${p.current_price:,.2f}"

                raw_strat = p.strategy or "7F_Meta"
                strat_display = raw_strat.split(" ")[0][:10]

                # Peak % excursion
                if p.entry_price > 0 and p.peak_price > 0:
                    if p.side.value == "LONG":
                        pk_pct = max(0.0, (p.peak_price - p.entry_price) / p.entry_price * 100.0)
                    else:
                        pk_pct = max(0.0, (p.entry_price - p.peak_price) / p.entry_price * 100.0)
                else:
                    pk_pct = max(0.0, pnl_pct)
                pk_style = "bold green" if pk_pct >= 2.5 else ("yellow" if pk_pct >= 1.5 else "dim")
                pk_text = Text(f"{pk_pct:+.2f}%", style=pk_style)

                # TP text & Estimated PnL
                if p.take_profit:
                    tp_price_fmt = f"${p.take_profit:,.4f}" if p.take_profit < 1.0 else f"${p.take_profit:,.2f}"
                    tp_pct_val = p.tp_pnl_pct or 0.0
                    tp_text = Text(f"{tp_price_fmt} ({tp_pct_val:+.0f}%)", style="bold green")
                else:
                    tp_text = Text("-", style="dim")

                # SL text & Estimated PnL
                if p.stop_loss:
                    sl_price_fmt = f"${p.stop_loss:,.4f}" if p.stop_loss < 1.0 else f"${p.stop_loss:,.2f}"
                    sl_pnl_val = p.sl_pnl_usd or 0.0
                    sl_pct_val = p.sl_pnl_pct or 0.0
                    if getattr(p, "is_profit_locked", False) or sl_pnl_val > 0:
                        sl_text = Text(f"🔒 {sl_price_fmt} (+${sl_pnl_val:,.2f} | {sl_pct_val:+.1f}%)", style="bold green")
                    else:
                        sl_text = Text(f"{sl_price_fmt} (${sl_pnl_val:+,.2f} | {sl_pct_val:+.0f}%)", style="bold red")
                else:
                    sl_text = Text("⏳ Arma @+2.5%", style="dim yellow" if pk_pct >= 1.5 else "dim")

                pos_table.add_row(
                    sym,
                    strat_display,
                    Text(p.side.value, style=side_style),
                    p_entry_fmt,
                    p_curr_fmt,
                    Text(f"${pnl:+,.2f} ({pnl_pct:+.1f}%)", style=pnl_color),
                    pk_text,
                    tp_text,
                    sl_text,
                )
        else:
            pos_table.add_row("-", "-", "FLAT", "$0.00", "$0.00", "$0.00 (0.0%)", "0.00%", "-", "-")

        layout["main_section"]["positions"].update(Panel(pos_table))

        # 3. Right Column: Market Pulse & Anomaly Radar Table
        pulse_table = Table(title="Radar de Mercado & Top Gainers/Losers 24h", expand=True, header_style="bold yellow")
        pulse_table.add_column("Símbolo", style="bold white", no_wrap=True)
        pulse_table.add_column("Precio", justify="right", no_wrap=True)
        pulse_table.add_column("24h %", justify="right", no_wrap=True)
        pulse_table.add_column("Vol 24h", justify="right", no_wrap=True)
        pulse_table.add_column("Radar Señal", justify="center", no_wrap=True)

        if market_pulse:
            # Display top 5 gainers, benchmarks (BTC/ETH), and top 4 losers (up to 11 rows)
            benchmarks = [p for p in market_pulse if p["symbol"] in ["BTC/USDT", "BTCUSDT", "ETH/USDT", "ETHUSDT"]]
            display_items = market_pulse[:5] + benchmarks[:2] + (market_pulse[-4:] if len(market_pulse) > 9 else [])
            seen = set()
            unique_items = []
            for item in display_items:
                if item["symbol"] not in seen:
                    seen.add(item["symbol"])
                    unique_items.append(item)

            for item in unique_items[:11]:
                sym = item["symbol"]
                chg = float(item["change_24h"])
                chg_style = "bold green" if chg >= 0 else "bold red"
                vol_m = float(item["volume_24h"]) / 1_000_000.0

                # Determine anomaly radar status
                if chg >= 25.0:
                    radar = Text("🔥 PUMP SQUEEZE", style="bold white on red")
                elif chg <= -15.0:
                    radar = Text("❄️ OVERSOLD BOUNCE", style="bold white on blue")
                elif chg >= 8.0:
                    radar = Text("⚡ BREAKOUT", style="bold yellow")
                else:
                    radar = Text("NORMAL", style="dim")

                pulse_table.add_row(
                    sym,
                    f"${float(item['price']):,.2f}" if float(item['price']) >= 1.0 else f"${float(item['price']):.4f}",
                    Text(f"{chg:+.2f}%", style=chg_style),
                    f"${vol_m:,.1f}M",
                    radar,
                )
        else:
            pulse_table.add_row("Sincronizando...", "$0.00", "0.00%", "$0.0M", Text("Sincronizando...", style="dim"))

        layout["main_section"]["market_pulse"].update(Panel(pulse_table))

        # 4. Strategies / 7-Factor Causal Online Weights Table
        if sf_telemetry and sf_telemetry.get("factor_weights"):
            fw_map = sf_telemetry.get("factor_weights", {})
            ft_map = sf_telemetry.get("factor_tstats", {})
            strat_table = Table(
                title="Motor Causal Online de 7 Factores (t-stat 36h/96h/240h + Beta-BTC Neutral)",
                expand=True,
                header_style="bold cyan",
            )
            strat_table.add_column("Factor Ortogonal", style="bold white", no_wrap=True)
            strat_table.add_column("Horizonte / Mecanismo", style="dim", no_wrap=True)
            strat_table.add_column("Peso Online", justify="right", no_wrap=True)
            strat_table.add_column("t-Stat Causal", justify="right", no_wrap=True)
            strat_table.add_column("Estado", justify="center", no_wrap=True)
            strat_table.add_column("Posiciones", justify="center", no_wrap=True)

            factor_defs = [
                ("macro", "📈 F1_Macro_Idio_PB", "168h-720h Idio + 4h/24h PB"),
                ("med", "🚀 F2_Med_Idio_PB", "72h-168h Idio + 12h PB"),
                ("fast", "⚡ F3_Fast_Swing_PB", "12h-48h VolSurge + 15m/4h PB"),
                ("rev", "⚖️ F4_Rev_24_72h", "24h-72h Cross-Sect MeanRev"),
                ("carry", "💰 F5_Carry_Neg", "72h Contrarian Funding Arb"),
                ("liq", "🌊 F6_Liq_Surge", "168h vs 720h DollarVol Exp"),
                ("size", "🏛️ F7_HighLiq_Size", "168h Institutional Quality"),
            ]
            top_f_key = max(fw_map, key=fw_map.get) if fw_map else "macro"
            total_open_pos = len(positions)

            for f_key, f_name, f_desc in factor_defs:
                w_val = float(fw_map.get(f_key, 0.0)) * 100.0
                t_val = float(ft_map.get(f_key, 0.0))
                is_active = w_val >= 1.0
                w_style = "bold green" if w_val >= 20.0 else ("bold yellow" if is_active else "dim")
                t_style = "bold green" if t_val > 0.5 else ("yellow" if t_val >= 0.0 else "red")
                status_txt = Text("● ACTIVO", style="bold green") if is_active else Text("○ OFF (ReLU)", style="dim")
                pos_cnt = total_open_pos if f_key == top_f_key else 0
                pos_style = "bold yellow" if pos_cnt > 0 else "dim"

                strat_table.add_row(
                    f_name,
                    f_desc,
                    Text(f"{w_val:5.1f}%", style=w_style),
                    Text(f"{t_val:+.2f}", style=t_style),
                    status_txt,
                    Text(f"{pos_cnt} (Líder)" if pos_cnt > 0 else "Ensamble", style=pos_style),
                )
        else:
            strat_table = Table(title="Estrategias Cuantitativas en Uso", expand=True, header_style="bold cyan")
            strat_table.add_column("Estrategia", style="bold white", no_wrap=True)
            strat_table.add_column("Categoría", style="dim", no_wrap=True)
            strat_table.add_column("Estado", justify="center", no_wrap=True)
            strat_table.add_column("Posiciones", justify="center", no_wrap=True)
            strat_table.add_column("Trades (24h)", justify="center", no_wrap=True)
            strat_table.add_column("PnL Real. (24h)", justify="right", no_wrap=True)

            strat_defs = [
                {"id": "momentum", "name": "📈 Cross-Sectional Mom", "cat": "Meta Allocator", "status": "ACTIVA"},
                {"id": "stat_arb", "name": "⚖️ StatArb Reversion", "cat": "Meta Allocator", "status": "ACTIVA"},
                {"id": "carry", "name": "💰 Funding Carry Arb", "cat": "Meta Allocator", "status": "ACTIVA"},
                {"id": "pairs", "name": "🔗 Cointegrated Pairs", "cat": "StatArb Pares", "status": "STANDBY"},
                {"id": "early_breakout", "name": "⚡ Early Breakout", "cat": "Cola Ilíquida", "status": "STANDBY"},
                {"id": "funding_fade", "name": "🔥 Pump & Funding Fade", "cat": "Cola Ilíquida", "status": "STANDBY"},
                {"id": "oversold_bounce", "name": "❄️ Oversold Bounce", "cat": "Cola Ilíquida", "status": "STANDBY"},
            ]

            def _strat_match(strat_val: str | None, symbol: str, side: str = "") -> str:
                if strat_val:
                    sv = strat_val.lower()
                    if "breakout" in sv or "early" in sv:
                        return "early_breakout"
                    if "fade" in sv or "pump" in sv:
                        return "funding_fade"
                    if "bounce" in sv or "oversold" in sv:
                        return "oversold_bounce"
                    if "pair" in sv:
                        return "pairs"
                    if "carry" in sv:
                        return "carry"
                    if "stat_arb" in sv or "statarb" in sv or "reversion" in sv or "mean" in sv:
                        return "stat_arb"
                    if "momentum" in sv:
                        return "momentum"
                return "momentum"

            for s in strat_defs:
                sid = s["id"]
                pos_count = sum(1 for p in positions.values() if _strat_match(p.strategy, p.symbol, p.side.value) == sid)
                matching_trades = [t for t in (trades_24h or []) if _strat_match(t.get("strategy"), t.get("symbol", ""), t.get("side", "")) == sid]
                trd_count = len(matching_trades)
                rpnl = sum(float(t.get("realized_pnl", 0.0)) for t in matching_trades)

                pos_style = "bold yellow" if pos_count > 0 else "dim"
                trd_style = "bold white" if trd_count > 0 else "dim"

                if rpnl > 0:
                    pnl_text = Text(f"+${rpnl:,.2f}", style="bold green")
                elif rpnl < 0:
                    pnl_text = Text(f"-${abs(rpnl):,.2f}", style="bold red")
                else:
                    pnl_text = Text("$0.00", style="dim")

                status_style = "bold green" if s["status"] == "ACTIVA" else "yellow"

                strat_table.add_row(
                    s["name"],
                    s["cat"],
                    Text(f"● {s['status']}", style=status_style),
                    Text(str(pos_count), style=pos_style),
                    Text(str(trd_count), style=trd_style),
                    pnl_text,
                )

        layout["strategies_and_funding"]["strategies"].update(Panel(strat_table))

        # 5. Funding Table
        fnd_table = Table(title="Liquidaciones 8h Funding", expand=True, header_style="bold green")
        fnd_table.add_column("Símbolo", no_wrap=True)
        fnd_table.add_column("Tasa", justify="right", no_wrap=True)
        fnd_table.add_column("Lado", justify="center", no_wrap=True)
        fnd_table.add_column("Cashflow", justify="right", no_wrap=True)

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

        layout["strategies_and_funding"]["funding"].update(Panel(fnd_table))

        # 6. Orders Table
        ord_table = Table(title="Auditoría de Órdenes & Execution Alpha", expand=True, header_style="bold blue")
        ord_table.add_column("ID", style="dim", no_wrap=True)
        ord_table.add_column("Símbolo", no_wrap=True)
        ord_table.add_column("Lado", justify="center", no_wrap=True)
        ord_table.add_column("Tipo", justify="center", no_wrap=True)
        ord_table.add_column("Precio", justify="right", no_wrap=True)
        ord_table.add_column("Estado", justify="center", no_wrap=True)
        ord_table.add_column("Rol", justify="center", no_wrap=True)
        ord_table.add_column("Fee Pagada", justify="right", no_wrap=True)
        ord_table.add_column("Ahorro Maker", justify="right", no_wrap=True)

        for r in orders:
            side_color = "green" if r["side"] == "BUY" else "red"
            status_color = "green" if r["status"] == "FILLED" else ("yellow" if r["status"] == "PENDING" else "dim")
            role_str = r["role"] or "-"
            role_style = "bold cyan" if role_str == "MAKER" else "bold magenta"
            fee = float(r["fee"])
            savings = fee if role_str == "MAKER" else 0.0

            p_val = float(r["price"] or 0)
            p_str = f"${p_val:,.4f}" if (0 < p_val < 1.0) else f"${p_val:,.2f}"

            ord_table.add_row(
                r["id"][:12],
                r["symbol"],
                Text(r["side"], style=side_color),
                r["order_type"],
                p_str,
                Text(r["status"], style=status_color),
                Text(role_str, style=role_style),
                f"${fee:,.2f}",
                Text(f"+${savings:,.2f}" if savings > 0 else "-", style="green" if savings > 0 else "dim"),
            )
        layout["orders_and_trades"]["orders"].update(Panel(ord_table))

        # 7. Trades Table
        trade_table = Table(title="Auditoría de Trades & PnL Realizado", expand=True, header_style="bold green")
        trade_table.add_column("ID", style="dim", no_wrap=True)
        trade_table.add_column("Hora UTC", justify="center", no_wrap=True)
        trade_table.add_column("Símbolo", no_wrap=True)
        trade_table.add_column("Evento", justify="center", no_wrap=True)
        trade_table.add_column("Lado", justify="center", no_wrap=True)
        trade_table.add_column("Precio", justify="right", no_wrap=True)
        trade_table.add_column("Cant.", justify="right", no_wrap=True)
        trade_table.add_column("Rol", justify="center", no_wrap=True)
        trade_table.add_column("Fee", justify="right", no_wrap=True)
        trade_table.add_column("PnL Real.", justify="right", no_wrap=True)

        if trades:
            for t in trades:
                oid = str(t.get("order_id", ""))
                rpnl = float(t.get("realized_pnl", 0.0))
                side = t.get("side", "BUY")

                if "tp_" in oid:
                    ev_text = Text("🎯 TAKE-PROFIT", style="bold green")
                elif "sl_" in oid:
                    ev_text = Text("🛑 STOP-LOSS", style="bold red")
                elif side == "BUY" and rpnl == 0:
                    ev_text = Text("ENTRADA LONG", style="green")
                elif side == "SELL" and rpnl == 0:
                    ev_text = Text("ENTRADA SHORT", style="red")
                else:
                    ev_text = Text("CIERRE", style="yellow")

                side_style = "green" if side == "BUY" else "red"
                p_val = float(t.get("price", 0.0))
                p_str = f"${p_val:,.4f}" if p_val < 1.0 else f"${p_val:,.2f}"
                role_str = "MAKER" if t.get("is_maker") else "TAKER"
                role_style = "bold cyan" if role_str == "MAKER" else "bold magenta"
                fee_val = float(t.get("fee_paid", 0.0))
                qty_val = float(t.get("quantity", 0.0))

                if rpnl > 0:
                    pnl_text = Text(f"+${rpnl:,.2f}", style="bold green")
                elif rpnl < 0:
                    pnl_text = Text(f"-${abs(rpnl):,.2f}", style="bold red")
                else:
                    pnl_text = Text("-", style="dim")

                ts_str = str(t.get("timestamp", ""))
                try:
                    dt = datetime.fromisoformat(ts_str)
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = ts_str[-8:] if len(ts_str) >= 8 else ts_str

                trade_table.add_row(
                    t["id"][:12],
                    time_str,
                    t["symbol"],
                    ev_text,
                    Text(side, style=side_style),
                    p_str,
                    f"{qty_val:,.2f}",
                    Text(role_str, style=role_style),
                    f"${fee_val:,.4f}",
                    pnl_text,
                )
        else:
            trade_table.add_row("-", "--:--:--", "-", "-", "-", "$0.00", "0.00", "-", "$0.00", "-")

        layout["orders_and_trades"]["trades"].update(Panel(trade_table))

        # 6. Footer / Countdowns
        now = datetime.now(timezone.utc)
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

    def _build_plain_layout(
        self,
        account,
        positions,
        orders,
        trades,
        trades_24h,
        funding,
        rebalance,
        heartbeat,
        market_pulse,
        sf_telemetry=None,
    ):
        """Plain ANSI fallback for terminals without rich."""
        equity = float(account["total_equity"])
        cash = float(account["cash_balance"])
        print("\n" + "=" * 80)
        print(" CRYPTO-ALPHA-ENGINE: LIVE SHADOW TRADING (Binance Mainnet Read-Only)")
        if heartbeat:
            print(f" Status: ONLINE | Streams: {heartbeat.get('symbols_count')} | Ticks: {heartbeat.get('total_ticks')} (~{heartbeat.get('ticks_per_sec')} t/s)")
        print(f" Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | uPnL: ${account['unrealized_pnl']:+,.2f} | PnL: ${account['realized_pnl']:+,.2f}")
        print(" Posiciones Abiertas:")
        for sym, p in positions.items():
            print(f"  - {sym} ({getattr(p, 'strategy', 'Meta')}): {p.side.value} {p.quantity} @ ${p.entry_price:,.4f} (uPnL: ${p.unrealized_pnl:+,.2f})")
        print(" Últimos Trades:")
        for t in trades[:6]:
            print(f"  - [{t['id'][:10]}] {t['symbol']} {t['side']} {float(t['quantity']):,.2f} @ ${float(t['price']):,.4f} | PnL: ${float(t.get('realized_pnl', 0.0)):+,.2f}")
        print("=" * 80 + "\n")

    def run_live(self, refresh_rate: float = 2.0) -> None:
        """Starts live dashboard loop."""
        if not HAS_RICH:
            print("Running in non-rich mode...")
            while True:
                self.render_snapshot()
                time.sleep(refresh_rate)

        refresh_per_second = max(1, int(1.0 / refresh_rate)) if refresh_rate < 1.0 else 1
        with Live(self.render_snapshot(), refresh_per_second=refresh_per_second, console=self.console) as live:
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
