"""
Institutional End-Of-Day (EOD) Performance Tear Sheet Generator.

Calculates and formats:
- Daily Return %, Net PnL, Peak Equity, Current Drawdown %
- Active Gross Leverage & Net Exposure
- Trade Execution Win Rate, Total Trades, Profit Factor
- Net Funding Settlement Yield
- Rolling Annualized Sharpe Ratio
- Clean ASCII equity distribution and Telegram/Discord Markdown presentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.alerts.dispatcher import AlertMessage, AlertPriority


@dataclass
class DailyTearSheet:
    """Calculated metrics for an end-of-day portfolio performance tear sheet."""

    date_str: str
    starting_equity: float
    ending_equity: float
    net_pnl: float
    return_pct: float
    peak_equity: float
    drawdown_pct: float
    gross_leverage: float
    net_exposure: float
    total_trades: int
    winning_trades: int
    win_rate_pct: float
    profit_factor: float
    net_funding_fees: float
    sharpe_ratio: float | None = None
    positions_breakdown: dict[str, float] | None = None

    def to_markdown(self) -> str:
        """Formats the tear sheet into high-fidelity GitHub Flavored / Telegram Markdown."""
        pnl_sign = "+" if self.net_pnl >= 0 else "-"
        ret_sign = "+" if self.return_pct >= 0 else "-"
        sharpe_str = f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "N/A"

        md = (
            f"📊 **PORTFOLIO EOD TEAR SHEET — {self.date_str}**\n\n"
            f"• **Starting Equity**: `${self.starting_equity:,.2f}`\n"
            f"• **Ending Equity**: `${self.ending_equity:,.2f}`\n"
            f"• **Daily Net PnL**: `{pnl_sign}${abs(self.net_pnl):,.2f}` ({ret_sign}{abs(self.return_pct * 100):.2f}%)\n"
            f"• **Peak Equity**: `${self.peak_equity:,.2f}`\n"
            f"• **Current Drawdown**: `{self.drawdown_pct * 100:.2f}%`\n"
            f"• **Annualized Sharpe**: `{sharpe_str}`\n\n"
            f"**Risk & Exposure Metrics**:\n"
            f"• **Gross Leverage**: `{self.gross_leverage:.2f}x`\n"
            f"• **Net Dollar Exposure**: `${self.net_exposure:+,.2f}`\n"
            f"• **Net Funding Fees**: `${self.net_funding_fees:+,.2f}`\n\n"
            f"**Execution Performance**:\n"
            f"• **Total Trades**: `{self.total_trades}`\n"
            f"• **Win Rate**: `{self.win_rate_pct:.1f}%` ({self.winning_trades}/{self.total_trades})\n"
            f"• **Profit Factor**: `{self.profit_factor:.2f}`\n"
        )

        if self.positions_breakdown:
            md += "\n**Active Position Allocations**:\n"
            for sym, notional in self.positions_breakdown.items():
                md += f"  - `{sym}`: `${notional:+,.2f}`\n"

        return md

    def to_alert_message(self) -> AlertMessage:
        """Converts tear sheet into standard AlertMessage for Telegram/Discord dispatch."""
        pnl_sign = "+" if self.net_pnl >= 0 else "-"
        title = f"DAILY TEAR SHEET: {pnl_sign}${abs(self.net_pnl):,.2f} ({self.return_pct * 100:+.2f}%)"
        return AlertMessage(
            title=title,
            body=self.to_markdown(),
            priority=AlertPriority.INFO,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "date": self.date_str,
                "net_pnl": self.net_pnl,
                "return_pct": self.return_pct,
                "drawdown_pct": self.drawdown_pct,
                "sharpe_ratio": self.sharpe_ratio,
            },
        )


class EODTearSheetGenerator:
    """Computes daily institutional performance metrics from portfolio and trade history."""

    @classmethod
    def generate(
        cls,
        date_str: str | None = None,
        starting_equity: float = 100_000.0,
        ending_equity: float = 101_250.0,
        peak_equity: float = 102_000.0,
        trades: list[dict[str, Any]] | None = None,
        positions: dict[str, dict[str, Any]] | None = None,
        net_funding_fees: float = 0.0,
        historical_returns: list[float] | None = None,
    ) -> DailyTearSheet:
        """Constructs a DailyTearSheet from intraday trade records and account balances."""
        today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        net_pnl = ending_equity - starting_equity
        return_pct = net_pnl / starting_equity if starting_equity > 0 else 0.0

        drawdown_pct = max(0.0, (peak_equity - ending_equity) / peak_equity) if peak_equity > 0 else 0.0

        # Exposure and Leverage
        gross_notional = 0.0
        net_exposure = 0.0
        positions_breakdown: dict[str, float] = {}

        if positions:
            for sym, pos_data in positions.items():
                qty = float(pos_data.get("quantity") or pos_data.get("size") or 0.0)
                price = float(pos_data.get("price") or pos_data.get("current_price") or 0.0)
                notional = qty * price
                gross_notional += abs(notional)
                net_exposure += notional
                positions_breakdown[sym] = notional

        gross_leverage = gross_notional / ending_equity if ending_equity > 0 else 0.0

        # Trade analytics
        trade_list = trades or []
        total_trades = len(trade_list)
        winning_trades = 0
        gross_gains = 0.0
        gross_losses = 0.0

        for t in trade_list:
            pnl = float(t.get("pnl") or t.get("realized_pnl") or 0.0)
            if pnl > 0:
                winning_trades += 1
                gross_gains += pnl
            elif pnl < 0:
                gross_losses += abs(pnl)

        win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (gross_gains if gross_gains > 0 else 1.0)

        # Annualized Sharpe ratio calculation
        sharpe_ratio: float | None = None
        if historical_returns and len(historical_returns) >= 5:
            mean_ret = sum(historical_returns) / len(historical_returns)
            variance = sum((r - mean_ret) ** 2 for r in historical_returns) / (len(historical_returns) - 1)
            std_ret = math.sqrt(variance)
            if std_ret > 1e-8:
                sharpe_ratio = (mean_ret / std_ret) * math.sqrt(365)  # 365 trading days in crypto

        return DailyTearSheet(
            date_str=today,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            net_pnl=net_pnl,
            return_pct=return_pct,
            peak_equity=peak_equity,
            drawdown_pct=drawdown_pct,
            gross_leverage=gross_leverage,
            net_exposure=net_exposure,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate_pct=win_rate_pct,
            profit_factor=profit_factor,
            net_funding_fees=net_funding_fees,
            sharpe_ratio=sharpe_ratio,
            positions_breakdown=positions_breakdown if positions_breakdown else None,
        )
