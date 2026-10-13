"""
Standardized Alert Message Templates for Critical Incidents, Executions, and Summaries.
"""

from __future__ import annotations

from typing import Any

from src.alerts.dispatcher import AlertMessage, AlertPriority


def format_circuit_breaker_alert(
    breaker_name: str,
    metric_value: float,
    threshold: float,
    symbol: str | None = None,
    action_taken: str = "Execution Halted / Order Rejected",
) -> AlertMessage:
    """Builds a high-priority emergency alert when a safety circuit breaker trips."""
    sym_text = f" on `{symbol}`" if symbol else ""
    body = (
        f"**Safety Rule**: `{breaker_name}`{sym_text}\n"
        f"**Observed Metric**: `{metric_value:,.2f}`\n"
        f"**Circuit Breaker Ceiling**: `{threshold:,.2f}`\n"
        f"**Action Taken**: *{action_taken}*\n"
        f"Immediate inspection required before resuming automated order routing."
    )
    return AlertMessage(
        title=f"CIRCUIT BREAKER TRIPPED: {breaker_name}",
        body=body,
        priority=AlertPriority.CRITICAL,
        metadata={
            "breaker": breaker_name,
            "metric": metric_value,
            "threshold": threshold,
            "symbol": symbol,
        },
    )


def format_position_drift_alert(
    symbol: str,
    local_qty: float,
    exchange_qty: float,
    drift_pct: float,
) -> AlertMessage:
    """Builds an emergency alert when local internal accounting deviates from exchange truth."""
    body = (
        f"**Asset**: `{symbol}`\n"
        f"**Local Internal Position**: `{local_qty:+.4f}`\n"
        f"**Exchange Mainnet Position**: `{exchange_qty:+.4f}`\n"
        f"**Relative Discrepancy**: `{drift_pct * 100:.2f}%`\n"
        f"**Status**: *Trading PAUSED on asset. Position reconciliation invoked.*"
    )
    return AlertMessage(
        title=f"POSITION RECONCILIATION DRIFT DETECTED: {symbol}",
        body=body,
        priority=AlertPriority.CRITICAL,
        metadata={
            "symbol": symbol,
            "local_qty": local_qty,
            "exchange_qty": exchange_qty,
            "drift_pct": drift_pct,
        },
    )


def format_connection_outage_alert(
    exchange: str,
    duration_sec: float,
    fallback_active: bool = True,
) -> AlertMessage:
    """Builds an alert when WebSocket disconnects or REST fallback is triggered."""
    priority = AlertPriority.WARNING if fallback_active else AlertPriority.CRITICAL
    status = "REST polling fallback ACTIVE" if fallback_active else "DEADLINE EXCEEDED: Connection DEAD"
    body = (
        f"**Exchange**: `{exchange.upper()}`\n"
        f"**Downtime Duration**: `{duration_sec:.1f}s`\n"
        f"**Resilience Mode**: *{status}*\n"
    )
    return AlertMessage(
        title=f"EXCHANGE WS DISCONNECT: {exchange.upper()}",
        body=body,
        priority=priority,
        metadata={
            "exchange": exchange,
            "duration_sec": duration_sec,
            "fallback_active": fallback_active,
        },
    )


def format_trade_execution_alert(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    notional: float,
    fee: float,
    slippage_bps: float = 0.0,
) -> AlertMessage:
    """Builds informational confirmation upon filled trade execution."""
    body = (
        f"**Symbol**: `{symbol}`\n"
        f"**Side**: `{side.upper()}`\n"
        f"**Filled Quantity**: `{quantity:,.4f}`\n"
        f"**Fill Price**: `${price:,.4f}`\n"
        f"**Notional Value**: `${notional:,.2f}`\n"
        f"**Estimated Fee**: `${fee:,.4f}`\n"
        f"**Estimated Slippage**: `{slippage_bps:.1f} bps`\n"
    )
    return AlertMessage(
        title=f"TRADE FILL: {side.upper()} {symbol}",
        body=body,
        priority=AlertPriority.INFO,
        metadata={
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "notional": notional,
            "fee": fee,
            "slippage_bps": slippage_bps,
        },
    )


def format_funding_settlement_alert(
    net_cashflow: float,
    payments: list[dict[str, Any]],
) -> AlertMessage:
    """Builds funding fee settlement summary alert."""
    direction = "RECEIVED (+)" if net_cashflow >= 0 else "PAID (-)"
    body = (
        f"**Net Funding Cashflow**: `${net_cashflow:+,.4f}` ({direction})\n"
        f"**Instruments Settled**: `{len(payments)}`\n"
    )
    for p in payments[:5]:
        sym = p.get("symbol", "UNKNOWN")
        amt = float(p.get("amount", 0.0))
        body += f"• `{sym}`: `${amt:+,.4f}`\n"
    if len(payments) > 5:
        body += f"*...and {len(payments) - 5} more instruments.*\n"

    sign = "+$" if net_cashflow >= 0 else "-$"
    return AlertMessage(
        title=f"FUNDING SETTLEMENT: {sign}{abs(net_cashflow):,.2f}",
        body=body,
        priority=AlertPriority.INFO,
        metadata={
            "net_cashflow": net_cashflow,
            "num_payments": len(payments),
        },
    )
