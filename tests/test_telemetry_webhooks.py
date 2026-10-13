"""
Unit tests for AlertDispatcher, Telegram / Discord Webhooks, Priority Queue Routing,
Incident Templates, and Daily EOD Performance Tear Sheets.
"""

import time
from unittest.mock import MagicMock, patch

from src.alerts.dispatcher import (
    AlertDispatcher,
    AlertMessage,
    AlertPriority,
    WebhookConfig,
)
from src.alerts.tearsheet import EODTearSheetGenerator
from src.alerts.templates import (
    format_circuit_breaker_alert,
    format_connection_outage_alert,
    format_funding_settlement_alert,
    format_position_drift_alert,
    format_trade_execution_alert,
)


def test_alert_priority_queue_ordering():
    """Verifies that CRITICAL emergency alerts are delivered ahead of INFO notices."""
    delivered_priorities: list[AlertPriority] = []

    def handle_delivered(msg: AlertMessage, _channel: str):
        delivered_priorities.append(msg.priority)

    config = WebhookConfig(mock_mode=True)
    dispatcher = AlertDispatcher(config=config, on_delivered=handle_delivered)

    # Dispatch in reverse order: INFO, WARNING, CRITICAL
    dispatcher.dispatch_info("Info Alert 1", "Regular update")
    dispatcher.dispatch_warning("Warning Alert 2", "Minor issue")
    dispatcher.dispatch_critical("Critical Alert 3", "Emergency halt")

    # Start worker to process queue
    dispatcher.start()
    time.sleep(0.2)
    dispatcher.stop(timeout=1.0)

    # First delivered item must be CRITICAL, then WARNING, then INFO
    assert len(delivered_priorities) == 3
    assert delivered_priorities[0] == AlertPriority.CRITICAL
    assert delivered_priorities[1] == AlertPriority.WARNING
    assert delivered_priorities[2] == AlertPriority.INFO


def test_telegram_and_discord_post_payloads():
    """Verifies payload construction for Telegram Markdown and Discord Embeds."""
    config = WebhookConfig(
        telegram_bot_token="test_bot_token_123",
        telegram_chat_id="12345678",
        discord_webhook_url="https://discord.com/api/webhooks/mock/test",
    )
    dispatcher = AlertDispatcher(config=config)

    critical_msg = AlertMessage(
        title="Circuit Breaker Tripped",
        body="Drawdown exceeded 15%",
        priority=AlertPriority.CRITICAL,
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        # Test Telegram post
        tg_ok = dispatcher._post_telegram(critical_msg)
        assert tg_ok
        tg_call = mock_post.call_args_list[-1]
        assert "api.telegram.org" in tg_call[0][0]
        assert tg_call[1]["json"]["chat_id"] == "12345678"
        assert "🚨" in tg_call[1]["json"]["text"]

        # Test Discord post
        dc_ok = dispatcher._post_discord(critical_msg)
        assert dc_ok
        dc_call = mock_post.call_args_list[-1]
        assert "discord.com" in dc_call[0][0]
        embed = dc_call[1]["json"]["embeds"][0]
        assert embed["color"] == 0xFF0000  # Red for critical


def test_standardized_alert_templates():
    """Verifies formatting of circuit breaker, drift, outage, trade fill, and funding alerts."""
    # 1. Circuit breaker alert
    cb_alert = format_circuit_breaker_alert(
        breaker_name="MaxGrossLeverage",
        metric_value=3.2,
        threshold=2.5,
        symbol="BTC/USDT",
    )
    assert cb_alert.priority == AlertPriority.CRITICAL
    assert "CIRCUIT BREAKER TRIPPED" in cb_alert.title
    assert "BTC/USDT" in cb_alert.body

    # 2. Position drift alert
    drift_alert = format_position_drift_alert(
        symbol="ETH/USDT",
        local_qty=2.5,
        exchange_qty=2.1,
        drift_pct=0.16,
    )
    assert drift_alert.priority == AlertPriority.CRITICAL
    assert "16.00%" in drift_alert.body

    # 3. Connection outage alert
    outage_alert = format_connection_outage_alert(
        exchange="binance",
        duration_sec=15.2,
        fallback_active=True,
    )
    assert outage_alert.priority == AlertPriority.WARNING
    assert "REST polling fallback ACTIVE" in outage_alert.body

    # 4. Trade execution alert
    fill_alert = format_trade_execution_alert(
        symbol="SOL/USDT",
        side="buy",
        quantity=50.0,
        price=145.20,
        notional=7260.0,
        fee=2.90,
        slippage_bps=2.1,
    )
    assert fill_alert.priority == AlertPriority.INFO
    assert "BUY SOL/USDT" in fill_alert.title
    assert "$7,260.00" in fill_alert.body

    # 5. Funding settlement alert
    funding_alert = format_funding_settlement_alert(
        net_cashflow=18.50,
        payments=[{"symbol": "BTC/USDT", "amount": 12.0}, {"symbol": "ETH/USDT", "amount": 6.5}],
    )
    assert funding_alert.priority == AlertPriority.INFO
    assert "+$18.50" in funding_alert.title


def test_eod_tearsheet_generator():
    """Verifies daily performance analytics, Sharpe calculation, and markdown generation."""
    trades = [
        {"pnl": 350.0},
        {"pnl": 150.0},
        {"pnl": -100.0},
        {"pnl": 200.0},
    ]
    positions = {
        "BTCUSDT": {"quantity": 0.5, "price": 60000.0},
        "ETHUSDT": {"quantity": -5.0, "price": 3000.0},
    }
    hist_returns = [0.01, 0.005, -0.002, 0.008, 0.004, 0.012]

    tearsheet = EODTearSheetGenerator.generate(
        date_str="2026-10-13",
        starting_equity=100_000.0,
        ending_equity=100_600.0,
        peak_equity=101_000.0,
        trades=trades,
        positions=positions,
        net_funding_fees=25.40,
        historical_returns=hist_returns,
    )

    assert tearsheet.net_pnl == 600.0
    assert tearsheet.return_pct == 0.006  # 0.6%
    assert tearsheet.total_trades == 4
    assert tearsheet.winning_trades == 3
    assert tearsheet.win_rate_pct == 75.0
    assert tearsheet.profit_factor == 7.0  # 700 / 100
    assert tearsheet.gross_leverage > 0.4
    assert tearsheet.sharpe_ratio is not None and tearsheet.sharpe_ratio > 0

    # Markdown representation
    md = tearsheet.to_markdown()
    assert "PORTFOLIO EOD TEAR SHEET" in md
    assert "$100,600.00" in md
    assert "75.0%" in md

    # AlertMessage conversion
    alert = tearsheet.to_alert_message()
    assert alert.priority == AlertPriority.INFO
    assert "+$600.00" in alert.title
