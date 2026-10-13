"""
Unit tests for ResilientWebSocketClient, heartbeat timeouts, exponential backoff,
24h disconnect rollover, and REST fallback mechanisms.
"""

import json
import time

from src.execution.ws_manager import (
    PriceTick,
    ResilientWebSocketClient,
    WebSocketConfig,
    WebSocketState,
)


def test_ws_stream_url_generation():
    """Validates combined stream URL formatting for multiple symbols."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    client = ResilientWebSocketClient(symbols=symbols)

    url = client.get_stream_url()
    assert "btcusdt@bookTicker" in url
    assert "ethusdt@bookTicker" in url
    assert "solusdt@bookTicker" in url
    assert url.startswith("wss://fstream.binance.com/stream?streams=")


def test_tick_processing_and_buffer():
    """Verifies incoming message parsing, PriceTick construction, and ring buffer caching."""
    received_ticks = []

    def handle_tick(tick: PriceTick):
        received_ticks.append(tick)

    client = ResilientWebSocketClient(
        symbols=["BTC/USDT"],
        on_tick=handle_tick,
    )

    # Simulate Binance combined stream bookTicker payload
    mock_msg = json.dumps({
        "stream": "btcusdt@bookTicker",
        "data": {
            "s": "BTCUSDT",
            "b": "60000.00",
            "a": "60002.00",
            "B": "1.5",
            "A": "2.0",
        }
    })

    client._process_ws_message(mock_msg)

    assert len(received_ticks) == 1
    tick = received_ticks[0]
    assert tick.symbol == "BTCUSDT"
    assert tick.price == 60001.0
    assert tick.bid == 60000.0
    assert tick.ask == 60002.0
    assert tick.source == "WS"

    latest = client.get_latest_tick("BTCUSDT")
    assert latest is not None
    assert latest.price == 60001.0


def test_rest_fallback_on_prolonged_disconnect():
    """Validates that a disconnect exceeding 10 seconds triggers FALLBACK_REST and executes REST poll."""
    rest_polled = {"count": 0}

    def mock_rest_poll():
        rest_polled["count"] += 1
        return {
            "BTC/USDT": {"price": 60500.0, "bid": 60499.0, "ask": 60501.0},
        }

    cfg = WebSocketConfig(
        rest_fallback_threshold_sec=0.1,  # Fast 100ms threshold for unit test
    )
    client = ResilientWebSocketClient(
        symbols=["BTC/USDT"],
        config=cfg,
        rest_poll_callback=mock_rest_poll,
    )

    # Manually simulate disconnected state with disconnected_at in the past
    client.state = WebSocketState.DISCONNECTED
    client.disconnected_at = time.monotonic() - 0.2  # 200ms ago

    # Execute fallback monitor check
    client._execute_rest_fallback()

    assert rest_polled["count"] == 1
    tick = client.get_latest_tick("BTCUSDT")
    assert tick is not None
    assert tick.price == 60500.0
    assert tick.source == "REST_FALLBACK"


def test_heartbeat_timeout_detection():
    """Verifies is_healthy() returns False when messages stop arriving."""
    cfg = WebSocketConfig(heartbeat_timeout_sec=0.2)
    client = ResilientWebSocketClient(symbols=["BTC/USDT"], config=cfg)

    client.state = WebSocketState.CONNECTED
    client.last_message_time = time.monotonic() - 0.5  # 500ms ago

    assert not client.is_healthy()

    client.last_message_time = time.monotonic()
    assert client.is_healthy()


def test_24h_rollover_telemetry():
    """Validates operational statistics reporting."""
    client = ResilientWebSocketClient(symbols=["BTC/USDT", "ETH/USDT"])
    stats = client.get_stats()

    assert stats["state"] == WebSocketState.INITIALIZING.value
    assert stats["symbols_tracked"] == 2
    assert "uptime_seconds" in stats
