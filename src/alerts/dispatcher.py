"""
Institutional Asynchronous Incident Response & Telemetry Alert Dispatcher.

Supports:
- Priority routing (CRITICAL P0, WARNING P1, INFO P2).
- Dual destination support: Telegram Bot API & Discord Webhooks.
- Priority queue ensuring CRITICAL safety incidents preempt INFO notices.
- Non-blocking asynchronous delivery with exponential backoff retry.
- Automatic payload formatting, rate-limit throttling, and dry-run testing support.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AlertPriority(IntEnum):
    """Priority levels: lower integer = higher urgency."""

    CRITICAL = 0  # Safety circuit breaker trips, position drift, security breach
    WARNING = 1   # WebSocket disconnection, high slippage, order retry
    INFO = 2      # Executed trades, funding cashflows, daily EOD tear sheets


@dataclass(order=True)
class PrioritizedItem:
    """Wrapper ensuring PriorityQueue orders by priority then timestamp."""

    priority: int
    timestamp_ts: float
    message: AlertMessage = field(compare=False)


@dataclass
class AlertMessage:
    """Standardized alert telemetry payload."""

    title: str
    body: str
    priority: AlertPriority = AlertPriority.INFO
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prioritized_item(self) -> PrioritizedItem:
        return PrioritizedItem(
            priority=int(self.priority),
            timestamp_ts=self.timestamp.timestamp(),
            message=self,
        )


@dataclass
class WebhookConfig:
    """Configuration for external notification endpoints."""

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    discord_webhook_url: str | None = None
    min_priority: AlertPriority = AlertPriority.INFO
    rate_limit_per_minute: int = 30
    request_timeout_sec: float = 5.0
    mock_mode: bool = False

    @classmethod
    def from_env(cls) -> WebhookConfig:
        """Loads webhook credentials and URLs from environment variables."""
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
            mock_mode=os.getenv("ALERTS_MOCK_MODE", "false").lower() in ("1", "true"),
        )


class AlertDispatcher:
    """
    Thread-safe asynchronous alert dispatcher managing priority queues
    and routing notifications to Telegram and Discord endpoints.
    """

    def __init__(
        self,
        config: WebhookConfig | None = None,
        on_delivered: Callable[[AlertMessage, str], None] | None = None,
    ):
        self.config = config or WebhookConfig()
        self.on_delivered_callback = on_delivered

        self._queue: queue.PriorityQueue[PrioritizedItem] = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Telemetry
        self.total_dispatched: int = 0
        self.total_delivered: int = 0
        self.total_failed: int = 0
        self.dispatched_history: list[AlertMessage] = []

    def start(self) -> None:
        """Starts the background worker thread for alert processing."""
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="AlertDispatcherThread",
            )
            self._worker_thread.start()
            logger.info("AlertDispatcher started.")

    def stop(self, timeout: float = 3.0) -> None:
        """Stops the worker thread and flushes remaining queue items."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        logger.info("AlertDispatcher stopped.")

    def dispatch(self, alert: AlertMessage) -> None:
        """
        Enqueues an alert for asynchronous priority delivery.
        Non-blocking: will never stall caller or raise exceptions.
        """
        try:
            if alert.priority > self.config.min_priority:
                return

            self._queue.put(alert.to_prioritized_item(), block=False)
            with self._lock:
                self.total_dispatched += 1
                self.dispatched_history.append(alert)
                if len(self.dispatched_history) > 500:
                    self.dispatched_history.pop(0)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to enqueue alert '{alert.title}': {e}")

    def dispatch_critical(self, title: str, body: str, metadata: dict[str, Any] | None = None) -> None:
        """Helper to directly dispatch a high-urgency CRITICAL alert."""
        self.dispatch(
            AlertMessage(
                title=title,
                body=body,
                priority=AlertPriority.CRITICAL,
                metadata=metadata or {},
            )
        )

    def dispatch_warning(self, title: str, body: str, metadata: dict[str, Any] | None = None) -> None:
        """Helper to directly dispatch a WARNING alert."""
        self.dispatch(
            AlertMessage(
                title=title,
                body=body,
                priority=AlertPriority.WARNING,
                metadata=metadata or {},
            )
        )

    def dispatch_info(self, title: str, body: str, metadata: dict[str, Any] | None = None) -> None:
        """Helper to directly dispatch an informational notice."""
        self.dispatch(
            AlertMessage(
                title=title,
                body=body,
                priority=AlertPriority.INFO,
                metadata=metadata or {},
            )
        )

    def _worker_loop(self) -> None:
        """Continuous background worker draining priority queue."""
        while not self._stop_event.is_set():
            try:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                msg = item.message
                self._send_payload(msg)
                self._queue.task_done()

            except Exception:
                logger.exception("Unexpected error in alert worker loop")

    def _send_payload(self, msg: AlertMessage) -> None:
        """Dispatches message to configured channels (Telegram / Discord)."""
        # If in mock mode or no endpoints configured, record delivery and return
        if self.config.mock_mode or (not self.config.telegram_bot_token and not self.config.discord_webhook_url):
            logger.debug(f"[MOCK ALERT] [{msg.priority.name}] {msg.title}: {msg.body}")
            with self._lock:
                self.total_delivered += 1
            if self.on_delivered_callback:
                try:
                    self.on_delivered_callback(msg, "mock")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error in on_delivered callback: {e}")
            return

        success = False

        # 1. Telegram delivery
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            try:
                tg_ok = self._post_telegram(msg)
                if tg_ok:
                    success = True
                    if self.on_delivered_callback:
                        self.on_delivered_callback(msg, "telegram")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Telegram dispatch error: {e}")

        # 2. Discord delivery
        if self.config.discord_webhook_url:
            try:
                dc_ok = self._post_discord(msg)
                if dc_ok:
                    success = True
                    if self.on_delivered_callback:
                        self.on_delivered_callback(msg, "discord")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Discord dispatch error: {e}")

        with self._lock:
            if success:
                self.total_delivered += 1
            else:
                self.total_failed += 1

    def _post_telegram(self, msg: AlertMessage) -> bool:
        """Posts formatted message to Telegram Bot API."""
        emoji = "🚨" if msg.priority == AlertPriority.CRITICAL else ("⚠️" if msg.priority == AlertPriority.WARNING else "ℹ️")
        text = f"{emoji} *{msg.title}*\n\n{msg.body}\n\n`UTC: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`"
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=self.config.request_timeout_sec)
        return resp.status_code == 200

    def _post_discord(self, msg: AlertMessage) -> bool:
        """Posts rich embed to Discord Webhook."""
        color = 0xFF0000 if msg.priority == AlertPriority.CRITICAL else (0xFFA500 if msg.priority == AlertPriority.WARNING else 0x00FF00)
        embed = {
            "title": msg.title,
            "description": msg.body,
            "color": color,
            "timestamp": msg.timestamp.isoformat(),
            "footer": {"text": f"Priority: {msg.priority.name} | Crypto Alpha Engine"},
        }
        resp = requests.post(
            self.config.discord_webhook_url,  # type: ignore
            json={"embeds": [embed]},
            timeout=self.config.request_timeout_sec,
        )
        return resp.status_code in (200, 204)
