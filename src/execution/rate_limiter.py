"""
Institutional Exchange Rate Limiting Engine.

Implements:
1. Token-Bucket & Leaky-Bucket Request Weight Limiter for Binance & Bybit.
2. IP Request Weight protection (2,400 weight/min default).
3. Order placement rate limiting (1,200 orders/min, 300 orders/10s).
4. Proactive high-watermark backoff (throttling when >85% capacity).
5. Dynamic HTTP 429 (Too Many Requests) & 418 (IP Ban) cooldown handling.
6. Thread-safe and async-compatible request scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for exchange rate limits (Binance USDS-M Futures defaults)."""

    ip_weight_limit_per_minute: int = 2400
    """Binance IP request weight limit per 60-second window."""

    order_limit_per_minute: int = 1200
    """Maximum order placements allowed per 60-second window."""

    order_limit_per_10s: int = 300
    """Maximum order placements allowed per 10-second window."""

    high_watermark_ratio: float = 0.85
    """Capacity ratio above which non-urgent requests are throttled with added delay."""

    default_cooldown_on_429_seconds: float = 60.0
    """Safety cooldown duration when receiving HTTP 429 / 418 if Retry-After is not present."""


class TokenBucket:
    """
    Thread-safe Token Bucket rate limiter.
    Tokens refill continuously at `refill_rate` (tokens per second) up to `capacity`.
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self.last_update
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_update = now

    def can_consume(self, tokens: float = 1.0) -> bool:
        """Returns True if requested tokens are immediately available without waiting."""
        with self._lock:
            self._refill(time.monotonic())
            return self.tokens >= tokens

    def acquire(self, tokens: float = 1.0, wait: bool = True, timeout: float | None = None) -> bool:
        """
        Synchronously acquires tokens.
        If `wait` is True, sleeps until tokens are available or timeout is reached.
        """
        tokens = float(tokens)
        start_time = time.monotonic()

        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not wait:
                    return False

                # Calculate required sleep time
                deficit = tokens - self.tokens
                sleep_sec = deficit / self.refill_rate

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + sleep_sec > timeout:
                    return False

            time.sleep(min(sleep_sec, 0.5))

    async def async_acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """Asynchronously acquires tokens with non-blocking sleep."""
        tokens = float(tokens)
        start_time = time.monotonic()

        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                deficit = tokens - self.tokens
                sleep_sec = deficit / self.refill_rate

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + sleep_sec > timeout:
                    return False

            await asyncio.sleep(min(sleep_sec, 0.25))

    @property
    def utilization(self) -> float:
        """Returns current utilization ratio (1.0 = fully consumed, 0.0 = completely free)."""
        with self._lock:
            self._refill(time.monotonic())
            available = max(0.0, self.tokens)
            return 1.0 - (available / self.capacity)


class ExchangeRateLimiter:
    """
    Production-grade Rate Limiter guarding REST and WebSocket requests against Binance 429/418 bans.
    Tracks multi-dimensional limits (IP request weights, 1m order counts, 10s order counts)
    and enforces high-watermark throttling and ban cooldowns.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()

        # 1. IP Weight Bucket (e.g. 2400 weight / 60s = 40 weight / sec)
        self.ip_bucket = TokenBucket(
            capacity=self.config.ip_weight_limit_per_minute,
            refill_rate=self.config.ip_weight_limit_per_minute / 60.0,
        )

        # 2. 1-Minute Order Placement Bucket (e.g. 1200 orders / 60s = 20 orders / sec)
        self.order_1m_bucket = TokenBucket(
            capacity=self.config.order_limit_per_minute,
            refill_rate=self.config.order_limit_per_minute / 60.0,
        )

        # 3. 10-Second Order Placement Bucket (e.g. 300 orders / 10s = 30 orders / sec)
        self.order_10s_bucket = TokenBucket(
            capacity=self.config.order_limit_per_10s,
            refill_rate=self.config.order_limit_per_10s / 10.0,
        )

        self._cooldown_until: float = 0.0
        self._lock = threading.Lock()
        self.last_used_weight_1m: int = 0
        self.total_requests_throttled: int = 0
        self.ban_events_count: int = 0

    def is_cooling_down(self) -> bool:
        """Returns True if the limiter is under an active 429/418 cooldown."""
        return time.monotonic() < self._cooldown_until

    def get_remaining_cooldown_seconds(self) -> float:
        """Returns remaining cooldown time in seconds (0.0 if not cooling down)."""
        return max(0.0, self._cooldown_until - time.monotonic())

    def trigger_cooldown(self, seconds: float | None = None, reason: str = "Rate Limit 429") -> None:
        """Activates emergency cooldown to protect against IP banning."""
        cd = float(seconds) if seconds is not None and seconds > 0 else self.config.default_cooldown_on_429_seconds
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + cd)
            self.ban_events_count += 1
        logger.warning(f"🚨 RateLimiter triggered {cd:.1f}s cooldown ({reason}). Cool down until +{cd:.1f}s")

    def acquire(
        self,
        weight: int = 1,
        is_order: bool = False,
        priority: str = "NORMAL",
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """
        Synchronously acquires permission to dispatch an API call.

        Parameters
        ----------
        weight : int
            Endpoint weight (e.g. 1 for ticker/single order, 5 for klines/depth).
        is_order : bool
            True if the request is an order placement or modification.
        priority : str
            "URGENT" (e.g. emergency stop/cancel), "NORMAL", or "LOW" (polling).
        wait : bool
            Whether to block until tokens are available.
        timeout : Optional[float]
            Maximum time to wait before failing.
        """
        start = time.monotonic()

        # Check emergency cooldown
        while self.is_cooling_down():
            if priority == "URGENT":
                # Emergency orders bypass cooldown if possible, but log warning
                logger.critical("URGENT request bypassing active rate limit cooldown!")
                break
            if not wait:
                return False
            remain = self.get_remaining_cooldown_seconds()
            if timeout is not None and remain > timeout:
                return False
            time.sleep(min(remain, 0.5))

        # Check high-watermark throttling for low-priority requests
        util = self.ip_bucket.utilization
        if util > self.config.high_watermark_ratio and priority != "URGENT":
            with self._lock:
                self.total_requests_throttled += 1
            # Inject proportional backoff delay (e.g. 100ms - 500ms)
            delay = 0.1 + (util - self.config.high_watermark_ratio) * 2.0
            time.sleep(delay)

        # Acquire IP weight
        ip_ok = self.ip_bucket.acquire(tokens=float(weight), wait=wait, timeout=timeout)
        if not ip_ok:
            return False

        # If order, acquire order count tokens
        if is_order:
            elapsed = time.monotonic() - start
            sub_timeout = max(0.0, timeout - elapsed) if timeout is not None else None
            ok_1m = self.order_1m_bucket.acquire(tokens=1.0, wait=wait, timeout=sub_timeout)
            if not ok_1m:
                return False

            elapsed = time.monotonic() - start
            sub_timeout = max(0.0, timeout - elapsed) if timeout is not None else None
            ok_10s = self.order_10s_bucket.acquire(tokens=1.0, wait=wait, timeout=sub_timeout)
            if not ok_10s:
                return False

        return True

    async def async_acquire(
        self,
        weight: int = 1,
        is_order: bool = False,
        priority: str = "NORMAL",
        timeout: float | None = None,
    ) -> bool:
        """Asynchronously acquires rate limit capacity."""
        start = time.monotonic()

        # Wait for cooldown if active
        while self.is_cooling_down():
            if priority == "URGENT":
                break
            remain = self.get_remaining_cooldown_seconds()
            if timeout is not None and remain > timeout:
                return False
            await asyncio.sleep(min(remain, 0.25))

        # High watermark backoff
        util = self.ip_bucket.utilization
        if util > self.config.high_watermark_ratio and priority != "URGENT":
            with self._lock:
                self.total_requests_throttled += 1
            delay = 0.1 + (util - self.config.high_watermark_ratio) * 2.0
            await asyncio.sleep(delay)

        ip_ok = await self.ip_bucket.async_acquire(tokens=float(weight), timeout=timeout)
        if not ip_ok:
            return False

        if is_order:
            elapsed = time.monotonic() - start
            sub_timeout = max(0.0, timeout - elapsed) if timeout is not None else None
            ok_1m = await self.order_1m_bucket.async_acquire(tokens=1.0, timeout=sub_timeout)
            if not ok_1m:
                return False

            elapsed = time.monotonic() - start
            sub_timeout = max(0.0, timeout - elapsed) if timeout is not None else None
            ok_10s = await self.order_10s_bucket.async_acquire(tokens=1.0, timeout=sub_timeout)
            if not ok_10s:
                return False

        return True

    def on_http_response(self, status_code: int, headers: dict[str, Any] | None = None) -> None:
        """
        Inspects exchange HTTP response headers and status codes to dynamically update rate limits.
        Parses `X-MBX-USED-WEIGHT-1M`, `Retry-After`, and handles 429/418 status codes.
        """
        headers = headers or {}
        header_map = {k.lower(): v for k, v in headers.items()}

        # 1. Update Binance reported 1-minute used weight
        used_weight_str = header_map.get("x-mbx-used-weight-1m")
        if used_weight_str is not None:
            try:
                used_val = int(used_weight_str)
                self.last_used_weight_1m = used_val
                # Sync bucket if exchange reported significantly higher usage
                with self.ip_bucket._lock:
                    est_remaining = max(0.0, float(self.config.ip_weight_limit_per_minute - used_val))
                    self.ip_bucket.tokens = min(self.ip_bucket.tokens, est_remaining)
            except ValueError:
                pass

        # 2. Handle HTTP 429 (Rate Limit Breached) or 418 (IP Ban Notice)
        if status_code in (429, 418):
            retry_after_str = header_map.get("retry-after")
            cooldown_sec = self.config.default_cooldown_on_429_seconds
            if retry_after_str is not None:
                try:
                    cooldown_sec = max(5.0, float(retry_after_str))
                except ValueError:
                    pass

            reason = "IP Ban Warning (HTTP 418)" if status_code == 418 else "Rate Limit Exceeded (HTTP 429)"
            self.trigger_cooldown(cooldown_sec, reason=reason)

    def get_stats(self) -> dict[str, Any]:
        """Returns rate limiter operational statistics."""
        return {
            "ip_weight_utilization": self.ip_bucket.utilization,
            "order_1m_utilization": self.order_1m_bucket.utilization,
            "order_10s_utilization": self.order_10s_bucket.utilization,
            "last_used_weight_1m": self.last_used_weight_1m,
            "is_cooling_down": self.is_cooling_down(),
            "cooldown_remaining_sec": self.get_remaining_cooldown_seconds(),
            "total_throttled": self.total_requests_throttled,
            "ban_events_count": self.ban_events_count,
        }
