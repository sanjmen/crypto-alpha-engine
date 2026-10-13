"""
Unit tests for institutional TokenBucket and ExchangeRateLimiter.
Validates multi-bucket capacity, request weight limits, high-watermark throttling,
and HTTP 429 / 418 cooldowns.
"""

import asyncio
import time

from src.execution.rate_limiter import (
    ExchangeRateLimiter,
    RateLimitConfig,
    TokenBucket,
)


def test_token_bucket_refill_and_consumption():
    """Verifies continuous token refill and immediate consumption."""
    # Capacity 10 tokens, refill 10 tokens / sec
    bucket = TokenBucket(capacity=10.0, refill_rate=10.0)

    assert bucket.can_consume(5.0)
    assert bucket.acquire(5.0, wait=False)
    assert bucket.tokens <= 5.1

    # Cannot consume 10 immediately
    assert not bucket.can_consume(10.0)
    assert not bucket.acquire(10.0, wait=False)

    # Wait 0.5s for refill
    time.sleep(0.5)
    assert bucket.can_consume(4.0)
    assert bucket.acquire(4.0, wait=False)


def test_token_bucket_async_acquire():
    """Tests non-blocking asynchronous token acquisition."""
    async def _test():
        bucket = TokenBucket(capacity=2.0, refill_rate=10.0)

        ok1 = await bucket.async_acquire(2.0)
        assert ok1

        # Empty now, async_acquire will wait briefly
        start = time.monotonic()
        ok2 = await bucket.async_acquire(1.0, timeout=1.0)
        elapsed = time.monotonic() - start

        assert ok2
        assert elapsed >= 0.05

    asyncio.run(_test())


def test_exchange_rate_limiter_ip_and_order_limits():
    """Verifies multi-dimensional limits (IP weights and order limits)."""
    cfg = RateLimitConfig(
        ip_weight_limit_per_minute=100,
        order_limit_per_minute=20,
        order_limit_per_10s=5,
    )
    limiter = ExchangeRateLimiter(config=cfg)

    # Acquire standard request weight (e.g. klines weight 5)
    assert limiter.acquire(weight=5, is_order=False)
    assert limiter.ip_bucket.tokens < 100.0

    # Acquire order placement (weight 1 + order counts)
    for _ in range(5):
        assert limiter.acquire(weight=1, is_order=True, wait=False)

    # 10s order limit reached (5 orders)
    assert not limiter.acquire(weight=1, is_order=True, wait=False)


def test_http_429_cooldown_and_retry_after():
    """Validates dynamic cooldown trigger upon receiving HTTP 429/418 with Retry-After header."""
    limiter = ExchangeRateLimiter()

    assert not limiter.is_cooling_down()

    # Simulate exchange returning HTTP 429 with 3-second Retry-After
    limiter.on_http_response(status_code=429, headers={"Retry-After": "2", "X-MBX-USED-WEIGHT-1M": "2390"})

    assert limiter.is_cooling_down()
    assert limiter.get_remaining_cooldown_seconds() > 0.5
    assert limiter.last_used_weight_1m == 2390

    # Non-urgent requests blocked during cooldown
    assert not limiter.acquire(weight=1, priority="NORMAL", wait=False)

    # URGENT requests can bypass during emergencies
    assert limiter.acquire(weight=1, priority="URGENT", wait=False)


def test_high_watermark_throttling_statistics():
    """Verifies high-watermark throttling and telemetry tracking."""
    cfg = RateLimitConfig(
        ip_weight_limit_per_minute=10,
        high_watermark_ratio=0.5,
    )
    limiter = ExchangeRateLimiter(config=cfg)

    # Consume 6 tokens (>50% watermark)
    limiter.acquire(weight=6, wait=False)
    assert limiter.ip_bucket.utilization >= 0.5

    stats = limiter.get_stats()
    assert stats["ip_weight_utilization"] >= 0.5
    assert not stats["is_cooling_down"]
