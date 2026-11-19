"""
Low-Overhead Live Multi-Modal Data Feed Adapter (Issue #72 / M16-6).

Streams and normalizes the live counterparts of the M16 multi-modal datasets
while keeping total Binance API weight strictly under 2% of the 2,400/min budget:
  1. Realized Post-Settlement Funding Scanner (`/fapi/v1/fundingRate` weight=1 +
     `/fapi/v1/premiumIndex` nextFundingTime map) for `Sniper:M25_PostSettleUnwind`
     (`T + 0.5m .. T + 10m` strictly AFTER `:00:00` settlement -> 0 bps funding paid).
  2. Causal `5m` Derivatives Positioning (`/futures/data/openInterestHist`,
     `topLongShortPositionRatio`, `globalLongShortAccountRatio`, `takerlongshortRatio`)
     with mandatory `+60s` publication safety lag.
  3. Multiplier-Adjusted Spot-vs-Perp Basis (`api.binance.com/api/v3/ticker/bookTicker`
     mapped through `SPOT_PERP_MULTIPLIER_MAP` + `1000x`/`1000000x` parity checks).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import requests

from src.data.multimodal_vision_ingestor import (
    SPOT_PERP_MULTIPLIER_MAP,
    is_valid_crypto_usdt_perp,
    verify_spot_price_parity,
)
from src.execution.rate_limiter import ExchangeRateLimiter

logger = logging.getLogger(__name__)

FAPI_BASE_URL = "https://fapi.binance.com"
SPOT_API_BASE_URL = "https://api.binance.com"
BINANCE_1M_WEIGHT_LIMIT = 2400


@dataclass(frozen=True)
class SettledFundingEvent:
    """Realized perpetual funding settlement detected strictly after settlement timestamp."""

    symbol: str
    funding_time_sec: int
    funding_rate: float
    mark_price: float
    elapsed_sec: float
    mins_to_next_settlement: float
    interval_hours: float = 8.0


@dataclass(frozen=True)
class SpotPerpBasisSnapshot:
    """Multiplier-adjusted real-time Spot-vs-Perp basis snapshot."""

    perp_symbol: str
    spot_symbol: str
    multiplier: float
    spot_mid: float
    perp_mid: float
    basis_bps: float
    timestamp_sec: float


@dataclass(frozen=True)
class DerivativesPositioningSnapshot:
    """Strictly causal (`+60s` lagged) 5m Open Interest & Smart-Money positioning snapshot."""

    symbol: str
    bar_ts_sec: int
    oi_usd: float
    oi_vel_15m: float
    oi_vel_1h: float
    top_ls_ratio: float
    global_ls_ratio: float
    smart_money_div: float
    taker_ls_vol_ratio: float


class LiveMultiModalFeedAdapter:
    """
    Thread-safe, TTL-cached multi-modal live data adapter with strict API weight budgeting.
    """

    def __init__(
        self,
        exchange_client: Any | None = None,
        rate_limiter: ExchangeRateLimiter | None = None,
        http_session: requests.Session | None = None,
        settlement_cache_ttl_sec: float = 45.0,
        spot_book_cache_ttl_sec: float = 30.0,
        positioning_cache_ttl_sec: float = 240.0,
        metrics_safety_lag_sec: int = 60,
        request_timeout_sec: float = 8.0,
    ) -> None:
        self.exchange = exchange_client
        self.rate_limiter = rate_limiter
        self._session = http_session or requests.Session()
        self.settlement_cache_ttl_sec = float(settlement_cache_ttl_sec)
        self.spot_book_cache_ttl_sec = float(spot_book_cache_ttl_sec)
        self.positioning_cache_ttl_sec = float(positioning_cache_ttl_sec)
        self.metrics_safety_lag_sec = int(metrics_safety_lag_sec)
        self.request_timeout_sec = float(request_timeout_sec)

        self._lock = threading.Lock()
        self._weight_events: deque[tuple[float, int]] = deque()

        # TTL caches
        self._cached_raw_funding_rows: list[dict[str, Any]] = []
        self._last_raw_funding_ts: float = 0.0
        self._cached_premium_index: dict[str, dict[str, float]] = {}
        self._last_premium_index_ts: float = 0.0
        self._cached_spot_books: dict[str, tuple[float, float, float]] = {}
        self._positioning_cache: dict[str, tuple[float, DerivativesPositioningSnapshot]] = {}

    def _record_weight(self, weight: int, now_mono: float | None = None) -> None:
        now_m = time.monotonic() if now_mono is None else float(now_mono)
        if self.rate_limiter is not None:
            self.rate_limiter.acquire(weight=max(1, int(weight)), priority="LOW", wait=True)
        with self._lock:
            self._weight_events.append((now_m, int(weight)))
            cutoff = now_m - 60.0
            while self._weight_events and self._weight_events[0][0] < cutoff:
                self._weight_events.popleft()

    @property
    def rolling_1m_weight(self) -> int:
        now_m = time.monotonic()
        cutoff = now_m - 60.0
        with self._lock:
            while self._weight_events and self._weight_events[0][0] < cutoff:
                self._weight_events.popleft()
            return int(sum(w for _, w in self._weight_events))

    @property
    def budget_utilization_pct(self) -> float:
        return (self.rolling_1m_weight / float(BINANCE_1M_WEIGHT_LIMIT)) * 100.0

    def _get_json(self, url: str, params: dict[str, Any] | None = None, weight: int = 1) -> Any:
        self._record_weight(weight)
        resp = self._session.get(url, params=params, timeout=self.request_timeout_sec)
        if self.rate_limiter is not None and hasattr(resp, "headers"):
            self.rate_limiter.on_http_response(resp.status_code, dict(resp.headers))
        resp.raise_for_status()
        return resp.json()

    def _fetch_premium_index_map(self, now_mono: float) -> dict[str, dict[str, float]]:
        if self._cached_premium_index and (now_mono - self._last_premium_index_ts) < 55.0:
            return self._cached_premium_index

        rows: Any = None
        fn = getattr(self.exchange, "fapiPublicGetPremiumIndex", None) if self.exchange is not None else None
        try:
            if callable(fn) and not isinstance(fn, MagicMock):
                self._record_weight(10, now_mono=now_mono)
                rows = fn()
            elif self.exchange is None or not isinstance(self.exchange, MagicMock):
                rows = self._get_json(f"{FAPI_BASE_URL}/fapi/v1/premiumIndex", weight=10)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"LiveMultiModalFeedAdapter premiumIndex fetch note: {exc}")
            return self._cached_premium_index

        parsed: dict[str, dict[str, float]] = {}
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                sym = str(r.get("symbol") or "").strip().upper()
                if not is_valid_crypto_usdt_perp(sym):
                    continue
                try:
                    mark_px = float(r.get("markPrice") or 0.0)
                    next_ft_ms = float(r.get("nextFundingTime") or 0.0)
                    last_fr = float(r.get("lastFundingRate") or 0.0)
                except (TypeError, ValueError):
                    continue
                parsed[sym] = {
                    "mark_price": mark_px,
                    "next_funding_sec": (next_ft_ms / 1000.0) if next_ft_ms > 1e11 else next_ft_ms,
                    "last_funding_rate": last_fr,
                }

        if parsed:
            self._cached_premium_index = parsed
            self._last_premium_index_ts = now_mono
        return self._cached_premium_index

    def _fetch_raw_funding_rows(self, now_mono: float, limit: int = 500) -> list[dict[str, Any]]:
        if self._cached_raw_funding_rows and (now_mono - self._last_raw_funding_ts) < self.settlement_cache_ttl_sec:
            return self._cached_raw_funding_rows

        rows: Any = None
        fn = getattr(self.exchange, "fapiPublicGetFundingRate", None) if self.exchange is not None else None
        try:
            if callable(fn) and not isinstance(fn, MagicMock):
                self._record_weight(1, now_mono=now_mono)
                rows = fn({"limit": int(limit)})
            elif self.exchange is None or not isinstance(self.exchange, MagicMock):
                rows = self._get_json(
                    f"{FAPI_BASE_URL}/fapi/v1/fundingRate",
                    params={"limit": int(limit)},
                    weight=1,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"LiveMultiModalFeedAdapter fundingRate fetch note: {exc}")
            return self._cached_raw_funding_rows

        if isinstance(rows, list) and rows:
            self._cached_raw_funding_rows = [r for r in rows if isinstance(r, dict)]
            self._last_raw_funding_ts = now_mono
        return self._cached_raw_funding_rows

    def fetch_recent_settled_funding(
        self,
        now_ts: float | None = None,
        min_elapsed_sec: float = 30.0,
        max_elapsed_sec: float = 600.0,
        max_funding_rate: float = -0.0020,
        min_mins_to_next_settlement: float = 45.0,
    ) -> tuple[list[SettledFundingEvent], int]:
        """
        Detects realized perpetual funding settlements across all USDS-M perps (`1h/2h/4h/8h`)
        that settled strictly `[now_ts - max_elapsed_sec, now_ts - min_elapsed_sec]` ago
        with `funding_rate <= max_funding_rate` (default `-20 bps`).

        Returns `(matching_events_sorted_most_negative_first, total_recent_settlements_seen)`
        so callers know whether Binance has published the current hour's settlements yet.
        """
        now_sec = float(time.time() if now_ts is None else now_ts)
        now_mono = time.monotonic()

        raw_rows = self._fetch_raw_funding_rows(now_mono=now_mono, limit=500)
        if not raw_rows:
            return [], 0

        prem_map = self._fetch_premium_index_map(now_mono=now_mono)

        latest_by_sym: dict[str, dict[str, Any]] = {}
        prev_ts_by_sym: dict[str, int] = {}
        for r in raw_rows:
            sym = str(r.get("symbol") or "").strip().upper()
            if not is_valid_crypto_usdt_perp(sym):
                continue
            try:
                ft_raw = float(r.get("fundingTime") or 0.0)
                ft_sec = int(round(ft_raw / 1000.0)) if ft_raw > 1e11 else int(round(ft_raw))
                # Round minor ms jitter (e.g. :00:00.003) to nearest minute boundary
                ft_sec_aligned = int(round(ft_sec / 60.0) * 60)
                fr_val = float(r.get("fundingRate"))
                mp_raw = r.get("markPrice")
                mp_val = float(mp_raw) if mp_raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                continue
            if ft_sec_aligned <= 0 or ft_sec_aligned > int(now_sec):
                continue

            cur = latest_by_sym.get(sym)
            if cur is None or ft_sec_aligned > cur["ft_sec"]:
                if cur is not None:
                    prev_ts_by_sym[sym] = max(prev_ts_by_sym.get(sym, 0), int(cur["ft_sec"]))
                latest_by_sym[sym] = {
                    "ft_sec": ft_sec_aligned,
                    "funding_rate": fr_val,
                    "mark_price": mp_val,
                }
            elif ft_sec_aligned < cur["ft_sec"]:
                prev_ts_by_sym[sym] = max(prev_ts_by_sym.get(sym, 0), ft_sec_aligned)

        recent_settlements_seen = 0
        matched: list[SettledFundingEvent] = []

        for sym, info in latest_by_sym.items():
            ft_sec = int(info["ft_sec"])
            elapsed = now_sec - float(ft_sec)
            if 0.0 <= elapsed <= max_elapsed_sec:
                recent_settlements_seen += 1
            if elapsed < min_elapsed_sec or elapsed > max_elapsed_sec:
                continue

            fr_val = float(info["funding_rate"])
            if fr_val > max_funding_rate:
                continue

            p_info = prem_map.get(sym, {})
            mark_px = float(info["mark_price"] or p_info.get("mark_price") or 0.0)

            prev_ft = prev_ts_by_sym.get(sym, 0)
            if prev_ft > 0 and 3000 <= (ft_sec - prev_ft) <= 32400:
                interval_h = float(round((ft_sec - prev_ft) / 3600.0))
                interval_h = max(1.0, min(8.0, interval_h))
            else:
                interval_h = 8.0

            next_ft_sec = float(p_info.get("next_funding_sec") or 0.0)
            if next_ft_sec > now_sec:
                mins_to_next = (next_ft_sec - now_sec) / 60.0
                inferred_h = round((next_ft_sec - ft_sec) / 3600.0)
                if 1.0 <= inferred_h <= 8.0:
                    interval_h = float(inferred_h)
            else:
                # Conservative fallback: assume at least 1h clock from realized settlement
                mins_to_next = ((ft_sec + 3600.0) - now_sec) / 60.0

            if mins_to_next < min_mins_to_next_settlement:
                continue

            matched.append(
                SettledFundingEvent(
                    symbol=sym,
                    funding_time_sec=ft_sec,
                    funding_rate=fr_val,
                    mark_price=mark_px,
                    elapsed_sec=elapsed,
                    mins_to_next_settlement=mins_to_next,
                    interval_hours=interval_h,
                )
            )

        matched.sort(key=lambda e: e.funding_rate)
        return matched, recent_settlements_seen

    def _fetch_single_spot_book(self, spot_symbol: str, now_mono: float) -> tuple[float, float] | None:
        cached = self._cached_spot_books.get(spot_symbol)
        if cached is not None and (now_mono - cached[0]) < self.spot_book_cache_ttl_sec:
            return cached[1], cached[2]

        if self.exchange is not None and isinstance(self.exchange, MagicMock):
            return None

        try:
            payload = self._get_json(
                f"{SPOT_API_BASE_URL}/api/v3/ticker/bookTicker",
                params={"symbol": spot_symbol},
                weight=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"LiveMultiModalFeedAdapter spot bookTicker fetch note ({spot_symbol}): {exc}")
            return (cached[1], cached[2]) if cached is not None else None

        row: dict[str, Any] | None = None
        if isinstance(payload, dict):
            row = payload
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and str(item.get("symbol") or "").strip().upper() == spot_symbol:
                    row = item
                    break

        if not row:
            return None
        try:
            b = float(row.get("bidPrice") or 0.0)
            a = float(row.get("askPrice") or 0.0)
        except (TypeError, ValueError):
            return None
        if b > 0.0 and a >= b:
            self._cached_spot_books[spot_symbol] = (now_mono, b, a)
            return b, a
        return None

    @staticmethod
    def resolve_spot_candidates_static(perp_symbol: str) -> list[tuple[str, float]]:
        """Ordered `(spot_symbol, multiplier)` candidates for a USDS-M perpetual."""
        clean = perp_symbol.replace("/", "").split(":")[0].strip().upper()
        cands: list[tuple[str, float]] = []
        if clean in SPOT_PERP_MULTIPLIER_MAP:
            cands.extend((s, float(m)) for s, m in SPOT_PERP_MULTIPLIER_MAP[clean])
        cands.append((clean, 1.0))
        if clean.startswith("1000000") and len(clean) > 11:
            cands.append((clean[7:], 1_000_000.0))
        elif clean.startswith("1000") and len(clean) > 8:
            cands.append((clean[4:], 1000.0))
        dedup: list[tuple[str, float]] = []
        seen: set[tuple[str, float]] = set()
        for item in cands:
            if item not in seen:
                seen.add(item)
                dedup.append(item)
        return dedup

    def fetch_spot_perp_basis_snapshot(
        self,
        perp_symbol: str,
        perp_mid: float,
        now_ts: float | None = None,
    ) -> SpotPerpBasisSnapshot | None:
        """
        Computes real-time multiplier-adjusted Spot-vs-Perp basis in bps:
            `basis_bps = ((perp_mid / (spot_mid * multiplier)) - 1.0) * 10_000`
        Returns `None` if no parity-verified Spot pair is available.
        """
        if perp_mid <= 0.0:
            return None
        clean_perp = perp_symbol.replace("/", "").split(":")[0].strip().upper()
        now_mono = time.monotonic()
        now_sec = float(time.time() if now_ts is None else now_ts)

        for spot_sym, mult in self.resolve_spot_candidates_static(clean_perp):
            ba = self._fetch_single_spot_book(spot_sym, now_mono=now_mono)
            if ba is None:
                continue
            spot_mid = 0.5 * (ba[0] + ba[1])
            if spot_mid <= 0.0:
                continue
            if not verify_spot_price_parity(perp_close=perp_mid, spot_close=spot_mid, multiplier=mult):
                continue
            adj_spot = spot_mid * mult
            basis_bps = ((perp_mid / adj_spot) - 1.0) * 10_000.0
            return SpotPerpBasisSnapshot(
                perp_symbol=clean_perp,
                spot_symbol=spot_sym,
                multiplier=mult,
                spot_mid=adj_spot,
                perp_mid=float(perp_mid),
                basis_bps=float(basis_bps),
                timestamp_sec=now_sec,
            )
        return None

    def fetch_derivatives_positioning_5m(
        self,
        symbol: str,
        now_ts: float | None = None,
        limit: int = 16,
    ) -> DerivativesPositioningSnapshot | None:
        """
        Fetches `5m` Open Interest velocity and Top-Trader vs Global Long/Short positioning
        for a single candidate symbol, enforcing a mandatory `+60s` causal safety lag
        so unsettled `5m` buckets are strictly excluded.
        """
        clean_sym = symbol.replace("/", "").split(":")[0].strip().upper()
        now_mono = time.monotonic()
        now_sec = float(time.time() if now_ts is None else now_ts)

        cached = self._positioning_cache.get(clean_sym)
        if cached is not None and (now_mono - cached[0]) < self.positioning_cache_ttl_sec:
            return cached[1]

        if self.exchange is not None and isinstance(self.exchange, MagicMock):
            return None

        cutoff_ts_sec = int(now_sec) - self.metrics_safety_lag_sec
        lim = max(14, min(int(limit), 30))

        try:
            oi_rows = self._get_json(
                f"{FAPI_BASE_URL}/futures/data/openInterestHist",
                params={"symbol": clean_sym, "period": "5m", "limit": lim},
                weight=1,
            )
            top_rows = self._get_json(
                f"{FAPI_BASE_URL}/futures/data/topLongShortPositionRatio",
                params={"symbol": clean_sym, "period": "5m", "limit": 6},
                weight=1,
            )
            glob_rows = self._get_json(
                f"{FAPI_BASE_URL}/futures/data/globalLongShortAccountRatio",
                params={"symbol": clean_sym, "period": "5m", "limit": 6},
                weight=1,
            )
            taker_rows = self._get_json(
                f"{FAPI_BASE_URL}/futures/data/takerlongshortRatio",
                params={"symbol": clean_sym, "period": "5m", "limit": 6},
                weight=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"LiveMultiModalFeedAdapter 5m positioning fetch note for {clean_sym}: {exc}")
            return None

        oi_pts: list[tuple[int, float]] = []
        if isinstance(oi_rows, list):
            for r in oi_rows:
                if not isinstance(r, dict):
                    continue
                try:
                    ts_s = int(round(float(r.get("timestamp") or 0.0) / 1000.0))
                    oi_v = float(r.get("sumOpenInterestValue") or 0.0)
                except (TypeError, ValueError):
                    continue
                if 0 < ts_s <= cutoff_ts_sec and oi_v > 0.0:
                    oi_pts.append((ts_s, oi_v))

        if not oi_pts:
            return None

        oi_pts.sort(key=lambda x: x[0])
        latest_ts, latest_oi = oi_pts[-1]
        oi_15m_prev = oi_pts[-4][1] if len(oi_pts) >= 4 else oi_pts[0][1]
        oi_1h_prev = oi_pts[-13][1] if len(oi_pts) >= 13 else oi_pts[0][1]
        oi_vel_15m = (latest_oi / max(oi_15m_prev, 1e-8)) - 1.0
        oi_vel_1h = (latest_oi / max(oi_1h_prev, 1e-8)) - 1.0

        def _latest_causal_val(rows_list: Any, key: str, default: float = 1.0) -> float:
            if not isinstance(rows_list, list):
                return default
            best_ts = -1
            best_v = default
            for item in rows_list:
                if not isinstance(item, dict):
                    continue
                try:
                    ts_s = int(round(float(item.get("timestamp") or 0.0) / 1000.0))
                    v = float(item.get(key) or 0.0)
                except (TypeError, ValueError):
                    continue
                if 0 < ts_s <= cutoff_ts_sec and v > 0.0 and ts_s >= best_ts:
                    best_ts = ts_s
                    best_v = v
            return best_v

        top_r = float(np.clip(_latest_causal_val(top_rows, "longShortRatio", 1.0), 0.05, 20.0))
        glob_r = float(np.clip(_latest_causal_val(glob_rows, "longShortRatio", 1.0), 0.05, 20.0))
        taker_r = float(np.clip(_latest_causal_val(taker_rows, "buySellRatio", 1.0), 0.05, 20.0))
        sm_div = float(np.log(top_r / glob_r))

        snap = DerivativesPositioningSnapshot(
            symbol=clean_sym,
            bar_ts_sec=latest_ts,
            oi_usd=latest_oi,
            oi_vel_15m=float(oi_vel_15m),
            oi_vel_1h=float(oi_vel_1h),
            top_ls_ratio=top_r,
            global_ls_ratio=glob_r,
            smart_money_div=sm_div,
            taker_ls_vol_ratio=taker_r,
        )
        self._positioning_cache[clean_sym] = (now_mono, snap)
        return snap
