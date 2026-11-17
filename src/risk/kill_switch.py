"""
Production Emergency Kill-Switch, Cross-Margin Synchronization & Net Exposure Circuit Breaker.

Enforces institutional live safety guardrails on Binance USD-M Perpetual Futures:
1. Pre-trade Cross-Margin (`CROSSED`) and per-symbol leverage synchronization.
2. Rolling 24h Drawdown Circuit Breaker (-12% hard stop -> cancel all open orders & flatten `reduceOnly=True`).
3. Net Dollar Exposure Asymmetry Guard (|Long_USD - Short_USD| <= 15% of Equity).
4. Maintenance Margin Ratio Guard (maintMargin / marginBalance <= 50%).
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Any

from src.domain.entities import Portfolio
from src.domain.enums import PositionSide

logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime | None) -> datetime:
    """Normalizes a datetime (or None) into a timezone-aware UTC datetime."""
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


@dataclass
class KillSwitchStatus:
    """Audit result of a Kill-Switch health check."""

    triggered: bool
    reason: str
    rolling_24h_dd_pct: float
    net_exposure_usd: float
    net_exposure_pct: float
    margin_ratio_pct: float
    flattened_symbols: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def rolling_24h_drawdown_pct(self) -> float:
        return self.rolling_24h_dd_pct


class EmergencyKillSwitch:
    """
    Independent hardware/software safety circuit breaker for Live Production.
    """

    def __init__(
        self,
        max_24h_drawdown_pct: float = 0.12,
        max_net_exposure_pct: float = 0.15,
        max_margin_ratio: float = 0.50,
        lockout_hours: int = 24,
        equity_floor_usd: float = 0.0,
    ) -> None:
        self.max_24h_drawdown_pct = float(max_24h_drawdown_pct)
        self.max_net_exposure_pct = float(max_net_exposure_pct)
        self.max_margin_ratio = float(max_margin_ratio)
        self.lockout_hours = int(lockout_hours)
        self.equity_floor_usd = float(equity_floor_usd)

        self._equity_window_24h: deque[tuple[datetime, float]] = deque()
        self.halted_until: datetime | None = None
        self.last_halt_reason: str = ""
        self._last_asymmetry_warn_ts: datetime | None = None

    def is_locked(self, now: datetime | None = None) -> bool:
        """Returns True if the kill-switch is currently in an emergency lockout window."""
        now_dt = _ensure_utc(now)
        return bool(self.halted_until and now_dt < _ensure_utc(self.halted_until))

    def record_equity(self, equity: float, timestamp: datetime | None = None) -> float:
        """
        Records an equity observation and returns the rolling 24h drawdown fraction (0.0 to 1.0).
        """
        now_dt = _ensure_utc(timestamp)
        cutoff = now_dt - timedelta(hours=24)
        self._equity_window_24h.append((now_dt, float(equity)))

        while self._equity_window_24h and self._equity_window_24h[0][0] < cutoff:
            self._equity_window_24h.popleft()

        peak_24h = max((eq for _, eq in self._equity_window_24h), default=float(equity))
        if peak_24h <= 0:
            return 0.0
        return max(0.0, (peak_24h - float(equity)) / peak_24h)

    @staticmethod
    def compute_net_exposure(portfolio: Portfolio) -> tuple[float, float, float]:
        """Returns (long_usd, short_usd, net_usd) across all active portfolio positions."""
        long_usd = 0.0
        short_usd = 0.0
        for position in portfolio.positions.values():
            if position.side == PositionSide.FLAT or position.quantity <= 0:
                continue
            price = position.current_price if position.current_price > 0 else position.entry_price
            notional = abs(position.quantity * price)
            if position.side == PositionSide.LONG:
                long_usd += notional
            elif position.side == PositionSide.SHORT:
                short_usd += notional
        return long_usd, short_usd, long_usd - short_usd

    def sync_margin_and_leverage(
        self,
        exchange_client: Any,
        symbols: list[str],
        effective_leverage: float,
        dry_run: bool = True,
        enforce_one_way_mode: bool = True,
    ) -> dict[str, int]:
        """
        Synchronizes One-Way Position Mode (`dualSidePosition='false'`), `CROSSED` margin mode,
        and sets per-symbol exchange leverage bracket to `min(5, ceil(effective_leverage))`
        on Binance Futures (`fapi`).
        """
        target_leverage_int = max(1, min(5, int(math.ceil(effective_leverage))))
        synced: dict[str, int] = {}

        if not dry_run and enforce_one_way_mode:
            try:
                is_hedged = None
                if hasattr(exchange_client, "fetch_position_mode"):
                    mode_info = exchange_client.fetch_position_mode()
                    if isinstance(mode_info, dict):
                        is_hedged = mode_info.get("hedged")
                if is_hedged is not False and hasattr(exchange_client, "set_position_mode"):
                    exchange_client.set_position_mode(False)
            except Exception as err:  # noqa: BLE001
                err_str = str(err)
                if "-4059" in err_str or "No need to change position side" in err_str:
                    logger.debug("Position mode already set to One-Way Mode (-4059).")
                else:
                    logger.warning(f"Could not enforce One-Way position mode: {err}")

        for symbol in symbols:
            synced[symbol] = target_leverage_int
            if dry_run:
                continue
            try:
                if hasattr(exchange_client, "set_margin_mode"):
                    exchange_client.set_margin_mode("cross", symbol)
            except Exception as err:  # noqa: BLE001
                logger.debug(f"Margin mode already CROSSED or unchanged for {symbol}: {err}")
            try:
                if hasattr(exchange_client, "set_leverage"):
                    exchange_client.set_leverage(target_leverage_int, symbol)
            except Exception as err:  # noqa: BLE001
                logger.warning(f"Could not set leverage={target_leverage_int}x for {symbol}: {err}")

        return synced

    def evaluate_and_enforce(
        self,
        portfolio: Portfolio,
        broker: Any = None,
        margin_ratio: float = 0.0,
        timestamp: datetime | None = None,
    ) -> KillSwitchStatus:
        """
        Evaluates all 3 hard safety circuit breakers:
        1. Rolling 24h drawdown >= `max_24h_drawdown_pct` (12%)
        2. Maintenance margin ratio >= `max_margin_ratio` (50%)
        3. Net dollar exposure asymmetry > `max_net_exposure_pct` (15% of gross/equity)
        """
        now_dt = _ensure_utc(timestamp)
        equity = max(float(portfolio.total_equity), 1.0)
        drawdown_24h = self.record_equity(equity, timestamp=now_dt)
        long_usd, short_usd, net_usd = self.compute_net_exposure(portfolio)
        gross_usd = long_usd + short_usd
        net_pct = abs(net_usd) / max(gross_usd, equity, 1.0)

        if self.is_locked(now_dt):
            return KillSwitchStatus(
                triggered=True,
                reason=f"LOCKED: {self.last_halt_reason}",
                rolling_24h_dd_pct=drawdown_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
            )

        trigger_reason = ""
        is_floor_breach = False
        raw_equity = float(portfolio.total_equity)
        if self.equity_floor_usd > 0.0 and 0.0 < raw_equity <= self.equity_floor_usd:
            trigger_reason = f"Hard Equity Floor breached: ${raw_equity:,.2f} <= ${self.equity_floor_usd:,.2f}"
            is_floor_breach = True
        elif drawdown_24h >= self.max_24h_drawdown_pct:
            trigger_reason = (
                f"24h Drawdown {drawdown_24h * 100:.2f}% breached limit {self.max_24h_drawdown_pct * 100:.2f}%"
            )
        elif margin_ratio >= self.max_margin_ratio:
            trigger_reason = f"Margin Ratio {margin_ratio * 100:.2f}% breached limit {self.max_margin_ratio * 100:.2f}%"

        flattened: list[str] = []
        if trigger_reason:
            lockout_hrs = max(self.lockout_hours, 8760) if is_floor_breach else self.lockout_hours
            self.halted_until = now_dt + timedelta(hours=lockout_hrs)
            self.last_halt_reason = trigger_reason
            logger.critical(f"🚨 EMERGENCY KILL-SWITCH TRIGGERED: {trigger_reason}")
            if broker is not None and hasattr(broker, "emergency_flatten_all"):
                flattened = broker.emergency_flatten_all()
            return KillSwitchStatus(
                triggered=True,
                reason=trigger_reason,
                rolling_24h_dd_pct=drawdown_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
                flattened_symbols=flattened,
            )

        if net_pct > self.max_net_exposure_pct:
            asymmetry_reason = (
                f"Net Exposure Asymmetry {net_pct * 100:.2f}% (${net_usd:+,.2f}) "
                f"exceeded {self.max_net_exposure_pct * 100:.2f}% limit"
            )
            if self._last_asymmetry_warn_ts is None or (now_dt - self._last_asymmetry_warn_ts).total_seconds() >= 300.0:
                logger.warning(f"⚠️ {asymmetry_reason}")
                self._last_asymmetry_warn_ts = now_dt
            if broker is not None and hasattr(broker, "hedge_excess_exposure"):
                broker.hedge_excess_exposure(net_usd)
            return KillSwitchStatus(
                triggered=True,
                reason=asymmetry_reason,
                rolling_24h_dd_pct=drawdown_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
            )

        return KillSwitchStatus(
            triggered=False,
            reason="OK",
            rolling_24h_dd_pct=drawdown_24h * 100.0,
            net_exposure_usd=net_usd,
            net_exposure_pct=net_pct * 100.0,
            margin_ratio_pct=margin_ratio * 100.0,
        )
