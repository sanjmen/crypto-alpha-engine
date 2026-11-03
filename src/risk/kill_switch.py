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
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.domain.entities import Portfolio
from src.domain.enums import PositionSide

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchStatus:
    """Audit result of a Kill-Switch health check."""

    triggered: bool
    reason: str
    rolling_24h_dd_pct: float
    net_exposure_usd: float
    net_exposure_pct: float
    margin_ratio_pct: float
    flattened_symbols: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    ) -> None:
        self.max_24h_drawdown_pct = float(max_24h_drawdown_pct)
        self.max_net_exposure_pct = float(max_net_exposure_pct)
        self.max_margin_ratio = float(max_margin_ratio)
        self.lockout_hours = int(lockout_hours)

        self._equity_window_24h: Deque[Tuple[datetime, float]] = deque()
        self.halted_until: Optional[datetime] = None
        self.last_halt_reason: str = ""

    def is_locked(self, now: Optional[datetime] = None) -> bool:
        """Returns True if the kill-switch is currently in an emergency lockout window."""
        now_dt = now or datetime.now(timezone.utc)
        if self.halted_until and now_dt < self.halted_until:
            return True
        return False

    def record_equity(self, equity: float, timestamp: Optional[datetime] = None) -> float:
        """
        Records an equity observation and returns the rolling 24h drawdown fraction (0.0 to 1.0).
        """
        now_dt = timestamp or datetime.now(timezone.utc)
        cutoff = now_dt - timedelta(hours=24)
        self._equity_window_24h.append((now_dt, float(equity)))

        while self._equity_window_24h and self._equity_window_24h[0][0] < cutoff:
            self._equity_window_24h.popleft()

        peak_24h = max((eq for _, eq in self._equity_window_24h), default=float(equity))
        if peak_24h <= 0:
            return 0.0
        return max(0.0, (peak_24h - float(equity)) / peak_24h)

    @staticmethod
    def compute_net_exposure(portfolio: Portfolio) -> Tuple[float, float, float]:
        """Returns (long_usd, short_usd, net_usd) across all active portfolio positions."""
        long_usd = 0.0
        short_usd = 0.0
        for pos in portfolio.positions.values():
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue
            px = pos.current_price if pos.current_price > 0 else pos.entry_price
            notional = abs(pos.quantity * px)
            if pos.side == PositionSide.LONG:
                long_usd += notional
            elif pos.side == PositionSide.SHORT:
                short_usd += notional
        return long_usd, short_usd, long_usd - short_usd

    def sync_margin_and_leverage(
        self,
        exchange_client: Any,
        symbols: List[str],
        effective_leverage: float,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        """
        Synchronizes `CROSSED` margin mode and sets per-symbol exchange leverage bracket
        to `min(5, ceil(effective_leverage))` on Binance Futures (`fapi`).
        """
        target_int_lev = max(1, min(5, int(math.ceil(effective_leverage))))
        synced: Dict[str, int] = {}

        for sym in symbols:
            synced[sym] = target_int_lev
            if dry_run:
                continue
            try:
                if hasattr(exchange_client, "set_margin_mode"):
                    exchange_client.set_margin_mode("cross", sym)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Margin mode already CROSSED or unchanged for {sym}: {e}")
            try:
                if hasattr(exchange_client, "set_leverage"):
                    exchange_client.set_leverage(target_int_lev, sym)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not set leverage={target_int_lev}x for {sym}: {e}")

        return synced

    def evaluate_and_enforce(
        self,
        portfolio: Portfolio,
        broker: Any = None,
        margin_ratio: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> KillSwitchStatus:
        """
        Evaluates all 3 hard safety circuit breakers:
        1. Rolling 24h drawdown >= `max_24h_drawdown_pct` (12%)
        2. Maintenance margin ratio >= `max_margin_ratio` (50%)
        3. Net dollar exposure asymmetry > `max_net_exposure_pct` (15% of equity)

        If (1) or (2) triggers, cancels all open orders and flattens all positions (`reduceOnly=True`).
        """
        now_dt = timestamp or datetime.now(timezone.utc)
        equity = max(float(portfolio.total_equity), 1.0)
        dd_24h = self.record_equity(equity, timestamp=now_dt)
        _, _, net_usd = self.compute_net_exposure(portfolio)
        net_pct = abs(net_usd) / equity

        if self.is_locked(now_dt):
            return KillSwitchStatus(
                triggered=True,
                reason=f"LOCKED: {self.last_halt_reason}",
                rolling_24h_dd_pct=dd_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
            )

        trigger_reason = ""
        if dd_24h >= self.max_24h_drawdown_pct:
            trigger_reason = (
                f"24h Drawdown {dd_24h*100:.2f}% breached limit {self.max_24h_drawdown_pct*100:.2f}%"
            )
        elif margin_ratio >= self.max_margin_ratio:
            trigger_reason = (
                f"Margin Ratio {margin_ratio*100:.2f}% breached limit {self.max_margin_ratio*100:.2f}%"
            )

        flattened: List[str] = []
        if trigger_reason:
            self.halted_until = now_dt + timedelta(hours=self.lockout_hours)
            self.last_halt_reason = trigger_reason
            logger.critical(f"🚨 EMERGENCY KILL-SWITCH TRIGGERED: {trigger_reason}")
            if broker is not None and hasattr(broker, "emergency_flatten_all"):
                flattened = broker.emergency_flatten_all()
            return KillSwitchStatus(
                triggered=True,
                reason=trigger_reason,
                rolling_24h_dd_pct=dd_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
                flattened_symbols=flattened,
            )

        # Check Net Exposure Asymmetry Guard
        if net_pct > self.max_net_exposure_pct:
            asym_reason = (
                f"Net Exposure Asymmetry {net_pct*100:.2f}% (${net_usd:+,.2f}) "
                f"exceeded {self.max_net_exposure_pct*100:.2f}% limit"
            )
            logger.warning(f"⚠️ {asym_reason}")
            if broker is not None and hasattr(broker, "hedge_excess_exposure"):
                broker.hedge_excess_exposure(net_usd)
            return KillSwitchStatus(
                triggered=True,
                reason=asym_reason,
                rolling_24h_dd_pct=dd_24h * 100.0,
                net_exposure_usd=net_usd,
                net_exposure_pct=net_pct * 100.0,
                margin_ratio_pct=margin_ratio * 100.0,
            )

        return KillSwitchStatus(
            triggered=False,
            reason="OK",
            rolling_24h_dd_pct=dd_24h * 100.0,
            net_exposure_usd=net_usd,
            net_exposure_pct=net_pct * 100.0,
            margin_ratio_pct=margin_ratio * 100.0,
        )
