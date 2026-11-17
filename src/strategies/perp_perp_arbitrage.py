"""
Institutional Cross-Exchange Perpetual-Perpetual Funding Spread Arbitrage Engine.

Harvests diverging funding rate differentials between Binance USDT-M and Bybit Linear
perpetuals delta-neutrally without spot margin borrow fees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ArbitrageAction(str, Enum):
    ENTER_SHORT_BINANCE_LONG_BYBIT = "ENTER_SHORT_BINANCE_LONG_BYBIT"
    ENTER_SHORT_BYBIT_LONG_BINANCE = "ENTER_SHORT_BYBIT_LONG_BINANCE"
    EXIT_ARBITRAGE = "EXIT_ARBITRAGE"
    NO_ACTION = "NO_ACTION"


@dataclass
class PerpPerpSpread:
    """Snapshot of cross-exchange funding rate and mark price spread."""

    symbol: str
    timestamp: datetime
    binance_funding_8h: float
    bybit_funding_8h: float
    binance_funding_ann: float
    bybit_funding_ann: float
    spread_8h: float
    spread_ann: float
    binance_price: float
    bybit_price: float
    basis_bps: float
    action: ArbitrageAction = ArbitrageAction.NO_ACTION


@dataclass
class TransferEvent:
    """Record of cross-exchange collateral rebalance transfer."""

    timestamp: datetime
    from_exchange: str
    to_exchange: str
    amount: float
    fee: float
    remaining_from_collateral: float
    new_to_collateral: float


@dataclass
class CrossExchangeCollateralManager:
    """
    Manages dual-exchange collateral pools, computes margin ratios, and
    executes modeled rebalancing transfers when margin buffers erode.
    """

    binance_collateral: float
    bybit_collateral: float
    initial_total_collateral: float = field(init=False)
    min_margin_buffer: float = 0.25
    maintenance_margin_rate: float = 0.05
    transfer_fee: float = 1.0
    transfer_history: list[TransferEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.initial_total_collateral = self.binance_collateral + self.bybit_collateral

    @property
    def total_equity(self) -> float:
        return self.binance_collateral + self.bybit_collateral

    def check_margin_status(
        self,
        binance_notional: float,
        bybit_notional: float,
    ) -> tuple[float, float, bool]:
        """
        Calculates margin buffer on each exchange:
        buffer = (Collateral - MaintenanceMargin) / MaintenanceMargin
        Returns (binance_buffer, bybit_buffer, rebalance_needed).
        """
        binance_maint_margin = max(binance_notional * self.maintenance_margin_rate, 1.0)
        bybit_maint_margin = max(bybit_notional * self.maintenance_margin_rate, 1.0)

        binance_buffer = (self.binance_collateral - binance_maint_margin) / binance_maint_margin
        bybit_buffer = (self.bybit_collateral - bybit_maint_margin) / bybit_maint_margin

        rebalance_needed = binance_buffer < self.min_margin_buffer or bybit_buffer < self.min_margin_buffer
        return binance_buffer, bybit_buffer, rebalance_needed

    def rebalance_collateral(
        self,
        from_exchange: str,
        to_exchange: str,
        amount: float,
    ) -> TransferEvent:
        """Transfers collateral from surplus exchange to depleted exchange, deducting network fee."""
        if from_exchange.lower() == "binance":
            actual_amount = min(amount, self.binance_collateral * 0.6)
            self.binance_collateral -= actual_amount
            net_received = max(0.0, actual_amount - self.transfer_fee)
            self.bybit_collateral += net_received
        else:
            actual_amount = min(amount, self.bybit_collateral * 0.6)
            self.bybit_collateral -= actual_amount
            net_received = max(0.0, actual_amount - self.transfer_fee)
            self.binance_collateral += net_received

        event = TransferEvent(
            timestamp=datetime.now(timezone.utc),
            from_exchange=from_exchange,
            to_exchange=to_exchange,
            amount=actual_amount,
            fee=self.transfer_fee,
            remaining_from_collateral=(
                self.binance_collateral if from_exchange.lower() == "binance" else self.bybit_collateral
            ),
            new_to_collateral=(
                self.bybit_collateral if from_exchange.lower() == "binance" else self.binance_collateral
            ),
        )
        self.transfer_history.append(event)
        logger.info(
            f"Collateral rebalance: transferred ${actual_amount:,.2f} from {from_exchange} "
            f"to {to_exchange} (fee: ${self.transfer_fee:.2f})."
        )
        return event


class CrossExchangePerpArbitrageEngine:
    """
    Evaluates and executes delta-neutral funding rate spread arbitrage
    between Binance USDT-M Futures and Bybit Linear Perpetuals.
    """

    def __init__(
        self,
        binance_client: Any = None,
        bybit_client: Any = None,
        min_entry_spread_ann: float = 0.12,
        exit_spread_ann: float = 0.02,
        max_price_basis_bps: float = 25.0,
        taker_fee_pct: float = 0.0004,
    ) -> None:
        self.binance = binance_client
        self.bybit = bybit_client
        self.min_entry_spread_ann = min_entry_spread_ann
        self.exit_spread_ann = exit_spread_ann
        self.max_price_basis_bps = max_price_basis_bps
        self.taker_fee_pct = taker_fee_pct

    def compute_spread(
        self,
        symbol: str,
        binance_funding_8h: float,
        bybit_funding_8h: float,
        binance_price: float,
        bybit_price: float,
        timestamp: datetime | None = None,
        current_position: ArbitrageAction = ArbitrageAction.NO_ACTION,
    ) -> PerpPerpSpread:
        """
        Computes 8h and annualized cross-exchange funding spread and determines action.
        """
        ts = timestamp or datetime.now(timezone.utc)
        ann_mult = 3.0 * 365.0

        binance_ann = binance_funding_8h * ann_mult
        bybit_ann = bybit_funding_8h * ann_mult

        spread_8h = binance_funding_8h - bybit_funding_8h
        spread_ann = spread_8h * ann_mult

        mid_price = (binance_price + bybit_price) / 2.0
        basis_bps = (abs(binance_price - bybit_price) / mid_price) * 10_000.0 if mid_price > 0 else 0.0

        action = ArbitrageAction.NO_ACTION

        if current_position == ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT:
            if spread_ann < self.exit_spread_ann:
                action = ArbitrageAction.EXIT_ARBITRAGE
            else:
                action = ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT
        elif current_position == ArbitrageAction.ENTER_SHORT_BYBIT_LONG_BINANCE:
            if (-spread_ann) < self.exit_spread_ann:
                action = ArbitrageAction.EXIT_ARBITRAGE
            else:
                action = ArbitrageAction.ENTER_SHORT_BYBIT_LONG_BINANCE
        elif basis_bps <= self.max_price_basis_bps:
            if spread_ann >= self.min_entry_spread_ann:
                action = ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT
            elif (-spread_ann) >= self.min_entry_spread_ann:
                action = ArbitrageAction.ENTER_SHORT_BYBIT_LONG_BINANCE

        return PerpPerpSpread(
            symbol=symbol,
            timestamp=ts,
            binance_funding_8h=binance_funding_8h,
            bybit_funding_8h=bybit_funding_8h,
            binance_funding_ann=binance_ann,
            bybit_funding_ann=bybit_ann,
            spread_8h=spread_8h,
            spread_ann=spread_ann,
            binance_price=binance_price,
            bybit_price=bybit_price,
            basis_bps=basis_bps,
            action=action,
        )

    def fetch_market_spread(self, symbol: str) -> PerpPerpSpread:
        """
        Fetches live prices and funding rates via dual CCXT clients.
        """
        if not self.binance or not self.bybit:
            raise ValueError("Binance and Bybit CCXT clients must be configured.")

        binance_rate_info = self.binance.fetch_funding_rate(symbol)
        binance_ticker = self.binance.fetch_ticker(symbol)
        binance_funding = float(binance_rate_info.get("fundingRate") or 0.0)
        binance_price = float(binance_ticker.get("last") or binance_ticker.get("close") or 0.0)

        bybit_rate_info = self.bybit.fetch_funding_rate(symbol)
        bybit_ticker = self.bybit.fetch_ticker(symbol)
        bybit_funding = float(bybit_rate_info.get("fundingRate") or 0.0)
        bybit_price = float(bybit_ticker.get("last") or bybit_ticker.get("close") or 0.0)

        return self.compute_spread(
            symbol=symbol,
            binance_funding_8h=binance_funding,
            bybit_funding_8h=bybit_funding,
            binance_price=binance_price,
            bybit_price=bybit_price,
        )

    def simulate_arbitrage_series(
        self,
        spread_series: list[PerpPerpSpread],
        initial_capital: float = 100_000.0,
        leverage: float = 2.0,
    ) -> dict[str, Any]:
        """
        Simulates backtested PnL, cashflows, and collateral dynamics across historical funding periods.
        """
        collateral = CrossExchangeCollateralManager(
            binance_collateral=initial_capital / 2.0,
            bybit_collateral=initial_capital / 2.0,
        )

        total_notional_capacity = initial_capital * leverage
        per_leg_notional = total_notional_capacity / 2.0

        current_state = ArbitrageAction.NO_ACTION
        total_funding_earned = 0.0
        total_fees_paid = 0.0
        active_periods = 0

        pnl_history: list[float] = []

        for spread in spread_series:
            new_action = self.compute_spread(
                symbol=spread.symbol,
                binance_funding_8h=spread.binance_funding_8h,
                bybit_funding_8h=spread.bybit_funding_8h,
                binance_price=spread.binance_price,
                bybit_price=spread.bybit_price,
                timestamp=spread.timestamp,
                current_position=current_state,
            ).action

            if current_state == ArbitrageAction.NO_ACTION and new_action in (
                ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT,
                ArbitrageAction.ENTER_SHORT_BYBIT_LONG_BINANCE,
            ):
                entry_fees = per_leg_notional * self.taker_fee_pct * 2.0
                total_fees_paid += entry_fees
                collateral.binance_collateral -= entry_fees / 2.0
                collateral.bybit_collateral -= entry_fees / 2.0
                current_state = new_action

            if current_state != ArbitrageAction.NO_ACTION:
                active_periods += 1
                if current_state == ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT:
                    cf_binance = per_leg_notional * spread.binance_funding_8h
                    cf_bybit = -per_leg_notional * spread.bybit_funding_8h
                else:
                    cf_bybit = per_leg_notional * spread.bybit_funding_8h
                    cf_binance = -per_leg_notional * spread.binance_funding_8h

                net_period_funding = cf_binance + cf_bybit
                total_funding_earned += net_period_funding

                collateral.binance_collateral += cf_binance
                collateral.bybit_collateral += cf_bybit

                bn_buf, by_buf, rebal_needed = collateral.check_margin_status(per_leg_notional, per_leg_notional)
                if rebal_needed:
                    if (
                        bn_buf < collateral.min_margin_buffer
                        and collateral.bybit_collateral > collateral.binance_collateral
                    ):
                        diff = (collateral.bybit_collateral - collateral.binance_collateral) / 2.0
                        collateral.rebalance_collateral("bybit", "binance", diff)
                    elif (
                        by_buf < collateral.min_margin_buffer
                        and collateral.binance_collateral > collateral.bybit_collateral
                    ):
                        diff = (collateral.binance_collateral - collateral.bybit_collateral) / 2.0
                        collateral.rebalance_collateral("binance", "bybit", diff)

            if current_state != ArbitrageAction.NO_ACTION and new_action == ArbitrageAction.EXIT_ARBITRAGE:
                exit_fees = per_leg_notional * self.taker_fee_pct * 2.0
                total_fees_paid += exit_fees
                collateral.binance_collateral -= exit_fees / 2.0
                collateral.bybit_collateral -= exit_fees / 2.0
                current_state = ArbitrageAction.NO_ACTION

            pnl_history.append(collateral.total_equity)

        net_profit = collateral.total_equity - initial_capital
        total_rebalance_fees = sum(t.fee for t in collateral.transfer_history)

        return {
            "initial_capital": initial_capital,
            "final_equity": collateral.total_equity,
            "net_profit": net_profit,
            "return_pct": net_profit / max(initial_capital, 1.0),
            "total_funding_earned": total_funding_earned,
            "total_fees_paid": total_fees_paid,
            "total_rebalance_fees": total_rebalance_fees,
            "active_funding_periods": active_periods,
            "rebalances_count": len(collateral.transfer_history),
            "collateral_breakdown": {
                "binance": collateral.binance_collateral,
                "bybit": collateral.bybit_collateral,
            },
        }
