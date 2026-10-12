"""
Non-Linear Market Impact & Microstructure Slippage Engine.

Implements:
1. Institutional Square-Root Market Impact Law:
   ΔP / P = Y * σ * sqrt(Q / V)  (Almgren-Chriss, Bouchaud, Kyle)
2. Calibration from Binance Vision L2 bookDepth snapshots and tick aggregates.
3. Liquidity-Aware Dynamic Slippage conditioned on Corwin-Schultz spread & Amihud illiquidity.
4. Capital Capacity Ceiling per symbol (enforcing <= 2-5% top-of-book depth per trade).
5. TWAP/VWAP order slicing recommendations for capacity breaches.
6. Flash crash stress regimes and liquidity shock modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.entities import Order

logger = logging.getLogger(__name__)


@dataclass
class MarketImpactParameters:
    """Institutional parameter set for non-linear market impact modeling."""

    Y_coefficient: float = 0.60
    """Dimensionless impact coefficient Y (typically 0.50 - 0.75 in crypto perpetuals)."""

    alpha_exponent: float = 0.50
    """Square-root power law exponent (alpha = 0.50)."""

    min_slippage_bps: float = 0.50
    """Minimum half-spread floor in basis points (0.005% minimum tick friction)."""

    corwin_schultz_weight: float = 0.50
    """Weight applied to Corwin-Schultz spread (0.50 represents half-spread)."""

    amihud_lambda: float = 0.05
    """Multiplier for Amihud illiquidity impact adjustment."""

    max_depth_fraction: float = 0.05
    """Maximum fraction of top-of-book (1%) depth allowed per single trade (5%)."""

    max_volume_fraction: float = 0.02
    """Maximum fraction of bar volume allowed per single trade (2%)."""

    stress_vol_multiplier: float = 3.0
    """Volatility ratio above baseline that triggers flash crash stress mode."""

    stress_impact_multiplier: float = 2.5
    """Slippage expansion multiplier applied during flash crash / liquidity shocks."""


@dataclass
class CapacityReport:
    """Report on order size capacity utilization against market depth."""

    symbol: str
    order_size_usd: float
    max_allowed_size_usd: float
    depth_1pct_usd: float
    bar_volume_usd: float
    capacity_utilized_pct: float
    within_capacity: bool
    recommended_chunks: int
    chunk_size_usd: float
    reason: str


@dataclass
class SlippageEstimate:
    """Comprehensive decomposition of dynamic execution slippage."""

    symbol: str
    order_size_usd: float
    side: str
    mark_price: float
    fill_price: float
    slippage_pct: float
    slippage_bps: float
    half_spread_bps: float
    impact_bps: float
    amihud_bps: float
    stress_penalty_bps: float
    total_slippage_usd: float
    capacity_report: CapacityReport
    is_flash_crash: bool = False
    metadata: dict = field(default_factory=dict)


class CapitalCapacityCeiling:
    """
    Computes institutional capital capacity ceilings per symbol.
    Guarantees individual trades do not consume more than 2-5% of top-of-book depth,
    preventing market manipulation and catastrophic price slippage.
    """

    def __init__(
        self,
        max_depth_fraction: float = 0.05,
        max_volume_fraction: float = 0.02,
        default_depth_to_volume_ratio: float = 0.08,
    ):
        self.max_depth_fraction = max_depth_fraction
        self.max_volume_fraction = max_volume_fraction
        self.default_depth_to_volume_ratio = default_depth_to_volume_ratio

    def estimate_depth_from_volume(self, bar_volume_usd: float) -> float:
        """
        Estimates 1% top-of-book order book depth from bar volume when L2 depth is unavailable.
        Empirically in BTC/ETH perpetuals, 1% depth is ~5-15% of 1-hour bar volume.
        """
        return max(0.0, bar_volume_usd * self.default_depth_to_volume_ratio)

    def compute_max_order_size(
        self,
        symbol: str,
        depth_1pct_usd: float | None = None,
        bar_volume_usd: float | None = None,
    ) -> float:
        """
        Calculates maximum allowed single-order dollar notional.
        Takes minimum of depth-based ceiling and volume-based ceiling.
        """
        limits = []

        if depth_1pct_usd is not None and depth_1pct_usd > 0:
            limits.append(depth_1pct_usd * self.max_depth_fraction)
        elif bar_volume_usd is not None and bar_volume_usd > 0:
            est_depth = self.estimate_depth_from_volume(bar_volume_usd)
            limits.append(est_depth * self.max_depth_fraction)

        if bar_volume_usd is not None and bar_volume_usd > 0:
            limits.append(bar_volume_usd * self.max_volume_fraction)

        if not limits:
            # Fallback conservative ceiling if no market data is provided
            return 100_000.0

        return float(min(limits))

    def evaluate_capacity(
        self,
        symbol: str,
        order_size_usd: float,
        depth_1pct_usd: float | None = None,
        bar_volume_usd: float | None = None,
    ) -> CapacityReport:
        """Evaluates whether an order conforms to the capital capacity ceiling."""
        order_notional = float(abs(order_size_usd))
        bar_vol = float(bar_volume_usd) if bar_volume_usd is not None and bar_volume_usd > 0 else 0.0

        if depth_1pct_usd is not None and depth_1pct_usd > 0:
            effective_depth = float(depth_1pct_usd)
        elif bar_vol > 0:
            effective_depth = self.estimate_depth_from_volume(bar_vol)
        else:
            effective_depth = 1_000_000.0  # Default nominal depth

        max_allowed = self.compute_max_order_size(
            symbol,
            depth_1pct_usd=effective_depth,
            bar_volume_usd=bar_vol if bar_vol > 0 else None,
        )

        within_capacity = order_notional <= max_allowed
        capacity_utilization = (order_notional / max_allowed * 100.0) if max_allowed > 0 else 100.0

        if within_capacity:
            recommended_chunks = 1
            chunk_size = order_notional
            reason = f"Order ${order_notional:,.2f} is within ceiling (${max_allowed:,.2f}, {capacity_utilization:.1f}% util)"
        else:
            recommended_chunks = int(np.ceil(order_notional / max_allowed))
            chunk_size = order_notional / recommended_chunks
            reason = (
                f"Order ${order_notional:,.2f} exceeds capacity ceiling (${max_allowed:,.2f}). "
                f"Requires {recommended_chunks} TWAP/VWAP slices of ~${chunk_size:,.2f}."
            )

        return CapacityReport(
            symbol=symbol,
            order_size_usd=order_notional,
            max_allowed_size_usd=max_allowed,
            depth_1pct_usd=effective_depth,
            bar_volume_usd=bar_vol,
            capacity_utilized_pct=capacity_utilization,
            within_capacity=within_capacity,
            recommended_chunks=recommended_chunks,
            chunk_size_usd=chunk_size,
            reason=reason,
        )

    def slice_twap_orders(
        self,
        order_size_usd: float,
        max_chunk_usd: float,
    ) -> list[float]:
        """Slices an oversized order into equal execution chunks."""
        total = float(abs(order_size_usd))
        if total <= max_chunk_usd or max_chunk_usd <= 0:
            return [total]

        num_chunks = int(np.ceil(total / max_chunk_usd))
        chunk_val = total / num_chunks
        return [chunk_val] * num_chunks


class NonLinearMarketImpactModel:
    """
    Production Non-Linear Market Impact & Microstructure Slippage Model.
    Calculates dynamic slippage combining:
      1. Corwin-Schultz effective half-spread
      2. Institutional Square-Root Law of Market Impact (ΔP / P = Y * σ * sqrt(Q / V))
      3. Amihud illiquidity ratio adjustments
      4. Capital capacity ceiling enforcement
      5. Flash crash / liquidity shock volatility expansion
    """

    def __init__(
        self,
        params: MarketImpactParameters | None = None,
        capacity_ceiling: CapitalCapacityCeiling | None = None,
    ):
        self.params = params or MarketImpactParameters()
        self.capacity_ceiling = capacity_ceiling or CapitalCapacityCeiling(
            max_depth_fraction=self.params.max_depth_fraction,
            max_volume_fraction=self.params.max_volume_fraction,
        )
        self.calibrated_symbols: dict[str, MarketImpactParameters] = {}

    def calibrate_from_book_depth(
        self,
        book_depth_df: pd.DataFrame,
        reference_volatility: float,
        reference_volume: float,
        fit_exponent: bool = False,
    ) -> MarketImpactParameters:
        """
        Calibrates square-root law impact coefficient Y (and optionally power alpha)
        directly from Binance Data Vision L2 bookDepth snapshots.

        book_depth_df contains columns:
          - 'percentage': price distance from mid (e.g. 1, 2, 3, 4, 5 for asks, -1..-5 for bids)
          - 'notional': cumulative USD depth at that percentage
        """
        df = book_depth_df.copy()
        if "percentage" not in df.columns or "notional" not in df.columns:
            raise ValueError("book_depth_df must contain 'percentage' and 'notional' columns")

        # Focus on positive side (or aggregate absolute percentages)
        df["abs_pct"] = df["percentage"].abs()
        df = df[df["abs_pct"] > 0]

        mean_notionals = df.groupby("abs_pct")["notional"].mean()
        if len(mean_notionals) < 2:
            return self.params

        pct_deltas = mean_notionals.index.values / 100.0  # [0.01, 0.02, ...]
        cum_notionals = mean_notionals.values

        vol = max(1e-5, float(reference_volatility))
        vol_ref = max(1.0, float(reference_volume))

        if fit_exponent:
            # OLS log(delta) = log(Y * vol) + alpha * log(Q / V)
            log_q_v = np.log(cum_notionals / vol_ref)
            log_deltas = np.log(pct_deltas)
            poly = np.polyfit(log_q_v, log_deltas, 1)
            alpha_fitted = float(np.clip(poly[0], 0.30, 0.80))
            intercept = poly[1]
            Y_fitted = float(np.clip(np.exp(intercept) / vol, 0.10, 2.0))
        else:
            alpha_fitted = 0.50
            # Solve Y = delta / (vol * sqrt(Q / V))
            q_v_sqrt = np.sqrt(cum_notionals / vol_ref)
            y_levels = pct_deltas / (vol * q_v_sqrt)
            # Take median across depth levels (1% to 5%)
            Y_fitted = float(np.clip(np.median(y_levels), 0.10, 2.0))

        calibrated = MarketImpactParameters(
            Y_coefficient=Y_fitted,
            alpha_exponent=alpha_fitted,
            min_slippage_bps=self.params.min_slippage_bps,
            corwin_schultz_weight=self.params.corwin_schultz_weight,
            amihud_lambda=self.params.amihud_lambda,
            max_depth_fraction=self.params.max_depth_fraction,
            max_volume_fraction=self.params.max_volume_fraction,
            stress_vol_multiplier=self.params.stress_vol_multiplier,
            stress_impact_multiplier=self.params.stress_impact_multiplier,
        )
        return calibrated

    def calibrate_from_agg_trades(
        self,
        trades_df: pd.DataFrame,
        reference_volatility: float,
        window_trades: int = 100,
    ) -> MarketImpactParameters:
        """
        Calibrates impact coefficient Y from tick-level aggTrades clusters.
        Measures contemporaneous price displacement per signed dollar flow.
        """
        if "notional" not in trades_df.columns or "price" not in trades_df.columns:
            return self.params

        df = trades_df.copy()
        if "tick_sign" not in df.columns:
            if "is_buyer_maker" in df.columns:
                df["tick_sign"] = np.where(df["is_buyer_maker"], -1, 1)
            else:
                df["tick_sign"] = 1

        df["signed_notional"] = df["notional"] * df["tick_sign"]
        chunks = len(df) // window_trades
        if chunks < 3:
            return self.params

        y_estimates = []
        vol = max(1e-5, float(reference_volatility))

        for i in range(chunks):
            sub = df.iloc[i * window_trades : (i + 1) * window_trades]
            flow = sub["signed_notional"].sum()
            tot_vol = sub["notional"].sum()
            if tot_vol <= 0:
                continue

            p_start = sub["price"].iloc[0]
            p_end = sub["price"].iloc[-1]
            ret = (p_end - p_start) / p_start

            # Check if sign matches flow
            if np.sign(ret) == np.sign(flow) and abs(flow) > 0:
                q_ratio = abs(flow) / tot_vol
                y_sample = abs(ret) / (vol * np.sqrt(q_ratio))
                if 0.05 <= y_sample <= 5.0:
                    y_estimates.append(y_sample)

        if not y_estimates:
            return self.params

        y_calibrated = float(np.clip(np.median(y_estimates), 0.20, 1.50))
        return MarketImpactParameters(
            Y_coefficient=y_calibrated,
            alpha_exponent=0.50,
            min_slippage_bps=self.params.min_slippage_bps,
            corwin_schultz_weight=self.params.corwin_schultz_weight,
            amihud_lambda=self.params.amihud_lambda,
        )

    def estimate_slippage(
        self,
        order_size_usd: float,
        symbol: str,
        volatility: float,
        volume: float,
        mark_price: float = 100.0,
        side: str = "BUY",
        depth_1pct_usd: float | None = None,
        cs_spread: float | None = None,
        amihud: float | None = None,
        is_flash_crash: bool = False,
    ) -> SlippageEstimate:
        """
        Computes realistic dynamic non-linear slippage for a market/taker order.

        Parameters
        ----------
        order_size_usd : float
            Dollar notional of the trade (price * quantity).
        symbol : str
            Trading pair identifier (e.g. 'BTCUSDT').
        volatility : float
            Asset return volatility over reference window (e.g. 1h bar volatility: 0.005 = 0.5%).
        volume : float
            Total traded volume in USD over the reference window.
        mark_price : float
            Current mid / mark price.
        side : str
            "BUY" or "SELL".
        depth_1pct_usd : Optional[float]
            Available L2 depth in USD within 1% of mid price.
        cs_spread : Optional[float]
            Corwin-Schultz high-low bid-ask spread estimate (in decimal, e.g. 0.0002 = 2 bps).
        amihud : Optional[float]
            Amihud illiquidity ratio (|r| / volume).
        is_flash_crash : bool
            Explicit flag or detector trigger for liquidity stress / flash crashes.
        """
        p = self.calibrated_symbols.get(symbol, self.params)
        q_usd = float(abs(order_size_usd))
        v_usd = max(100.0, float(volume))
        vol = max(1e-5, float(volatility))

        # 1. Capacity Ceiling Check
        cap_report = self.capacity_ceiling.evaluate_capacity(
            symbol=symbol,
            order_size_usd=q_usd,
            depth_1pct_usd=depth_1pct_usd,
            bar_volume_usd=v_usd,
        )

        # 2. Corwin-Schultz Half-Spread Component
        if cs_spread is not None and cs_spread > 0:
            effective_cs_spread = float(cs_spread)
        else:
            # Baseline spread: 1.0 bp
            effective_cs_spread = 0.0001

        half_spread_pct = p.corwin_schultz_weight * effective_cs_spread
        half_spread_bps = half_spread_pct * 10_000.0

        # 3. Square-Root Market Impact Component: I(Q) = Y * sigma * (Q / V)^alpha
        participation_rate = np.clip(q_usd / v_usd, 1e-7, 1.0)
        impact_pct = p.Y_coefficient * vol * (participation_rate**p.alpha_exponent)
        impact_bps = impact_pct * 10_000.0

        # 4. Amihud Illiquidity Term
        if amihud is not None and amihud > 0:
            # Amihud term scaled by notional
            amihud_pct = min(0.005, p.amihud_lambda * float(amihud) * q_usd)
            amihud_bps = amihud_pct * 10_000.0
        else:
            amihud_pct = 0.0
            amihud_bps = 0.0

        # 5. Flash Crash / Liquidity Shock Stress Regimes
        stress_penalty_bps = 0.0
        # Automatically detect flash crash if volatility spikes > stress threshold
        auto_stress = is_flash_crash or (vol > 0.025)  # e.g. >2.5% bar volatility is extreme shock
        if auto_stress:
            # Liquidity evaporates: expand impact and spread
            expansion = p.stress_impact_multiplier
            raw_total_bps = (half_spread_bps + impact_bps + amihud_bps)
            stress_total_bps = raw_total_bps * expansion
            stress_penalty_bps = stress_total_bps - raw_total_bps
        else:
            stress_total_bps = half_spread_bps + impact_bps + amihud_bps

        # Enforce minimum tick friction floor
        total_slippage_bps = max(p.min_slippage_bps, stress_total_bps)
        total_slippage_pct = total_slippage_bps / 10_000.0

        # Calculate fill price
        is_buy_order = side.upper() in ["BUY", "LONG"]
        if is_buy_order:
            fill_price = mark_price * (1.0 + total_slippage_pct)
        else:
            fill_price = mark_price * (1.0 - total_slippage_pct)

        total_cost_usd = q_usd * total_slippage_pct

        return SlippageEstimate(
            symbol=symbol,
            order_size_usd=q_usd,
            side=side.upper(),
            mark_price=mark_price,
            fill_price=fill_price,
            slippage_pct=total_slippage_pct,
            slippage_bps=total_slippage_bps,
            half_spread_bps=half_spread_bps,
            impact_bps=impact_bps,
            amihud_bps=amihud_bps,
            stress_penalty_bps=stress_penalty_bps,
            total_slippage_usd=total_cost_usd,
            capacity_report=cap_report,
            is_flash_crash=auto_stress,
            metadata={
                "participation_rate": participation_rate,
                "volatility": vol,
                "volume": v_usd,
                "Y": p.Y_coefficient,
                "alpha": p.alpha_exponent,
            },
        )

    def apply_execution_slippage(
        self,
        order: Order,
        mark_price: float,
        volatility: float,
        volume: float,
        depth_1pct_usd: float | None = None,
        cs_spread: float | None = None,
        amihud: float | None = None,
        is_flash_crash: bool = False,
    ) -> tuple[float, SlippageEstimate]:
        """
        Calculates execution fill price for an Order entity with dynamic slippage.
        Returns (fill_price, SlippageEstimate).
        """
        order_notional = float(mark_price * order.quantity)
        side_str = order.side.name if hasattr(order.side, "name") else str(order.side)

        estimate = self.estimate_slippage(
            order_size_usd=order_notional,
            symbol=order.symbol,
            volatility=volatility,
            volume=volume,
            mark_price=mark_price,
            side=side_str,
            depth_1pct_usd=depth_1pct_usd,
            cs_spread=cs_spread,
            amihud=amihud,
            is_flash_crash=is_flash_crash,
        )

        return estimate.fill_price, estimate


def compute_order_slippage(
    order_size_usd: float,
    symbol: str = "BTCUSDT",
    volatility: float = 0.005,
    volume: float = 50_000_000.0,
    mark_price: float = 100.0,
    side: str = "BUY",
    depth_1pct_usd: float | None = None,
    cs_spread: float | None = None,
    amihud: float | None = None,
    is_flash_crash: bool = False,
) -> SlippageEstimate:
    """Convenience function for calculating dynamic market impact slippage."""
    model = NonLinearMarketImpactModel()
    return model.estimate_slippage(
        order_size_usd=order_size_usd,
        symbol=symbol,
        volatility=volatility,
        volume=volume,
        mark_price=mark_price,
        side=side,
        depth_1pct_usd=depth_1pct_usd,
        cs_spread=cs_spread,
        amihud=amihud,
        is_flash_crash=is_flash_crash,
    )
