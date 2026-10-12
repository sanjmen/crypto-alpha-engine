"""
Unit tests for non-linear market impact modeling, square-root law calibration,
Corwin-Schultz / Amihud dynamic conditioning, and capital capacity ceilings.
"""

import numpy as np
import pandas as pd
import pytest

from src.domain.enums import OrderSide, OrderType
from src.execution.database import ShadowDatabase
from src.execution.live_broker import CircuitBreakerError
from src.execution.shadow_broker import ShadowExecutionBroker
from src.features.market_impact import (
    CapitalCapacityCeiling,
    NonLinearMarketImpactModel,
)
from src.features.microstructure import (
    rolling_amihud_illiquidity,
    rolling_corwin_schultz_spread,
)


def test_square_root_scaling_property():
    """Validates the fundamental square root property: doubling size increases impact by sqrt(2)."""
    model = NonLinearMarketImpactModel()

    q1 = 100_000.0
    q2 = 200_000.0
    vol = 0.005
    volume = 50_000_000.0

    est1 = model.estimate_slippage(order_size_usd=q1, symbol="BTCUSDT", volatility=vol, volume=volume, cs_spread=0.0)
    est2 = model.estimate_slippage(order_size_usd=q2, symbol="BTCUSDT", volatility=vol, volume=volume, cs_spread=0.0)

    ratio = est2.impact_bps / est1.impact_bps
    assert np.isclose(ratio, np.sqrt(2.0), rtol=1e-3)


def test_corwin_schultz_and_amihud_conditioning():
    """Verifies dynamic conditioning on high-low Corwin-Schultz spread and Amihud illiquidity."""
    model = NonLinearMarketImpactModel()

    # Base test with 1 bp spread
    est_tight = model.estimate_slippage(
        order_size_usd=50_000.0,
        symbol="BTCUSDT",
        volatility=0.005,
        volume=50_000_000.0,
        cs_spread=0.0001,  # 1 bp
        amihud=0.0,
    )

    # Wide spread (10 bps)
    est_wide = model.estimate_slippage(
        order_size_usd=50_000.0,
        symbol="BTCUSDT",
        volatility=0.005,
        volume=50_000_000.0,
        cs_spread=0.0010,  # 10 bps
        amihud=0.0,
    )

    assert est_wide.half_spread_bps > est_tight.half_spread_bps
    assert est_wide.slippage_bps > est_tight.slippage_bps

    # Amihud illiquidity premium
    est_illiquid = model.estimate_slippage(
        order_size_usd=50_000.0,
        symbol="BTCUSDT",
        volatility=0.005,
        volume=50_000_000.0,
        cs_spread=0.0001,
        amihud=1e-8,
    )
    assert est_illiquid.amihud_bps > 0.0
    assert est_illiquid.slippage_bps > est_tight.slippage_bps


def test_capital_capacity_ceiling_and_slicing():
    """Ensures capital capacity ceiling rejects oversized single trades and recommends TWAP slices."""
    ceiling = CapitalCapacityCeiling(max_depth_fraction=0.05, max_volume_fraction=0.02)

    # 1% depth is $1,000,000; 5% ceiling is $50,000
    report_ok = ceiling.evaluate_capacity(
        symbol="ETHUSDT",
        order_size_usd=30_000.0,
        depth_1pct_usd=1_000_000.0,
        bar_volume_usd=5_000_000.0,
    )
    assert report_ok.within_capacity
    assert report_ok.recommended_chunks == 1
    assert report_ok.chunk_size_usd == 30_000.0

    # Order of $150,000 exceeds $50,000 ceiling -> 3 chunks
    report_breach = ceiling.evaluate_capacity(
        symbol="ETHUSDT",
        order_size_usd=150_000.0,
        depth_1pct_usd=1_000_000.0,
        bar_volume_usd=5_000_000.0,
    )
    assert not report_breach.within_capacity
    assert report_breach.recommended_chunks == 3
    assert np.isclose(report_breach.chunk_size_usd, 50_000.0)

    # Slicing helper
    slices = ceiling.slice_twap_orders(order_size_usd=150_000.0, max_chunk_usd=50_000.0)
    assert len(slices) == 3
    assert np.isclose(sum(slices), 150_000.0)


def test_calibrate_from_book_depth_data():
    """Tests parameter calibration from Binance L2 bookDepth snapshots."""
    # Synthetic bookDepth DataFrame simulating 1% to 5% depth
    records = []
    base_notionals = {1: 50_000_000.0, 2: 120_000_000.0, 3: 180_000_000.0, 4: 230_000_000.0, 5: 270_000_000.0}
    for pct, notional in base_notionals.items():
        records.append({"percentage": pct, "depth": notional / 60_000.0, "notional": notional})
        records.append({"percentage": -pct, "depth": notional / 60_000.0, "notional": notional})

    df = pd.DataFrame(records)
    model = NonLinearMarketImpactModel()

    daily_vol = 0.03
    daily_vol_usd = 2_500_000_000.0
    calibrated_params = model.calibrate_from_book_depth(
        book_depth_df=df,
        reference_volatility=daily_vol,
        reference_volume=daily_vol_usd,
        fit_exponent=False,
    )

    assert calibrated_params.Y_coefficient > 0.0
    assert calibrated_params.alpha_exponent == 0.50


def test_flash_crash_liquidity_shock_expansion():
    """Validates that during flash crash conditions, non-linear slippage expands dramatically."""
    model = NonLinearMarketImpactModel()

    order_size = 500_000.0
    symbol = "SOLUSDT"

    # Normal regime: 0.5% hourly vol, $20M volume, 1 bp spread
    est_normal = model.estimate_slippage(
        order_size_usd=order_size,
        symbol=symbol,
        volatility=0.005,
        volume=20_000_000.0,
        cs_spread=0.0001,
        is_flash_crash=False,
    )

    # Flash crash regime: 4.0% hourly vol, $5M volume (evaporated book), 20 bp spread
    est_crash = model.estimate_slippage(
        order_size_usd=order_size,
        symbol=symbol,
        volatility=0.040,
        volume=5_000_000.0,
        cs_spread=0.0020,
        is_flash_crash=True,
    )

    assert est_crash.is_flash_crash
    assert est_crash.slippage_bps > est_normal.slippage_bps * 5.0
    assert est_crash.stress_penalty_bps > 0.0


def test_shadow_broker_dynamic_impact_execution(tmp_path):
    """Verifies ShadowExecutionBroker executes orders using dynamic impact and enforces capacity ceilings."""
    db_path = str(tmp_path / "test_shadow.db")
    db = ShadowDatabase(db_path=db_path)

    impact_model = NonLinearMarketImpactModel()
    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=200_000.0,
        impact_model=impact_model,
        enforce_capacity_ceiling=True,
        max_order_notional=100_000.0,
    )

    # Configure microstructure for BTCUSDT: $50M bar volume, $2M 1% depth
    broker.set_market_price("BTCUSDT", price=50_000.0)
    broker.set_symbol_microstructure(
        symbol="BTCUSDT",
        volatility=0.008,
        volume=50_000_000.0,
        depth_1pct_usd=2_000_000.0,  # 5% depth ceiling is $100,000
        cs_spread=0.0002,             # 2 bps spread
    )

    # Small order: 0.5 BTC = $25,000 (well within capacity)
    order_small = broker.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.5,
    )
    assert order_small.average_fill_price > 50_000.0

    # Oversized order: 3 BTC = $150,000 (exceeds max_order_notional & capacity ceiling)
    with pytest.raises(CircuitBreakerError):
        broker.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=3.0,
        )


def test_rolling_microstructure_estimators():
    """Tests rolling Corwin-Schultz and rolling Amihud helper estimators."""
    np.random.seed(42)
    n = 60
    highs = 100.0 + np.random.uniform(0.5, 2.0, n)
    lows = 100.0 - np.random.uniform(0.5, 2.0, n)
    returns = np.random.normal(0, 0.01, n)
    volumes = np.random.uniform(1_000_000, 5_000_000, n)

    roll_cs = rolling_corwin_schultz_spread(highs, lows, window=10)
    assert len(roll_cs) == n
    assert np.all(roll_cs >= 0.0)

    roll_amihud = rolling_amihud_illiquidity(returns, volumes, window=10)
    assert len(roll_amihud) == n
    assert np.all(roll_amihud >= 0.0)
