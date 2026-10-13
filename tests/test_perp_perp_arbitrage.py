"""
Unit tests for CrossExchangePerpArbitrageEngine, PerpPerpSpread computation,
Delta-neutral paired execution, and Collateral Rebalancing.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.strategies.perp_perp_arbitrage import (
    ArbitrageAction,
    CrossExchangeCollateralManager,
    CrossExchangePerpArbitrageEngine,
    PerpPerpSpread,
)


def test_perp_perp_spread_computation():
    """Verifies cross-exchange 8h and annualized funding spread and entry trigger."""
    engine = CrossExchangePerpArbitrageEngine(
        min_entry_spread_ann=0.15,  # 15% hurdle
        exit_spread_ann=0.03,
        max_price_basis_bps=20.0,
    )

    # Binance 0.03% (0.0003), Bybit 0.01% (0.0001) -> spread_8h = 0.0002
    # Annualized spread = 0.0002 * 3 * 365 = 0.219 (21.9% > 15%)
    spread = engine.compute_spread(
        symbol="BTC/USDT",
        binance_funding_8h=0.0003,
        bybit_funding_8h=0.0001,
        binance_price=60000.0,
        bybit_price=60010.0,  # ~1.6 bps diff
    )

    assert abs(spread.spread_8h - 0.0002) < 1e-7
    assert abs(spread.spread_ann - 0.219) < 1e-4
    assert spread.basis_bps < 5.0
    assert spread.action == ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT


def test_perp_perp_spread_negative_divergence():
    """Verifies trigger when Bybit funding rate is significantly higher than Binance."""
    engine = CrossExchangePerpArbitrageEngine(min_entry_spread_ann=0.15)

    # Bybit 0.04% (0.0004), Binance 0.01% (0.0001) -> spread_8h = -0.0003
    # Spread is -32.85% (Bybit higher by >15%)
    spread = engine.compute_spread(
        symbol="ETH/USDT",
        binance_funding_8h=0.0001,
        bybit_funding_8h=0.0004,
        binance_price=3000.0,
        bybit_price=3000.5,
    )

    assert spread.spread_ann < -0.30
    assert spread.action == ArbitrageAction.ENTER_SHORT_BYBIT_LONG_BINANCE


def test_perp_perp_spread_exit_threshold():
    """Verifies position exit when spread compresses below exit threshold."""
    engine = CrossExchangePerpArbitrageEngine(exit_spread_ann=0.03)

    # Currently in SHORT_BINANCE_LONG_BYBIT, spread drops to 0.00001 * 3 * 365 = 1.09% < 3%
    spread = engine.compute_spread(
        symbol="SOL/USDT",
        binance_funding_8h=0.00011,
        bybit_funding_8h=0.00010,  # 8h diff = 0.00001 (1.09% ann)
        binance_price=150.0,
        bybit_price=150.0,
        current_position=ArbitrageAction.ENTER_SHORT_BINANCE_LONG_BYBIT,
    )

    assert spread.action == ArbitrageAction.EXIT_ARBITRAGE


def test_collateral_manager_margin_check_and_rebalance():
    """Verifies margin buffer monitoring and cross-exchange transfer mechanics."""
    collateral = CrossExchangeCollateralManager(
        binance_collateral=50_000.0,
        bybit_collateral=50_000.0,
        min_margin_buffer=0.25,
        maintenance_margin_rate=0.05,
        transfer_fee=1.0,
    )

    # Initial healthy status: 50k collateral with 200k notional (MM = 10k, buffer = 4.0)
    bn_buf, by_buf, rebal_needed = collateral.check_margin_status(200_000.0, 200_000.0)
    assert not rebal_needed
    assert bn_buf >= 4.0
    assert by_buf >= 4.0

    # Simulate market move where Binance leg loses $42,000 equity (down to $8k, MM = 10k -> negative buffer)
    collateral.binance_collateral = 8_000.0
    collateral.bybit_collateral = 92_000.0  # Bybit leg gained the mirror amount

    bn_buf, by_buf, rebal_needed = collateral.check_margin_status(200_000.0, 200_000.0)
    assert rebal_needed
    assert bn_buf < collateral.min_margin_buffer

    # Execute rebalance transfer from Bybit to Binance of $30,000
    transfer = collateral.rebalance_collateral("bybit", "binance", 30_000.0)
    assert transfer.fee == 1.0
    assert transfer.amount == 30_000.0
    assert collateral.binance_collateral == 8_000.0 + (30_000.0 - 1.0)
    assert collateral.bybit_collateral == 92_000.0 - 30_000.0
    assert len(collateral.transfer_history) == 1


def test_cross_exchange_simulation():
    """Verifies end-to-end backtest simulation of delta-neutral funding arbitrage."""
    engine = CrossExchangePerpArbitrageEngine(
        min_entry_spread_ann=0.12,
        exit_spread_ann=0.02,
        taker_fee_pct=0.0004,
    )

    # 30 periods (10 days) of 0.03% Binance vs 0.01% Bybit funding
    series: list[PerpPerpSpread] = []
    base_time = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)

    for i in range(30):
        # Last period spread narrows to trigger exit
        byb_f = 0.01 if i < 28 else 0.03
        s = engine.compute_spread(
            symbol="BTC/USDT",
            binance_funding_8h=0.0003,
            bybit_funding_8h=byb_f if byb_f == 0.01 else 0.0003,
            binance_price=60000.0,
            bybit_price=60005.0,
            timestamp=base_time,
        )
        series.append(s)

    results = engine.simulate_arbitrage_series(series, initial_capital=100_000.0, leverage=2.0)

    assert results["total_funding_earned"] > 0.0
    assert results["total_fees_paid"] > 0.0
    # Funding earned should exceed the 4 taker fee executions
    assert results["net_profit"] > 0.0
    assert results["return_pct"] > 0.0
    assert results["active_funding_periods"] > 20


def test_fetch_market_spread_with_mock_ccxt():
    """Verifies live ingestion via mocked CCXT exchange clients."""
    mock_binance = MagicMock()
    mock_bybit = MagicMock()

    mock_binance.fetch_funding_rate.return_value = {"fundingRate": 0.00025}
    mock_binance.fetch_ticker.return_value = {"last": 65000.0}

    mock_bybit.fetch_funding_rate.return_value = {"fundingRate": 0.00010}
    mock_bybit.fetch_ticker.return_value = {"last": 65008.0}

    engine = CrossExchangePerpArbitrageEngine(
        binance_client=mock_binance,
        bybit_client=mock_bybit,
    )

    spread = engine.fetch_market_spread("BTC/USDT")
    assert spread.symbol == "BTC/USDT"
    assert spread.binance_funding_8h == 0.00025
    assert spread.bybit_funding_8h == 0.00010
    assert abs(spread.spread_8h - 0.00015) < 1e-7
    assert spread.binance_price == 65000.0
    assert spread.bybit_price == 65008.0
