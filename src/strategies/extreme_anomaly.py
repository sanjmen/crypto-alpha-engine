"""
Extreme Anomaly & Illiquid Tail Alpha Strategies.
Addresses Issue: Extreme 24h Gainers & Losers (Pump & Dump Exploitation).

Provides 3 core statistical strategies with dedicated parameter presets
for both Liquid Core assets and the Illiquid Tail (6 strategies total):
1. Oversold Flush & Bounce Strategy (LONG on liquidation flushes / severe dumps).
2. Early Breakout Momentum Strategy (LONG on early volume/price breakout before UI ranking).
3. Pump & Funding Squeeze Fade Strategy (SHORT on overextended pumps with extreme funding rates).

Features configurable position sizing with a conservative test phase (e.g. $15 USDT for 48-96h)
and full production scaling (up to $100 USDT).
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.domain.entities import Portfolio, Signal
from src.ports.model_ports import IStrategy

# Canonical liquid assets vs illiquid tail classification
LIQUID_CORE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT"
]

ILLIQUID_TAIL_SYMBOLS = [
    "OGNUSDT", "RLCUSDT", "GTCUSDT", "BATUSDT", "MEMEUSDT", "CTSIUSDT",
    "1INCHUSDT", "MAGICUSDT", "FLOWUSDT", "SKLUSDT", "IOTAUSDT", "KSMUSDT",
    "CHZUSDT", "ZILUSDT", "COMPUSDT", "CRVUSDT"
]


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes relative strength index (RSI) on price series."""
    if len(series) < period + 1:
        return pd.Series(50.0, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Where avg_loss is 0, RSI is 100 (or 50 if both gain and loss are 0)
    rs = np.where(avg_loss == 0.0, np.where(avg_gain == 0.0, 1.0, 1000.0), avg_gain / (avg_loss + 1e-8))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=series.index).bfill().fillna(50.0)


class OversoldBounceStrategy(IStrategy):
    """
    Oversold Flush & Bounce Strategy (LONG).
    Harvests violent mean-reversion bounces after rapid liquidation flushes.
    """

    def __init__(
        self,
        dump_threshold_24h: float = -0.15,      # 24h price drop (e.g. -15% or -18%)
        rsi_threshold: float = 28.0,            # RSI oversold ceiling
        max_position_usd: float = 15.0,         # Test mode: $15 USDT (Prod: $100 USDT)
        max_portfolio_allocation: float = 0.15, # Max 15% of portfolio per asset
        symbols_filter: Optional[List[str]] = None,
        name: str = "OversoldBounce",
    ):
        self.dump_threshold_24h = dump_threshold_24h
        self.rsi_threshold = rsi_threshold
        self.max_position_usd = max_position_usd
        self.max_portfolio_allocation = max_portfolio_allocation
        self.symbols_filter = set(symbols_filter) if symbols_filter else None
        self.name = name

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if self.symbols_filter and sym not in self.symbols_filter:
                continue
            if len(df) < 26:
                continue

            closes = df["close"]
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))
            rsi_series = calculate_rsi(closes, period=14)
            rsi_curr = float(rsi_series.iloc[-1])

            # Flush condition: severe 24h drop and oversold RSI
            if ret_24h <= self.dump_threshold_24h and rsi_curr <= self.rsi_threshold:
                # Severity score between 0.5 and 1.0 based on dump depth
                score = float(np.clip(abs(ret_24h) / abs(self.dump_threshold_24h), 0.5, 1.0))
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=score,  # Positive indicates LONG
                        confidence=0.85,
                        horizon_minutes=1440,  # 24h holding period
                        metadata={
                            "ret_24h": ret_24h,
                            "rsi": rsi_curr,
                            "entry_price": float(closes.iloc[-1]),
                            "stop_loss": float(closes.iloc[-1] * 0.95),
                            "take_profit": float(closes.iloc[-1] * 1.06),
                        },
                    )
                )

        return signals

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        if not signals:
            return allocations

        total_equity = max(portfolio.total_equity, 100.0)
        cap_usd = min(self.max_position_usd, total_equity * self.max_portfolio_allocation)

        for sig in signals:
            allocations[sig.symbol] = round(cap_usd * sig.score, 2)

        return allocations


class EarlyBreakoutStrategy(IStrategy):
    """
    Early Breakout Momentum Strategy (LONG).
    Catches volume and price breakouts at the beginning of an expansion (+8% in 4h with 2.5x volume),
    before the asset hits the +30% or +40% retail FOMO screen.
    """

    def __init__(
        self,
        pump_threshold_4h: float = 0.08,        # +8% in 4 hours
        volume_multiplier: float = 2.5,         # Volume >= 2.5x 24h average
        max_24h_pump_ceiling: float = 0.22,     # Must NOT have already pumped > 22% (prevents late FOMO)
        max_position_usd: float = 15.0,         # Test mode: $15 USDT (Prod: $100 USDT)
        max_portfolio_allocation: float = 0.15,
        symbols_filter: Optional[List[str]] = None,
        name: str = "EarlyBreakout",
    ):
        self.pump_threshold_4h = pump_threshold_4h
        self.volume_multiplier = volume_multiplier
        self.max_24h_pump_ceiling = max_24h_pump_ceiling
        self.max_position_usd = max_position_usd
        self.max_portfolio_allocation = max_portfolio_allocation
        self.symbols_filter = set(symbols_filter) if symbols_filter else None
        self.name = name

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if self.symbols_filter and sym not in self.symbols_filter:
                continue
            if len(df) < 26:
                continue

            closes = df["close"]
            volumes = df["volume"]

            ret_4h = float((closes.iloc[-1] - closes.iloc[-5]) / (closes.iloc[-5] + 1e-8))
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))

            vol_curr = float(volumes.iloc[-1])
            vol_mean_24h = float(volumes.iloc[-25:-1].mean()) + 1e-6
            vol_ratio = vol_curr / vol_mean_24h

            # Early breakout condition
            if ret_4h >= self.pump_threshold_4h and vol_ratio >= self.volume_multiplier:
                if ret_24h <= self.max_24h_pump_ceiling:
                    signals.append(
                        Signal(
                            timestamp=now,
                            symbol=sym,
                            score=1.0,  # Positive indicates LONG
                            confidence=0.75,
                            horizon_minutes=720,  # 12h horizon
                            metadata={
                                "ret_4h": ret_4h,
                                "vol_ratio": vol_ratio,
                                "entry_price": float(closes.iloc[-1]),
                                "stop_loss": float(closes.iloc[-1] * 0.96),
                                "take_profit": float(closes.iloc[-1] * 1.15),
                            },
                        )
                    )

        return signals

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        if not signals:
            return allocations

        total_equity = max(portfolio.total_equity, 100.0)
        cap_usd = min(self.max_position_usd, total_equity * self.max_portfolio_allocation)

        for sig in signals:
            allocations[sig.symbol] = round(cap_usd, 2)

        return allocations


class PumpFundingFadeStrategy(IStrategy):
    """
    Pump & Funding Squeeze Fade Strategy (SHORT).
    Fades overextended blow-off tops where retail FOMO has driven the funding rate
    to extreme positive levels, profiting both from the inevitable price collapse
    and collecting the large funding payment.
    """

    def __init__(
        self,
        pump_threshold_24h: float = 0.25,       # +25% in 24h
        min_funding_rate: float = 0.0005,       # +0.05% per 8h (>= 54% annualized)
        rsi_ceiling: float = 72.0,              # Overbought RSI exhaustion
        max_position_usd: float = 15.0,         # Test mode: $15 USDT (Prod: $100 USDT)
        max_portfolio_allocation: float = 0.15,
        symbols_filter: Optional[List[str]] = None,
        name: str = "PumpFundingFade",
    ):
        self.pump_threshold_24h = pump_threshold_24h
        self.min_funding_rate = min_funding_rate
        self.rsi_ceiling = rsi_ceiling
        self.max_position_usd = max_position_usd
        self.max_portfolio_allocation = max_portfolio_allocation
        self.symbols_filter = set(symbols_filter) if symbols_filter else None
        self.name = name

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Signal]:
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if self.symbols_filter and sym not in self.symbols_filter:
                continue
            if len(df) < 26:
                continue

            closes = df["close"]
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))
            rsi_series = calculate_rsi(closes, period=14)
            rsi_curr = float(rsi_series.iloc[-1])

            # Extract latest funding rate
            f_rate = 0.0
            if funding_data and sym in funding_data and not funding_data[sym].empty:
                f_df = funding_data[sym]
                f_rate = float(f_df["funding_rate"].iloc[-1]) if "funding_rate" in f_df.columns else 0.0
            elif "funding_rate" in df.columns:
                f_rate = float(df["funding_rate"].iloc[-1])

            # Condition: violent pump + overbought RSI + extreme positive funding
            if ret_24h >= self.pump_threshold_24h and f_rate >= self.min_funding_rate and rsi_curr >= self.rsi_ceiling:
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=-1.0,  # Negative indicates SHORT
                        confidence=0.90,
                        horizon_minutes=1440,
                        metadata={
                            "ret_24h": ret_24h,
                            "funding_rate": f_rate,
                            "rsi": rsi_curr,
                            "entry_price": float(closes.iloc[-1]),
                            "stop_loss": float(closes.iloc[-1] * 1.08),
                            "take_profit": float(closes.iloc[-1] * 0.92),
                        },
                    )
                )

        return signals

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        if not signals:
            return allocations

        total_equity = max(portfolio.total_equity, 100.0)
        cap_usd = min(self.max_position_usd, total_equity * self.max_portfolio_allocation)

        for sig in signals:
            # Short target is negative dollar notional
            allocations[sig.symbol] = round(-cap_usd * abs(sig.score), 2)

        return allocations


# -----------------------------------------------------------------------------
# Factory Presets for the 6 Distinct Strategies
# -----------------------------------------------------------------------------

def create_liquid_oversold_bounce(test_mode: bool = True) -> OversoldBounceStrategy:
    """Strategy 1: Liquid Core Oversold Flush & Bounce (LONG on BTC/ETH/Major Alts)."""
    return OversoldBounceStrategy(
        dump_threshold_24h=-0.12,
        rsi_threshold=28.0,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=LIQUID_CORE_SYMBOLS,
        name="LiquidOversoldBounce",
    )


def create_illiquid_oversold_bounce(test_mode: bool = True) -> OversoldBounceStrategy:
    """Strategy 2: Illiquid Tail Oversold Flush & Bounce (LONG on micro-cap liquidations)."""
    return OversoldBounceStrategy(
        dump_threshold_24h=-0.18,
        rsi_threshold=25.0,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=ILLIQUID_TAIL_SYMBOLS,
        name="IlliquidOversoldBounce",
    )


def create_liquid_early_breakout(test_mode: bool = True) -> EarlyBreakoutStrategy:
    """Strategy 3: Liquid Core Early Breakout (LONG on volume momentum)."""
    return EarlyBreakoutStrategy(
        pump_threshold_4h=0.07,
        volume_multiplier=2.5,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=LIQUID_CORE_SYMBOLS,
        name="LiquidEarlyBreakout",
    )


def create_illiquid_early_breakout(test_mode: bool = True) -> EarlyBreakoutStrategy:
    """Strategy 4: Illiquid Tail Early Breakout (LONG on confirmed high-volume micro-caps)."""
    return EarlyBreakoutStrategy(
        pump_threshold_4h=0.10,
        volume_multiplier=3.5,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=ILLIQUID_TAIL_SYMBOLS,
        name="IlliquidEarlyBreakout",
    )


def create_liquid_funding_fade(test_mode: bool = True) -> PumpFundingFadeStrategy:
    """Strategy 5: Liquid Core Funding Fade (SHORT on overextended large-caps)."""
    return PumpFundingFadeStrategy(
        pump_threshold_24h=0.20,
        min_funding_rate=0.0004,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=LIQUID_CORE_SYMBOLS,
        name="LiquidFundingFade",
    )


def create_illiquid_funding_fade(test_mode: bool = True) -> PumpFundingFadeStrategy:
    """Strategy 6: Illiquid Tail Funding Fade (SHORT on micro-cap pump & dumps with funding spike)."""
    return PumpFundingFadeStrategy(
        pump_threshold_24h=0.28,
        min_funding_rate=0.0006,
        max_position_usd=15.0 if test_mode else 100.0,
        symbols_filter=ILLIQUID_TAIL_SYMBOLS,
        name="IlliquidFundingFade",
    )
