"""
Extreme Anomaly & Illiquid Tail Alpha Strategies.
Addresses Extreme 24h Gainers & Losers (Pump & Dump Exploitation).

Provides statistical anomaly strategies with dedicated parameter presets
for both Liquid Core assets and the Illiquid Tail:
1. Oversold Flush & Bounce Strategy (LONG on liquidation flushes / severe dumps).
2. Early Breakout Momentum Strategy (LONG on early volume/price breakout before UI ranking).
3. Pump & Funding Squeeze Fade Strategy (SHORT on overextended pumps with extreme funding rates).
4. Extreme Funding Rate Harvester Strategy (LONG/SHORT on extreme perpetual funding rates).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.entities import Portfolio, Signal
from src.ports.model_ports import IStrategy

LIQUID_CORE_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "APTUSDT",
]

ILLIQUID_TAIL_SYMBOLS = [
    "OGNUSDT",
    "RLCUSDT",
    "GTCUSDT",
    "BATUSDT",
    "MEMEUSDT",
    "CTSIUSDT",
    "1INCHUSDT",
    "MAGICUSDT",
    "FLOWUSDT",
    "SKLUSDT",
    "IOTAUSDT",
    "KSMUSDT",
    "CHZUSDT",
    "ZILUSDT",
    "COMPUSDT",
    "CRVUSDT",
]

FUNDING_COLUMN_CANDIDATES = ("rate", "funding_rate", "fundingRate", "lastFundingRate", "last_funding_rate")


def _extract_latest_funding_rate(funding_df: pd.DataFrame) -> float:
    """Safely extracts the latest numeric funding rate from a funding DataFrame."""
    for col in FUNDING_COLUMN_CANDIDATES:
        if col in funding_df.columns and not funding_df[col].empty:
            val = pd.to_numeric(funding_df[col].iloc[-1], errors="coerce")
            if pd.notna(val) and np.isfinite(float(val)):
                return float(val)
    return 0.0


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes strictly causal relative strength index (RSI) on price series."""
    if len(series) < period + 1:
        return pd.Series(50.0, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = np.where(avg_loss == 0.0, np.where(avg_gain == 0.0, 1.0, 1000.0), avg_gain / (avg_loss + 1e-8))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=series.index).ffill().fillna(50.0)


def compute_anomaly_confluence_size(
    strategy_type: str,
    metadata: dict[str, float],
    mkt_disp_24h: float = 0.030,
    min_notional: float = 10.0,
    max_notional: float = 25.0,
) -> tuple[float, float]:
    """
    Computes the multi-factor confluence score in [0.0, 1.0] and maps it linearly
    to a dynamic position size in `[min_notional, max_notional]`.
    Returns `(recommended_notional, confluence_score)`.
    """
    ret_24h = float(metadata.get("ret_24h", 0.0))
    ret_4h = float(metadata.get("ret_4h", 0.0))
    rsi = float(metadata.get("rsi", 50.0))
    vol_ratio = float(metadata.get("vol_ratio", 1.5))
    funding_rate = float(metadata.get("funding_rate", 0.0))
    dispersion_score = float(np.clip((mkt_disp_24h - 0.020) / 0.030, 0.0, 1.0))

    strategy_key = strategy_type.lower()
    if "bounce" in strategy_key or "drop" in strategy_key or "oversold" in strategy_key:
        depth_score = float(np.clip((abs(ret_24h) - 0.18) / 0.08, 0.0, 1.0))
        rsi_score = float(np.clip((25.0 - rsi) / 12.0, 0.0, 1.0))
        vol_score = float(np.clip((vol_ratio - 1.2) / 3.0, 0.0, 1.0))
        confluence = float(
            np.clip(0.35 * depth_score + 0.30 * rsi_score + 0.20 * vol_score + 0.15 * dispersion_score, 0.0, 1.0)
        )
    elif "breakout" in strategy_key:
        impulse_score = float(np.clip((ret_4h - 0.10) / 0.06, 0.0, 1.0))
        vol_score = float(np.clip((vol_ratio - 3.5) / 4.5, 0.0, 1.0))
        sweetspot_score = float(1.0 - np.clip(abs(ret_24h - 0.16) / 0.06, 0.0, 1.0))
        confluence = float(
            np.clip(
                0.35 * impulse_score + 0.30 * vol_score + 0.20 * sweetspot_score + 0.15 * dispersion_score,
                0.0,
                1.0,
            )
        )
    elif "harvester" in strategy_key:
        abs_funding = abs(funding_rate)
        funding_score = float(np.clip((abs_funding - 0.0012) / 0.0035, 0.0, 1.0))
        vol_score = float(np.clip((vol_ratio - 1.0) / 3.0, 0.0, 1.0))
        if funding_rate < 0:
            momentum_score = float(np.clip((ret_4h + 0.03) / 0.08, 0.0, 1.0))
        else:
            momentum_score = float(np.clip((rsi - 58.0) / 22.0, 0.0, 1.0))
        confluence = float(
            np.clip(
                0.50 * funding_score + 0.20 * vol_score + 0.15 * momentum_score + 0.15 * dispersion_score,
                0.0,
                1.0,
            )
        )
    else:
        pump_score = float(np.clip((ret_24h - 0.25) / 0.20, 0.0, 1.0))
        funding_score = float(np.clip((funding_rate - 0.0005) / 0.0015, 0.0, 1.0))
        rsi_score = float(np.clip((rsi - 72.0) / 15.0, 0.0, 1.0))
        confluence = float(
            np.clip(0.35 * pump_score + 0.35 * funding_score + 0.15 * rsi_score + 0.15 * dispersion_score, 0.0, 1.0)
        )

    span = max(0.0, float(max_notional) - float(min_notional))
    notional = float(np.round(float(min_notional) + span * confluence, 2))
    return notional, round(confluence, 4)


class BaseAnomalyStrategy(IStrategy):
    """
    Shared base class for event-driven anomaly strategies.
    Encapsulates universe filtering and capped dollar-weight allocation.
    """

    def __init__(
        self,
        max_position_usd: float = 15.0,
        max_portfolio_allocation: float = 0.15,
        symbols_filter: list[str] | None = None,
        name: str = "BaseAnomaly",
    ) -> None:
        self.max_position_usd = max_position_usd
        self.max_portfolio_allocation = max_portfolio_allocation
        self.symbols_filter = set(symbols_filter) if symbols_filter else None
        self.name = name

    def _passes_universe_filter(self, symbol: str, df: pd.DataFrame, min_bars: int = 26) -> bool:
        if self.symbols_filter and symbol not in self.symbols_filter:
            return False
        return len(df) >= min_bars

    def allocate_weights(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        allocations: dict[str, float] = {}
        if not signals:
            return allocations

        total_equity = max(portfolio.total_equity, 100.0)
        cap_usd = min(self.max_position_usd, total_equity * self.max_portfolio_allocation)

        for sig in signals:
            sign = 1.0 if sig.score >= 0 else -1.0
            allocations[sig.symbol] = round(sign * cap_usd * abs(sig.score), 2)

        return allocations


class OversoldBounceStrategy(BaseAnomalyStrategy):
    """
    Oversold Flush & Bounce Strategy (LONG).
    Harvests violent mean-reversion bounces after rapid liquidation flushes.
    """

    def __init__(
        self,
        dump_threshold_24h: float = -0.15,
        rsi_threshold: float = 28.0,
        max_position_usd: float = 15.0,
        max_portfolio_allocation: float = 0.15,
        symbols_filter: list[str] | None = None,
        name: str = "OversoldBounce",
    ) -> None:
        super().__init__(
            max_position_usd=max_position_usd,
            max_portfolio_allocation=max_portfolio_allocation,
            symbols_filter=symbols_filter,
            name=name,
        )
        self.dump_threshold_24h = dump_threshold_24h
        self.rsi_threshold = rsi_threshold

    def generate_signals(self, market_data: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if not self._passes_universe_filter(sym, df, min_bars=26):
                continue

            closes = df["close"]
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))
            ret_4h = float((closes.iloc[-1] - closes.iloc[-5]) / (closes.iloc[-5] + 1e-8)) if len(closes) >= 5 else 0.0
            ret_1h = float((closes.iloc[-1] - closes.iloc[-2]) / (closes.iloc[-2] + 1e-8)) if len(closes) >= 2 else 0.0
            bar_pos = 0.5
            if {"high", "low"}.issubset(df.columns):
                rng_1h = float(df["high"].iloc[-1]) - float(df["low"].iloc[-1])
                if rng_1h > 1e-8:
                    bar_pos = (float(closes.iloc[-1]) - float(df["low"].iloc[-1])) / rng_1h
            rsi_curr = float(calculate_rsi(closes, period=14).iloc[-1])
            vol_ratio = 1.5
            if "volume" in df.columns:
                volumes = df["volume"]
                vol_curr = float(volumes.iloc[-1])
                vol_mean_24h = float(volumes.iloc[-25:-1].mean()) + 1e-6
                vol_ratio = vol_curr / vol_mean_24h

            if ret_1h < -0.040 or (ret_1h < -0.025 and bar_pos < 0.25):
                continue

            if ret_24h <= self.dump_threshold_24h and rsi_curr <= self.rsi_threshold:
                score = float(np.clip(abs(ret_24h) / abs(self.dump_threshold_24h), 0.5, 1.0))
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=score,
                        confidence=0.85,
                        horizon_minutes=1440,
                        metadata={
                            "ret_24h": ret_24h,
                            "ret_4h": ret_4h,
                            "rsi": rsi_curr,
                            "vol_ratio": vol_ratio,
                            "entry_price": float(closes.iloc[-1]),
                            "stop_loss": float(closes.iloc[-1] * 0.95),
                            "take_profit": float(closes.iloc[-1] * 1.06),
                        },
                    )
                )

        return signals


class EarlyBreakoutStrategy(BaseAnomalyStrategy):
    """
    Early Breakout Momentum Strategy (LONG).
    Catches volume and price breakouts at the beginning of an expansion (+8% in 4h with 2.5x volume),
    before the asset hits the +30% or +40% retail FOMO screen.
    """

    def __init__(
        self,
        pump_threshold_4h: float = 0.08,
        volume_multiplier: float = 2.5,
        max_24h_pump_ceiling: float = 0.22,
        max_position_usd: float = 15.0,
        max_portfolio_allocation: float = 0.15,
        symbols_filter: list[str] | None = None,
        name: str = "EarlyBreakout",
    ) -> None:
        super().__init__(
            max_position_usd=max_position_usd,
            max_portfolio_allocation=max_portfolio_allocation,
            symbols_filter=symbols_filter,
            name=name,
        )
        self.pump_threshold_4h = pump_threshold_4h
        self.volume_multiplier = volume_multiplier
        self.max_24h_pump_ceiling = max_24h_pump_ceiling

    def generate_signals(self, market_data: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if not self._passes_universe_filter(sym, df, min_bars=26):
                continue

            closes = df["close"]
            volumes = df["volume"]

            ret_4h = float((closes.iloc[-1] - closes.iloc[-5]) / (closes.iloc[-5] + 1e-8))
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))
            ret_1h = float((closes.iloc[-1] - closes.iloc[-2]) / (closes.iloc[-2] + 1e-8)) if len(closes) >= 2 else 0.0
            high_4h = float(df["high"].iloc[-4:].max()) if "high" in df.columns else float(closes.iloc[-4:].max())
            up_bars_4h = int((closes.diff().iloc[-4:] > 0).sum())

            vol_curr = float(volumes.iloc[-1])
            vol_mean_24h = float(volumes.iloc[-25:-1].mean()) + 1e-6
            vol_ratio = vol_curr / vol_mean_24h

            if float(closes.iloc[-1]) < high_4h * 0.965 or ret_1h < 0.0 or ret_1h > 0.09 or up_bars_4h < 2:
                continue

            if (
                ret_4h >= self.pump_threshold_4h
                and vol_ratio >= self.volume_multiplier
                and ret_24h <= self.max_24h_pump_ceiling
            ):
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=1.0,
                        confidence=0.75,
                        horizon_minutes=720,
                        metadata={
                            "ret_24h": ret_24h,
                            "ret_4h": ret_4h,
                            "vol_ratio": vol_ratio,
                            "entry_price": float(closes.iloc[-1]),
                            "stop_loss": float(closes.iloc[-1] * 0.96),
                            "take_profit": float(closes.iloc[-1] * 1.15),
                        },
                    )
                )

        return signals


class PumpFundingFadeStrategy(BaseAnomalyStrategy):
    """
    Pump & Funding Squeeze Fade Strategy (SHORT).
    Fades overextended blow-off tops where retail FOMO has driven the funding rate
    to extreme positive levels.
    """

    def __init__(
        self,
        pump_threshold_24h: float = 0.25,
        min_funding_rate: float = 0.0005,
        rsi_ceiling: float = 72.0,
        max_position_usd: float = 15.0,
        max_portfolio_allocation: float = 0.15,
        symbols_filter: list[str] | None = None,
        name: str = "PumpFundingFade",
    ) -> None:
        super().__init__(
            max_position_usd=max_position_usd,
            max_portfolio_allocation=max_portfolio_allocation,
            symbols_filter=symbols_filter,
            name=name,
        )
        self.pump_threshold_24h = pump_threshold_24h
        self.min_funding_rate = min_funding_rate
        self.rsi_ceiling = rsi_ceiling

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        funding_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[Signal]:
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if not self._passes_universe_filter(sym, df, min_bars=26):
                continue

            closes = df["close"]
            ret_24h = float((closes.iloc[-1] - closes.iloc[-25]) / (closes.iloc[-25] + 1e-8))
            ret_1h = float((closes.iloc[-1] - closes.iloc[-2]) / (closes.iloc[-2] + 1e-8)) if len(closes) >= 2 else 0.0
            rsi_curr = float(calculate_rsi(closes, period=14).iloc[-1])

            if ret_1h > 0.025:
                continue

            f_rate = 0.0
            if funding_data and sym in funding_data and not funding_data[sym].empty:
                f_rate = _extract_latest_funding_rate(funding_data[sym])
            elif "funding_rate" in df.columns:
                f_rate = float(df["funding_rate"].iloc[-1])

            if ret_24h >= self.pump_threshold_24h and f_rate >= self.min_funding_rate and rsi_curr >= self.rsi_ceiling:
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=-1.0,
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


class ExtremeFundingHarvesterStrategy(BaseAnomalyStrategy):
    """
    Extreme Funding Rate Harvester across USDS-M Perpetuals with Beta-Neutral Hedge metadata:
    - Deep Negative Funding Squeeze (funding_rate <= -0.15% per interval): LONG entry when
      price is stabilizing (ret_4h > 0, ret_1h >= 0, RSI >= 35) to harvest negative funding + short squeeze.
    - Extreme Positive Funding Blow-Off (funding_rate >= +0.15% per interval): SHORT entry when
      momentum is stalling (ret_4h <= +2%, ret_1h <= +0.5%, RSI >= 60) to harvest positive funding + post-settlement fade.
    """

    def __init__(
        self,
        neg_funding_threshold: float = -0.0015,
        pos_funding_threshold: float = 0.0015,
        max_position_usd: float = 45.0,
        max_portfolio_allocation: float = 0.045,
        symbols_filter: list[str] | None = None,
        name: str = "ExtremeFundingHarvester",
    ) -> None:
        super().__init__(
            max_position_usd=max_position_usd,
            max_portfolio_allocation=max_portfolio_allocation,
            symbols_filter=symbols_filter,
            name=name,
        )
        self.neg_funding_threshold = neg_funding_threshold
        self.pos_funding_threshold = pos_funding_threshold

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        funding_data: dict[str, pd.DataFrame] | None = None,
        benchmark_returns_1h: pd.Series | None = None,
    ) -> list[Signal]:
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)

        for sym, df in market_data.items():
            if not self._passes_universe_filter(sym, df, min_bars=25):
                continue

            current_funding = 0.0
            if funding_data and sym in funding_data and not funding_data[sym].empty:
                current_funding = _extract_latest_funding_rate(funding_data[sym])

            if self.neg_funding_threshold < current_funding < self.pos_funding_threshold:
                continue

            closes = df["close"].astype(float)
            vols = df["volume"].astype(float)
            ret_24h = (closes.iloc[-1] / max(closes.iloc[-25], 1e-12)) - 1.0
            ret_4h = (closes.iloc[-1] / max(closes.iloc[-5], 1e-12)) - 1.0
            ret_1h = (closes.iloc[-1] / max(closes.iloc[-2], 1e-12)) - 1.0 if len(closes) >= 2 else 0.0
            rsi_val = float(calculate_rsi(closes, period=14).iloc[-1])
            avg_vol_20h = float(vols.iloc[-24:-4].mean()) if len(vols) >= 24 else 1.0
            recent_vol = float(vols.iloc[-4:].mean())
            vol_ratio = recent_vol / (avg_vol_20h + 1e-8)

            beta_est = 1.0
            if benchmark_returns_1h is not None and len(benchmark_returns_1h) >= 12:
                asset_rets = closes.pct_change().dropna().tail(24)
                bench_rets = benchmark_returns_1h.dropna().tail(len(asset_rets))
                if len(asset_rets) >= 10 and len(bench_rets) == len(asset_rets):
                    b_var = float(np.var(bench_rets.values))
                    if b_var > 1e-8:
                        cov = float(np.cov(asset_rets.values, bench_rets.values)[0, 1])
                        beta_est = float(np.clip(cov / b_var, 0.50, 1.50))

            if (
                current_funding <= self.neg_funding_threshold
                and ret_4h > 0.0
                and ret_1h >= 0.0
                and ret_24h >= -0.10
                and rsi_val >= 35.0
            ):
                raw_score = min(abs(current_funding) / abs(self.neg_funding_threshold) * 0.65, 1.0)
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=float(np.clip(raw_score, 0.25, 1.0)),
                        confidence=0.88,
                        horizon_minutes=480,
                        metadata={
                            "direction": "LONG",
                            "ret_24h": float(ret_24h),
                            "ret_4h": float(ret_4h),
                            "rsi": float(rsi_val),
                            "vol_ratio": float(vol_ratio),
                            "funding_rate": float(current_funding),
                            "beta_est": float(beta_est),
                        },
                    )
                )
            elif (
                current_funding >= self.pos_funding_threshold and ret_4h <= 0.02 and ret_1h <= 0.005 and rsi_val >= 60.0
            ):
                raw_score = -min(current_funding / self.pos_funding_threshold * 0.65, 1.0)
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=float(np.clip(raw_score, -1.0, -0.25)),
                        confidence=0.88,
                        horizon_minutes=480,
                        metadata={
                            "direction": "SHORT",
                            "ret_24h": float(ret_24h),
                            "ret_4h": float(ret_4h),
                            "rsi": float(rsi_val),
                            "vol_ratio": float(vol_ratio),
                            "funding_rate": float(current_funding),
                            "beta_est": float(beta_est),
                        },
                    )
                )

        return signals


def create_extreme_funding_harvester(test_mode: bool = False) -> ExtremeFundingHarvesterStrategy:
    """Factory preset for Extreme Funding Rate Harvester across USDS-M Perpetuals."""
    return ExtremeFundingHarvesterStrategy(
        neg_funding_threshold=-0.0015,
        pos_funding_threshold=0.0015,
        max_position_usd=25.0 if test_mode else 135.0,
        max_portfolio_allocation=0.045,
        symbols_filter=None,
        name="ExtremeFundingHarvester",
    )
