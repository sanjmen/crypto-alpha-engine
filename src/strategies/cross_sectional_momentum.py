"""
Cross-Sectional Factor Momentum Strategy.
Computes multi-horizon momentum, Parkinson volatility, and microstructure features,
generates dollar-neutral alpha signals via CrossSectionalAlphaPredictor,
and sizes positions via inverse-volatility weighting.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.entities import Portfolio, Signal
from src.features.microstructure import amihud_illiquidity
from src.features.volatility import parkinson_volatility
from src.models.alpha_predictor import CrossSectionalAlphaPredictor
from src.ports.model_ports import IStrategy
from src.risk.vol_targeting import VolatilityTargetingSizer


class CrossSectionalMomentumStrategy(IStrategy):
    """
    Dollar-neutral systematic cross-sectional momentum strategy.
    Ranks crypto assets on composite momentum and microstructure quality,
    going long highest expected returns and short lowest expected returns.
    """

    def __init__(
        self,
        predictor: CrossSectionalAlphaPredictor | None = None,
        target_annual_vol: float = 0.20,
        max_leverage: float = 2.0,
    ) -> None:
        self.predictor = predictor or CrossSectionalAlphaPredictor()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

    def extract_features(self, bars_df: pd.DataFrame, symbol: str) -> pd.Series | None:
        """
        Extracts point-in-time quantitative factors for an asset from its OHLCV bars.
        bars_df must have ['open', 'high', 'low', 'close', 'volume'].
        """
        if len(bars_df) < 72:
            return None

        closes = bars_df["close"].values
        highs = bars_df["high"].values
        lows = bars_df["low"].values
        volumes = bars_df["volume"].values

        ret_12h = float((closes[-1] - closes[-12]) / (closes[-12] + 1e-8))
        ret_24h = float((closes[-1] - closes[-24]) / (closes[-24] + 1e-8))
        ret_72h = float((closes[-1] - closes[-72]) / (closes[-72] + 1e-8))

        vol_parkinson = float(parkinson_volatility(highs[-24:], lows[-24:], annualize=True))

        abs_rets = np.abs(np.diff(closes[-25:]) / (closes[-25:-1] + 1e-8))
        amihud = float(amihud_illiquidity(abs_rets, volumes[-24:]))

        return pd.Series(
            {
                "symbol": symbol,
                "ret_12h": ret_12h,
                "ret_24h": ret_24h,
                "ret_72h": ret_72h,
                "vol_parkinson": vol_parkinson,
                "amihud": amihud,
            }
        )

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        Processes universe market data dict {symbol: bars_df} and generates
        dollar-neutral alpha signals.
        """
        feature_rows: list[pd.Series] = []
        for symbol, bars_df in market_data.items():
            features = self.extract_features(bars_df, symbol)
            if features is not None:
                feature_rows.append(features)

        if not feature_rows:
            return []

        universe_features = pd.DataFrame(feature_rows)

        if self.predictor.ridge_model is None:
            composite_score = (
                0.5 * universe_features["ret_24h"]
                + 0.3 * universe_features["ret_72h"]
                + 0.2 * universe_features["ret_12h"]
            )
            demeaned = composite_score - composite_score.mean()
            raw_max_abs = float(demeaned.abs().max())
            denom = raw_max_abs if (np.isfinite(raw_max_abs) and raw_max_abs > 1e-12) else 1.0
            normalized_scores = (demeaned / denom).fillna(0.0).values

            now = datetime.now(timezone.utc)
            return [
                Signal(
                    timestamp=now,
                    symbol=sym,
                    score=float(score),
                    confidence=float(min(1.0, abs(score))),
                    horizon_minutes=60,
                )
                for sym, score in zip(universe_features["symbol"], normalized_scores, strict=False)
            ]

        return self.predictor.predict(universe_features)

    def allocate_weights(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """
        Sizes dollar weights across universe using inverse-volatility targeting.
        """
        fractional_weights = self.risk_manager.size_positions(signals, volatilities, market_neutral=True)
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
