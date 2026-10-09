"""
Cross-Sectional Factor Momentum Strategy.
Computes multi-horizon momentum, Parkinson volatility, and microstructure features,
generates dollar-neutral alpha signals via CrossSectionalAlphaPredictor,
and sizes positions via inverse-volatility weighting.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.domain.entities import Signal, Portfolio
from src.features.volatility import parkinson_volatility
from src.features.microstructure import amihud_illiquidity
from src.models.alpha_predictor import CrossSectionalAlphaPredictor
from src.risk.vol_targeting import VolatilityTargetingSizer
from src.ports.model_ports import IStrategy


class CrossSectionalMomentumStrategy(IStrategy):
    """
    Dollar-neutral systematic cross-sectional momentum strategy.
    Ranks crypto assets on composite momentum and microstructure quality,
    going long highest expected returns and short lowest expected returns.
    """

    def __init__(
        self,
        predictor: Optional[CrossSectionalAlphaPredictor] = None,
        target_annual_vol: float = 0.20,
        max_leverage: float = 2.0,
    ):
        self.predictor = predictor or CrossSectionalAlphaPredictor()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

    def extract_features(self, df_bars: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """
        Extracts point-in-time quantitative factors for an asset from its OHLCV bars.
        df_bars must have ['open', 'high', 'low', 'close', 'volume'].
        """
        if len(df_bars) < 72:
            return None

        closes = df_bars["close"].values
        highs = df_bars["high"].values
        lows = df_bars["low"].values
        volumes = df_bars["volume"].values

        # 1. Multi-horizon momentum returns
        ret_12h = float((closes[-1] - closes[-12]) / (closes[-12] + 1e-8))
        ret_24h = float((closes[-1] - closes[-24]) / (closes[-24] + 1e-8))
        ret_72h = float((closes[-1] - closes[-72]) / (closes[-72] + 1e-8))

        # 2. Parkinson Volatility (24 bars)
        vol_parkinson = float(parkinson_volatility(
            highs[-24:], lows[-24:], annualize=True
        ))

        # 3. Amihud Illiquidity (24 bars)
        abs_rets = np.abs(np.diff(closes[-25:]) / (closes[-25:-1] + 1e-8))
        amihud = float(amihud_illiquidity(abs_rets, volumes[-24:]))

        return pd.Series({
            "symbol": symbol,
            "ret_12h": ret_12h,
            "ret_24h": ret_24h,
            "ret_72h": ret_72h,
            "vol_parkinson": vol_parkinson,
            "amihud": amihud,
        })

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Processes universe market data dict {symbol: df_bars} and generates
        dollar-neutral alpha signals.
        """
        feature_rows = []
        for sym, df in market_data.items():
            feat = self.extract_features(df, sym)
            if feat is not None:
                feature_rows.append(feat)

        if not feature_rows:
            return []

        universe_features = pd.DataFrame(feature_rows)

        # If predictor is not yet trained, default to composite rank on momentum
        if self.predictor.ridge_model is None:
            composite_score = (
                0.5 * universe_features["ret_24h"]
                + 0.3 * universe_features["ret_72h"]
                + 0.2 * universe_features["ret_12h"]
            )
            # Demean to dollar-neutral
            demeaned = composite_score - composite_score.mean()
            norm = (demeaned / (demeaned.abs().max() or 1.0)).values

            now = datetime.now(timezone.utc)
            return [
                Signal(
                    timestamp=now,
                    symbol=sym,
                    score=float(score),
                    confidence=float(min(1.0, abs(score))),
                    horizon_minutes=60,
                )
                for sym, score in zip(universe_features["symbol"], norm)
            ]

        return self.predictor.predict(universe_features)

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Sizes dollar weights across universe using inverse-volatility targeting.
        """
        fractional_weights = self.risk_manager.size_positions(
            signals, volatilities, market_neutral=True
        )
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
