"""
CCXT Market Data Provider Adapter.
Fetches free public OHLCV candles and funding rates from Binance and Bybit.
"""

from datetime import datetime, timezone
import logging
import time
from typing import List, Optional
import ccxt

from src.domain.entities import Bar, FundingRate
from src.domain.enums import OrderSide, Timeframe

logger = logging.getLogger(__name__)

TOP_30_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "SUI/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "DOT/USDT",
    "MATIC/USDT", "LTC/USDT", "BCH/USDT", "UNI/USDT", "AAVE/USDT",
    "INJ/USDT", "RENDER/USDT", "FET/USDT", "TIA/USDT", "SEI/USDT",
    "RUNE/USDT", "ICP/USDT", "KAS/USDT", "STX/USDT", "TAO/USDT",
]


class CCXTMarketDataProvider:
    """
    Adapter implementing IDataProvider via the CCXT library.
    Utilizes free public REST endpoints without requiring private API keys.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        enable_rate_limit: bool = True,
        is_futures: bool = True,
    ):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "enableRateLimit": enable_rate_limit,
            "timeout": 30000,
        }
        if is_futures:
            config["options"] = {"defaultType": "future"}

        self.exchange: ccxt.Exchange = exchange_class(config)

    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H1,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Bar]:
        """
        Fetches OHLCV candlestick bars from the exchange.
        """
        since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000) if since else None

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe.value,
                since=since_ms,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} on {self.exchange_id}: {e}")
            return []

        bars: List[Bar] = []
        for row in ohlcv:
            ts_ms, o, h, l, c, v = row
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                    symbol=symbol,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v),
                )
            )
        return bars

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[FundingRate]:
        """
        Fetches historical funding rates for perpetual futures contracts.
        """
        if not self.exchange.has.get("fetchFundingRateHistory", False):
            logger.warning(f"{self.exchange_id} does not support fetchFundingRateHistory")
            return []

        try:
            history = self.exchange.fetch_funding_rate_history(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return []

        rates: List[FundingRate] = []
        for entry in history:
            ts_ms = entry.get("timestamp") or 0
            rate = float(entry.get("fundingRate") or 0.0)
            rates.append(
                FundingRate(
                    timestamp=datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                    symbol=symbol,
                    rate=rate,
                    next_funding_time=datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                )
            )
        return rates
