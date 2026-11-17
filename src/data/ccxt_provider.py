"""
CCXT Market Data Provider Adapter.
Fetches free public OHLCV candles and funding rates from Binance and Bybit.
"""

from datetime import datetime, timedelta, timezone
import logging

import ccxt

from src.domain.entities import Bar, FundingRate
from src.domain.enums import Timeframe

logger = logging.getLogger(__name__)

TOP_30_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "SUI/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "UNI/USDT",
    "AAVE/USDT",
    "INJ/USDT",
    "RENDER/USDT",
    "FET/USDT",
    "TIA/USDT",
    "SEI/USDT",
    "RUNE/USDT",
    "ICP/USDT",
    "KAS/USDT",
    "STX/USDT",
    "TAO/USDT",
]


def _to_utc_timestamp_ms(dt: datetime) -> int:
    """Converts a naive or timezone-aware datetime to UTC milliseconds since epoch."""
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return int(utc_dt.timestamp() * 1000)


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
    ) -> None:
        self.exchange_id = exchange_id
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unsupported CCXT exchange_id: '{exchange_id}'")
        exchange_class = getattr(ccxt, exchange_id)
        config: dict[str, object] = {
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
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        """
        Fetches OHLCV candlestick bars from the exchange.
        """
        since_ms = _to_utc_timestamp_ms(since) if since is not None else None

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe.value,
                since=since_ms,
                limit=limit,
            )
        except Exception as err:  # noqa: BLE001
            logger.error(f"Error fetching OHLCV for {symbol} on {self.exchange_id}: {err}")
            return []

        bars: list[Bar] = []
        for row in ohlcv:
            timestamp_ms, open_price, high_price, low_price, close_price, volume = row
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc),
                    symbol=symbol,
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume),
                )
            )
        return bars

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[FundingRate]:
        """
        Fetches historical funding rates for perpetual futures contracts.
        """
        if not self.exchange.has.get("fetchFundingRateHistory", False):
            logger.warning(f"{self.exchange_id} does not support fetchFundingRateHistory")
            return []

        try:
            history = self.exchange.fetch_funding_rate_history(symbol=symbol, limit=limit)
        except Exception as err:  # noqa: BLE001
            logger.error(f"Error fetching funding rate for {symbol}: {err}")
            return []

        rates: list[FundingRate] = []
        for entry in history:
            raw_ts_ms = entry.get("timestamp")
            if raw_ts_ms is None:
                continue
            timestamp_ms = float(raw_ts_ms)
            rate = float(entry.get("fundingRate") or 0.0)
            settlement_dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            next_ts_ms = entry.get("nextFundingTimestamp")
            next_funding_dt = (
                datetime.fromtimestamp(float(next_ts_ms) / 1000.0, tz=timezone.utc)
                if next_ts_ms is not None
                else settlement_dt + timedelta(hours=8)
            )
            rates.append(
                FundingRate(
                    timestamp=settlement_dt,
                    symbol=symbol,
                    rate=rate,
                    next_funding_time=next_funding_dt,
                )
            )
        return rates
