"""
Market-Wide Aggregate Basket Dollar Bar Synthesizer.
Solves the Cross-Sectional Desynchronization Paradox (Issue #33).
Accumulates global traded dollar notional across the entire multi-asset universe
and emits synchronized cross-sectional snapshots whenever market information reaches $D_market.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class MarketPulseSnapshot:
    """Represents a synchronized cross-sectional market information pulse."""
    bar_index: int
    timestamp: pd.Timestamp
    market_notional: float
    duration_bars: int
    asset_closes: Dict[str, float]
    asset_returns: Dict[str, float]


class MarketWideBasketDollarBarSynthesizer:
    """
    Synthesizes synchronous cross-sectional Dollar Bars from multi-asset market data.
    Guarantees that all N assets share the exact same timestamp t_k,
    enabling rigorous cross-sectional ranking and factor investing without chronological time distortion.
    """

    def __init__(
        self,
        market_dollar_threshold: float = 250_000_000.0,  # $250M default market pulse
        min_asset_notional: float = 1_000.0,
    ):
        self.market_dollar_threshold = float(market_dollar_threshold)
        self.min_asset_notional = float(min_asset_notional)
        self.reset()

    def reset(self) -> None:
        """Resets the market-wide accumulator state."""
        self.cum_market_notional = 0.0
        self.bar_index = 0
        self.ticks_or_bars_count = 0
        self.asset_states: Dict[str, Dict[str, Any]] = {}

    def _init_asset_state(self, sym: str, p: float, q: float, d: float) -> Dict[str, Any]:
        return {
            "open": p,
            "high": p,
            "low": p,
            "close": p,
            "volume": q,
            "notional": d,
        }

    def process_synchronized_bars(
        self,
        bars_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Synthesizes basket dollar bars from multi-asset time bars (e.g. 15m or 1h klines).
        Args:
            bars_dict: Dict mapping symbol -> DataFrame with index=timestamp, columns=[open, high, low, close, volume, (quote_volume/notional)]
        Returns:
            market_summary_df: DataFrame of market pulses (timestamp, market_notional, duration)
            synced_asset_bars: Dict mapping symbol -> DataFrame with exact same aligned timestamps!
        """
        if not bars_dict:
            return pd.DataFrame(), {}

        # 1. Determine universal timeline
        valid_symbols = [s for s in bars_dict if not bars_dict[s].empty]
        if not valid_symbols:
            return pd.DataFrame(), {}

        # Build union timeline sorted ascending
        all_timestamps = set()
        for sym in valid_symbols:
            all_timestamps.update(bars_dict[sym].index)
        timeline = pd.DatetimeIndex(sorted(all_timestamps))

        pulses: List[Dict[str, Any]] = []
        synced_asset_records: Dict[str, List[Dict[str, Any]]] = {sym: [] for sym in valid_symbols}

        prev_closes: Dict[str, float] = {}

        for ts in timeline:
            self.ticks_or_bars_count += 1
            step_market_notional = 0.0

            # Collect step activity across all assets
            for sym in valid_symbols:
                df = bars_dict[sym]
                if ts not in df.index:
                    continue

                row = df.loc[ts]
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row["volume"])
                d = float(row.get("quote_volume", row.get("notional", c * v)))

                step_market_notional += d

                # Accumulate per-asset bar state
                if sym not in self.asset_states:
                    self.asset_states[sym] = self._init_asset_state(sym, o, v, d)
                    self.asset_states[sym]["high"] = h
                    self.asset_states[sym]["low"] = l
                    self.asset_states[sym]["close"] = c
                else:
                    st = self.asset_states[sym]
                    if h > st["high"]:
                        st["high"] = h
                    if l < st["low"]:
                        st["low"] = l
                    st["close"] = c
                    st["volume"] += v
                    st["notional"] += d

            self.cum_market_notional += step_market_notional

            # Check market-wide threshold breach
            if self.cum_market_notional >= self.market_dollar_threshold:
                self.bar_index += 1

                pulses.append({
                    "bar_index": self.bar_index,
                    "timestamp": ts,
                    "market_notional": self.cum_market_notional,
                    "duration_steps": self.ticks_or_bars_count,
                    "active_assets": len(self.asset_states),
                })

                # Record per-asset synchronized bar
                for sym in valid_symbols:
                    if sym in self.asset_states:
                        st = self.asset_states[sym]
                        prev_c = prev_closes.get(sym, st["open"])
                        ret = (st["close"] / prev_c - 1.0) if prev_c > 0 else 0.0

                        synced_asset_records[sym].append({
                            "timestamp": ts,
                            "open": st["open"],
                            "high": st["high"],
                            "low": st["low"],
                            "close": st["close"],
                            "volume": st["volume"],
                            "notional": st["notional"],
                            "return": ret,
                        })
                        prev_closes[sym] = st["close"]
                    else:
                        # Asset had no volume in this pulse -> forward-fill close, 0 volume
                        last_c = prev_closes.get(sym, 1.0)
                        synced_asset_records[sym].append({
                            "timestamp": ts,
                            "open": last_c,
                            "high": last_c,
                            "low": last_c,
                            "close": last_c,
                            "volume": 0.0,
                            "notional": 0.0,
                            "return": 0.0,
                        })

                # Carry remainder or reset
                remainder = self.cum_market_notional - self.market_dollar_threshold
                self.cum_market_notional = max(0.0, remainder)
                self.ticks_or_bars_count = 0
                self.asset_states = {}

        if not pulses:
            return pd.DataFrame(), {}

        market_summary_df = pd.DataFrame(pulses).set_index("timestamp")

        synced_dfs: Dict[str, pd.DataFrame] = {}
        for sym, records in synced_asset_records.items():
            if records:
                df_sym = pd.DataFrame(records).set_index("timestamp")
                synced_dfs[sym] = df_sym

        return market_summary_df, synced_dfs

    def extract_cross_sectional_matrix(
        self,
        synced_asset_bars: Dict[str, pd.DataFrame],
        metric: str = "close",
    ) -> pd.DataFrame:
        """
        Extracts a wide N x T cross-sectional matrix for any feature (close, return, notional).
        Guaranteed to have perfectly aligned columns and rows without NaNs.
        """
        matrix_cols = {}
        for sym, df in synced_asset_bars.items():
            if metric in df.columns:
                matrix_cols[sym] = df[metric]
        return pd.DataFrame(matrix_cols).ffill().fillna(0.0)
