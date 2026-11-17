"""
Market-Wide Aggregate Basket Dollar Bar Synthesizer.
Solves the Cross-Sectional Desynchronization Paradox (Issue #33).
Accumulates global traded dollar notional across the entire multi-asset universe
and emits synchronized cross-sectional snapshots whenever market information reaches $D_market.
"""

from typing import Any

import numpy as np
import pandas as pd


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
    ) -> None:
        self.market_dollar_threshold = float(market_dollar_threshold)
        self.min_asset_notional = float(min_asset_notional)
        self.reset()

    def reset(self) -> None:
        """Resets the market-wide accumulator state."""
        self.cum_market_notional = 0.0
        self.bar_index = 0
        self.ticks_or_bars_count = 0
        self.asset_states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _init_asset_state(price: float, volume: float, notional: float) -> dict[str, Any]:
        return {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
            "notional": notional,
        }

    def process_synchronized_bars(
        self,
        bars_dict: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Synthesizes basket dollar bars from multi-asset time bars (e.g. 15m or 1h klines).
        """
        if not bars_dict:
            return pd.DataFrame(), {}

        valid_symbols = [sym for sym in bars_dict if not bars_dict[sym].empty]
        if not valid_symbols:
            return pd.DataFrame(), {}

        all_timestamps = set()
        prev_closes: dict[str, float] = {}
        for sym in valid_symbols:
            df_sym = bars_dict[sym]
            all_timestamps.update(df_sym.index)
            first_valid = df_sym["close"].dropna()
            if not first_valid.empty:
                prev_closes[sym] = float(first_valid.iloc[0])

        timeline = pd.DatetimeIndex(sorted(all_timestamps))
        pulses: list[dict[str, Any]] = []
        synced_asset_records: dict[str, list[dict[str, Any]]] = {sym: [] for sym in valid_symbols}

        for ts in timeline:
            self.ticks_or_bars_count += 1
            step_market_notional = 0.0

            for sym in valid_symbols:
                df_sym = bars_dict[sym]
                if ts not in df_sym.index:
                    continue

                row = df_sym.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                open_px = float(row["open"])
                high_px = float(row["high"])
                low_px = float(row["low"])
                close_px = float(row["close"])
                vol = float(row["volume"])

                raw_notional = row.get("quote_volume")
                if raw_notional is None or pd.isna(raw_notional):
                    raw_notional = row.get("notional")
                if raw_notional is None or pd.isna(raw_notional):
                    raw_notional = close_px * vol
                notional = float(raw_notional)
                if not np.isfinite(notional) or notional < 0.0:
                    notional = 0.0

                step_market_notional += notional

                if sym not in self.asset_states:
                    state = self._init_asset_state(open_px, vol, notional)
                    state["high"] = high_px
                    state["low"] = low_px
                    state["close"] = close_px
                    self.asset_states[sym] = state
                else:
                    state = self.asset_states[sym]
                    if high_px > state["high"]:
                        state["high"] = high_px
                    if low_px < state["low"]:
                        state["low"] = low_px
                    state["close"] = close_px
                    state["volume"] += vol
                    state["notional"] += notional

            self.cum_market_notional += step_market_notional

            if self.cum_market_notional >= self.market_dollar_threshold:
                self.bar_index += 1

                pulses.append(
                    {
                        "bar_index": self.bar_index,
                        "timestamp": ts,
                        "market_notional": self.cum_market_notional,
                        "duration_steps": self.ticks_or_bars_count,
                        "active_assets": len(self.asset_states),
                    }
                )

                for sym in valid_symbols:
                    if sym in self.asset_states:
                        state = self.asset_states[sym]
                        prev_close = prev_closes.get(sym, state["open"])
                        ret = (state["close"] / prev_close - 1.0) if prev_close > 0 else 0.0

                        synced_asset_records[sym].append(
                            {
                                "timestamp": ts,
                                "open": state["open"],
                                "high": state["high"],
                                "low": state["low"],
                                "close": state["close"],
                                "volume": state["volume"],
                                "notional": state["notional"],
                                "return": ret,
                            }
                        )
                        prev_closes[sym] = state["close"]
                    else:
                        last_close = prev_closes.get(sym, 0.0)
                        synced_asset_records[sym].append(
                            {
                                "timestamp": ts,
                                "open": last_close,
                                "high": last_close,
                                "low": last_close,
                                "close": last_close,
                                "volume": 0.0,
                                "notional": 0.0,
                                "return": 0.0,
                            }
                        )

                remainder = self.cum_market_notional - self.market_dollar_threshold
                self.cum_market_notional = max(0.0, remainder)
                self.ticks_or_bars_count = 0
                self.asset_states = {}

        if not pulses:
            return pd.DataFrame(), {}

        market_summary_df = pd.DataFrame(pulses).set_index("timestamp")

        synced_dfs: dict[str, pd.DataFrame] = {}
        for sym, records in synced_asset_records.items():
            if records:
                synced_dfs[sym] = pd.DataFrame(records).set_index("timestamp")

        return market_summary_df, synced_dfs

    def extract_cross_sectional_matrix(
        self,
        synced_asset_bars: dict[str, pd.DataFrame],
        metric: str = "close",
    ) -> pd.DataFrame:
        """
        Extracts a wide N x T cross-sectional matrix for any feature (close, return, notional).
        Guaranteed to have perfectly aligned columns and rows without NaNs.
        """
        matrix_cols = {sym: df[metric] for sym, df in synced_asset_bars.items() if metric in df.columns}
        mat = pd.DataFrame(matrix_cols).ffill()
        if metric in ("open", "high", "low", "close"):
            return mat.bfill().fillna(0.0)
        return mat.fillna(0.0)
