"""
Marcos López de Prado's Triple Barrier Method & Sample Weighting Engine.
References: Advances in Financial Machine Learning (AFML), Chapters 3 & 4.

Provides dynamic volatility-scaled profit-taking, stop-loss, and vertical holding period barriers,
concurrent event tracking, sample uniqueness calculation, and return-attributed sample weights.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    """
    Configuration parameters for the Triple Barrier Method.
    pt: Profit-taking barrier multiplier (e.g. 1.5 * sigma)
    sl: Stop-loss barrier multiplier (e.g. 1.0 * sigma)
    holding_period_bars: Maximum duration before vertical barrier expiration (default 24 bars)
    min_ret: Minimum required target volatility floor (default 0.001 = 10 bps)
    strict_pt: If True, only trades hitting profit-take barrier are labeled 1.
               If False, trades expiring with positive return at vertical barrier are also labeled 1.
    conservative_touch: If True and both PT and SL are breached in the same bar, assumes SL hit first.
    """
    pt: float = 1.5
    sl: float = 1.0
    holding_period_bars: int = 24
    min_ret: float = 0.001
    strict_pt: bool = False
    conservative_touch: bool = True


class TripleBarrierLabeler:
    """
    Applies horizontal volatility-scaled barriers and vertical time limits to financial paths.
    Generates meta-labels (0/1) or directional labels (-1/0/1), execution timestamps,
    and AFML Chapter 4 sample uniqueness weights.
    """

    def __init__(self, config: TripleBarrierConfig | None = None):
        self.config = config or TripleBarrierConfig()

    def get_daily_volatility(
        self,
        close: pd.Series,
        span: int = 24,
    ) -> pd.Series:
        """
        Computes rolling exponential moving standard deviation of log returns.
        Equivalent to Marcos López de Prado's getDailyVol.
        """
        log_rets = np.log(close / close.shift(1)).fillna(0.0)
        vol = log_rets.ewm(span=span, min_periods=max(5, span // 3)).std()
        return vol.bfill().fillna(self.config.min_ret)

    def apply_barriers(
        self,
        close: pd.Series,
        events: pd.DataFrame,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        volatility: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Evaluates the Triple Barrier Method for each trade initiation event.

        Args:
            close: Series of close prices indexed by timestamp.
            events: DataFrame indexed by event timestamp (t0).
                    Can include columns 'side' (+1 for Long, -1 for Short; default +1)
                    and optionally 'holding_period' or custom barriers.
            high: Optional high price Series for intrabar breach detection.
            low: Optional low price Series for intrabar breach detection.
            volatility: Optional Series of volatility scaled per bar (e.g., Garman-Klass, Parkinson, or EWMA).
                        If not provided, computed via rolling EWMA std of close prices.

        Returns:
            pd.DataFrame with columns:
                - 't1': Timestamp of the barrier touch (exit time)
                - 'ret': Realized return of the trade accounting for side
                - 'label': Binary target (1 = successful trade, 0 = unsuccessful trade)
                - 'touch_type': 'pt' (profit take), 'sl' (stop loss), or 'vertical' (holding period expired)
                - 'side': Trade direction (+1 or -1)
                - 'barrier_pt': Absolute price level for profit take
                - 'barrier_sl': Absolute price level for stop loss
        """
        if len(events) == 0:
            return pd.DataFrame(
                columns=["t1", "ret", "label", "touch_type", "side", "barrier_pt", "barrier_sl"]
            )

        # Align series indices
        close_idx = close.index
        if volatility is None:
            vol = self.get_daily_volatility(close)
        else:
            vol = volatility.reindex(close_idx).bfill().ffill()

        high_s = high.reindex(close_idx) if high is not None else close
        low_s = low.reindex(close_idx) if low is not None else close

        results = []
        loc_map = {ts: idx for idx, ts in enumerate(close_idx)}

        for t0 in events.index:
            if t0 not in loc_map:
                continue

            t0_idx = loc_map[t0]
            p0 = float(close.iloc[t0_idx])
            if np.isnan(p0) or p0 <= 0:
                continue

            side = 1.0
            if "side" in events.columns:
                side_val = events.loc[t0, "side"]
                # Handle possible duplicate indices in events
                if isinstance(side_val, pd.Series):
                    side_val = side_val.iloc[0]
                side = 1.0 if float(side_val) >= 0 else -1.0

            # Target volatility at t0
            vol_val = float(vol.iloc[t0_idx])
            sigma = max(vol_val, self.config.min_ret)

            # Barrier levels
            pt_mult = self.config.pt * sigma
            sl_mult = self.config.sl * sigma

            if side > 0:
                upper_target = p0 * (1.0 + pt_mult)
                lower_target = p0 * (1.0 - sl_mult)
            else:
                upper_target = p0 * (1.0 - pt_mult)  # Short profit takes when price drops
                lower_target = p0 * (1.0 + sl_mult)  # Short stops out when price rises

            # Vertical barrier limit
            h_bars = self.config.holding_period_bars
            end_idx = min(len(close_idx) - 1, t0_idx + h_bars)

            # Trace path
            touch_time = close_idx[end_idx]
            touch_type = "vertical"
            touch_price = float(close.iloc[end_idx])

            for step_idx in range(t0_idx + 1, end_idx + 1):
                step_high = float(high_s.iloc[step_idx])
                step_low = float(low_s.iloc[step_idx])
                step_time = close_idx[step_idx]

                if side > 0:
                    pt_hit = step_high >= upper_target
                    sl_hit = step_low <= lower_target

                    if pt_hit and sl_hit:
                        if self.config.conservative_touch:
                            touch_type = "sl"
                            touch_price = lower_target
                        else:
                            touch_type = "pt"
                            touch_price = upper_target
                        touch_time = step_time
                        break
                    elif pt_hit:
                        touch_type = "pt"
                        touch_price = upper_target
                        touch_time = step_time
                        break
                    elif sl_hit:
                        touch_type = "sl"
                        touch_price = lower_target
                        touch_time = step_time
                        break

                else:  # Short position
                    pt_hit = step_low <= upper_target
                    sl_hit = step_high >= lower_target

                    if pt_hit and sl_hit:
                        if self.config.conservative_touch:
                            touch_type = "sl"
                            touch_price = lower_target
                        else:
                            touch_type = "pt"
                            touch_price = upper_target
                        touch_time = step_time
                        break
                    elif pt_hit:
                        touch_type = "pt"
                        touch_price = upper_target
                        touch_time = step_time
                        break
                    elif sl_hit:
                        touch_type = "sl"
                        touch_price = lower_target
                        touch_time = step_time
                        break

            # Calculate realized return
            price_ret = (touch_price - p0) / p0
            trade_ret = price_ret * side

            # Determine label Y in {0, 1}
            if touch_type == "pt":
                label = 1
            elif touch_type == "sl":
                label = 0
            else:  # Vertical barrier timeout
                if not self.config.strict_pt and trade_ret > 0:
                    label = 1
                else:
                    label = 0

            results.append({
                "t0": t0,
                "t1": touch_time,
                "ret": trade_ret,
                "label": int(label),
                "touch_type": touch_type,
                "side": int(side),
                "barrier_pt": upper_target,
                "barrier_sl": lower_target,
            })

        df_res = pd.DataFrame(results)
        if len(df_res) > 0:
            df_res = df_res.set_index("t0")
        return df_res

    @staticmethod
    def compute_sample_uniqueness(
        events: pd.DataFrame,
        price_index: pd.Index,
    ) -> pd.Series:
        """
        Computes Marcos López de Prado's Sample Uniqueness (AFML Ch. 4).
        Solves label concurrency: when multiple trades overlap in time, their
        information is correlated. Uniqueness weights each observation inversely
        by the number of concurrent active labels.

        Args:
            events: DataFrame with event start index (t0) and 't1' column (exit time).
            price_index: Full chronological index of price bars.

        Returns:
            pd.Series indexed by t0 containing average uniqueness score in (0.0, 1.0].
        """
        if len(events) == 0:
            return pd.Series(dtype=np.float64)

        # 1. Count concurrency c_t across all bars
        # Map timestamps to integer positions for fast array operations
        time_to_idx = {ts: idx for idx, ts in enumerate(price_index)}
        concurrency = np.zeros(len(price_index), dtype=np.int32)

        for t0, row in events.iterrows():
            if t0 in time_to_idx and row["t1"] in time_to_idx:
                i0 = time_to_idx[t0]
                i1 = time_to_idx[row["t1"]]
                if i1 >= i0:
                    concurrency[i0:i1 + 1] += 1

        # 2. Compute average uniqueness per event row
        uniqueness_vals = np.ones(len(events), dtype=np.float64)
        for row_pos, (t0, row) in enumerate(events.iterrows()):
            if t0 in time_to_idx and row["t1"] in time_to_idx:
                i0 = time_to_idx[t0]
                i1 = time_to_idx[row["t1"]]
                if i1 >= i0:
                    event_concurrency = concurrency[i0:i1 + 1]
                    weights_t = 1.0 / np.maximum(event_concurrency, 1)
                    uniqueness_vals[row_pos] = float(np.mean(weights_t))

        return pd.Series(uniqueness_vals, index=events.index)

    @staticmethod
    def compute_sample_weights(
        events: pd.DataFrame,
        price_index: pd.Index,
        attribute_returns: bool = True,
    ) -> pd.Series:
        """
        Computes sample weights taking into account both average uniqueness
        and return attribution (AFML Ch. 4).

        w_i = u_i * |r_i| normalized such that mean(w) == 1.0
        """
        uniqueness = TripleBarrierLabeler.compute_sample_uniqueness(events, price_index)
        if len(uniqueness) == 0:
            return pd.Series(dtype=np.float64)

        weights = uniqueness.copy()
        if attribute_returns and "ret" in events.columns:
            rets = events["ret"].abs().values
            weights = pd.Series(weights.values * rets, index=events.index)

        # Normalize to average 1.0
        mean_w = weights.mean()
        if mean_w > 1e-12:
            weights = weights / mean_w
        else:
            weights = pd.Series(1.0, index=events.index)

        return weights
