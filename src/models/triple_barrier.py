"""
Marcos López de Prado's Triple Barrier Method & Sample Weighting Engine.
References: Advances in Financial Machine Learning (AFML), Chapters 3 & 4.

Provides dynamic volatility-scaled profit-taking, stop-loss, and vertical holding period barriers,
concurrent event tracking, sample uniqueness calculation, and return-attributed sample weights.
"""

from dataclasses import dataclass
from typing import Any

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

    def __init__(self, config: TripleBarrierConfig | None = None) -> None:
        self.config = config or TripleBarrierConfig()

    def get_daily_volatility(
        self,
        close: pd.Series,
        span: int = 24,
    ) -> pd.Series:
        """
        Computes strictly causal rolling exponential moving standard deviation of log returns.
        Equivalent to Marcos López de Prado's getDailyVol without backward-fill leakage.
        """
        log_rets = np.log(close / close.shift(1)).fillna(0.0)
        vol = log_rets.ewm(span=span, min_periods=max(5, span // 3)).std()
        return vol.ffill().fillna(self.config.min_ret)

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
        """
        if len(events) == 0:
            return pd.DataFrame(columns=["t1", "ret", "label", "touch_type", "side", "barrier_pt", "barrier_sl"])

        close_idx = close.index
        if volatility is None:
            vol = self.get_daily_volatility(close)
        else:
            vol = volatility.reindex(close_idx).ffill().fillna(self.config.min_ret)

        high_series = high.reindex(close_idx) if high is not None else close
        low_series = low.reindex(close_idx) if low is not None else close

        results: list[dict[str, Any]] = []
        loc_map = {ts: idx for idx, ts in enumerate(close_idx)}

        for t0 in events.index:
            if t0 not in loc_map:
                continue

            t0_idx = loc_map[t0]
            entry_price = float(close.iloc[t0_idx])
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            side = 1.0
            if "side" in events.columns:
                side_val = events.loc[t0, "side"]
                if isinstance(side_val, pd.Series):
                    side_val = side_val.iloc[0]
                side = 1.0 if float(side_val) >= 0 else -1.0

            vol_val = float(vol.iloc[t0_idx])
            sigma = max(vol_val if np.isfinite(vol_val) else 0.0, self.config.min_ret)

            pt_mult = self.config.pt * sigma
            sl_mult = self.config.sl * sigma

            if side > 0:
                profit_take_price = entry_price * (1.0 + pt_mult)
                stop_loss_price = entry_price * (1.0 - sl_mult)
            else:
                profit_take_price = entry_price * (1.0 - pt_mult)
                stop_loss_price = entry_price * (1.0 + sl_mult)

            holding_bars = self.config.holding_period_bars
            if "holding_period" in events.columns:
                hp_val = events.loc[t0, "holding_period"]
                if isinstance(hp_val, pd.Series):
                    hp_val = hp_val.iloc[0]
                if pd.notna(hp_val) and int(hp_val) > 0:
                    holding_bars = int(hp_val)

            end_idx = min(len(close_idx) - 1, t0_idx + holding_bars)

            touch_time = close_idx[end_idx]
            touch_type = "vertical"
            touch_price = float(close.iloc[end_idx])

            for step_idx in range(t0_idx + 1, end_idx + 1):
                step_high = float(high_series.iloc[step_idx])
                step_low = float(low_series.iloc[step_idx])
                step_time = close_idx[step_idx]

                if side > 0:
                    pt_hit = step_high >= profit_take_price
                    sl_hit = step_low <= stop_loss_price
                else:
                    pt_hit = step_low <= profit_take_price
                    sl_hit = step_high >= stop_loss_price

                if pt_hit and sl_hit:
                    if self.config.conservative_touch:
                        touch_type = "sl"
                        touch_price = stop_loss_price
                    else:
                        touch_type = "pt"
                        touch_price = profit_take_price
                    touch_time = step_time
                    break
                if pt_hit:
                    touch_type = "pt"
                    touch_price = profit_take_price
                    touch_time = step_time
                    break
                if sl_hit:
                    touch_type = "sl"
                    touch_price = stop_loss_price
                    touch_time = step_time
                    break

            price_ret = (touch_price - entry_price) / entry_price
            trade_ret = price_ret * side

            if touch_type == "pt":
                label = 1
            elif touch_type == "sl":
                label = 0
            else:
                label = 1 if (not self.config.strict_pt and trade_ret > 0) else 0

            results.append(
                {
                    "t0": t0,
                    "t1": touch_time,
                    "ret": trade_ret,
                    "label": int(label),
                    "touch_type": touch_type,
                    "side": int(side),
                    "barrier_pt": profit_take_price,
                    "barrier_sl": stop_loss_price,
                }
            )

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
        """
        if len(events) == 0:
            return pd.Series(dtype=np.float64)

        time_to_idx = {ts: idx for idx, ts in enumerate(price_index)}
        concurrency = np.zeros(len(price_index), dtype=np.int32)
        spans: list[tuple[int, int] | None] = []

        for t0, t1 in zip(events.index, events["t1"].values, strict=False):
            if t0 in time_to_idx and t1 in time_to_idx:
                i0 = time_to_idx[t0]
                i1 = time_to_idx[t1]
                if i1 >= i0:
                    concurrency[i0 : i1 + 1] += 1
                    spans.append((i0, i1))
                    continue
            spans.append(None)

        uniqueness_vals = np.ones(len(events), dtype=np.float64)
        for row_pos, span in enumerate(spans):
            if span is not None:
                i0, i1 = span
                event_concurrency = concurrency[i0 : i1 + 1]
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
        and return attribution (AFML Ch. 4):
        w_i = u_i * |r_i| normalized such that mean(w) == 1.0
        """
        uniqueness = TripleBarrierLabeler.compute_sample_uniqueness(events, price_index)
        if len(uniqueness) == 0:
            return pd.Series(dtype=np.float64)

        weights = uniqueness.copy()
        if attribute_returns and "ret" in events.columns:
            rets = events["ret"].abs().values
            weights = pd.Series(weights.values * rets, index=events.index)

        mean_w = float(weights.mean())
        if mean_w > 1e-12:
            weights = weights / mean_w
        else:
            weights = pd.Series(1.0, index=events.index)

        return weights
