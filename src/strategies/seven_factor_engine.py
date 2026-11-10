"""
7-Factor Causal Online Alpha Engine with 72h BTC Beta Neutralization & Sub-Hourly Pullback.

Implements the empirical discoveries from the 22-month (2025-2026) research:
1. Rolling 72h OLS Bitcoin Beta decomposition (idiosyncratic residual returns).
2. Seven Orthogonal Cross-Sectional Alpha Factors:
   - F1 (macro): 168h/336h/720h Idiosyncratic & Raw Trend + Pullback (4h/24h)
   - F2 (med):   72h/168h Idiosyncratic Momentum + 12h Pullback
   - F3 (fast):  12h/24h/48h Vol-Surge Swing Momentum + Sub-Hourly/4h Pullback
   - F4 (rev):   24h/72h Cross-Sectional Mean Reversion
   - F5 (carry): 72h Contrarian Perpetual Funding Carry
   - F6 (liq):   168h vs 720h Institutional Dollar-Volume Expansion
   - F7 (size):  168h Institutional Dollar-Volume Quality
3. Causal Online Factor Momentum Meta-Allocator (trailing 36h, 96h, 240h net t-statistics
   with structural priors and ReLU^2 activation).
4. Strictly Dollar-Neutral Tail-Power Portfolio Construction (power_factor=1.7, power_comb=1.3)
   with 72h Portfolio Realized Volatility Targeting (clip[0.55, 1.15]).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


FACTOR_NAMES: Tuple[str, ...] = (
    "macro",
    "med",
    "fast",
    "rev",
    "carry",
    "liq",
    "size",
)

FACTOR_LABELS: Dict[str, str] = {
    "macro": "F1_Macro_Idio_PB (7d-30d)",
    "med": "F2_Med_Idio_PB (3d-7d)",
    "fast": "F3_Fast_Swing_PB (12h-48h)",
    "rev": "F4_Rev_24_72h (MeanRev)",
    "carry": "F5_Carry_Neg (72h Fund)",
    "liq": "F6_Liq_Surge (Vol Exp)",
    "size": "F7_HighLiq_Size (Quality)",
}

DEFAULT_PRIORS: Dict[str, float] = {
    "macro": 0.25,
    "med": 0.14,
    "fast": 0.19,
    "rev": 0.16,
    "carry": 0.18,
    "liq": 0.04,
    "size": 0.22,
}

ALT_SECTOR_CLUSTERS: Dict[str, Tuple[str, ...]] = {
    "AI_COMPUTE": ("FETUSDT", "RENDERUSDT"),
    "FAST_L1": ("SOLUSDT", "SUIUSDT", "AVAXUSDT"),
    "ALT_L1": ("NEARUSDT", "APTUSDT"),
    "PAY_MEME": ("XRPUSDT", "DOGEUSDT", "ADAUSDT"),
    "INFRA_L2": ("LINKUSDT", "ARBUSDT"),
}


@dataclass
class SevenFactorAllocationResult:
    """Output snapshot of the 7-Factor Causal Online Engine at a rebalance timestamp."""

    unit_weights: Dict[str, float]
    vol_scalar: float
    port_vol_72h: float
    median_vol: float
    conviction: float
    factor_weights: Dict[str, float]
    factor_tstats: Dict[str, float]
    asset_betas: Dict[str, float]
    portfolio_beta: float
    asset_composite_scores: Dict[str, float] = field(default_factory=dict)
    cmf_tstat: float = 0.0
    cmf_blend_weight: float = 0.0


def make_continuous_neutral_weights(
    score_df: pd.DataFrame,
    inv_vol_df: pd.DataFrame,
    gross_leverage: float = 1.0,
    power: float = 1.7,
) -> pd.DataFrame:
    """
    Cross-sectionally demeans and standardizes scores per row, raises to `power`
    to concentrate capital in high-conviction tails, weights by inverse volatility,
    and enforces strict 100% dollar neutrality:
        sum(longs) = +gross_leverage / 2
        sum(shorts) = -gross_leverage / 2
        net_exposure = 0.0 at every timestamp.
    """
    mean_s = score_df.mean(axis=1)
    std_s = score_df.std(axis=1).replace(0, 1.0).fillna(1.0)
    z = score_df.sub(mean_s, axis=0).div(std_s, axis=0).clip(-3.5, 3.5).fillna(0.0)

    tilted = np.sign(z) * (np.abs(z) ** power) * inv_vol_df

    pos = tilted.clip(lower=0.0)
    neg = (-tilted).clip(lower=0.0)

    pos_sum = pos.sum(axis=1)
    neg_sum = neg.sum(axis=1)
    valid = (pos_sum > 1e-10) & (neg_sum > 1e-10)

    wl = pos.div(pos_sum.replace(0, 1.0), axis=0) * (gross_leverage / 2.0)
    ws = neg.div(neg_sum.replace(0, 1.0), axis=0) * (gross_leverage / 2.0)

    return (wl - ws).mul(valid.astype(float), axis=0)


def cap_side_weights_waterfill(
    w_df: pd.DataFrame,
    max_side_share: Optional[float] = 0.30,
    max_cluster_share: Optional[float] = None,
    sector_clusters: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> pd.DataFrame:
    """
    Caps each asset's share of the Long leg (and Short leg) at `max_side_share` (e.g. 30%)
    and each correlated altcoin sector cluster at `max_cluster_share` (e.g. 48%),
    iteratively redistributing clipped excess proportionally across remaining unclipped
    assets on the same leg (water-filling), preserving strict 100% dollar neutrality.
    """
    has_side_cap = max_side_share is not None and 0.0 < max_side_share < 1.0
    has_clust_cap = max_cluster_share is not None and 0.0 < max_cluster_share < 1.0
    if not has_side_cap and not has_clust_cap:
        return w_df

    clusters_map = sector_clusters if sector_clusters is not None else ALT_SECTOR_CLUSTERS
    clean_cols = [str(c).split(":")[0].replace("/", "") for c in w_df.columns]
    cluster_indices: List[List[int]] = []
    if has_clust_cap and clusters_map:
        for syms in clusters_map.values():
            c_idx = [clean_cols.index(s) for s in syms if s in clean_cols]
            if len(c_idx) >= 2:
                cluster_indices.append(c_idx)

    arr = w_df.values.copy()
    n_rows, _ = arr.shape

    def _waterfill_leg(vec: np.ndarray, total_leg: float) -> np.ndarray:
        if total_leg <= 1e-10:
            return vec
        cap_single = total_leg * float(max_side_share) if has_side_cap else total_leg
        cap_clust = total_leg * float(max_cluster_share) if has_clust_cap else total_leg
        active_cnt = int((vec > 1e-10).sum())

        for _ in range(5):
            changed = False
            if has_side_cap and active_cnt >= 4:
                over = vec > cap_single + 1e-9
                if np.any(over) and not np.all(over):
                    excess = float((vec[over] - cap_single).sum())
                    vec[over] = cap_single
                    under = (~over) & (vec > 1e-10)
                    if np.any(under):
                        vec[under] += excess * (vec[under] / float(vec[under].sum()))
                        changed = True

            if has_clust_cap and cluster_indices:
                in_capped_clust = np.zeros(len(vec), dtype=bool)
                clust_excess = 0.0
                for c_idx in cluster_indices:
                    c_sum = float(vec[c_idx].sum())
                    if c_sum > cap_clust + 1e-9:
                        clust_excess += c_sum - cap_clust
                        vec[c_idx] *= cap_clust / c_sum
                        in_capped_clust[c_idx] = True
                        changed = True
                if clust_excess > 1e-9:
                    under_c = (~in_capped_clust) & (vec > 1e-10) & (vec < cap_single - 1e-6)
                    if np.any(under_c):
                        vec[under_c] += clust_excess * (vec[under_c] / float(vec[under_c].sum()))

            if not changed:
                break

        if has_side_cap and active_cnt >= 4:
            vec = np.minimum(vec, cap_single)
        return vec

    for i in range(n_rows):
        row = arr[i]
        pos = np.maximum(row, 0.0)
        neg = np.maximum(-row, 0.0)
        p_sum = float(pos.sum())
        n_sum = float(neg.sum())
        if p_sum > 1e-10:
            pos = _waterfill_leg(pos, p_sum)
        if n_sum > 1e-10:
            neg = _waterfill_leg(neg, n_sum)
        leg = min(float(pos.sum()), float(neg.sum()))
        if leg > 1e-10:
            pos = pos * (leg / float(pos.sum()))
            neg = neg * (leg / float(neg.sum()))
            arr[i] = pos - neg
        else:
            arr[i] = 0.0
    return pd.DataFrame(arr, index=w_df.index, columns=w_df.columns)


class SevenFactorOnlineEngine:
    """
    Vectorized Causal Online 7-Factor Cross-Sectional Engine.
    """

    def __init__(
        self,
        lookbacks: Tuple[int, int, int] = (36, 96, 240),
        priors: Optional[Dict[str, float]] = None,
        factor_power: float = 1.7,
        comb_power: float = 1.3,
        vol_scalar_bounds: Tuple[float, float] = (0.55, 1.15),
        conviction_bounds: Tuple[float, float] = (0.35, 1.15),
        factor_turnover_penalty: float = 0.0001,
        comb_ema_span: Optional[float] = 3.0,
        trend_waterfill_cap: Optional[float] = 0.30,
        trend_waterfill_threshold: float = 0.20,
        alt_cluster_cap: Optional[float] = 0.48,
        f6_min_tstat: Optional[float] = 0.75,
        min_tstat_by_factor: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lookbacks = lookbacks
        self.priors = pd.Series(priors or DEFAULT_PRIORS)
        self.factor_power = factor_power
        self.comb_power = comb_power
        self.vol_scalar_bounds = vol_scalar_bounds
        self.conviction_bounds = conviction_bounds
        self.factor_turnover_penalty = factor_turnover_penalty
        self.comb_ema_span = comb_ema_span
        self.trend_waterfill_cap = trend_waterfill_cap
        self.trend_waterfill_threshold = trend_waterfill_threshold
        self.alt_cluster_cap = alt_cluster_cap
        self.f6_min_tstat = f6_min_tstat
        self.min_tstat_by_factor: Dict[str, float] = dict(min_tstat_by_factor or {})
        if f6_min_tstat is not None and "liq" not in self.min_tstat_by_factor:
            self.min_tstat_by_factor["liq"] = float(f6_min_tstat)

    @staticmethod
    def _find_btc_column(columns: List[str]) -> Optional[str]:
        for c in columns:
            if c.startswith("BTC/") or c == "BTCUSDT":
                return c
        return columns[0] if columns else None

    def _build_aligned_matrices(
        self,
        bars_1h: Dict[str, pd.DataFrame],
        funding_rates: Optional[Dict[str, Union[float, pd.Series, pd.DataFrame]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aligns symbol DataFrames onto a common UTC hourly DatetimeIndex.
        Returns (close, volume, dollar_vol, raw_funding, funding_per_bar).
        """
        close_dict: Dict[str, pd.Series] = {}
        vol_dict: Dict[str, pd.Series] = {}

        for sym, df in bars_1h.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            if "timestamp" in d.columns:
                idx = pd.to_datetime(d["timestamp"], utc=True)
                d = d.set_index(idx)
            elif not isinstance(d.index, pd.DatetimeIndex):
                d.index = pd.to_datetime(d.index, utc=True)
            elif d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            d = d[~d.index.duplicated(keep="last")].sort_index()
            close_dict[sym] = d["close"].astype(float)
            vol_dict[sym] = d["volume"].astype(float)

        if not close_dict:
            raise ValueError("No valid 1h bar series provided to SevenFactorOnlineEngine.")

        close = pd.DataFrame(close_dict).sort_index().ffill().bfill()
        volume = pd.DataFrame(vol_dict).reindex(close.index).fillna(0.0)
        dollar_vol = close * volume

        raw_funding = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        funding_per_bar = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        if funding_rates:
            for sym in close.columns:
                fr_val = funding_rates.get(sym)
                if fr_val is None:
                    continue
                if isinstance(fr_val, (int, float, np.floating)):
                    raw_funding[sym] = float(fr_val)
                    # Funding settles every 8h (00:00, 08:00, 16:00 UTC)
                    is_settlement = (close.index.hour % 8 == 0)
                    funding_per_bar.loc[is_settlement, sym] = float(fr_val)
                elif isinstance(fr_val, pd.Series):
                    s = fr_val.copy()
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index, utc=True)
                    elif s.index.tz is None:
                        s.index = s.index.tz_localize("UTC")
                    s = s[~s.index.duplicated(keep="last")].sort_index()
                    reindexed = s.reindex(close.index)
                    funding_per_bar[sym] = reindexed.fillna(0.0)
                    raw_funding[sym] = reindexed.ffill().fillna(0.0)
                elif isinstance(fr_val, pd.DataFrame) and "fundingRate" in fr_val.columns:
                    d_f = fr_val.copy()
                    if "timestamp" in d_f.columns:
                        d_f = d_f.set_index(pd.to_datetime(d_f["timestamp"], utc=True))
                    d_f.index = d_f.index.round("1h")
                    d_f = d_f[~d_f.index.duplicated(keep="last")].sort_index()
                    reindexed = d_f["fundingRate"].astype(float).reindex(close.index)
                    funding_per_bar[sym] = reindexed.fillna(0.0)
                    raw_funding[sym] = reindexed.ffill().fillna(0.0)

        return close, volume, dollar_vol, raw_funding, funding_per_bar

    @staticmethod
    def _extract_high_low(
        bars_1h: Dict[str, pd.DataFrame],
        close: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts aligned hourly high and low matrices from `bars_1h`, falling back
        to `close` when synthetic test DataFrames only provide close and volume.
        """
        high_dict: Dict[str, pd.Series] = {}
        low_dict: Dict[str, pd.Series] = {}
        for sym, df in bars_1h.items():
            if df is None or df.empty or sym not in close.columns:
                continue
            d = df.copy()
            if "timestamp" in d.columns:
                idx = pd.to_datetime(d["timestamp"], utc=True)
                d = d.set_index(idx)
            elif not isinstance(d.index, pd.DatetimeIndex):
                d.index = pd.to_datetime(d.index, utc=True)
            elif d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            d = d[~d.index.duplicated(keep="last")].sort_index()
            if "high" in d.columns and "low" in d.columns:
                high_dict[sym] = d["high"].astype(float)
                low_dict[sym] = d["low"].astype(float)

        if not high_dict:
            return close.copy(), close.copy()

        high = pd.DataFrame(high_dict).reindex(close.index).ffill().bfill().fillna(close)
        low = pd.DataFrame(low_dict).reindex(close.index).ffill().bfill().fillna(close)
        for col in close.columns:
            if col not in high.columns:
                high[col] = close[col]
                low[col] = close[col]
        return high[close.columns], low[close.columns]

    @staticmethod
    def _extract_open_interest(
        open_interest: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]],
        close: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Aligns hourly Open Interest contract series (`sumOpenInterest` or `open_interest`)
        onto `close.index` for Issue #46 Squeeze/Knife Guard & Breakout Confirmation.
        """
        if not open_interest:
            return None
        oi_dict: Dict[str, pd.Series] = {}
        for sym in close.columns:
            oi_val = open_interest.get(sym)
            if oi_val is None:
                continue
            if isinstance(oi_val, pd.Series):
                s = oi_val.copy()
                if not isinstance(s.index, pd.DatetimeIndex):
                    s.index = pd.to_datetime(s.index, utc=True)
                elif s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")
                s = s[~s.index.duplicated(keep="last")].sort_index()
                oi_dict[sym] = s.astype(float)
            elif isinstance(oi_val, pd.DataFrame) and not oi_val.empty:
                d_o = oi_val.copy()
                if "timestamp" in d_o.columns:
                    ts = d_o["timestamp"]
                    if pd.api.types.is_numeric_dtype(ts):
                        unit = "ms" if float(ts.dropna().max()) > 1e11 else "s"
                        d_o = d_o.set_index(pd.to_datetime(ts, unit=unit, utc=True))
                    else:
                        d_o = d_o.set_index(pd.to_datetime(ts, utc=True))
                elif not isinstance(d_o.index, pd.DatetimeIndex):
                    d_o.index = pd.to_datetime(d_o.index, utc=True)
                elif d_o.index.tz is None:
                    d_o.index = d_o.index.tz_localize("UTC")
                d_o.index = d_o.index.round("1h")
                d_o = d_o[~d_o.index.duplicated(keep="last")].sort_index()
                col = "sumOpenInterest" if "sumOpenInterest" in d_o.columns else (
                    "open_interest" if "open_interest" in d_o.columns else d_o.columns[0]
                )
                oi_dict[sym] = d_o[col].astype(float)
        if not oi_dict:
            return None
        oi_df = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for sym, s in oi_dict.items():
            oi_df[sym] = s.reindex(close.index).ffill()
        return oi_df

    @staticmethod
    def _extract_taker_flow(
        taker_flow: Optional[Dict[str, pd.DataFrame]],
        bars_1h: Dict[str, pd.DataFrame],
        close: pd.DataFrame,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Aligns hourly `quote_volume` and `taker_buy_quote_volume` onto `close.index`
        for Issue #48 True Taker Buy/Sell CVD & Institutional Limit Absorption.
        """
        qvol_dict: Dict[str, pd.Series] = {}
        tbuy_dict: Dict[str, pd.Series] = {}
        source_map = taker_flow if taker_flow else bars_1h
        if not source_map:
            return None

        for sym in close.columns:
            df = source_map.get(sym)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "quote_volume" not in df.columns or "taker_buy_quote_volume" not in df.columns:
                continue
            d_t = df.copy()
            if "timestamp" in d_t.columns:
                ts = d_t["timestamp"]
                if pd.api.types.is_numeric_dtype(ts):
                    unit = "ms" if float(ts.dropna().max()) > 1e11 else "s"
                    d_t = d_t.set_index(pd.to_datetime(ts, unit=unit, utc=True))
                else:
                    d_t = d_t.set_index(pd.to_datetime(ts, utc=True))
            elif not isinstance(d_t.index, pd.DatetimeIndex):
                d_t.index = pd.to_datetime(d_t.index, utc=True)
            elif d_t.index.tz is None:
                d_t.index = d_t.index.tz_localize("UTC")
            d_t.index = d_t.index.round("1h")
            d_t = d_t[~d_t.index.duplicated(keep="last")].sort_index()
            qvol_dict[sym] = d_t["quote_volume"].astype(float)
            tbuy_dict[sym] = d_t["taker_buy_quote_volume"].astype(float)

        if not qvol_dict:
            return None

        qvol_df = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        tbuy_df = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for sym in qvol_dict:
            s_q = qvol_dict[sym].reindex(close.index)
            s_b = tbuy_dict[sym].reindex(close.index)
            if s_q.notna().sum() == 0:
                s_q = qvol_dict[sym].copy()
                s_b = tbuy_dict[sym].copy()
                s_q.index = s_q.index + pd.Timedelta(hours=1)
                s_b.index = s_b.index + pd.Timedelta(hours=1)
                s_q = s_q.reindex(close.index)
                s_b = s_b.reindex(close.index)
            qvol_df[sym] = s_q.ffill()
            tbuy_df[sym] = s_b.ffill()
        return qvol_df, tbuy_df

    @staticmethod
    def _extract_premium_index(
        premium_index: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]],
        close: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Aligns hourly Spot-Perp Premium Index (`close` or `premium_index`) onto `close.index`
        for Issue #49 Spot-Perp Basis Spread & Premium Index Alpha.
        """
        if not premium_index:
            return None
        prem_dict: Dict[str, pd.Series] = {}
        for sym in close.columns:
            p_val = premium_index.get(sym)
            if p_val is None:
                continue
            if isinstance(p_val, pd.Series):
                s = p_val.copy()
                if not isinstance(s.index, pd.DatetimeIndex):
                    s.index = pd.to_datetime(s.index, utc=True)
                elif s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")
                s = s[~s.index.duplicated(keep="last")].sort_index()
                prem_dict[sym] = s.astype(float)
            elif isinstance(p_val, pd.DataFrame) and not p_val.empty:
                d_p = p_val.copy()
                if "timestamp" in d_p.columns:
                    ts = d_p["timestamp"]
                    if pd.api.types.is_numeric_dtype(ts):
                        unit = "ms" if float(ts.dropna().max()) > 1e11 else "s"
                        d_p = d_p.set_index(pd.to_datetime(ts, unit=unit, utc=True))
                    else:
                        d_p = d_p.set_index(pd.to_datetime(ts, utc=True))
                elif not isinstance(d_p.index, pd.DatetimeIndex):
                    d_p.index = pd.to_datetime(d_p.index, utc=True)
                elif d_p.index.tz is None:
                    d_p.index = d_p.index.tz_localize("UTC")
                d_p.index = d_p.index.round("1h")
                d_p = d_p[~d_p.index.duplicated(keep="last")].sort_index()
                col = "close" if "close" in d_p.columns else (
                    "premium_index" if "premium_index" in d_p.columns else d_p.columns[0]
                )
                prem_dict[sym] = d_p[col].astype(float)
        if not prem_dict:
            return None
        prem_df = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        for sym, s in prem_dict.items():
            s_re = s.reindex(close.index)
            if s_re.notna().sum() == 0:
                s_shift = s.copy()
                s_shift.index = s_shift.index + pd.Timedelta(hours=1)
                s_re = s_shift.reindex(close.index)
            prem_df[sym] = s_re.ffill().fillna(0.0)
        return prem_df

    def _compute_subhourly_pullback_adjustment(
        self,
        bars_15m: Optional[Dict[str, pd.DataFrame]],
        symbols: List[str],
        vol_24h_latest: pd.Series,
    ) -> pd.Series:
        """
        Computes high-resolution sub-hourly pullback dip adjustment from 15m klines
        (30m and 2h volatility-normalized returns + 1h extreme shock filter)
        for the latest rebalance bar.
        """
        adj = pd.Series(0.0, index=symbols)
        if not bars_15m:
            return adj

        for sym in symbols:
            df15 = bars_15m.get(sym)
            if df15 is None or len(df15) < 9:
                continue
            closes = df15["close"].astype(float).values
            v24 = max(float(vol_24h_latest.get(sym, 0.02)), 1e-4)
            # 30m return = 2 bars of 15m, 1h = 4 bars, 2h = 8 bars
            r_30m = np.log(closes[-1] / max(closes[-3], 1e-12)) / (v24 * np.sqrt(0.5))
            r_1h = np.log(closes[-1] / max(closes[-5], 1e-12)) / (v24 * 1.0)
            r_2h = np.log(closes[-1] / max(closes[-9], 1e-12)) / (v24 * np.sqrt(2.0))
            shock_1h = r_1h if abs(r_1h) > 2.0 else 0.0
            # Negative of short-term spike = positive entry on controlled pullback dip
            adj[sym] = -float(0.05 * r_30m + 0.08 * r_2h + 0.08 * shock_1h)

        return adj

    def compute_matrices(
        self,
        bars_1h: Dict[str, pd.DataFrame],
        funding_rates: Optional[Dict[str, Union[float, pd.Series, pd.DataFrame]]] = None,
        bars_15m: Optional[Dict[str, pd.DataFrame]] = None,
        open_interest: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
        taker_flow: Optional[Dict[str, pd.DataFrame]] = None,
        premium_index: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Computes full historical matrices for the 7 factors, online factor weights,
        unit portfolio weights, and 72h volatility scalar.
        """
        close, volume, dollar_vol, raw_funding, funding = self._build_aligned_matrices(
            bars_1h, funding_rates
        )
        high, low = self._extract_high_low(bars_1h, close)
        oi_contracts = self._extract_open_interest(open_interest, close)
        taker_pair = self._extract_taker_flow(taker_flow, bars_1h, close)
        prem_close = self._extract_premium_index(premium_index, close)
        idx = close.index
        cols = list(close.columns)

        # Institutional Close Location Value (CLV) & Chaikin Money Flow (CMF) over 24h, 72h, 168h
        hl_span = (high - low).replace(0, np.nan).ffill().fillna(close * 0.005).replace(0, 1e-8)
        clv = ((2.0 * close - high - low) / hl_span).clip(-1.0, 1.0).fillna(0.0)
        cmf_24 = (clv * dollar_vol).rolling(24, min_periods=4).sum() / (
            dollar_vol.rolling(24, min_periods=4).sum() + 1e-8
        )
        cmf_72 = (clv * dollar_vol).rolling(72, min_periods=12).sum() / (
            dollar_vol.rolling(72, min_periods=12).sum() + 1e-8
        )
        cmf_168 = (clv * dollar_vol).rolling(168, min_periods=24).sum() / (
            dollar_vol.rolling(168, min_periods=24).sum() + 1e-8
        )

        rets_1h = close.pct_change().fillna(0.0)
        log_rets = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        vol_24h = rets_1h.rolling(24, min_periods=4).std().replace(0, np.nan).ffill().fillna(0.02)
        inv_vol = (1.0 / vol_24h).clip(upper=100.0)

        vol_surge = (
            dollar_vol.rolling(6, min_periods=1).mean()
            / (dollar_vol.rolling(48, min_periods=6).mean() + 1e-8)
        ).clip(0.5, 2.5)
        vol_surge_wk = (
            dollar_vol.rolling(24, min_periods=4).mean()
            / (dollar_vol.rolling(168, min_periods=24).mean() + 1e-8)
        ).clip(0.5, 2.5)

        btc_col = self._find_btc_column(cols)
        btc_ret = rets_1h[btc_col] if btc_col else rets_1h.mean(axis=1)
        btc_var = btc_ret.rolling(72, min_periods=12).var().replace(0, 1e-8)
        beta_72 = (
            rets_1h.rolling(72, min_periods=12)
            .cov(btc_ret)
            .div(btc_var, axis=0)
            .clip(0.2, 2.5)
            .fillna(1.0)
        )
        resid_ret = rets_1h.sub(beta_72.mul(btc_ret, axis=0))

        def z_h(s: pd.DataFrame, h: int) -> pd.DataFrame:
            min_p = max(2, min(h, 12))
            return (s.rolling(h, min_periods=min_p).sum() / (vol_24h * np.sqrt(h))).fillna(0.0)

        horizons = [4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720]
        zr = {h: z_h(log_rets, h) for h in horizons}
        zi = {h: z_h(resid_ret, h) for h in horizons}

        # 7 Orthogonal Cross-Sectional Factors
        f_macro = 0.70 * (
            (0.35 * zi[168] + 0.40 * zi[336] + 0.25 * zi[720]) * np.sqrt(vol_surge_wk)
            - (0.15 * zi[4] + 0.20 * zi[24])
        ) + 0.30 * (
            (0.35 * zr[168] + 0.40 * zr[336] + 0.25 * zr[720]) * np.sqrt(vol_surge_wk)
            - (0.15 * zr[4] + 0.20 * zr[24])
        )

        f_med = (0.40 * zi[72] + 0.60 * zi[168]) * np.sqrt(vol_surge_wk) - 0.25 * zi[12]

        f_fast = 0.60 * (
            (0.35 * zr[12] + 0.45 * zr[24] + 0.20 * zr[48]) * np.sqrt(vol_surge)
            - 0.25 * zr[4]
        ) + 0.40 * (
            (0.30 * zi[12] + 0.40 * zi[24] + 0.30 * zi[72]) * np.sqrt(vol_surge)
            - 0.20 * zi[4]
        )

        # Incorporate sub-hourly 15m pullback precision into latest bar of f_fast if provided
        if bars_15m and len(f_fast) > 0:
            sub_adj = self._compute_subhourly_pullback_adjustment(
                bars_15m, cols, vol_24h.iloc[-1]
            )
            f_fast.iloc[-1] = f_fast.iloc[-1] + sub_adj

        f_rev = -(0.40 * zi[24] + 0.60 * zi[72]) - 0.25 * zr[4]

        ann_fund = raw_funding.rolling(72, min_periods=1).mean() * 3.0 * 365.0
        f_carry = -ann_fund.where(ann_fund.abs() >= 0.05, 0.0)

        f_liq = np.log(dollar_vol.rolling(168, min_periods=12).mean() + 1.0) - np.log(
            dollar_vol.rolling(720, min_periods=24).mean() + 1.0
        )
        f_size = np.log(dollar_vol.rolling(168, min_periods=12).mean() + 1.0)

        factor_scores: Dict[str, pd.DataFrame] = {
            "macro": f_macro,
            "med": f_med,
            "fast": f_fast,
            "rev": f_rev,
            "carry": f_carry,
            "liq": f_liq,
            "size": f_size,
        }

        w_unit = {
            k: make_continuous_neutral_weights(
                sc, inv_vol, gross_leverage=1.0, power=self.factor_power
            )
            for k, sc in factor_scores.items()
        }

        f_bar_ret = {}
        for k, w in w_unit.items():
            w_exec = w.shift(1).fillna(0.0)
            turn = w_exec.diff().abs().sum(axis=1).fillna(0.0)
            f_bar_ret[k] = (
                (w_exec * rets_1h).sum(axis=1)
                - (w_exec * funding).sum(axis=1)
                - turn * self.factor_turnover_penalty
            )
        df_frets = pd.DataFrame(f_bar_ret, index=idx)

        t_df = pd.DataFrame(0.0, index=idx, columns=df_frets.columns)
        for lb in self.lookbacks:
            min_p = max(6, lb // 4)
            m = df_frets.rolling(lb, min_periods=min_p).mean()
            s = df_frets.rolling(lb, min_periods=min_p).std().replace(0, 1e-8)
            t_df += (m / s * np.sqrt(lb)).fillna(0.0) / float(len(self.lookbacks))

        # Retoque 1A: Selective Factor Weight Smoothing on non-trend factors (rev=12h, liq=16h, size=6h)
        # Prevents single-bar 36h rolling window spikes in F6_Liq / F7_Size from flipping the portfolio
        # during brief 3h counter-trend dips, while keeping macro/med/fast trend breakouts 100% reactive.
        t_df_raw = t_df.copy()
        if "rev" in t_df.columns:
            t_df["rev"] = t_df["rev"].ewm(span=12, min_periods=1).mean()
        if "liq" in t_df.columns:
            t_df["liq"] = t_df["liq"].ewm(span=16, min_periods=1).mean()
        if "size" in t_df.columns:
            t_df["size"] = t_df["size"].ewm(span=6, min_periods=1).mean()

        priors_aligned = self.priors.reindex(t_df.columns).fillna(0.10)
        raw_fw = np.maximum(0.0, t_df.add(priors_aligned, axis=1)) ** 2.0

        # Lever #1 (Issue #52): High-Conviction t-stat Activation Gate on F6_Liq_Surge (t_stat >= +0.75)
        for f_name, min_t in self.min_tstat_by_factor.items():
            if f_name in raw_fw.columns and min_t is not None:
                gate_mask = (t_df_raw[f_name] >= float(min_t)) & (t_df[f_name] >= float(min_t))
                raw_fw[f_name] = raw_fw[f_name].where(gate_mask, 0.0)

        if "rev" in raw_fw.columns:
            raw_fw["rev"] = raw_fw["rev"].ewm(span=12, min_periods=1).mean()
        if "liq" in raw_fw.columns:
            raw_fw["liq"] = raw_fw["liq"].ewm(span=16, min_periods=1).mean()
        if "size" in raw_fw.columns:
            raw_fw["size"] = raw_fw["size"].ewm(span=6, min_periods=1).mean()

        for f_name, min_t in self.min_tstat_by_factor.items():
            if f_name in raw_fw.columns and min_t is not None:
                gate_mask = (t_df_raw[f_name] >= float(min_t)) & (t_df[f_name] >= float(min_t))
                raw_fw[f_name] = raw_fw[f_name].where(gate_mask, 0.0)

        # Fallback to un-gated priors if all factors are simultaneously gated to 0 on a bar
        zero_rows = raw_fw.sum(axis=1) <= 1e-12
        if zero_rows.any():
            fallback_fw = pd.DataFrame(
                np.tile((priors_aligned.values ** 2.0), (int(zero_rows.sum()), 1)),
                index=raw_fw.index[zero_rows],
                columns=raw_fw.columns,
            )
            for f_name, min_t in self.min_tstat_by_factor.items():
                if f_name in fallback_fw.columns and min_t is not None:
                    fallback_fw[f_name] = 0.0
            if (fallback_fw.sum(axis=1) <= 1e-12).any():
                fallback_fw = pd.DataFrame(
                    np.tile((priors_aligned.values ** 2.0), (int(zero_rows.sum()), 1)),
                    index=raw_fw.index[zero_rows],
                    columns=raw_fw.columns,
                )
            raw_fw.loc[zero_rows] = fallback_fw

        fw_sum = raw_fw.sum(axis=1).replace(0, 1e-8)
        fw_norm = raw_fw.div(fw_sum, axis=0)

        conviction = (raw_fw.sum(axis=1) / (float(priors_aligned.sum()) ** 2.0)).clip(
            self.conviction_bounds[0], self.conviction_bounds[1]
        )

        comb_w = pd.DataFrame(0.0, index=idx, columns=cols)
        for k in FACTOR_NAMES:
            comb_w += w_unit[k].mul(fw_norm[k], axis=0)

        # Mejora 2: Targeted Falling-Knife & Parabolic Squeeze Guard when F4_Rev dominates (> 48% weight)
        rev_heavy = fw_norm["rev"] > 0.48
        is_knife = (zi[168] < -1.30) & (zr[168] < -1.30) & (zi[72] < -0.50) & (cmf_24 < 0.0)
        is_rocket = (zi[168] > 0.65) & (zr[168] > 0.85) & (zi[24] > 1.25) & (cmf_24 > 0.05)
        comb_w = comb_w.where(~(rev_heavy.values[:, None] & (comb_w > 0.0) & is_knife), 0.15 * comb_w)
        comb_w = comb_w.where(~(rev_heavy.values[:, None] & (comb_w < 0.0) & is_rocket), 0.35 * comb_w)

        # Mejora 3: Dynamic Online F8_OrderFlow_CMF (Chaikin Accumulation/Distribution) Overlay
        f_cmf_score = 0.50 * cmf_24 + 0.30 * cmf_72 + 0.20 * cmf_168
        w_cmf = make_continuous_neutral_weights(
            f_cmf_score, inv_vol, gross_leverage=1.0, power=self.factor_power
        )
        w_cmf_exec = w_cmf.shift(1).fillna(0.0)
        cmf_turn = w_cmf_exec.diff().abs().sum(axis=1).fillna(0.0)
        cmf_ret = (
            (w_cmf_exec * rets_1h).sum(axis=1)
            - (w_cmf_exec * funding).sum(axis=1)
            - cmf_turn * self.factor_turnover_penalty
        )
        t_cmf = pd.Series(0.0, index=idx)
        for lb in self.lookbacks:
            min_p = max(6, lb // 4)
            m_c = cmf_ret.rolling(lb, min_periods=min_p).mean()
            s_c = cmf_ret.rolling(lb, min_periods=min_p).std().replace(0, 1e-8)
            t_cmf += (m_c / s_c * np.sqrt(lb)).fillna(0.0) / float(len(self.lookbacks))

        alpha_cmf_dyn = ((t_cmf - 0.45).clip(lower=0.0) * 0.08).clip(upper=0.10)
        comb_w = comb_w.mul(1.0 - alpha_cmf_dyn, axis=0) + w_cmf.mul(alpha_cmf_dyn, axis=0)

        # Retoque 1B: Sudden Rev-Exit Hysteresis
        # When F4_Rev was dominant 4h ago (> 45% weight) and drops sharply (< 25%) without a confirmed
        # macro trend (t_df_raw["macro"] < 0.25), blend 55% of the prior 4h signal and cap conviction
        # to prevent selling the bottom of an intraday washout.
        if len(comb_w) > 4 and "rev" in fw_norm.columns and "macro" in t_df_raw.columns:
            rev_drop = (
                (fw_norm["rev"].shift(4).fillna(0.0) > 0.45)
                & (fw_norm["rev"] < 0.25)
                & (t_df_raw["macro"] < 0.25)
            )
            if rev_drop.any():
                comb_h = 0.55 * comb_w.shift(4).fillna(0.0) + 0.45 * comb_w
                comb_w = comb_w.copy()
                comb_w.loc[rev_drop] = comb_h.loc[rev_drop]
                conviction = conviction.where(
                    ~rev_drop, np.minimum(conviction, conviction.shift(4).fillna(0.50))
                )

        # Issue #46 (M13 Ola 2): Real-Time Open Interest (OI) Squeeze/Knife Guard & Confirmation
        if oi_contracts is not None:
            oi_chg_4h = oi_contracts.pct_change(4, fill_method=None).fillna(0.0)
            oi_chg_12h = oi_contracts.pct_change(12, fill_method=None).fillna(0.0)
            oi_chg_24h = oi_contracts.pct_change(24, fill_method=None).fillna(0.0)

            sq_short = (
                ((oi_chg_4h >= 0.06) | (oi_chg_12h >= 0.10))
                & (ann_fund < -0.02)
                & (zr[4] >= 0.80)
            )
            fl_long = (
                (fw_norm["rev"].values[:, None] > 0.35)
                & (oi_chg_24h >= 0.06)
                & (zr[12] <= -1.40)
                & (cmf_24 < -0.05)
            )
            oi_confirm = (
                (oi_chg_12h >= 0.04)
                & (zi[24] >= 0.75)
                & (cmf_24 > 0.05)
            )
            oi_flush_bounce = (
                (oi_chg_12h <= -0.05)
                & (zi[24] <= -1.20)
                & (cmf_24 > 0.02)
            )
            comb_w = (
                comb_w
                .where(~((comb_w < 0.0) & sq_short), 0.25 * comb_w)
                .where(~((comb_w > 0.0) & fl_long), 0.25 * comb_w)
                .where(~((comb_w > 0.0) & oi_confirm), 1.15 * comb_w)
                .where(~((comb_w > 0.0) & oi_flush_bounce), 1.15 * comb_w)
            )

        # Issues #48 & #49 (M13 Ola 3): True Taker Buy/Sell CVD Iceberg Absorption & Spot-Perp Basis Alpha
        ask_distrib = pd.DataFrame(False, index=idx, columns=cols)
        bid_absorb = pd.DataFrame(False, index=idx, columns=cols)
        taker_cvd_24 = pd.DataFrame(0.0, index=idx, columns=cols)
        has_cvd = taker_pair is not None
        if has_cvd:
            qvol_df, tbuy_df = taker_pair
            net_taker_usd = (2.0 * tbuy_df - qvol_df).fillna(0.0)
            tot_qvol_usd = qvol_df.fillna(dollar_vol)
            taker_cvd_24 = (
                net_taker_usd.rolling(24, min_periods=4).sum()
                / (tot_qvol_usd.rolling(24, min_periods=4).sum() + 1e-8)
            ).clip(-1.0, 1.0).fillna(0.0)
            limit_abs_24 = (cmf_24 - taker_cvd_24).fillna(0.0)
            ask_distrib = (
                (limit_abs_24 <= -0.10)
                & (taker_cvd_24 >= 0.015)
                & (zi[12] <= -0.20)
            )
            bid_absorb = (
                (limit_abs_24 >= 0.12)
                & (taker_cvd_24 <= -0.020)
                & (zi[12] >= 0.30)
            )

        perp_froth = pd.DataFrame(False, index=idx, columns=cols)
        spot_led_short_guard = pd.DataFrame(False, index=idx, columns=cols)
        spot_led_long_confirm = pd.DataFrame(False, index=idx, columns=cols)
        has_basis = prem_close is not None
        if has_basis:
            prem_12h = prem_close.rolling(12, min_periods=3).mean().fillna(0.0)
            prem_24h = prem_close.rolling(24, min_periods=4).mean().fillna(0.0)
            prem_168h = prem_close.rolling(168, min_periods=24).mean().fillna(0.0)
            prem_std_168 = (
                prem_close.rolling(168, min_periods=24).std().replace(0, 1e-4).fillna(1e-4)
            )
            prem_z12 = ((prem_12h - prem_168h) / prem_std_168).clip(-3.5, 3.5).fillna(0.0)
            prem_z24 = ((prem_24h - prem_168h) / prem_std_168).clip(-3.5, 3.5).fillna(0.0)
            perp_froth = (
                (prem_z12 >= 0.60)
                & (prem_12h >= 0.0001)
                & (zr[4] <= -0.50)
                & (cmf_24 < 0.0)
            )
            spot_led_short_guard = (
                (zi[24] >= 0.65)
                & (prem_24h <= -0.0003)
                & (prem_z24 <= -0.30)
                & (cmf_24 > 0.0)
            )
            spot_led_long_confirm = (
                (zi[24] >= 0.40)
                & (prem_24h <= -0.0003)
                & (prem_z24 <= -0.30)
                & (cmf_24 > 0.0)
            )

        if has_cvd and not has_basis:
            comb_w = (
                comb_w
                .where(~((comb_w > 0.0) & ask_distrib), 0.55 * comb_w)
                .where(~((comb_w < 0.0) & bid_absorb), 0.60 * comb_w)
            )
        elif has_basis and not has_cvd:
            comb_w = (
                comb_w
                .where(~((comb_w > 0.0) & perp_froth), 0.55 * comb_w)
                .where(~((comb_w < 0.0) & spot_led_short_guard), 0.55 * comb_w)
            )
        elif has_cvd and has_basis:
            comb_w = (
                comb_w
                .where(~((comb_w > 0.0) & (ask_distrib | perp_froth)), 0.55 * comb_w)
                .where(~((comb_w < 0.0) & (bid_absorb | spot_led_short_guard)), 0.60 * comb_w)
                .where(
                    ~((comb_w > 0.0) & spot_led_long_confirm & (taker_cvd_24 > 0.0)),
                    1.06 * comb_w,
                )
            )

        # Retoque 1C (Nov 1): Causal Cross-Sectional Composite Score Smoothing (EMA span=3h)
        # Prevents single-bar 24h return oscillations from flipping individual coins Long <-> Short
        # across consecutive 4h rebalances while preserving factor conviction.
        if self.comb_ema_span is not None and self.comb_ema_span > 1.0:
            comb_w = comb_w.ewm(span=self.comb_ema_span, min_periods=1).mean()

        ones_df = pd.DataFrame(1.0, index=idx, columns=cols)
        w_raw_1x = make_continuous_neutral_weights(
            comb_w, ones_df, gross_leverage=1.0, power=self.comb_power
        ).mul(conviction, axis=0)

        # Regime-Conditioned Water-Filling Concentration Cap (30% per asset when Trend >= 20%)
        # & Altcoin Sector Cluster Cap (48% per sector across all regimes — Issue #45).
        has_wf = self.trend_waterfill_cap is not None and 0.0 < self.trend_waterfill_cap < 1.0
        has_cl = self.alt_cluster_cap is not None and 0.0 < self.alt_cluster_cap < 1.0
        if has_wf or has_cl:
            w_wf_1x = cap_side_weights_waterfill(
                w_raw_1x,
                max_side_share=self.trend_waterfill_cap if has_wf else None,
                max_cluster_share=self.alt_cluster_cap if has_cl else None,
            )
            if (
                has_wf
                and self.trend_waterfill_threshold is not None
                and self.trend_waterfill_threshold > 0.0
                and all(k in fw_norm.columns for k in ("macro", "med", "fast"))
            ):
                if has_cl:
                    w_chop_1x = cap_side_weights_waterfill(
                        w_raw_1x,
                        max_side_share=None,
                        max_cluster_share=self.alt_cluster_cap,
                    )
                else:
                    w_chop_1x = w_raw_1x
                trend_share = fw_norm["macro"] + fw_norm["med"] + fw_norm["fast"]
                is_trend = (trend_share >= self.trend_waterfill_threshold).values[:, None]
                w_base_1x = pd.DataFrame(
                    np.where(is_trend, w_wf_1x.values, w_chop_1x.values),
                    index=idx,
                    columns=cols,
                )
            else:
                w_base_1x = w_wf_1x
        else:
            w_base_1x = w_raw_1x

        unit_ret_1h = (w_base_1x.shift(1).fillna(0.0) * rets_1h).sum(axis=1)
        port_vol_72h = (
            unit_ret_1h.rolling(72, min_periods=12).std().bfill().fillna(0.003)
        )
        median_vol = float(port_vol_72h.median()) if len(port_vol_72h) > 0 else 0.003
        if median_vol <= 1e-8 or np.isnan(median_vol):
            median_vol = 0.003

        vol_scalar = (
            (median_vol / port_vol_72h.replace(0, np.nan))
            .clip(self.vol_scalar_bounds[0], self.vol_scalar_bounds[1])
            .fillna(1.0)
        )

        return {
            "close": close,
            "rets_1h": rets_1h,
            "funding": funding,
            "w_base_1x": w_base_1x,
            "comb_w": comb_w,
            "fw_norm": fw_norm,
            "t_df": t_df,
            "conviction": conviction,
            "vol_scalar": vol_scalar,
            "port_vol_72h": port_vol_72h,
            "median_vol": pd.Series(median_vol, index=idx),
            "beta_72": beta_72,
            "cmf_24": cmf_24,
            "t_cmf": t_cmf,
            "alpha_cmf_dyn": alpha_cmf_dyn,
        }

    def allocate_latest(
        self,
        bars_1h: Dict[str, pd.DataFrame],
        funding_rates: Optional[Dict[str, Union[float, pd.Series, pd.DataFrame]]] = None,
        bars_15m: Optional[Dict[str, pd.DataFrame]] = None,
        open_interest: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
        taker_flow: Optional[Dict[str, pd.DataFrame]] = None,
        premium_index: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
    ) -> SevenFactorAllocationResult:
        """
        Evaluates the 7-Factor Causal Online Engine on the latest bar and returns
        the complete allocation & telemetry snapshot.
        """
        mats = self.compute_matrices(
            bars_1h=bars_1h,
            funding_rates=funding_rates,
            bars_15m=bars_15m,
            open_interest=open_interest,
            taker_flow=taker_flow,
            premium_index=premium_index,
        )

        w_latest = mats["w_base_1x"].iloc[-1].copy()
        fw_latest = mats["fw_norm"].iloc[-1]
        t_latest = mats["t_df"].iloc[-1]
        beta_latest = mats["beta_72"].iloc[-1]
        comb_latest = mats["comb_w"].iloc[-1]

        # Enforce soft portfolio BTC-beta cap (|beta_p| <= 0.15) while preserving 100% dollar neutrality
        port_beta = float((w_latest * beta_latest).sum())
        gross_target = float(w_latest.abs().sum())
        if abs(port_beta) > 0.15 and gross_target > 1e-8:
            b_centered = beta_latest - float(beta_latest.mean())
            b_var = float((b_centered ** 2).sum())
            if b_var > 1e-8:
                excess_beta = port_beta - np.sign(port_beta) * 0.10
                w_adj = w_latest - (excess_beta / b_var) * b_centered
                pos = w_adj.clip(lower=0.0)
                neg = (-w_adj).clip(lower=0.0)
                if pos.sum() > 1e-10 and neg.sum() > 1e-10:
                    w_latest = (
                        pos * ((gross_target / 2.0) / pos.sum())
                        - neg * ((gross_target / 2.0) / neg.sum())
                    )
                    port_beta = float((w_latest * beta_latest).sum())

        unit_weights = {str(k): float(v) for k, v in w_latest.items()}
        factor_weights = {str(k): float(v) for k, v in fw_latest.items()}
        factor_tstats = {str(k): float(v) for k, v in t_latest.items()}
        asset_betas = {str(k): float(v) for k, v in beta_latest.items()}
        asset_scores = {str(k): float(v) for k, v in comb_latest.items()}
        vol_scalar = float(mats["vol_scalar"].iloc[-1])
        port_vol_72h = float(mats["port_vol_72h"].iloc[-1])
        median_vol = float(mats["median_vol"].iloc[-1])
        conviction = float(mats["conviction"].iloc[-1])

        return SevenFactorAllocationResult(
            unit_weights=unit_weights,
            vol_scalar=vol_scalar,
            port_vol_72h=port_vol_72h,
            median_vol=median_vol,
            conviction=conviction,
            factor_weights=factor_weights,
            factor_tstats=factor_tstats,
            asset_betas=asset_betas,
            portfolio_beta=port_beta,
            asset_composite_scores=asset_scores,
            cmf_tstat=float(mats["t_cmf"].iloc[-1]),
            cmf_blend_weight=float(mats["alpha_cmf_dyn"].iloc[-1]),
        )
