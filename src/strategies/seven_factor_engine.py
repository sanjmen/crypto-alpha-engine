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
    "macro": 0.30,
    "med": 0.15,
    "fast": 0.15,
    "rev": 0.10,
    "carry": 0.15,
    "liq": 0.10,
    "size": 0.15,
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
    max_side_share: float = 0.30,
) -> pd.DataFrame:
    """
    Caps each asset's share of the Long leg (and Short leg) at `max_side_share` (e.g. 30%)
    and iteratively redistributes clipped excess proportionally across remaining unclipped
    assets on the same leg (water-filling), preserving strict 100% dollar neutrality.
    """
    if max_side_share is None or max_side_share >= 1.0:
        return w_df
    arr = w_df.values.copy()
    n_rows, _ = arr.shape
    for i in range(n_rows):
        row = arr[i]
        pos = np.maximum(row, 0.0)
        neg = np.maximum(-row, 0.0)
        p_sum = float(pos.sum())
        n_sum = float(neg.sum())
        if p_sum > 1e-10 and int((pos > 1e-10).sum()) >= 4:
            cap_p = p_sum * max_side_share
            for _ in range(3):
                over = pos > cap_p + 1e-9
                if not np.any(over) or np.all(over):
                    pos = np.minimum(pos, cap_p)
                    break
                excess = float((pos[over] - cap_p).sum())
                pos[over] = cap_p
                under = ~over & (pos > 1e-10)
                if not np.any(under):
                    break
                pos[under] += excess * (pos[under] / float(pos[under].sum()))
        if n_sum > 1e-10 and int((neg > 1e-10).sum()) >= 4:
            cap_n = n_sum * max_side_share
            for _ in range(3):
                over = neg > cap_n + 1e-9
                if not np.any(over) or np.all(over):
                    neg = np.minimum(neg, cap_n)
                    break
                excess = float((neg[over] - cap_n).sum())
                neg[over] = cap_n
                under = ~over & (neg > 1e-10)
                if not np.any(under):
                    break
                neg[under] += excess * (neg[under] / float(neg[under].sum()))
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
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Computes full historical matrices for the 7 factors, online factor weights,
        unit portfolio weights, and 72h volatility scalar.
        """
        close, volume, dollar_vol, raw_funding, funding = self._build_aligned_matrices(
            bars_1h, funding_rates
        )
        high, low = self._extract_high_low(bars_1h, close)
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
        if "rev" in raw_fw.columns:
            raw_fw["rev"] = raw_fw["rev"].ewm(span=12, min_periods=1).mean()
        if "liq" in raw_fw.columns:
            raw_fw["liq"] = raw_fw["liq"].ewm(span=16, min_periods=1).mean()
        if "size" in raw_fw.columns:
            raw_fw["size"] = raw_fw["size"].ewm(span=6, min_periods=1).mean()
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

        # Retoque 1C (Nov 1): Causal Cross-Sectional Composite Score Smoothing (EMA span=3h)
        # Prevents single-bar 24h return oscillations from flipping individual coins Long <-> Short
        # across consecutive 4h rebalances while preserving factor conviction.
        if self.comb_ema_span is not None and self.comb_ema_span > 1.0:
            comb_w = comb_w.ewm(span=self.comb_ema_span, min_periods=1).mean()

        ones_df = pd.DataFrame(1.0, index=idx, columns=cols)
        w_raw_1x = make_continuous_neutral_weights(
            comb_w, ones_df, gross_leverage=1.0, power=self.comb_power
        ).mul(conviction, axis=0)

        # Regime-Conditioned Water-Filling Concentration Cap (30% per side when Trend >= 20%)
        # Redistributes excess tail concentration across secondary leaders during trending regimes
        # without inflating low-conviction noise coins during pure mean-reversion chop.
        if self.trend_waterfill_cap is not None and 0.0 < self.trend_waterfill_cap < 1.0:
            w_wf_1x = cap_side_weights_waterfill(w_raw_1x, max_side_share=self.trend_waterfill_cap)
            if (
                self.trend_waterfill_threshold is not None
                and self.trend_waterfill_threshold > 0.0
                and all(k in fw_norm.columns for k in ("macro", "med", "fast"))
            ):
                trend_share = fw_norm["macro"] + fw_norm["med"] + fw_norm["fast"]
                is_trend = (trend_share >= self.trend_waterfill_threshold).values[:, None]
                w_base_1x = pd.DataFrame(
                    np.where(is_trend, w_wf_1x.values, w_raw_1x.values),
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
    ) -> SevenFactorAllocationResult:
        """
        Evaluates the 7-Factor Causal Online Engine on the latest bar and returns
        the complete allocation & telemetry snapshot.
        """
        mats = self.compute_matrices(
            bars_1h=bars_1h,
            funding_rates=funding_rates,
            bars_15m=bars_15m,
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
