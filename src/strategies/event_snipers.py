"""
Phase 2 Event-Driven Microstructure Sniper Engine (`M09`, `M23`, `M04`).

Implements the 3 orthogonal event-driven mechanisms that survived all 3 independent market eras
(2023-2024 Train, 2025 Validation, 2026 Sealed Holdout) across 82 Binance USD-M perpetuals
under exact 1-minute intrabar stop/trail physics and tier-realistic transaction costs:

1. `Sniper:M09_CoilBreakout` (`1h` -> `1m` brackets, Top-25 by causal 30d ADV):
   - Market-wide 24h cross-sectional return dispersion (`Top-30`, lagged `t-1`) compressed into
     bottom 20th percentile (`<= q20` over trailing 30d) — "coiled spring" regime.
   - Asset closes above prior 48h Donchian High (`LONG`, `taker_buy_share >= 0.56`) or below
     prior 48h Donchian Low (`SHORT`, `taker_buy_share <= 0.44`) on `quote_volume >= 2.2x`
     168h rolling median.
2. `Sniper:M23_BreadthThrust` (`1h` 82-symbol breadth sensor -> `1m` brackets on Tier-1 Majors):
   - `>= 65%` of Top-40 liquid perpetuals simultaneously close above their prior 24h High
     AND aggregate Top-40 hourly quote volume `>= 2.0x` 168h rolling median -> `LONG` Tier-1 Majors.
3. `Sniper:M04_NegFundSqueeze` (`15m` -> `1m` brackets, Top-60 by causal 30d ADV):
   - Trapped shorts paying `funding_rate <= -0.0004` (`-4 bps / 8h`) while closed 15m bar breaks
     above prior 24h (`96`-bar) High on `quote_volume >= 2.2x` 24h median and `taker_buy_share >= 0.56` -> `LONG`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SNIPER_M09_STRAT = "Sniper:M09_CoilBreakout"
SNIPER_M23_STRAT = "Sniper:M23_BreadthThrust"
SNIPER_M04_STRAT = "Sniper:M04_NegFundSqueeze"

SNIPER_STRATEGIES: frozenset[str] = frozenset(
    {
        SNIPER_M09_STRAT,
        SNIPER_M23_STRAT,
        SNIPER_M04_STRAT,
    }
)

SNIPER_UNIVERSE_82: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "APTUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "FTMUSDT",
    "ALGOUSDT",
    "EGLDUSDT",
    "KAVAUSDT",
    "IOTAUSDT",
    "MINAUSDT",
    "ICPUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "THETAUSDT",
    "ZILUSDT",
    "ARBUSDT",
    "OPUSDT",
    "MATICUSDT",
    "STRKUSDT",
    "ZKUSDT",
    "METUSDT",
    "ZROUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "CRVUSDT",
    "MKRUSDT",
    "SNXUSDT",
    "COMPUSDT",
    "1INCHUSDT",
    "SUSHIUSDT",
    "DYDXUSDT",
    "JUPUSDT",
    "ENAUSDT",
    "INJUSDT",
    "RUNEUSDT",
    "LDOUSDT",
    "PENDLEUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "TAOUSDT",
    "WLDUSDT",
    "GRTUSDT",
    "RLCUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ETCUSDT",
    "XMRUSDT",
    "ZECUSDT",
    "FILUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "1000FLOKIUSDT",
    "BONKUSDT",
    "WIFUSDT",
    "MEMEUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AXSUSDT",
    "GALAUSDT",
    "ENJUSDT",
    "CHZUSDT",
    "MAGICUSDT",
    "PYTHUSDT",
    "ONDOUSDT",
    "QNTUSDT",
    "BATUSDT",
    "SKLUSDT",
    "CTSIUSDT",
    "GTCUSDT",
    "OGNUSDT",
    "KSMUSDT",
    "FLOWUSDT",
    "WUSDT",
)

BREADTH_EXECUTION_MAJORS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
)


@dataclass(frozen=True)
class SniperBracketSpec:
    """Exact 1m-validated bracket and holding-horizon specification for an Event Sniper."""

    strategy: str
    stop_loss_pct: float
    take_profit_pct: float
    lock_trigger_pct: float
    lock_level_pct: float
    trail_trigger_pct: float
    trail_distance_pct: float
    max_hold_hours: float


SNIPER_BRACKET_PROFILES: dict[str, SniperBracketSpec] = {
    SNIPER_M09_STRAT: SniperBracketSpec(
        strategy=SNIPER_M09_STRAT,
        stop_loss_pct=0.020,
        take_profit_pct=0.140,
        lock_trigger_pct=0.025,
        lock_level_pct=0.010,
        trail_trigger_pct=0.040,
        trail_distance_pct=0.020,
        max_hold_hours=16.0,
    ),
    SNIPER_M23_STRAT: SniperBracketSpec(
        strategy=SNIPER_M23_STRAT,
        stop_loss_pct=0.020,
        take_profit_pct=0.120,
        lock_trigger_pct=0.022,
        lock_level_pct=0.009,
        trail_trigger_pct=0.038,
        trail_distance_pct=0.018,
        max_hold_hours=12.0,
    ),
    SNIPER_M04_STRAT: SniperBracketSpec(
        strategy=SNIPER_M04_STRAT,
        stop_loss_pct=0.024,
        take_profit_pct=0.220,
        lock_trigger_pct=0.028,
        lock_level_pct=0.012,
        trail_trigger_pct=0.045,
        trail_distance_pct=0.022,
        max_hold_hours=12.0,
    ),
}


@dataclass
class SniperCandidateSignal:
    """Actionable event-driven sniper entry signal."""

    symbol: str
    side: int  # +1 for LONG, -1 for SHORT
    strategy: str
    score: float
    reference_price: float
    adv_rank: int
    bracket: SniperBracketSpec
    reason: str


def _prep_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes OHLC + quote_volume + taker_buy_quote_volume indexed by UTC Timestamp."""
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    if "timestamp" not in d.columns:
        d = d.reset_index()
        if "timestamp" not in d.columns and "index" in d.columns:
            d = d.rename(columns={"index": "timestamp"})
    if "timestamp" not in d.columns:
        return pd.DataFrame()

    ts = d["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        mv = float(ts.dropna().max()) if len(ts.dropna()) > 0 else 0.0
        d["dt"] = pd.to_datetime(ts, unit=("ms" if mv > 1e11 else "s"), utc=True)
    else:
        d["dt"] = pd.to_datetime(ts, utc=True)

    d = d.dropna(subset=["dt", "close"]).drop_duplicates(subset=["dt"], keep="last").sort_values("dt").set_index("dt")
    if d.empty:
        return pd.DataFrame()

    c = pd.to_numeric(d["close"], errors="coerce").astype(np.float64)
    o = pd.to_numeric(d.get("open", c), errors="coerce").fillna(c).astype(np.float64)
    h = pd.to_numeric(d.get("high", c), errors="coerce").fillna(c).astype(np.float64)
    low = pd.to_numeric(d.get("low", c), errors="coerce").fillna(c).astype(np.float64)

    if "quote_volume" in d.columns:
        qv = pd.to_numeric(d["quote_volume"], errors="coerce").fillna(0.0).astype(np.float64)
    elif "qvol" in d.columns:
        qv = pd.to_numeric(d["qvol"], errors="coerce").fillna(0.0).astype(np.float64)
    elif "volume" in d.columns:
        qv = (pd.to_numeric(d["volume"], errors="coerce").fillna(0.0) * c).astype(np.float64)
    else:
        qv = pd.Series(0.0, index=d.index, dtype=np.float64)

    if "taker_buy_quote_volume" in d.columns:
        tb = pd.to_numeric(d["taker_buy_quote_volume"], errors="coerce").fillna(0.5 * qv).astype(np.float64)
    elif "tbqvol" in d.columns:
        tb = pd.to_numeric(d["tbqvol"], errors="coerce").fillna(0.5 * qv).astype(np.float64)
    elif "taker_buy_vol" in d.columns:
        tb = (pd.to_numeric(d["taker_buy_vol"], errors="coerce").fillna(0.0) * c).astype(np.float64)
    else:
        tb = (0.5 * qv).astype(np.float64)

    return pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "quote_volume": qv,
            "taker_buy_quote_volume": np.clip(tb, 0.0, np.maximum(qv, 0.0)),
        },
        index=d.index,
    )


class EventDrivenSniperEngine:
    """
    Phase 2 Multi-Mechanism Event-Driven Sniper Engine (`M09` + `M23` + `M04`).
    """

    def __init__(
        self,
        m09_dispersion_quantile: float = 0.20,
        m09_donchian_hours: int = 48,
        m09_max_rank: int = 25,
        m09_vol_mult: float = 2.2,
        m09_taker_buy_thresh: float = 0.56,
        m23_breadth_ratio: float = 0.65,
        m23_top_n: int = 40,
        m23_agg_vol_mult: float = 2.0,
        m04_max_funding_rate: float = -0.0004,
        m04_max_rank: int = 60,
        m04_vol_mult: float = 2.2,
        m04_taker_buy_thresh: float = 0.56,
    ) -> None:
        self.m09_dispersion_quantile = float(m09_dispersion_quantile)
        self.m09_donchian_hours = int(m09_donchian_hours)
        self.m09_max_rank = int(m09_max_rank)
        self.m09_vol_mult = float(m09_vol_mult)
        self.m09_taker_buy_thresh = float(m09_taker_buy_thresh)

        self.m23_breadth_ratio = float(m23_breadth_ratio)
        self.m23_top_n = int(m23_top_n)
        self.m23_agg_vol_mult = float(m23_agg_vol_mult)

        self.m04_max_funding_rate = float(m04_max_funding_rate)
        self.m04_max_rank = int(m04_max_rank)
        self.m04_vol_mult = float(m04_vol_mult)
        self.m04_taker_buy_thresh = float(m04_taker_buy_thresh)

    def evaluate_1h_snipers(
        self,
        bars_1h_map: dict[str, pd.DataFrame],
        blocked_symbols: set[str] | None = None,
    ) -> list[SniperCandidateSignal]:
        """
        Evaluates latest closed 1h bar across the multi-symbol universe for:
        1) `M23_BreadthThrust` (`>= 65%` of Top-40 breaking 24h high on `>= 2.0x` agg volume)
        2) `M09_CoilBreakout` (`t-1` dispersion `<= q20` + `48h` Donchian breakout + `>= 2.2x` vol + Taker-CVD)
        """
        if not bars_1h_map or len(bars_1h_map) < 10:
            return []

        blocked = {s.replace("/", "").split(":")[0] for s in (blocked_symbols or set())}
        frames: dict[str, pd.DataFrame] = {}
        for raw_sym, df_raw in bars_1h_map.items():
            clean_s = raw_sym.replace("/", "").split(":")[0]
            norm_df = _prep_kline_frame(df_raw)
            if len(norm_df) >= 72:
                frames[clean_s] = norm_df

        if len(frames) < 10:
            return []

        anchor = frames["BTCUSDT"] if "BTCUSDT" in frames else next(iter(frames.values()))
        idx = anchor.index
        if len(idx) < 72:
            return []

        syms = sorted(frames.keys())
        c_df = pd.DataFrame({s: frames[s]["close"].reindex(idx).ffill() for s in syms}, index=idx)
        h_df = pd.DataFrame({s: frames[s]["high"].reindex(idx).ffill() for s in syms}, index=idx)
        l_df = pd.DataFrame({s: frames[s]["low"].reindex(idx).ffill() for s in syms}, index=idx)
        qv_df = pd.DataFrame({s: frames[s]["quote_volume"].reindex(idx).fillna(0.0) for s in syms}, index=idx)
        tb_df = pd.DataFrame(
            {s: frames[s]["taker_buy_quote_volume"].reindex(idx).fillna(0.0) for s in syms},
            index=idx,
        )

        min_hist = min(168, max(48, len(idx) // 2))
        listed_ok = (c_df.notna().rolling(min_hist, min_periods=min_hist).sum() >= min_hist) & (
            qv_df.rolling(24, min_periods=12).sum() > 5e4
        )
        adv_win = min(720, max(72, len(idx) - 2))
        adv_30d = (qv_df.rolling(adv_win, min_periods=48).mean().shift(1) * 24.0).where(listed_ok, np.nan)
        rk_latest = (adv_30d.iloc[-1].rank(ascending=False, method="first").fillna(999.0) - 1.0).to_dict()

        signals: list[SniperCandidateSignal] = []

        # --- 1) M23: 82-Symbol Market Breadth Thrust Ignition ---
        h24_prev = h_df.shift(1).rolling(24, min_periods=16).max().iloc[-1]
        c_last = c_df.iloc[-1]
        qv_last = qv_df.iloc[-1]
        top40_syms = [
            s for s in syms if rk_latest.get(s, 999.0) < self.m23_top_n and np.isfinite(h24_prev.get(s, np.nan))
        ]
        if len(top40_syms) >= 12:
            brk_cnt = sum(1 for s in top40_syms if float(c_last[s]) > float(h24_prev[s]))
            breadth_frac = brk_cnt / float(len(top40_syms))
            agg_qv_series = qv_df[top40_syms].sum(axis=1)
            med_agg = float(agg_qv_series.iloc[:-1].tail(168).median()) if len(agg_qv_series) > 24 else 0.0
            cur_agg = float(agg_qv_series.iloc[-1])
            if breadth_frac >= self.m23_breadth_ratio and med_agg > 0.0 and cur_agg >= self.m23_agg_vol_mult * med_agg:
                for maj_sym in BREADTH_EXECUTION_MAJORS:
                    if maj_sym in frames and maj_sym not in blocked and float(c_last.get(maj_sym, 0.0)) > 0.0:
                        signals.append(
                            SniperCandidateSignal(
                                symbol=maj_sym,
                                side=1,
                                strategy=SNIPER_M23_STRAT,
                                score=float(breadth_frac * (cur_agg / med_agg)),
                                reference_price=float(c_last[maj_sym]),
                                adv_rank=int(rk_latest.get(maj_sym, 0)),
                                bracket=SNIPER_BRACKET_PROFILES[SNIPER_M23_STRAT],
                                reason=(
                                    f"BreadthThrust {breadth_frac * 100:.0f}% ({brk_cnt}/{len(top40_syms)}) "
                                    f"on {cur_agg / med_agg:.1f}x agg vol"
                                ),
                            )
                        )

        # --- 2) M09: Market Dispersion-Compression Coil + 48h Donchian Breakout ---
        r24 = c_df.pct_change(24, fill_method=None).clip(-0.5, 0.5)
        rk_mat = adv_30d.rank(axis=1, ascending=False, method="first").fillna(999.0) - 1.0
        disp_series = r24.where(rk_mat < 30).std(axis=1).dropna()
        if len(disp_series) >= 96:
            disp_prev = float(disp_series.iloc[-2])
            q_thresh = float(disp_series.iloc[:-2].tail(720).quantile(self.m09_dispersion_quantile))
            if np.isfinite(disp_prev) and np.isfinite(q_thresh) and disp_prev <= q_thresh:
                h_don = h_df.shift(1).rolling(self.m09_donchian_hours, min_periods=36).max().iloc[-1]
                l_don = l_df.shift(1).rolling(self.m09_donchian_hours, min_periods=36).min().iloc[-1]
                med_qv = qv_df.iloc[:-1].tail(168).median()
                tsh_last = tb_df.iloc[-1] / (qv_last + 1e-8)

                for sym in syms:
                    rk_s = int(rk_latest.get(sym, 999))
                    if rk_s >= self.m09_max_rank or sym in blocked:
                        continue
                    mq = float(med_qv.get(sym, 0.0))
                    cq = float(qv_last.get(sym, 0.0))
                    if mq <= 0.0 or cq < self.m09_vol_mult * mq:
                        continue
                    px = float(c_last[sym])
                    tsh = float(tsh_last[sym])
                    if px > float(h_don[sym]) and tsh >= self.m09_taker_buy_thresh:
                        signals.append(
                            SniperCandidateSignal(
                                symbol=sym,
                                side=1,
                                strategy=SNIPER_M09_STRAT,
                                score=float((cq / mq) * tsh),
                                reference_price=px,
                                adv_rank=rk_s,
                                bracket=SNIPER_BRACKET_PROFILES[SNIPER_M09_STRAT],
                                reason=(
                                    f"DispCoil({disp_prev * 100:.2f}%<={q_thresh * 100:.2f}%) + "
                                    f"48hHighBrk Vol={cq / mq:.1f}x TBuy={tsh * 100:.0f}%"
                                ),
                            )
                        )
                    elif px < float(l_don[sym]) and tsh <= (1.0 - self.m09_taker_buy_thresh):
                        signals.append(
                            SniperCandidateSignal(
                                symbol=sym,
                                side=-1,
                                strategy=SNIPER_M09_STRAT,
                                score=float((cq / mq) * (1.0 - tsh)),
                                reference_price=px,
                                adv_rank=rk_s,
                                bracket=SNIPER_BRACKET_PROFILES[SNIPER_M09_STRAT],
                                reason=(
                                    f"DispCoil({disp_prev * 100:.2f}%<={q_thresh * 100:.2f}%) + "
                                    f"48hLowBrk Vol={cq / mq:.1f}x TSell={(1.0 - tsh) * 100:.0f}%"
                                ),
                            )
                        )

        signals.sort(key=lambda s: s.score, reverse=True)
        deduped: list[SniperCandidateSignal] = []
        seen: set[str] = set()
        for sig in signals:
            if sig.symbol not in seen:
                deduped.append(sig)
                seen.add(sig.symbol)
        return deduped

    def evaluate_15m_squeeze_snipers(
        self,
        bars_15m_map: dict[str, pd.DataFrame],
        funding_rates: dict[str, float],
        adv_ranks: dict[str, int] | None = None,
        blocked_symbols: set[str] | None = None,
    ) -> list[SniperCandidateSignal]:
        """
        Evaluates latest closed 15m bar for `M04_NegFundSqueeze`:
        - `funding_rate <= -0.0004` (`-4 bps / 8h`)
        - `adv_rank < 60`
        - Closed 15m candle breaks above prior 24h (`96`-bar) High on `>= 2.2x` median volume
          and `taker_buy_share >= 0.56`.
        """
        if not bars_15m_map or not funding_rates:
            return []

        blocked = {s.replace("/", "").split(":")[0] for s in (blocked_symbols or set())}
        signals: list[SniperCandidateSignal] = []

        for raw_sym, df_raw in bars_15m_map.items():
            clean_s = raw_sym.replace("/", "").split(":")[0]
            if clean_s in blocked:
                continue
            rk_s = int((adv_ranks or {}).get(clean_s, 20))
            if rk_s >= self.m04_max_rank:
                continue
            fr_val = float(funding_rates.get(clean_s, funding_rates.get(raw_sym, 0.0)))
            if fr_val > self.m04_max_funding_rate:
                continue

            d15 = _prep_kline_frame(df_raw)
            if len(d15) < 52:
                continue

            c_last = float(d15["close"].iloc[-1])
            qv_last = float(d15["quote_volume"].iloc[-1])
            tb_last = float(d15["taker_buy_quote_volume"].iloc[-1])
            prev_h24 = float(d15["high"].iloc[:-1].tail(96).max())
            med_qv = float(d15["quote_volume"].iloc[:-1].tail(96).median())
            tsh = tb_last / (qv_last + 1e-8)

            if (
                c_last > 0.0
                and med_qv > 0.0
                and c_last > prev_h24
                and qv_last >= self.m04_vol_mult * med_qv
                and tsh >= self.m04_taker_buy_thresh
            ):
                signals.append(
                    SniperCandidateSignal(
                        symbol=clean_s,
                        side=1,
                        strategy=SNIPER_M04_STRAT,
                        score=float(abs(fr_val) * 1e4 * (qv_last / med_qv)),
                        reference_price=c_last,
                        adv_rank=rk_s,
                        bracket=SNIPER_BRACKET_PROFILES[SNIPER_M04_STRAT],
                        reason=(
                            f"NegFundSqueeze FR={fr_val * 10000:.1f}bps + 24hHighBrk "
                            f"Vol={qv_last / med_qv:.1f}x TBuy={tsh * 100:.0f}%"
                        ),
                    )
                )

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def scan_historical_1h_events(
        self,
        t_grid_1h: np.ndarray,
        syms: list[str],
        c_1h: np.ndarray,
        h_1h: np.ndarray,
        l_1h: np.ndarray,
        qv_1h: np.ndarray,
        tb_1h: np.ndarray,
        rank_1h: np.ndarray,
    ) -> list[tuple[int, int, int, str, float, int]]:
        """
        Vectorized historical event scanner using the exact `M09` and `M23` production thresholds.
        Returns list of `(entry_ts_sec, sym_idx, side, strategy_name, score, adv_rank)`.
        """
        idx_1h = pd.to_datetime(t_grid_1h, unit="s", utc=True)
        c_df = pd.DataFrame(c_1h, index=idx_1h, columns=syms)
        h_df = pd.DataFrame(h_1h, index=idx_1h, columns=syms)
        l_df = pd.DataFrame(l_1h, index=idx_1h, columns=syms)
        rk_df = pd.DataFrame(rank_1h, index=idx_1h, columns=syms)

        med_qv = pd.DataFrame(qv_1h).rolling(168, min_periods=24).median().to_numpy(dtype=np.float64)
        tsh = tb_1h / (qv_1h + 1e-8)

        # M09 arrays
        r24 = c_df.pct_change(24, fill_method=None).clip(-0.5, 0.5)
        disp24 = r24.where(rk_df < 30).std(axis=1)
        dq = disp24.rolling(24 * 30, min_periods=168).quantile(self.m09_dispersion_quantile)
        coiled = (disp24.shift(1) <= dq.shift(1)).fillna(False).to_numpy(dtype=bool)
        h_don = h_df.shift(1).rolling(self.m09_donchian_hours, min_periods=36).max().to_numpy(dtype=np.float64)
        l_don = l_df.shift(1).rolling(self.m09_donchian_hours, min_periods=36).min().to_numpy(dtype=np.float64)

        # M23 arrays
        h24_prev = h_df.shift(1).rolling(24, min_periods=16).max().to_numpy(dtype=np.float64)
        top_mask = rank_1h < self.m23_top_n
        brk_up = (c_1h > h24_prev) & top_mask
        breadth = brk_up.sum(axis=1) / np.maximum(top_mask.sum(axis=1), 1)
        agg_qv = np.where(top_mask, qv_1h, 0.0).sum(axis=1)
        med_agg_qv = pd.Series(agg_qv).shift(1).rolling(168, min_periods=24).median().to_numpy(dtype=np.float64)
        maj_indices = [syms.index(m) for m in BREADTH_EXECUTION_MAJORS if m in syms]

        n_1h, ns = c_1h.shape
        out_events: list[tuple[int, int, int, str, float, int]] = []

        for t in range(200, n_1h - 2):
            entry_ts = int(t_grid_1h[t]) + 3600
            if (
                breadth[t] >= self.m23_breadth_ratio
                and np.isfinite(med_agg_qv[t])
                and med_agg_qv[t] > 0.0
                and agg_qv[t] >= self.m23_agg_vol_mult * med_agg_qv[t]
            ):
                sc_23 = float(breadth[t] * (agg_qv[t] / med_agg_qv[t]))
                for mj in maj_indices:
                    if np.isfinite(c_1h[t, mj]) and c_1h[t, mj] > 0:
                        out_events.append((entry_ts, mj, 1, SNIPER_M23_STRAT, sc_23, int(rank_1h[t, mj])))

            if coiled[t]:
                for j in range(ns):
                    rk_j = int(rank_1h[t, j])
                    if (
                        rk_j >= self.m09_max_rank
                        or med_qv[t, j] <= 0.0
                        or qv_1h[t, j] < self.m09_vol_mult * med_qv[t, j]
                    ):
                        continue
                    v_ratio = float(qv_1h[t, j] / med_qv[t, j])
                    if c_1h[t, j] > h_don[t, j] and tsh[t, j] >= self.m09_taker_buy_thresh:
                        out_events.append((entry_ts, j, 1, SNIPER_M09_STRAT, v_ratio * float(tsh[t, j]), rk_j))
                    elif c_1h[t, j] < l_don[t, j] and tsh[t, j] <= (1.0 - self.m09_taker_buy_thresh):
                        out_events.append((entry_ts, j, -1, SNIPER_M09_STRAT, v_ratio * float(1.0 - tsh[t, j]), rk_j))

        return out_events

    def scan_historical_15m_events(
        self,
        t_grid_15m: np.ndarray,
        c_15m: np.ndarray,
        h_15m: np.ndarray,
        qv_15m: np.ndarray,
        tb_15m: np.ndarray,
        fr_15m: np.ndarray,
        rank_15m: np.ndarray,
    ) -> list[tuple[int, int, int, str, float, int]]:
        """
        Vectorized historical 15m event scanner using the exact `M04` production thresholds.
        Returns list of `(entry_ts_sec, sym_idx, side, strategy_name, score, adv_rank)`.
        """
        h24_15 = pd.DataFrame(h_15m).shift(1).rolling(96, min_periods=48).max().to_numpy(dtype=np.float64)
        med_qv15 = pd.DataFrame(qv_15m).shift(1).rolling(96, min_periods=32).median().to_numpy(dtype=np.float64)
        tsh15 = tb_15m / (qv_15m + 1e-8)

        cand_t, cand_j = np.where(
            (rank_15m < self.m04_max_rank)
            & (fr_15m <= self.m04_max_funding_rate)
            & (c_15m > h24_15)
            & (tsh15 >= self.m04_taker_buy_thresh)
            & (qv_15m >= self.m04_vol_mult * (med_qv15 + 1e-8))
        )
        n15 = len(t_grid_15m)
        out: list[tuple[int, int, int, str, float, int]] = []
        for t, j in zip(cand_t.tolist(), cand_j.tolist(), strict=False):
            if t < 100 or t >= n15 - 2:
                continue
            sc = float(abs(fr_15m[t, j]) * 1e4 * (qv_15m[t, j] / max(med_qv15[t, j], 1e-8)))
            out.append((int(t_grid_15m[t]) + 900, int(j), 1, SNIPER_M04_STRAT, sc, int(rank_15m[t, j])))
        return out
