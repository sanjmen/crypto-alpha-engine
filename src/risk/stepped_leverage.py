"""
Hysteresis Stepped Leverage Controller (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x)
with Bracketed Drawdown De-escalation, 72h Realized Volatility Targeting,
and Proportional Turnover Deadbanding.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class LeverageStateSnapshot:
    """Complete state and telemetry snapshot of the Stepped Leverage Controller."""

    initial_capital: float
    current_equity: float
    high_water_mark: float
    drawdown_pct: float
    unlocked_tier: float
    steps_unlocked: int
    steps_down: int
    active_base_leverage: float
    vol_scalar: float
    effective_leverage: float
    next_tier_leverage: float
    next_tier_hwm_target: float
    progress_to_next_tier_pct: float
    proportional_deadband: float
    max_single_asset_weight: float


class SteppedLeverageController:
    """
    Manages progressive leverage scaling funded by accumulated profits:
    - Starts at `start_leverage` (default 2.5x).
    - Unlocks `+0.5x` steps each time High-Water Mark (HWM) compounds by `gain_per_step` (+15%)
      up to `max_leverage` (default 5.0x).
    - Steps down by `-0.5x` for each `dd_bracket` (10%) of drawdown from HWM (down to `floor_leverage`).
    - Multiplies active base leverage by the 72h realized volatility scalar `vol_scalar` in [0.55, 1.15].
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        start_leverage: float = 2.5,
        max_leverage: float = 5.0,
        floor_leverage: float = 2.0,
        step_leverage: float = 0.5,
        gain_per_step: float = 0.15,
        dd_bracket: float = 0.10,
        base_deadband_ratio: float = 0.04 / 2.2,
        base_max_single_ratio: float = 0.45 / 2.2,
        max_leg_share: Optional[float] = 0.32,
        high_water_mark: Optional[float] = None,
    ) -> None:
        self.initial_capital = max(float(initial_capital), 1.0)
        self.start_leverage = float(start_leverage)
        self.max_leverage = float(max_leverage)
        self.floor_leverage = float(floor_leverage)
        self.step_leverage = float(step_leverage)
        self.gain_per_step = float(gain_per_step)
        self.dd_bracket = float(dd_bracket)
        self.base_deadband_ratio = float(base_deadband_ratio)
        self.base_max_single_ratio = float(base_max_single_ratio)
        self.max_leg_share = max_leg_share
        self.high_water_mark = max(
            float(high_water_mark) if high_water_mark is not None else self.initial_capital,
            self.initial_capital,
        )

    def update_state(
        self,
        current_equity: float,
        vol_scalar: float = 1.0,
    ) -> LeverageStateSnapshot:
        """
        Updates High-Water Mark with `current_equity` and computes the active
        leverage tier, drawdown de-escalation, and effective gross leverage.
        """
        eq = max(float(current_equity), 0.0)
        if eq > self.high_water_mark:
            self.high_water_mark = eq

        hwm_ratio = max(self.high_water_mark / self.initial_capital, 1.0)
        if self.step_leverage > 0.0 and self.gain_per_step > 0.0:
            max_steps = int(round((self.max_leverage - self.start_leverage) / self.step_leverage))
            raw_steps = int(np.floor(np.log(hwm_ratio) / np.log(1.0 + self.gain_per_step) + 1e-12))
            steps_unlocked = max(0, min(max_steps, raw_steps))
            unlocked_tier = min(
                self.max_leverage,
                self.start_leverage + steps_unlocked * self.step_leverage,
            )
        else:
            max_steps = 0
            steps_unlocked = 0
            unlocked_tier = self.start_leverage

        cur_dd = (
            max(0.0, (self.high_water_mark - eq) / self.high_water_mark)
            if self.high_water_mark > 0
            else 0.0
        )

        if self.dd_bracket > 0.0 and self.step_leverage > 0.0:
            steps_down = int(np.floor(cur_dd / self.dd_bracket + 1e-12))
            active_base = max(
                self.floor_leverage,
                unlocked_tier - steps_down * self.step_leverage,
            )
        else:
            steps_down = 0
            active_base = unlocked_tier

        v_scalar = float(np.clip(vol_scalar, 0.55, 1.15))
        eff_lev = float(min(self.max_leverage, active_base * v_scalar))

        # Next milestone telemetry
        if unlocked_tier < self.max_leverage and self.step_leverage > 0.0:
            next_tier = min(self.max_leverage, unlocked_tier + self.step_leverage)
            prev_milestone_hwm = self.initial_capital * ((1.0 + self.gain_per_step) ** steps_unlocked)
            next_milestone_hwm = self.initial_capital * ((1.0 + self.gain_per_step) ** (steps_unlocked + 1))
            span = max(next_milestone_hwm - prev_milestone_hwm, 1e-8)
            progress_pct = float(np.clip((self.high_water_mark - prev_milestone_hwm) / span * 100.0, 0.0, 100.0))
        else:
            next_tier = self.max_leverage
            next_milestone_hwm = self.high_water_mark
            progress_pct = 100.0

        prop_deadband = self.base_deadband_ratio * eff_lev
        max_single_w = self.base_max_single_ratio * eff_lev

        return LeverageStateSnapshot(
            initial_capital=self.initial_capital,
            current_equity=eq,
            high_water_mark=self.high_water_mark,
            drawdown_pct=cur_dd * 100.0,
            unlocked_tier=unlocked_tier,
            steps_unlocked=steps_unlocked,
            steps_down=steps_down,
            active_base_leverage=active_base,
            vol_scalar=v_scalar,
            effective_leverage=eff_lev,
            next_tier_leverage=next_tier,
            next_tier_hwm_target=float(next_milestone_hwm),
            progress_to_next_tier_pct=progress_pct,
            proportional_deadband=prop_deadband,
            max_single_asset_weight=max_single_w,
        )

    def scale_and_deadband_weights(
        self,
        unit_weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]],
        current_equity: float,
        vol_scalar: float = 1.0,
    ) -> Tuple[Dict[str, float], LeverageStateSnapshot]:
        """
        Scales 1x unit weights by `effective_leverage`, enforces per-asset weight caps
        and strict 100% dollar neutrality (sum(longs) == -sum(shorts)), and applies
        the proportional turnover deadband against `current_weights`.
        """
        snap = self.update_state(current_equity=current_equity, vol_scalar=vol_scalar)
        eff_lev = snap.effective_leverage
        max_single = snap.max_single_asset_weight
        db = snap.proportional_deadband

        symbols = sorted(unit_weights.keys())
        if not symbols:
            return {}, snap

        raw_w = np.array([float(unit_weights.get(s, 0.0)) * eff_lev for s in symbols], dtype=float)
        g_init = float(np.sum(np.abs(raw_w)))

        if g_init > 1e-8:
            target_g = min(g_init, eff_lev)
            pos_raw_cnt = int((raw_w > 1e-10).sum())
            neg_raw_cnt = int((raw_w < -1e-10).sum())
            use_active_leg_cap = (
                self.max_leg_share is not None
                and 0.0 < self.max_leg_share < 1.0
                and pos_raw_cnt >= 4
                and neg_raw_cnt >= 4
            )
            if use_active_leg_cap:
                max_single = float(self.max_leg_share) * (target_g / 2.0)
            else:
                max_single = snap.max_single_asset_weight
            clipped = np.clip(raw_w, -max_single, max_single)
            pos = np.maximum(clipped, 0.0)
            neg = np.minimum(clipped, 0.0)
            if pos.sum() > 1e-10 and np.abs(neg.sum()) > 1e-10:
                pos = pos * ((target_g / 2.0) / pos.sum())
                neg = neg * ((target_g / 2.0) / np.abs(neg.sum()))
                desired = np.clip(pos + neg, -max_single, max_single)
                # Final neutrality polish after single-asset clip
                pos2 = np.maximum(desired, 0.0)
                neg2 = np.minimum(desired, 0.0)
                leg_target = min(float(pos2.sum()), float(np.abs(neg2.sum())))
                if leg_target > 1e-10:
                    pos2 = pos2 * (leg_target / pos2.sum())
                    neg2 = neg2 * (leg_target / np.abs(neg2.sum()))
                    desired = pos2 + neg2
                else:
                    desired = np.zeros_like(raw_w)
            else:
                desired = np.zeros_like(raw_w)
        else:
            desired = np.zeros_like(raw_w)

        if current_weights:
            prev_w = np.array([float(current_weights.get(s, 0.0)) for s in symbols], dtype=float)
            delta = np.abs(desired - prev_w)
            upd = np.where((np.abs(desired) < 1e-9) | (delta >= db), desired, prev_w)

            # Re-balance long/short legs after deadband when net drift exceeds 2.5% (or cap gross <= eff_lev)
            pos_u = np.maximum(upd, 0.0)
            neg_u = np.minimum(upd, 0.0)
            if pos_u.sum() > 1e-10 and np.abs(neg_u.sum()) > 1e-10:
                net_drift = abs(float(pos_u.sum()) - float(np.abs(neg_u.sum())))
                gross_u = float(pos_u.sum()) + float(np.abs(neg_u.sum()))
                if net_drift > 0.025 or gross_u > eff_lev + 1e-6:
                    avg_leg = min(eff_lev / 2.0, 0.5 * gross_u)
                    pos_u = pos_u * (avg_leg / pos_u.sum())
                    neg_u = neg_u * (avg_leg / np.abs(neg_u.sum()))
                    upd = pos_u + neg_u
                # Enforce active-leg concentration cap after deadband so no single asset exceeds max_leg_share
                if (
                    self.max_leg_share is not None
                    and 0.0 < self.max_leg_share < 1.0
                    and int((upd > 1e-10).sum()) >= 4
                    and int((upd < -1e-10).sum()) >= 4
                ):
                    p_u = np.maximum(upd, 0.0)
                    n_u = np.maximum(-upd, 0.0)
                    if p_u.sum() > 1e-10 and n_u.sum() > 1e-10:
                        cap_p = float(p_u.sum()) * float(self.max_leg_share)
                        cap_n = float(n_u.sum()) * float(self.max_leg_share)
                        p_u = np.minimum(p_u, cap_p)
                        n_u = np.minimum(n_u, cap_n)
                        leg_u = min(float(p_u.sum()), float(n_u.sum()))
                        if leg_u > 1e-10:
                            upd = p_u * (leg_u / float(p_u.sum())) - n_u * (leg_u / float(n_u.sum()))
            else:
                upd = desired
        else:
            upd = desired

        final_weights = {sym: float(upd[i]) for i, sym in enumerate(symbols) if abs(upd[i]) > 1e-8}
        return final_weights, snap
