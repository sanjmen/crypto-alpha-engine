"""
Hysteresis Stepped Leverage Controller (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x)
with Bracketed Drawdown De-escalation, 72h Realized Volatility Targeting,
and Proportional Turnover Deadbanding.
"""

from dataclasses import dataclass

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

    @property
    def next_tier_high_water_mark_target(self) -> float:
        return self.next_tier_hwm_target


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
        max_leg_share: float | None = 0.32,
        min_leg_share: float = 0.05,
        max_leg_positions: int = 5,
        high_water_mark: float | None = None,
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
        self.min_leg_share = float(min_leg_share)
        self.max_leg_positions = int(max_leg_positions)
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
        equity_val = float(current_equity) if (np.isfinite(current_equity) and current_equity > 0.0) else 0.0
        if equity_val > self.high_water_mark:
            self.high_water_mark = equity_val

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
            steps_unlocked = 0
            unlocked_tier = self.start_leverage

        current_drawdown = (
            max(0.0, (self.high_water_mark - equity_val) / self.high_water_mark) if self.high_water_mark > 0 else 0.0
        )

        if self.dd_bracket > 0.0 and self.step_leverage > 0.0:
            steps_down = int(np.floor(current_drawdown / self.dd_bracket + 1e-12))
            active_base = max(
                self.floor_leverage,
                unlocked_tier - steps_down * self.step_leverage,
            )
        else:
            steps_down = 0
            active_base = unlocked_tier

        clamped_vol_scalar = float(np.clip(vol_scalar, 0.55, 1.15))
        effective_leverage = float(min(self.max_leverage, active_base * clamped_vol_scalar))

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

        proportional_deadband = self.base_deadband_ratio * effective_leverage
        max_single_weight = self.base_max_single_ratio * effective_leverage

        return LeverageStateSnapshot(
            initial_capital=self.initial_capital,
            current_equity=equity_val,
            high_water_mark=self.high_water_mark,
            drawdown_pct=current_drawdown * 100.0,
            unlocked_tier=unlocked_tier,
            steps_unlocked=steps_unlocked,
            steps_down=steps_down,
            active_base_leverage=active_base,
            vol_scalar=clamped_vol_scalar,
            effective_leverage=effective_leverage,
            next_tier_leverage=next_tier,
            next_tier_hwm_target=float(next_milestone_hwm),
            progress_to_next_tier_pct=progress_pct,
            proportional_deadband=proportional_deadband,
            max_single_asset_weight=max_single_weight,
        )

    def _keep_top_leg_mask(self, leg_magnitudes: np.ndarray) -> np.ndarray:
        """Returns a boolean mask keeping the top positions on one leg (>= 3 and >= min_leg_share)."""
        leg_total = float(leg_magnitudes.sum())
        if leg_total <= 1e-10 or int((leg_magnitudes > 1e-10).sum()) <= 3:
            return np.ones_like(leg_magnitudes, dtype=bool)

        leg_shares = leg_magnitudes / leg_total
        rank_order = np.argsort(-leg_magnitudes)
        keep_mask = np.zeros_like(leg_magnitudes, dtype=bool)
        for rank_idx, sym_idx in enumerate(rank_order):
            if leg_magnitudes[sym_idx] <= 1e-10:
                break
            if rank_idx < 3 or (rank_idx < self.max_leg_positions and leg_shares[sym_idx] >= self.min_leg_share):
                keep_mask[sym_idx] = True
        return keep_mask

    def scale_and_deadband_weights(
        self,
        unit_weights: dict[str, float],
        current_weights: dict[str, float] | None,
        current_equity: float,
        vol_scalar: float = 1.0,
    ) -> tuple[dict[str, float], LeverageStateSnapshot]:
        """
        Scales 1x unit weights by `effective_leverage`, enforces per-asset weight caps
        and strict 100% dollar neutrality (sum(longs) == -sum(shorts)), and applies
        the proportional turnover deadband against `current_weights`.
        """
        snapshot = self.update_state(current_equity=current_equity, vol_scalar=vol_scalar)
        effective_leverage = snapshot.effective_leverage
        max_single = snapshot.max_single_asset_weight
        deadband = snapshot.proportional_deadband

        symbols = sorted(unit_weights.keys())
        if not symbols:
            return {}, snapshot

        raw_weights = np.array([float(unit_weights.get(s, 0.0)) * effective_leverage for s in symbols], dtype=float)
        initial_gross = float(np.sum(np.abs(raw_weights)))

        if initial_gross > 1e-8:
            target_gross = min(initial_gross, effective_leverage)
            pos_raw_cnt = int((raw_weights > 1e-10).sum())
            neg_raw_cnt = int((raw_weights < -1e-10).sum())
            use_active_leg_cap = (
                self.max_leg_share is not None
                and 0.0 < self.max_leg_share < 1.0
                and pos_raw_cnt >= 4
                and neg_raw_cnt >= 4
            )
            if use_active_leg_cap:
                max_single = float(self.max_leg_share) * (target_gross / 2.0)

            raw_pruned = raw_weights.copy()
            if len(symbols) >= 8 and self.min_leg_share > 0.0:
                keep_pos = self._keep_top_leg_mask(np.maximum(raw_pruned, 0.0))
                keep_neg = self._keep_top_leg_mask(np.maximum(-raw_pruned, 0.0))
                raw_pruned = np.where((raw_pruned > 0.0) & (~keep_pos), 0.0, raw_pruned)
                raw_pruned = np.where((raw_pruned < 0.0) & (~keep_neg), 0.0, raw_pruned)

            clipped = np.clip(raw_pruned, -max_single, max_single)
            pos_leg = np.maximum(clipped, 0.0)
            neg_leg = np.minimum(clipped, 0.0)
            if pos_leg.sum() > 1e-10 and np.abs(neg_leg.sum()) > 1e-10:
                pos_leg = pos_leg * ((target_gross / 2.0) / pos_leg.sum())
                neg_leg = neg_leg * ((target_gross / 2.0) / np.abs(neg_leg.sum()))
                desired = np.clip(pos_leg + neg_leg, -max_single, max_single)
                pos_clipped = np.maximum(desired, 0.0)
                neg_clipped = np.minimum(desired, 0.0)
                leg_target = min(float(pos_clipped.sum()), float(np.abs(neg_clipped.sum())))
                if leg_target > 1e-10:
                    pos_clipped = pos_clipped * (leg_target / pos_clipped.sum())
                    neg_clipped = neg_clipped * (leg_target / np.abs(neg_clipped.sum()))
                    desired = pos_clipped + neg_clipped
                else:
                    desired = np.zeros_like(raw_weights)
            else:
                desired = np.zeros_like(raw_weights)
        else:
            desired = np.zeros_like(raw_weights)

        if current_weights:
            prev_weights = np.array([float(current_weights.get(s, 0.0)) for s in symbols], dtype=float)
            delta = np.abs(desired - prev_weights)
            sign_flip = (desired * prev_weights) < -1e-9
            sub_deadband_decay = (np.abs(desired) < 0.5 * deadband) & (np.abs(prev_weights) < deadband)
            updated = np.where(
                (np.abs(desired) < 1e-9) | (delta >= deadband) | sign_flip | sub_deadband_decay,
                desired,
                prev_weights,
            )

            pos_updated = np.maximum(updated, 0.0)
            neg_updated = np.minimum(updated, 0.0)
            if pos_updated.sum() > 1e-10 and np.abs(neg_updated.sum()) > 1e-10:
                net_drift = abs(float(pos_updated.sum()) - float(np.abs(neg_updated.sum())))
                gross_updated = float(pos_updated.sum()) + float(np.abs(neg_updated.sum()))
                if net_drift > 0.025 or gross_updated > effective_leverage + 1e-6:
                    avg_leg = min(effective_leverage / 2.0, 0.5 * gross_updated)
                    pos_updated = pos_updated * (avg_leg / pos_updated.sum())
                    neg_updated = neg_updated * (avg_leg / np.abs(neg_updated.sum()))
                    updated = pos_updated + neg_updated

                if (
                    self.max_leg_share is not None
                    and 0.0 < self.max_leg_share < 1.0
                    and int((updated > 1e-10).sum()) >= 4
                    and int((updated < -1e-10).sum()) >= 4
                ):
                    pos_cap_leg = np.maximum(updated, 0.0)
                    neg_cap_leg = np.maximum(-updated, 0.0)
                    if pos_cap_leg.sum() > 1e-10 and neg_cap_leg.sum() > 1e-10:
                        cap_pos = float(pos_cap_leg.sum()) * float(self.max_leg_share)
                        cap_neg = float(neg_cap_leg.sum()) * float(self.max_leg_share)
                        pos_cap_leg = np.minimum(pos_cap_leg, cap_pos)
                        neg_cap_leg = np.minimum(neg_cap_leg, cap_neg)
                        leg_target_u = min(float(pos_cap_leg.sum()), float(neg_cap_leg.sum()))
                        if leg_target_u > 1e-10:
                            updated = pos_cap_leg * (leg_target_u / float(pos_cap_leg.sum())) - neg_cap_leg * (
                                leg_target_u / float(neg_cap_leg.sum())
                            )
            else:
                updated = desired
        else:
            updated = desired

        final_weights = {sym: float(updated[i]) for i, sym in enumerate(symbols) if abs(updated[i]) > 1e-8}
        return final_weights, snapshot
