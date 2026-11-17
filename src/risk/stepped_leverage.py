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
        bhs_enter_top_k: int | None = None,
        bhs_hold_top_k: int | None = None,
        recovery_boost_equity: float | None = None,
        recovery_boost_leverage: float | None = None,
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
        self.bhs_enter_top_k = (
            int(bhs_enter_top_k) if bhs_enter_top_k is not None and int(bhs_enter_top_k) > 0 else None
        )
        self.bhs_hold_top_k = int(bhs_hold_top_k) if bhs_hold_top_k is not None and int(bhs_hold_top_k) > 0 else None
        eff_max_leg = int(max_leg_positions)
        if self.bhs_hold_top_k is not None:
            eff_max_leg = max(eff_max_leg, self.bhs_hold_top_k)
        self.max_leg_positions = eff_max_leg
        self.recovery_boost_equity = (
            float(recovery_boost_equity)
            if recovery_boost_equity is not None and float(recovery_boost_equity) > 0
            else None
        )
        self.recovery_boost_leverage = (
            float(recovery_boost_leverage)
            if recovery_boost_leverage is not None and float(recovery_boost_leverage) > 0
            else None
        )
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
        Supports optional two-phase safe recovery boost (`recovery_boost_equity` -> `recovery_boost_leverage`).
        """
        equity_val = float(current_equity) if (np.isfinite(current_equity) and current_equity > 0.0) else 0.0
        if equity_val > self.high_water_mark:
            self.high_water_mark = equity_val

        has_recovery_gate = (
            self.recovery_boost_equity is not None
            and self.recovery_boost_leverage is not None
            and self.recovery_boost_leverage > self.start_leverage
        )
        if has_recovery_gate and self.high_water_mark < float(self.recovery_boost_equity):
            # Phase 1 (Safe Recovery below threshold): hold conservative start_leverage
            steps_unlocked = 0
            unlocked_tier = min(self.max_leverage, self.start_leverage)
            ref_base_cap = float(self.recovery_boost_equity)
            ref_base_lev = float(self.recovery_boost_leverage)
            in_recovery_phase = True
        elif has_recovery_gate:
            # Phase 2 (Post-Recovery Boost unlocked): compound steps above recovery_boost_equity
            in_recovery_phase = False
            ref_base_cap = float(self.recovery_boost_equity)
            ref_base_lev = min(self.max_leverage, float(self.recovery_boost_leverage))
            hwm_ratio_post = max(self.high_water_mark / max(ref_base_cap, 1.0), 1.0)
            if self.step_leverage > 0.0 and self.gain_per_step > 0.0:
                max_steps = max(0, int(np.ceil((self.max_leverage - ref_base_lev) / self.step_leverage - 1e-12)))
                raw_steps = int(np.floor(np.log(hwm_ratio_post) / np.log(1.0 + self.gain_per_step) + 1e-12))
                steps_unlocked = max(0, min(max_steps, raw_steps))
                unlocked_tier = min(self.max_leverage, ref_base_lev + steps_unlocked * self.step_leverage)
            else:
                steps_unlocked = 0
                unlocked_tier = ref_base_lev
        else:
            in_recovery_phase = False
            ref_base_cap = self.initial_capital
            ref_base_lev = self.start_leverage
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

        if in_recovery_phase:
            next_tier = min(self.max_leverage, ref_base_lev)
            next_milestone_hwm = ref_base_cap
            span = max(next_milestone_hwm - min(self.initial_capital, equity_val), 1e-8)
            progress_pct = float(
                np.clip((self.high_water_mark - min(self.initial_capital, equity_val)) / span * 100.0, 0.0, 100.0)
            )
        elif unlocked_tier < self.max_leverage and self.step_leverage > 0.0:
            next_tier = min(self.max_leverage, unlocked_tier + self.step_leverage)
            prev_milestone_hwm = ref_base_cap * ((1.0 + self.gain_per_step) ** steps_unlocked)
            next_milestone_hwm = ref_base_cap * ((1.0 + self.gain_per_step) ** (steps_unlocked + 1))
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

    def _keep_top_leg_mask(
        self,
        leg_magnitudes: np.ndarray,
        already_open_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Returns a boolean mask keeping the top positions on one leg.
        When Novy-Marx & Velikov (2016) Buy/Hold Spread (`bhs_enter_top_k`, `bhs_hold_top_k`)
        is configured, new entries require `rank < bhs_enter_top_k`, while existing same-side
        positions are held without cliff-eviction while `rank < bhs_hold_top_k` (and `>= min_leg_share`).
        """
        leg_total = float(leg_magnitudes.sum())
        use_bhs = (
            self.bhs_enter_top_k is not None
            and self.bhs_hold_top_k is not None
            and self.bhs_hold_top_k > self.bhs_enter_top_k
        )
        min_keep = self.bhs_enter_top_k if use_bhs and self.bhs_enter_top_k is not None else 3
        if leg_total <= 1e-10 or int((leg_magnitudes > 1e-10).sum()) <= min_keep:
            return np.ones_like(leg_magnitudes, dtype=bool)

        leg_shares = leg_magnitudes / leg_total
        rank_order = np.argsort(-leg_magnitudes)
        keep_mask = np.zeros_like(leg_magnitudes, dtype=bool)
        for rank_idx, sym_idx in enumerate(rank_order):
            if leg_magnitudes[sym_idx] <= 1e-10:
                break
            if use_bhs:
                is_held = bool(already_open_mask[sym_idx]) if already_open_mask is not None else False
                if rank_idx < int(self.bhs_enter_top_k) or (
                    is_held and rank_idx < int(self.bhs_hold_top_k) and leg_shares[sym_idx] >= self.min_leg_share
                ):
                    keep_mask[sym_idx] = True
            else:
                if rank_idx < 3 or (rank_idx < self.max_leg_positions and leg_shares[sym_idx] >= self.min_leg_share):
                    keep_mask[sym_idx] = True
        return keep_mask

    def scale_and_deadband_weights(
        self,
        unit_weights: dict[str, float],
        current_weights: dict[str, float] | None,
        current_equity: float,
        vol_scalar: float = 1.0,
        recent_hourly_returns: object = None,
    ) -> tuple[dict[str, float], LeverageStateSnapshot]:
        """
        Scales 1x unit weights by `effective_leverage`, enforces per-asset weight caps
        and strict 100% dollar neutrality (sum(longs) == -sum(shorts)), and applies
        the proportional turnover deadband against `current_weights`.
        """
        if recent_hourly_returns is not None and abs(float(vol_scalar) - 1.0) <= 1e-9:
            arr_r = np.asarray(recent_hourly_returns, dtype=float)
            arr_r = arr_r[np.isfinite(arr_r)]
            if len(arr_r) >= 12:
                rv = float(np.std(arr_r, ddof=1))
                if rv > 1e-8:
                    vol_scalar = float(np.clip(0.003 / rv, 0.55, 1.15))

        snapshot = self.update_state(current_equity=current_equity, vol_scalar=vol_scalar)
        effective_leverage = snapshot.effective_leverage
        max_single = snapshot.max_single_asset_weight
        deadband = snapshot.proportional_deadband

        symbols = sorted(unit_weights.keys())
        if not symbols:
            return {}, snapshot

        raw_weights = np.array([float(unit_weights.get(s, 0.0)) * effective_leverage for s in symbols], dtype=float)
        prev_arr_for_bhs = (
            np.array([float(current_weights.get(s, 0.0)) for s in symbols], dtype=float)
            if current_weights
            else np.zeros_like(raw_weights)
        )
        initial_gross = float(np.sum(np.abs(raw_weights)))

        if initial_gross > 1e-8:
            target_gross = min(initial_gross, effective_leverage)
            raw_pruned = raw_weights.copy()
            if len(symbols) >= 8 and self.min_leg_share > 0.0:
                keep_pos = self._keep_top_leg_mask(
                    np.maximum(raw_pruned, 0.0),
                    already_open_mask=(prev_arr_for_bhs > 1e-6),
                )
                keep_neg = self._keep_top_leg_mask(
                    np.maximum(-raw_pruned, 0.0),
                    already_open_mask=(prev_arr_for_bhs < -1e-6),
                )
                raw_pruned = np.where((raw_pruned > 0.0) & (~keep_pos), 0.0, raw_pruned)
                raw_pruned = np.where((raw_pruned < 0.0) & (~keep_neg), 0.0, raw_pruned)

            pos_cnt = int((raw_pruned > 1e-10).sum())
            neg_cnt = int((raw_pruned < -1e-10).sum())
            has_leg_cap = self.max_leg_share is not None and 0.0 < self.max_leg_share < 1.0
            max_pos_single = (
                max(float(self.max_leg_share), (1.0 / pos_cnt) + 0.02) * (target_gross / 2.0)
                if (has_leg_cap and pos_cnt >= 2)
                else max_single
            )
            max_neg_single = (
                max(float(self.max_leg_share), (1.0 / neg_cnt) + 0.02) * (target_gross / 2.0)
                if (has_leg_cap and neg_cnt >= 2)
                else max_single
            )

            pos_leg = np.minimum(np.maximum(raw_pruned, 0.0), max_pos_single)
            neg_leg = np.maximum(np.minimum(raw_pruned, 0.0), -max_neg_single)
            if pos_leg.sum() > 1e-10 and np.abs(neg_leg.sum()) > 1e-10:
                pos_leg = pos_leg * ((target_gross / 2.0) / pos_leg.sum())
                neg_leg = neg_leg * ((target_gross / 2.0) / np.abs(neg_leg.sum()))
                pos_clipped = np.minimum(pos_leg, max_pos_single)
                neg_clipped = np.maximum(neg_leg, -max_neg_single)
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
                eff_gross_cap = target_gross if initial_gross > 1e-8 else effective_leverage
                signed_net = float(pos_updated.sum()) - float(np.abs(neg_updated.sum()))
                net_drift = abs(signed_net)
                gross_updated = float(pos_updated.sum()) + float(np.abs(neg_updated.sum()))

                # Anti-Churn Single-Leg Drift Absorption:
                # When 1-2 coins cross deadband causing moderate net drift, absorb the offset on the single
                # most-under-adjusted coin on the opposing leg before rescaling all other positions.
                if 0.025 < net_drift <= 2.5 * deadband and gross_updated <= eff_gross_cap + 1e-6:
                    if signed_net > 0:
                        # Longs > Shorts: either trim a Long with desired < updated, or add to a Short with desired < updated
                        gaps = np.where(
                            (updated < -1e-10) & (desired < updated),
                            updated - desired,
                            0.0,
                        )
                        if float(gaps.max()) > 1e-8:
                            best_i = int(np.argmax(gaps))
                            add_short = min(net_drift, float(gaps[best_i]))
                            updated[best_i] -= add_short
                    else:
                        gaps = np.where(
                            (updated > 1e-10) & (desired > updated),
                            desired - updated,
                            0.0,
                        )
                        if float(gaps.max()) > 1e-8:
                            best_i = int(np.argmax(gaps))
                            add_long = min(net_drift, float(gaps[best_i]))
                            updated[best_i] += add_long
                    pos_updated = np.maximum(updated, 0.0)
                    neg_updated = np.minimum(updated, 0.0)
                    net_drift = abs(float(pos_updated.sum()) - float(np.abs(neg_updated.sum())))
                    gross_updated = float(pos_updated.sum()) + float(np.abs(neg_updated.sum()))

                if net_drift > 0.025 or gross_updated > eff_gross_cap + 1e-6:
                    avg_leg = min(eff_gross_cap / 2.0, 0.5 * gross_updated)
                    pos_updated = pos_updated * (avg_leg / pos_updated.sum())
                    neg_updated = neg_updated * (avg_leg / np.abs(neg_updated.sum()))
                    updated = pos_updated + neg_updated

                has_leg_cap_u = self.max_leg_share is not None and 0.0 < self.max_leg_share < 1.0
                pos_u_cnt = int((updated > 1e-10).sum())
                neg_u_cnt = int((updated < -1e-10).sum())
                if has_leg_cap_u and (pos_u_cnt >= 2 or neg_u_cnt >= 2):
                    pos_cap_leg = np.maximum(updated, 0.0)
                    neg_cap_leg = np.maximum(-updated, 0.0)
                    if pos_cap_leg.sum() > 1e-10 and neg_cap_leg.sum() > 1e-10:
                        eff_p_share = (
                            max(float(self.max_leg_share), (1.0 / pos_u_cnt) + 0.02) if pos_u_cnt >= 2 else 1.0
                        )
                        eff_n_share = (
                            max(float(self.max_leg_share), (1.0 / neg_u_cnt) + 0.02) if neg_u_cnt >= 2 else 1.0
                        )
                        pos_cap_leg = np.minimum(pos_cap_leg, float(pos_cap_leg.sum()) * eff_p_share)
                        neg_cap_leg = np.minimum(neg_cap_leg, float(neg_cap_leg.sum()) * eff_n_share)
                        leg_target_u = min(float(pos_cap_leg.sum()), float(neg_cap_leg.sum()))
                        if leg_target_u > 1e-10:
                            updated = pos_cap_leg * (leg_target_u / float(pos_cap_leg.sum())) - neg_cap_leg * (
                                leg_target_u / float(neg_cap_leg.sum())
                            )
            else:
                updated = desired
        else:
            updated = desired

        # Unconditional < $160 Zombie Stub Pruning when Core gross notional >= $2,000
        eq_val = max(float(current_equity), 1.0)
        gross_usd = float(np.sum(np.abs(updated))) * eq_val
        if gross_usd >= 2000.0:
            min_stub_w = 160.0 / eq_val
            for sign in (1.0, -1.0):
                leg_mask = (updated * sign) > 1e-8
                if int(leg_mask.sum()) >= 2:
                    stub_mask = leg_mask & (np.abs(updated) < min_stub_w)
                    keep_mask = leg_mask & (np.abs(updated) >= min_stub_w)
                    if np.any(stub_mask) and np.any(keep_mask):
                        leg_sum_before = float(np.abs(updated[leg_mask]).sum())
                        updated[stub_mask] = 0.0
                        keep_sum = float(np.abs(updated[keep_mask]).sum())
                        if keep_sum > 1e-10:
                            updated[keep_mask] = updated[keep_mask] * (leg_sum_before / keep_sum)

        final_weights = {sym: float(updated[i]) for i, sym in enumerate(symbols) if abs(updated[i]) > 1e-8}
        return final_weights, snapshot
