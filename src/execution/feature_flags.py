"""
Production Feature Flags & Risk Shield Configuration.

Provides a unified, hot-reloadable configuration layer controllable via:
1. Environment variables (`FF_ENABLE_ANOMALY_CAPSULE`, `FF_EQUITY_FLOOR_USD`, etc.)
2. CLI arguments (`--disable-anomalies`, `--equity-floor-usd`, `--set-flag key=val`)
3. Runtime SQLite persistence (`feature_flags` table) for zero-downtime toggling.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import os
from typing import Any


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    text = str(val).strip().lower()
    if text in ("1", "true", "yes", "on", "enabled"):
        return True
    if text in ("0", "false", "no", "off", "disabled"):
        return False
    return default


def _parse_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


FLOAT_FLAG_NAMES = frozenset(
    {
        "equity_floor_usd",
        "chop_take_profit_pct",
        "chop_min_tp_pct",
        "chop_max_tp_pct",
    }
)


@dataclass
class FeatureFlags:
    """
    Runtime Feature Flags & Risk Shield guardrails:
    - `enable_anomaly_capsule` (default False in Mainnet) + `auto_close_anomalies_when_disabled` (True).
    - `enable_hard_equity_floor` (True) + `equity_floor_usd` ($2,600.00 hard stop in Mainnet).
    - `enable_invariant_audit` (True) — continuous verification of production invariants.
    """

    enable_anomaly_capsule: bool = False
    auto_close_anomalies_when_disabled: bool = True
    enable_hard_equity_floor: bool = True
    equity_floor_usd: float = 2600.0
    enable_intracycle_rebalance: bool = True
    enable_invariant_audit: bool = True
    enable_smart_money: bool = True
    enable_quarterly_basis: bool = True
    enable_chop_range_arb: bool = True
    enable_dynamic_barriers: bool = True
    enable_holistic_regime_shields: bool = True
    chop_take_profit_pct: float = 0.035
    chop_min_tp_pct: float = 0.028
    chop_max_tp_pct: float = 0.036

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, fallback: "FeatureFlags | None" = None) -> "FeatureFlags":
        base = fallback.to_dict() if fallback is not None else asdict(cls())
        if not data:
            return cls(**base)
        merged = dict(base)
        for field_info in fields(cls):
            name = field_info.name
            if name in data:
                if name in FLOAT_FLAG_NAMES:
                    merged[name] = _parse_float(data[name], merged[name])
                else:
                    merged[name] = _parse_bool(data[name], merged[name])
        return cls(**merged)

    @classmethod
    def from_env_and_args(
        cls,
        args: Any = None,
        mode: str = "shadow",
        environ: Mapping[str, str] | None = None,
        enable_institutional_data_feeds: bool = True,
        is_m15_core: bool | None = None,
    ) -> "FeatureFlags":
        env = environ if environ is not None else os.environ
        is_mainnet = mode == "mainnet"
        if is_m15_core is not None:
            enable_institutional_data_feeds = bool(is_m15_core)

        cli_anom = getattr(args, "enable_anomalies", None) if args is not None else None
        env_anom = env.get("FF_ENABLE_ANOMALY_CAPSULE", env.get("ENABLE_ANOMALY_CAPSULE"))
        if env_anom is not None:
            enable_anom = _parse_bool(env_anom, default=not is_mainnet)
        elif cli_anom is not None:
            enable_anom = bool(cli_anom)
        else:
            enable_anom = not is_mainnet

        if cli_anom is False:
            enable_anom = False

        auto_close_anom = _parse_bool(
            env.get("FF_AUTO_CLOSE_ANOMALIES_WHEN_DISABLED"),
            default=True,
        )

        cli_floor = getattr(args, "equity_floor_usd", None) if args is not None else None
        env_floor = env.get("FF_EQUITY_FLOOR_USD", env.get("EQUITY_FLOOR_USD"))
        default_floor = 2600.0 if is_mainnet else 0.0
        if cli_floor is not None:
            floor_usd = float(cli_floor)
        elif env_floor is not None:
            floor_usd = _parse_float(env_floor, default=default_floor)
        else:
            floor_usd = default_floor

        enable_floor = _parse_bool(
            env.get("FF_ENABLE_HARD_EQUITY_FLOOR"),
            default=(floor_usd > 0.0),
        )

        cli_intracycle = getattr(args, "enable_intracycle_rebalance", None) if args is not None else None
        env_intracycle = env.get("FF_ENABLE_INTRACYCLE_REBALANCE")
        if cli_intracycle is not None:
            enable_intracycle = bool(cli_intracycle)
        elif env_intracycle is not None:
            enable_intracycle = _parse_bool(env_intracycle, default=True)
        else:
            enable_intracycle = True

        enable_invariants = _parse_bool(
            env.get("FF_ENABLE_INVARIANT_AUDIT"),
            default=True,
        )
        enable_sm = _parse_bool(
            env.get("FF_ENABLE_SMART_MONEY"),
            default=bool(enable_institutional_data_feeds),
        )
        enable_qb = _parse_bool(
            env.get("FF_ENABLE_QUARTERLY_BASIS"),
            default=bool(enable_institutional_data_feeds),
        )
        cli_chop = getattr(args, "enable_chop_range_arb", None) if args is not None else None
        env_chop = env.get("FF_ENABLE_CHOP_RANGE_ARB")
        if env_chop is not None:
            enable_chop = _parse_bool(env_chop, default=True)
        elif cli_chop is not None:
            enable_chop = bool(cli_chop)
        else:
            enable_chop = True

        chop_tp = _parse_float(
            env.get("FF_CHOP_TAKE_PROFIT_PCT", getattr(args, "chop_tp", None) if args is not None else None),
            default=0.035,
        )
        enable_dyn_b = _parse_bool(
            env.get("FF_ENABLE_DYNAMIC_BARRIERS"),
            default=True,
        )
        enable_holistic = _parse_bool(
            env.get("FF_ENABLE_HOLISTIC_REGIME_SHIELDS"),
            default=True,
        )
        chop_min_tp = _parse_float(
            env.get("FF_CHOP_MIN_TP_PCT"),
            default=0.028,
        )
        chop_max_tp = _parse_float(
            env.get("FF_CHOP_MAX_TP_PCT"),
            default=0.036,
        )

        return cls(
            enable_anomaly_capsule=enable_anom,
            auto_close_anomalies_when_disabled=auto_close_anom,
            enable_hard_equity_floor=enable_floor,
            equity_floor_usd=floor_usd,
            enable_intracycle_rebalance=enable_intracycle,
            enable_invariant_audit=enable_invariants,
            enable_smart_money=enable_sm,
            enable_quarterly_basis=enable_qb,
            enable_chop_range_arb=enable_chop,
            enable_dynamic_barriers=enable_dyn_b,
            enable_holistic_regime_shields=enable_holistic,
            chop_take_profit_pct=chop_tp,
            chop_min_tp_pct=chop_min_tp,
            chop_max_tp_pct=chop_max_tp,
        )

    def apply_override(self, key: str, raw_value: Any) -> tuple[str, Any]:
        """Applies a single key=value override, returning (canonical_key, parsed_value)."""
        aliases = {
            "anomalies": "enable_anomaly_capsule",
            "enable_anomalies": "enable_anomaly_capsule",
            "anomaly_capsule": "enable_anomaly_capsule",
            "enable_anomaly_capsule": "enable_anomaly_capsule",
            "auto_close_anomalies": "auto_close_anomalies_when_disabled",
            "auto_close_anomalies_when_disabled": "auto_close_anomalies_when_disabled",
            "hard_equity_floor": "enable_hard_equity_floor",
            "enable_hard_equity_floor": "enable_hard_equity_floor",
            "equity_floor": "equity_floor_usd",
            "equity_floor_usd": "equity_floor_usd",
            "intracycle_rebalance": "enable_intracycle_rebalance",
            "enable_intracycle_rebalance": "enable_intracycle_rebalance",
            "invariant_audit": "enable_invariant_audit",
            "enable_invariant_audit": "enable_invariant_audit",
            "smart_money": "enable_smart_money",
            "enable_smart_money": "enable_smart_money",
            "quarterly_basis": "enable_quarterly_basis",
            "enable_quarterly_basis": "enable_quarterly_basis",
            "chop_range_arb": "enable_chop_range_arb",
            "enable_chop_range_arb": "enable_chop_range_arb",
            "dynamic_barriers": "enable_dynamic_barriers",
            "enable_dynamic_barriers": "enable_dynamic_barriers",
            "holistic_shields": "enable_holistic_regime_shields",
            "holistic_regime_shields": "enable_holistic_regime_shields",
            "enable_holistic_regime_shields": "enable_holistic_regime_shields",
            "chop_tp": "chop_take_profit_pct",
            "chop_take_profit_pct": "chop_take_profit_pct",
            "chop_min_tp": "chop_min_tp_pct",
            "chop_min_tp_pct": "chop_min_tp_pct",
            "chop_max_tp": "chop_max_tp_pct",
            "chop_max_tp_pct": "chop_max_tp_pct",
        }
        norm_key = aliases.get(key.strip().lower())
        if not norm_key:
            valid = ", ".join(sorted(asdict(self).keys()))
            raise ValueError(f"Unknown feature flag '{key}'. Valid flags: {valid}")

        if norm_key in FLOAT_FLAG_NAMES:
            parsed: Any = _parse_float(raw_value, getattr(self, norm_key))
        else:
            parsed = _parse_bool(raw_value, getattr(self, norm_key))

        setattr(self, norm_key, parsed)
        return norm_key, parsed
