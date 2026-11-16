"""
Production Feature Flags & Risk Shield Configuration (3 Reglas de Blindaje).

Provides a unified, hot-reloadable configuration layer controllable via:
1. Environment variables (`FF_ENABLE_ANOMALY_CAPSULE`, `FF_EQUITY_FLOOR_USD`, etc.)
2. CLI arguments (`--disable-anomalies`, `--equity-floor-usd`, `--set-flag key=val`)
3. Runtime SQLite persistence (`feature_flags` table) for zero-downtime toggling.
"""

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "enabled"):
        return True
    if s in ("0", "false", "no", "off", "disabled"):
        return False
    return default


def _parse_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@dataclass
class FeatureFlags:
    """
    Runtime Feature Flags & Risk Shield guardrails.

    Regla #1: `enable_anomaly_capsule` (default False in Mainnet) + `auto_close_anomalies_when_disabled` (True).
    Regla #2: `enable_hard_equity_floor` (True) + `equity_floor_usd` ($2,600.00 hard stop in Mainnet).
    Regla #3: `enable_invariant_audit` (True) — continuous verification of the 4 production invariants.
    """

    enable_anomaly_capsule: bool = False
    auto_close_anomalies_when_disabled: bool = True
    enable_hard_equity_floor: bool = True
    equity_floor_usd: float = 2600.0
    enable_intracycle_rebalance: bool = True
    enable_invariant_audit: bool = True
    enable_smart_money: bool = True
    enable_quarterly_basis: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, fallback: "FeatureFlags | None" = None) -> "FeatureFlags":
        base = fallback.to_dict() if fallback is not None else asdict(cls())
        if not data:
            return cls(**base)
        merged = dict(base)
        if "enable_anomaly_capsule" in data:
            merged["enable_anomaly_capsule"] = _parse_bool(
                data["enable_anomaly_capsule"], merged["enable_anomaly_capsule"]
            )
        if "auto_close_anomalies_when_disabled" in data:
            merged["auto_close_anomalies_when_disabled"] = _parse_bool(
                data["auto_close_anomalies_when_disabled"],
                merged["auto_close_anomalies_when_disabled"],
            )
        if "enable_hard_equity_floor" in data:
            merged["enable_hard_equity_floor"] = _parse_bool(
                data["enable_hard_equity_floor"], merged["enable_hard_equity_floor"]
            )
        if "equity_floor_usd" in data:
            merged["equity_floor_usd"] = _parse_float(
                data["equity_floor_usd"], merged["equity_floor_usd"]
            )
        if "enable_intracycle_rebalance" in data:
            merged["enable_intracycle_rebalance"] = _parse_bool(
                data["enable_intracycle_rebalance"], merged["enable_intracycle_rebalance"]
            )
        if "enable_invariant_audit" in data:
            merged["enable_invariant_audit"] = _parse_bool(
                data["enable_invariant_audit"], merged["enable_invariant_audit"]
            )
        if "enable_smart_money" in data:
            merged["enable_smart_money"] = _parse_bool(
                data["enable_smart_money"], merged["enable_smart_money"]
            )
        if "enable_quarterly_basis" in data:
            merged["enable_quarterly_basis"] = _parse_bool(
                data["enable_quarterly_basis"], merged["enable_quarterly_basis"]
            )
        return cls(**merged)

    @classmethod
    def from_env_and_args(
        cls,
        args: Any = None,
        mode: str = "shadow",
        environ: Mapping[str, str] | None = None,
        is_m15_core: bool = True,
    ) -> "FeatureFlags":
        env = environ if environ is not None else os.environ
        is_mainnet = (mode == "mainnet")

        # 1. Anomaly Capsule flag (Regla #1):
        # Priority: explicit CLI flag -> env var FF_ENABLE_ANOMALY_CAPSULE / ENABLE_ANOMALY_CAPSULE -> mode default
        cli_anom = getattr(args, "enable_anomalies", None) if args is not None else None
        env_anom = env.get("FF_ENABLE_ANOMALY_CAPSULE", env.get("ENABLE_ANOMALY_CAPSULE"))
        if env_anom is not None:
            enable_anom = _parse_bool(env_anom, default=not is_mainnet)
        elif cli_anom is not None:
            enable_anom = bool(cli_anom)
        else:
            enable_anom = not is_mainnet

        # If CLI explicitly passed --disable-anomalies, always honor False
        if cli_anom is False:
            enable_anom = False

        auto_close_anom = _parse_bool(
            env.get("FF_AUTO_CLOSE_ANOMALIES_WHEN_DISABLED"),
            default=True,
        )

        # 2. Hard Equity Floor (Regla #2):
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

        # 3. Intracycle Rebalance & Invariant Audit (Regla #3):
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
            default=bool(is_m15_core),
        )
        enable_qb = _parse_bool(
            env.get("FF_ENABLE_QUARTERLY_BASIS"),
            default=bool(is_m15_core),
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
        }
        norm_key = aliases.get(key.strip().lower())
        if not norm_key:
            valid = ", ".join(sorted(asdict(self).keys()))
            raise ValueError(f"Unknown feature flag '{key}'. Valid flags: {valid}")

        if norm_key == "equity_floor_usd":
            parsed: Any = _parse_float(raw_value, self.equity_floor_usd)
        else:
            parsed = _parse_bool(raw_value, getattr(self, norm_key))

        setattr(self, norm_key, parsed)
        return norm_key, parsed
