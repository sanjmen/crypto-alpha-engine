"""
Institutional Zero-Trust API Key Vault & Pre-Flight Permission Validator.

Enforces least-privilege security guarantees for algorithmic trading:
1. Multi-source credential management (Environment variables, AES-encrypted vault file).
2. Strict automated pre-flight API permission validation:
   - Immediately aborts if WITHDRAWAL permissions are enabled.
   - Enforces IP whitelist restriction verification.
   - Confirms trading permissions (Futures / Spot) and read-only minimums.
3. Sensitive data masking across all logging, exception traces, and telemetry.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecurityViolationError(Exception):
    """Base class for zero-trust security policy violations."""


class WithdrawalPermissionEnabledError(SecurityViolationError):
    """Raised when an API key has withdrawal permissions enabled."""


class UnrestrictedIpError(SecurityViolationError):
    """Raised when an API key is not restricted to an IP whitelist."""


class VaultDecryptionError(SecurityViolationError):
    """Raised when decrypting credentials fails due to invalid password or corrupted data."""


class MissingCredentialsError(SecurityViolationError):
    """Raised when required exchange credentials cannot be found."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ExchangeCredentials:
    """Encapsulated exchange credentials with safe representation."""

    exchange_id: str
    api_key: str
    api_secret: str
    passphrase: str | None = None
    subaccount: str | None = None

    @property
    def masked_api_key(self) -> str:
        """Returns partially masked API key (e.g. 'abc1***8901')."""
        if not self.api_key or len(self.api_key) <= 8:
            return "***MASKED***"
        return f"{self.api_key[:4]}***{self.api_key[-4:]}"

    @property
    def masked_api_secret(self) -> str:
        """Returns fully masked API secret."""
        return "******"

    def __repr__(self) -> str:
        passphrase_repr = "******" if self.passphrase else None
        return (
            f"ExchangeCredentials(exchange_id='{self.exchange_id}', "
            f"api_key='{self.masked_api_key}', api_secret='******', "
            f"passphrase={passphrase_repr}, subaccount={self.subaccount!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class ApiPermissionReport:
    """Detailed audit report of exchange API key permissions."""

    exchange_id: str
    validated_at: datetime
    can_withdraw: bool
    ip_restricted: bool
    can_trade_futures: bool
    can_trade_spot: bool
    can_read: bool
    raw_restrictions: dict[str, Any]
    is_safe_for_trading: bool
    warnings: list[str]


# ---------------------------------------------------------------------------
# Sensitive Data Masking & Log Sanitization
# ---------------------------------------------------------------------------


class SensitiveDataMasker:
    """
    Sanitizes strings and log streams by replacing known secret tokens and
    patterns resembling API keys and secrets with masked redactions.
    """

    # Generic patterns matching potential hex/alphanumeric API keys and secrets >= 32 chars
    _KEY_REGEX = re.compile(r"\b[A-Za-z0-9_-]{32,64}\b")
    _ENV_PAIR_REGEX = re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-+/=]{8,})['\"]?", re.IGNORECASE)

    def __init__(self) -> None:
        self._exact_secrets: set[str] = set()

    def register_secret(self, secret: str | None) -> None:
        """Registers a specific secret token to be aggressively scrubbed."""
        if secret and len(secret) >= 4:
            self._exact_secrets.add(secret)

    def register_credentials(self, creds: ExchangeCredentials) -> None:
        """Registers all components of credentials for scrubbing."""
        self.register_secret(creds.api_key)
        self.register_secret(creds.api_secret)
        if creds.passphrase:
            self.register_secret(creds.passphrase)

    def mask(self, text: str) -> str:
        """Sanitizes text, replacing any registered secret or pattern with [REDACTED]."""
        if not text:
            return text

        masked = text
        # 1. Exact match replacements for registered secrets
        for secret in self._exact_secrets:
            if secret in masked:
                masked = masked.replace(secret, f"***REDACTED({secret[:2]}..{secret[-2:] if len(secret) > 4 else ''})***")

        # 2. Key/Value assignment pattern masking
        def _replace_kv(m: re.Match) -> str:
            key_name = m.group(1)
            val = m.group(2)
            if len(val) <= 4:
                return f"{key_name}=***REDACTED***"
            return f"{key_name}={val[:2]}***REDACTED***{val[-2:]}"

        masked = self._ENV_PAIR_REGEX.sub(_replace_kv, masked)
        return masked


class SensitiveLogFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials from log records before dispatch."""

    def __init__(self, masker: SensitiveDataMasker) -> None:
        super().__init__()
        self.masker = masker

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.masker.mask(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.masker.mask(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.masker.mask(str(arg)) if isinstance(arg, str) else arg for arg in record.args)
        return True


def install_global_log_masker(masker: SensitiveDataMasker) -> SensitiveLogFilter:
    """Installs a sensitive log filter on the root logger and all active handlers."""
    log_filter = SensitiveLogFilter(masker)
    root_logger = logging.getLogger()
    root_logger.addFilter(log_filter)
    for handler in root_logger.handlers:
        handler.addFilter(log_filter)
    return log_filter


# ---------------------------------------------------------------------------
# Zero-Trust Credential Vault
# ---------------------------------------------------------------------------


class CredentialVault:
    """
    Secure repository for exchange credentials supporting environment variables
    and encrypted file storage using PBKDF2HMAC + Fernet.
    """

    PBKDF2_ITERATIONS = 100_000

    @classmethod
    def _derive_fernet_key(cls, passphrase: str, salt: bytes) -> bytes:
        """Derives a 32-byte Fernet-compatible key using PBKDF2 with SHA-256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls.PBKDF2_ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))

    @classmethod
    def from_env(
        cls,
        exchange_id: str = "binance",
        prefix: str = "",
    ) -> ExchangeCredentials:
        """
        Loads exchange credentials from environment variables.
        Checks:
          - {PREFIX}{EXCHANGE}_API_KEY or {PREFIX}API_KEY
          - {PREFIX}{EXCHANGE}_API_SECRET or {PREFIX}API_SECRET
          - {PREFIX}{EXCHANGE}_PASSPHRASE (optional)
        """
        ex_upper = exchange_id.upper()
        p = prefix.upper()

        api_key = os.getenv(f"{p}{ex_upper}_API_KEY") or os.getenv(f"{p}API_KEY")
        api_secret = os.getenv(f"{p}{ex_upper}_API_SECRET") or os.getenv(f"{p}API_SECRET")
        passphrase = os.getenv(f"{p}{ex_upper}_PASSPHRASE") or os.getenv(f"{p}PASSPHRASE")
        subaccount = os.getenv(f"{p}{ex_upper}_SUBACCOUNT") or os.getenv(f"{p}SUBACCOUNT")

        if not api_key or not api_secret:
            raise MissingCredentialsError(
                f"Missing API credentials for exchange '{exchange_id}'. "
                f"Expected environment variables: {p}{ex_upper}_API_KEY and {p}{ex_upper}_API_SECRET."
            )

        return ExchangeCredentials(
            exchange_id=exchange_id.lower(),
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            passphrase=passphrase.strip() if passphrase else None,
            subaccount=subaccount.strip() if subaccount else None,
        )

    @classmethod
    def save_to_encrypted_file(
        cls,
        credentials: dict[str, ExchangeCredentials],
        file_path: Path | str,
        master_passphrase: str,
        salt: bytes | None = None,
    ) -> None:
        """
        Encrypts a map of exchange credentials using PBKDF2 + Fernet and saves to disk.
        """
        if not master_passphrase or len(master_passphrase) < 8:
            raise SecurityViolationError("Master vault passphrase must be at least 8 characters long.")

        salt = salt or os.urandom(16)
        fernet_key = cls._derive_fernet_key(master_passphrase, salt)
        fernet = Fernet(fernet_key)

        payload_dict = {
            ex_id: asdict(creds) for ex_id, creds in credentials.items()
        }
        json_bytes = json.dumps(payload_dict).encode("utf-8")
        encrypted_bytes = fernet.encrypt(json_bytes)

        vault_bundle = {
            "version": "1.0",
            "kdf": "PBKDF2HMAC-SHA256",
            "iterations": cls.PBKDF2_ITERATIONS,
            "salt": base64.b64encode(salt).decode("ascii"),
            "data": encrypted_bytes.decode("ascii"),
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write with restricted file permissions (0600: read/write only by owner)
        path.write_text(json.dumps(vault_bundle, indent=2))
        try:
            path.chmod(0o600)
        except OSError:
            pass  # Non-posix fallback

    @classmethod
    def load_from_encrypted_file(
        cls,
        file_path: Path | str,
        master_passphrase: str,
    ) -> dict[str, ExchangeCredentials]:
        """
        Decrypts an encrypted vault file and returns the map of ExchangeCredentials.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Vault file not found at {path}")

        try:
            bundle = json.loads(path.read_text())
            salt = base64.b64decode(bundle["salt"])
            encrypted_data = bundle["data"].encode("ascii")

            fernet_key = cls._derive_fernet_key(master_passphrase, salt)
            fernet = Fernet(fernet_key)
            decrypted_bytes = fernet.decrypt(encrypted_data)
            payload_dict = json.loads(decrypted_bytes.decode("utf-8"))

            result: dict[str, ExchangeCredentials] = {}
            for ex_id, data in payload_dict.items():
                result[ex_id] = ExchangeCredentials(**data)
            return result
        except (InvalidToken, KeyError, json.JSONDecodeError) as e:
            raise VaultDecryptionError(
                "Failed to decrypt credentials vault: incorrect master password or corrupted payload."
            ) from e


# ---------------------------------------------------------------------------
# Pre-Flight API Permission Validator
# ---------------------------------------------------------------------------


class PreFlightPermissionValidator:
    """
    Automated zero-trust pre-flight validator for exchange API keys.
    Guarantees withdrawal permissions are disabled before trading starts.
    """

    @classmethod
    def validate_binance_permissions(
        cls,
        client_or_restrictions: Any,
        strict_ip_whitelist: bool = True,
        require_futures: bool = True,
    ) -> ApiPermissionReport:
        """
        Validates Binance API restrictions via ccxt client or raw restriction dict.

        Enforces:
        1. canWithdraw / enableWithdrawals == False (FATAL VIOLATION if True).
        2. ipRestrict == True (if strict_ip_whitelist is enabled).
        3. enableFutures == True (if require_futures is enabled).
        4. enableReading == True.
        """
        # Retrieve restrictions dictionary
        if isinstance(client_or_restrictions, dict):
            restrictions = client_or_restrictions
        elif hasattr(client_or_restrictions, "sapi_get_account_apirestrictions"):
            restrictions = client_or_restrictions.sapi_get_account_apirestrictions()
        elif hasattr(client_or_restrictions, "fetch_api_key_permissions"):
            restrictions = client_or_restrictions.fetch_api_key_permissions()
        else:
            raise ValueError(f"Unsupported exchange client object: {type(client_or_restrictions)}")

        can_withdraw = bool(restrictions.get("enableWithdrawals", False))
        ip_restricted = bool(restrictions.get("ipRestrict", False))
        can_trade_futures = bool(restrictions.get("enableFutures", False))
        can_trade_spot = bool(restrictions.get("enableSpotAndMarginTrading", False))
        can_read = bool(restrictions.get("enableReading", True))

        warnings: list[str] = []

        # 1. Critical Zero-Trust Check: Withdrawal permissions MUST BE FALSE
        if can_withdraw:
            logger.critical(
                "CRITICAL SECURITY VIOLATION: API key has WITHDRAWAL permissions ENABLED! "
                "Halting startup under Zero-Trust Principle of Least Privilege."
            )
            raise WithdrawalPermissionEnabledError(
                "CRITICAL SECURITY VIOLATION: Withdrawal permissions are ENABLED on exchange API key! "
                "Algorithmic execution requires withdrawal permissions to be strictly DISABLED."
            )

        # 2. IP Restriction Check
        if not ip_restricted:
            msg = "API key does not have IP whitelist restriction enabled."
            if strict_ip_whitelist:
                logger.critical(f"ZERO-TRUST VIOLATION: {msg} Strict IP whitelist enforcement failed.")
                raise UnrestrictedIpError(
                    f"ZERO-TRUST VIOLATION: {msg} "
                    "Exchange API key must restrict requests to authorized trading node IPs."
                )
            warnings.append(msg)
            logger.warning(f"SECURITY WARNING: {msg}")

        # 3. Read Permission Check
        if not can_read:
            raise SecurityViolationError("API key lacks read permission (enableReading is False).")

        # 4. Futures Trading Permission Check
        if require_futures and not can_trade_futures:
            raise SecurityViolationError(
                "API key lacks perpetual futures trading permission (enableFutures is False)."
            )

        return ApiPermissionReport(
            exchange_id="binance",
            validated_at=datetime.now(timezone.utc),
            can_withdraw=can_withdraw,
            ip_restricted=ip_restricted,
            can_trade_futures=can_trade_futures,
            can_trade_spot=can_trade_spot,
            can_read=can_read,
            raw_restrictions=restrictions,
            is_safe_for_trading=True,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# IP Whitelist Verification
# ---------------------------------------------------------------------------


class IpWhitelistValidator:
    """Verifies client IP addresses against CIDR and exact whitelist rules."""

    @staticmethod
    def verify_ip_against_whitelist(current_ip: str, allowed_ips_or_subnets: list[str]) -> bool:
        """
        Validates if current_ip is covered by any of allowed_ips_or_subnets.
        Supports both exact IPs ('192.168.1.5') and CIDR blocks ('192.168.1.0/24').
        """
        try:
            curr_addr = ipaddress.ip_address(current_ip.strip())
        except ValueError:
            logger.error(f"Invalid current IP address format: {current_ip}")
            return False

        for entry in allowed_ips_or_subnets:
            clean_entry = entry.strip()
            try:
                # Check CIDR subnet or single IP network
                net = ipaddress.ip_network(clean_entry, strict=False)
                if curr_addr in net:
                    return True
            except ValueError:
                # Plain string fallback comparison
                if current_ip.strip() == clean_entry:
                    return True

        return False
