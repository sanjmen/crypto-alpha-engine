"""
Unit tests for Zero-Trust API Key Vault, Pre-Flight Permission Validator,
Credential Masking, and IP Whitelist Verification.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.security.vault import (
    CredentialVault,
    ExchangeCredentials,
    IpWhitelistValidator,
    MissingCredentialsError,
    PreFlightPermissionValidator,
    SecurityViolationError,
    SensitiveDataMasker,
    SensitiveLogFilter,
    UnrestrictedIpError,
    VaultDecryptionError,
    WithdrawalPermissionEnabledError,
)


def test_exchange_credentials_masking():
    """Verifies credentials string representation and properties never leak secret keys."""
    creds = ExchangeCredentials(
        exchange_id="binance",
        api_key="vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A",
        api_secret="NhqPtMDs5NVvsHalHG8Nm5fqSLQIpkUWo2WaWMyFsOm6qKE2zvsw0MuIgwCIPy6u",
        passphrase="my_super_secret_passphrase",
    )

    # API key masking
    assert creds.masked_api_key == "vmPU***Eh8A"
    assert creds.masked_api_secret == "******"

    # Representation checks
    rep = repr(creds)
    assert creds.api_secret not in rep
    assert creds.passphrase not in rep
    assert "vmPU***Eh8A" in rep
    assert str(creds) == rep


def test_sensitive_data_masker_and_log_filter():
    """Verifies that SensitiveDataMasker and SensitiveLogFilter scrub registered secrets."""
    creds = ExchangeCredentials(
        exchange_id="binance",
        api_key="KEY_1234567890ABCDEF",
        api_secret="SECRET_9876543210ZYXWVU",
    )

    masker = SensitiveDataMasker()
    masker.register_credentials(creds)

    raw_text = (
        f"Connecting to Binance with apiKey={creds.api_key} and secret={creds.api_secret}."
    )
    masked = masker.mask(raw_text)

    assert creds.api_key not in masked
    assert creds.api_secret not in masked
    assert "***REDACTED" in masked

    # Test logging filter
    log_filter = SensitiveLogFilter(masker)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg=f"Failed auth for secret {creds.api_secret}",
        args=(),
        exc_info=None,
    )
    log_filter.filter(record)
    assert creds.api_secret not in record.msg


def test_credential_vault_env_loading(monkeypatch: pytest.MonkeyPatch):
    """Verifies loading credentials from environment variables."""
    monkeypatch.setenv("BINANCE_API_KEY", "env_test_api_key_123456")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_test_api_secret_654321")

    creds = CredentialVault.from_env(exchange_id="binance")
    assert creds.exchange_id == "binance"
    assert creds.api_key == "env_test_api_key_123456"
    assert creds.api_secret == "env_test_api_secret_654321"

    # Missing credentials should raise MissingCredentialsError
    monkeypatch.delenv("BINANCE_API_KEY")
    with pytest.raises(MissingCredentialsError):
        CredentialVault.from_env(exchange_id="binance")


def test_credential_vault_encrypted_file_roundtrip():
    """Verifies encryption and decryption of credentials in vault file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault_path = Path(tmp_dir) / "vault.enc"
        master_pw = "UltraSecureVaultPassphrase2026!"

        creds = {
            "binance": ExchangeCredentials(
                exchange_id="binance",
                api_key="binance_key_9999",
                api_secret="binance_sec_8888",
            ),
            "bybit": ExchangeCredentials(
                exchange_id="bybit",
                api_key="bybit_key_7777",
                api_secret="bybit_sec_6666",
            ),
        }

        # Save encrypted file
        CredentialVault.save_to_encrypted_file(creds, vault_path, master_pw)
        assert vault_path.exists()
        raw_content = vault_path.read_text()
        assert "binance_sec_8888" not in raw_content
        assert "bybit_sec_6666" not in raw_content

        # Load with correct password
        decrypted = CredentialVault.load_from_encrypted_file(vault_path, master_pw)
        assert "binance" in decrypted
        assert "bybit" in decrypted
        assert decrypted["binance"].api_key == "binance_key_9999"
        assert decrypted["binance"].api_secret == "binance_sec_8888"
        assert decrypted["bybit"].api_key == "bybit_key_7777"

        # Load with wrong password should raise VaultDecryptionError
        with pytest.raises(VaultDecryptionError):
            CredentialVault.load_from_encrypted_file(vault_path, "WrongPassword123!")


def test_preflight_permission_validator_withdrawal_fatal():
    """Verifies that withdrawal permissions enabled causes an immediate fatal abort."""
    mock_exchange = MagicMock()
    mock_exchange.sapi_get_account_apirestrictions.return_value = {
        "ipRestrict": True,
        "enableWithdrawals": True,  # FATAL VIOLATION
        "enableReading": True,
        "enableFutures": True,
    }

    with pytest.raises(WithdrawalPermissionEnabledError) as exc_info:
        PreFlightPermissionValidator.validate_binance_permissions(mock_exchange)

    assert "Withdrawal permissions are ENABLED" in str(exc_info.value)


def test_preflight_permission_validator_unrestricted_ip():
    """Verifies strict IP whitelist validation behavior."""
    unrestricted_payload = {
        "ipRestrict": False,  # Missing IP whitelist restriction
        "enableWithdrawals": False,
        "enableReading": True,
        "enableFutures": True,
    }

    # Strict mode should raise UnrestrictedIpError
    with pytest.raises(UnrestrictedIpError):
        PreFlightPermissionValidator.validate_binance_permissions(
            unrestricted_payload, strict_ip_whitelist=True
        )

    # Permissive mode should pass with warning
    report = PreFlightPermissionValidator.validate_binance_permissions(
        unrestricted_payload, strict_ip_whitelist=False
    )
    assert report.is_safe_for_trading
    assert not report.ip_restricted
    assert len(report.warnings) > 0


def test_preflight_permission_validator_valid_account():
    """Verifies successful validation of an institutional trading API key."""
    valid_payload = {
        "ipRestrict": True,
        "enableWithdrawals": False,
        "enableReading": True,
        "enableFutures": True,
        "enableSpotAndMarginTrading": False,
    }

    report = PreFlightPermissionValidator.validate_binance_permissions(valid_payload)
    assert report.is_safe_for_trading
    assert not report.can_withdraw
    assert report.ip_restricted
    assert report.can_trade_futures
    assert len(report.warnings) == 0


def test_preflight_permission_validator_missing_futures():
    """Verifies error raised when futures trading is required but disabled."""
    missing_futures_payload = {
        "ipRestrict": True,
        "enableWithdrawals": False,
        "enableReading": True,
        "enableFutures": False,
    }

    with pytest.raises(SecurityViolationError) as exc_info:
        PreFlightPermissionValidator.validate_binance_permissions(
            missing_futures_payload, require_futures=True
        )
    assert "enableFutures is False" in str(exc_info.value)


def test_ip_whitelist_validator():
    """Verifies exact IP and CIDR subnet whitelist matching."""
    whitelist = [
        "192.168.1.100",
        "10.0.0.0/16",
        "203.0.113.42",
    ]

    # Matching exact IP
    assert IpWhitelistValidator.verify_ip_against_whitelist("192.168.1.100", whitelist)
    assert IpWhitelistValidator.verify_ip_against_whitelist("203.0.113.42", whitelist)

    # Matching CIDR subnet
    assert IpWhitelistValidator.verify_ip_against_whitelist("10.0.5.21", whitelist)
    assert IpWhitelistValidator.verify_ip_against_whitelist("10.0.254.1", whitelist)

    # Non-matching IP
    assert not IpWhitelistValidator.verify_ip_against_whitelist("192.168.1.101", whitelist)
    assert not IpWhitelistValidator.verify_ip_against_whitelist("10.1.0.1", whitelist)
    assert not IpWhitelistValidator.verify_ip_against_whitelist("8.8.8.8", whitelist)
    assert not IpWhitelistValidator.verify_ip_against_whitelist("invalid-ip", whitelist)
