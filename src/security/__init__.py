"""
Zero-trust security vault, credential management, and pre-flight exchange API permission validation.
"""

from src.security.vault import (
    ApiPermissionReport,
    CredentialVault,
    ExchangeCredentials,
    IpWhitelistValidator,
    PreFlightPermissionValidator,
    SecurityViolationError,
    SensitiveDataMasker,
    SensitiveLogFilter,
    UnrestrictedIpError,
    VaultDecryptionError,
    WithdrawalPermissionEnabledError,
    install_global_log_masker,
)

__all__ = [
    "ApiPermissionReport",
    "CredentialVault",
    "ExchangeCredentials",
    "IpWhitelistValidator",
    "PreFlightPermissionValidator",
    "SecurityViolationError",
    "SensitiveDataMasker",
    "SensitiveLogFilter",
    "UnrestrictedIpError",
    "VaultDecryptionError",
    "WithdrawalPermissionEnabledError",
    "install_global_log_masker",
]
