from __future__ import annotations


class RAGNewsError(Exception):
    """Base exception for the application."""


class ConfigurationError(RAGNewsError):
    """Raised when runtime configuration is invalid."""


class ValidationError(RAGNewsError):
    """Raised when user or external input is invalid."""


class ProviderError(RAGNewsError):
    """Raised when an external model provider fails."""


class RepositoryError(RAGNewsError):
    """Raised when local persistence operations fail."""
