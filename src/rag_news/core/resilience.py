"""Centralized timeout and retry policy for external API calls."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from logging import getLogger
from typing import Awaitable, Callable, TypeVar

from rag_news.core.exceptions import ProviderError


logger = getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResilienceConfig:
    """Configuration for timeout and retry behavior.
    
    Attributes:
        base_timeout_sec: Base timeout in seconds for individual API calls.
        max_retries: Maximum number of retry attempts (excluding initial attempt).
        backoff_factor: Multiplier for exponential backoff: delay = base * (factor ** attempt).
        jitter_factor: Jitter factor for retry delays (0.1 = ±10% variation).
    """

    base_timeout_sec: float = 5.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    jitter_factor: float = 0.1

    def __post_init__(self) -> None:
        if self.base_timeout_sec <= 0:
            raise ValueError("base_timeout_sec must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if not 0.0 <= self.jitter_factor <= 1.0:
            raise ValueError("jitter_factor must be between 0.0 and 1.0")


def calculate_backoff_delay(
    attempt: int,
    base_delay: float,
    backoff_factor: float,
    jitter_factor: float,
) -> float:
    """Calculate exponential backoff delay with jitter.
    
    Args:
        attempt: Zero-indexed attempt number (0 = first retry, 1 = second retry, etc.).
        base_delay: Base delay in seconds.
        backoff_factor: Exponential backoff multiplier.
        jitter_factor: Jitter magnitude as fraction of base delay (0.0-1.0).
    
    Returns:
        Delay in seconds with jittered exponential backoff.
    
    Example:
        For backoff_factor=2.0, jitter_factor=0.1, base_delay=1.0:
        - attempt=0: 1.0 * 2^0 * (1 ± 0.1) = ~1.0 seconds
        - attempt=1: 1.0 * 2^1 * (1 ± 0.1) = ~2.0 seconds
        - attempt=2: 1.0 * 2^2 * (1 ± 0.1) = ~4.0 seconds
    """
    # Exponential backoff: base_delay * (backoff_factor ** attempt)
    exponential_delay = base_delay * (backoff_factor ** attempt)

    # Add jitter: ±(jitter_factor * exponential_delay)
    jitter_range = exponential_delay * jitter_factor
    jitter = random.uniform(-jitter_range, jitter_range)

    return max(0.0, exponential_delay + jitter)


def classify_error(exc: Exception) -> tuple[bool, str]:
    """Classify an exception to determine if it's retryable.
    
    Args:
        exc: Exception to classify.
    
    Returns:
        Tuple of (is_retryable, error_type):
        - is_retryable: True if the error may succeed on retry.
        - error_type: String classification of the error.
    
    Classification:
        - network: Transient network issues (ConnectionError, TimeoutError).
        - rate_limit: Rate limiting or quota exceeded.
        - provider: Provider-side errors (ProviderError, API errors).
        - validation: Client-side validation errors (ValidationError, ValueError).
        - unknown: Uncategorized errors.
    """
    error_name = type(exc).__name__

    # Network/timeout errors are retryable
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True, "network"
    if isinstance(exc, (ConnectionError, BrokenPipeError, ConnectionResetError)):
        return True, "network"

    # Some provider errors indicate transient issues
    if isinstance(exc, ProviderError):
        error_msg = str(exc).lower()
        # Retryable provider issues: rate limits, temporary unavailability, server errors
        if any(
            keyword in error_msg
            for keyword in [
                "rate_limit",
                "quota",
                "temporarily",
                "unavailable",
                "timeout",
                "503",
                "502",
                "500",
            ]
        ):
            return True, "rate_limit"
        # Non-retryable provider errors: authentication, validation, not found
        return False, "provider"

    # Check exception class names from external libraries
    if "RateLimitError" in error_name or "QuotaError" in error_name:
        return True, "rate_limit"

    if "APIError" in error_name and "401" not in str(exc) and "403" not in str(exc):
        # Generic API errors may be transient (unless auth failures)
        return True, "provider"

    # Validation and client errors are not retryable
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False, "validation"

    if "ValidationError" in error_name or "BadRequest" in error_name:
        return False, "validation"

    # Default: don't retry unknown errors
    return False, "unknown"


async def with_timeout_and_retry(
    operation_name: str,
    config: ResilienceConfig,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run an async operation with timeout and retry with exponential backoff.
    
    Wraps an async operation callable with:
    - Timeout enforcement
    - Automatic retry on transient errors
    - Jittered exponential backoff
    - Comprehensive logging
    
    Args:
        operation_name: Human-readable name for the operation (for logging).
        config: ResilienceConfig with timeout and retry parameters.
        operation: Async callable to execute for each attempt.
    
    Raises:
        TimeoutError: If operation exceeds timeout on all attempts.
        ProviderError: If operation fails after max retries.
        Original exception: If a non-retryable error occurs.
    
    Example:
        >>> config = ResilienceConfig(base_timeout_sec=5.0, max_retries=3)
        >>> await with_timeout_and_retry("tavily_search", config, some_api_call)
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            async with asyncio.timeout(config.base_timeout_sec):
                logger.debug(
                    "%s: attempt %d/%d (timeout=%fs)",
                    operation_name,
                    attempt + 1,
                    config.max_retries + 1,
                    config.base_timeout_sec,
                )
                result = await operation()
                logger.debug("%s: attempt %d succeeded", operation_name, attempt + 1)
                return result

        except asyncio.CancelledError:
            # Don't retry on cancellation
            logger.info("%s: operation cancelled", operation_name)
            raise

        except Exception as exc:
            last_exception = exc
            is_retryable, error_type = classify_error(exc)

            logger.warning(
                "%s: attempt %d failed with %s (%s, retryable=%s)",
                operation_name,
                attempt + 1,
                type(exc).__name__,
                error_type,
                is_retryable,
            )

            # Don't retry non-retryable errors
            if not is_retryable:
                logger.error(
                    "%s: non-retryable error; failing immediately",
                    operation_name,
                )
                raise

            # Don't sleep after final attempt
            if attempt >= config.max_retries:
                logger.error(
                    "%s: max retries (%d) exhausted after %s",
                    operation_name,
                    config.max_retries,
                    error_type,
                )
                break

            # Calculate backoff delay with jitter
            backoff_delay = calculate_backoff_delay(
                attempt=attempt,
                base_delay=config.base_timeout_sec,
                backoff_factor=config.backoff_factor,
                jitter_factor=config.jitter_factor,
            )

            logger.info(
                "%s: retrying after %.2fs (attempt %d/%d)",
                operation_name,
                backoff_delay,
                attempt + 1,
                config.max_retries + 1,
            )
            await asyncio.sleep(backoff_delay)

    # All retries exhausted
    if last_exception:
        raise ProviderError(
            f"{operation_name} failed after {config.max_retries + 1} attempts: "
            f"{type(last_exception).__name__}"
        ) from last_exception
    raise ProviderError(
        f"{operation_name} failed after {config.max_retries + 1} attempts"
    )
