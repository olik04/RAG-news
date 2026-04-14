"""Tests for the resilience module (timeout and retry policy)."""

from __future__ import annotations

import asyncio

import pytest

from rag_news.core.exceptions import ProviderError
from rag_news.core.resilience import (
    ResilienceConfig,
    calculate_backoff_delay,
    classify_error,
    with_timeout_and_retry,
)


class TestResilienceConfig:
    """Test ResilienceConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default resilience config values."""
        config = ResilienceConfig()
        assert config.base_timeout_sec == 5.0
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert config.jitter_factor == 0.1

    def test_custom_config(self) -> None:
        """Test custom resilience config values."""
        config = ResilienceConfig(
            base_timeout_sec=10.0,
            max_retries=5,
            backoff_factor=1.5,
            jitter_factor=0.2,
        )
        assert config.base_timeout_sec == 10.0
        assert config.max_retries == 5
        assert config.backoff_factor == 1.5
        assert config.jitter_factor == 0.2

    def test_invalid_timeout(self) -> None:
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="base_timeout_sec must be positive"):
            ResilienceConfig(base_timeout_sec=-1.0)

    def test_invalid_max_retries(self) -> None:
        """Test that invalid max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ResilienceConfig(max_retries=-1)

    def test_invalid_backoff_factor(self) -> None:
        """Test that invalid backoff_factor raises ValueError."""
        with pytest.raises(ValueError, match="backoff_factor must be >= 1.0"):
            ResilienceConfig(backoff_factor=0.5)

    def test_invalid_jitter_factor(self) -> None:
        """Test that invalid jitter_factor raises ValueError."""
        with pytest.raises(ValueError, match="jitter_factor must be between"):
            ResilienceConfig(jitter_factor=1.5)


class TestCalculateBackoffDelay:
    """Test exponential backoff delay calculation with jitter."""

    def test_first_attempt_backoff(self) -> None:
        """Test backoff delay for first retry (attempt=0)."""
        # With base_delay=1.0, backoff_factor=2.0, attempt=0:
        #   delay = 1.0 * (2.0 ** 0) = 1.0
        delay = calculate_backoff_delay(
            attempt=0,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter_factor=0.1,
        )
        # Expected: ~1.0 (±10% jitter = 0.9-1.1)
        assert 0.9 <= delay <= 1.1

    def test_second_attempt_backoff(self) -> None:
        """Test backoff delay for second retry (attempt=1)."""
        # With base_delay=1.0, backoff_factor=2.0, attempt=1:
        #   delay = 1.0 * (2.0 ** 1) = 2.0
        delay = calculate_backoff_delay(
            attempt=1,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter_factor=0.1,
        )
        # Expected: ~2.0 (±10% jitter = 1.8-2.2)
        assert 1.8 <= delay <= 2.2

    def test_third_attempt_backoff(self) -> None:
        """Test backoff delay for third retry (attempt=2)."""
        # With base_delay=1.0, backoff_factor=2.0, attempt=2:
        #   delay = 1.0 * (2.0 ** 2) = 4.0
        delay = calculate_backoff_delay(
            attempt=2,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter_factor=0.1,
        )
        # Expected: ~4.0 (±10% jitter = 3.6-4.4)
        assert 3.6 <= delay <= 4.4

    def test_jitter_bounds(self) -> None:
        """Test that jitter stays within expected bounds."""
        for _ in range(100):  # Multiple runs to test randomness
            delay = calculate_backoff_delay(
                attempt=1,
                base_delay=1.0,
                backoff_factor=2.0,
                jitter_factor=0.2,
            )
            # Base: 2.0, jitter range: ±0.4
            assert 1.6 <= delay <= 2.4

    def test_no_jitter(self) -> None:
        """Test backoff with no jitter."""
        delays = [
            calculate_backoff_delay(
                attempt=1,
                base_delay=1.0,
                backoff_factor=2.0,
                jitter_factor=0.0,
            )
            for _ in range(10)
        ]
        # All delays should be exactly 2.0 with no jitter
        assert all(d == 2.0 for d in delays)


class TestClassifyError:
    """Test error classification for retry decisions."""

    def test_timeout_error_is_retryable(self) -> None:
        """Test that TimeoutError is classified as retryable network error."""
        is_retryable, error_type = classify_error(TimeoutError("timeout"))
        assert is_retryable is True
        assert error_type == "network"

    def test_asyncio_timeout_is_retryable(self) -> None:
        """Test that asyncio.TimeoutError is classified as retryable."""
        is_retryable, error_type = classify_error(asyncio.TimeoutError())
        assert is_retryable is True
        assert error_type == "network"

    def test_connection_error_is_retryable(self) -> None:
        """Test that ConnectionError is classified as retryable network error."""
        is_retryable, error_type = classify_error(ConnectionError("failed"))
        assert is_retryable is True
        assert error_type == "network"

    def test_rate_limit_error_is_retryable(self) -> None:
        """Test that rate limit errors are retryable."""
        from rag_news.core.exceptions import ProviderError

        exc = ProviderError("rate_limit: too many requests")
        is_retryable, error_type = classify_error(exc)
        assert is_retryable is True
        assert error_type == "rate_limit"

    def test_provider_error_not_retryable(self) -> None:
        """Test that non-retryable provider errors are not retried."""
        from rag_news.core.exceptions import ProviderError

        exc = ProviderError("invalid model configuration")
        is_retryable, error_type = classify_error(exc)
        assert is_retryable is False
        assert error_type == "provider"

    def test_validation_error_not_retryable(self) -> None:
        """Test that validation errors are not retried."""
        is_retryable, error_type = classify_error(ValueError("invalid input"))
        assert is_retryable is False
        assert error_type == "validation"

    def test_unknown_error_not_retryable(self) -> None:
        """Test that unknown errors default to not retryable."""
        is_retryable, error_type = classify_error(RuntimeError("something broke"))
        assert is_retryable is False
        assert error_type == "unknown"


class TestWithTimeoutAndRetry:
    """Test the with_timeout_and_retry execution helper."""

    @pytest.mark.asyncio
    async def test_successful_call_on_first_attempt(self) -> None:
        """Test successful call completes without retry."""
        config = ResilienceConfig(max_retries=3)
        call_count = 0

        async def operation() -> None:
            nonlocal call_count
            call_count += 1

        await with_timeout_and_retry("test_operation", config, operation)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_on_flaky_call(self) -> None:
        """Test timeout error is raised when all retries fail."""
        config = ResilienceConfig(
            base_timeout_sec=0.01,  # Very short timeout
            max_retries=1,
        )
        call_count = 0

        async def slow_operation() -> None:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(1.0)  # Much longer than timeout

        with pytest.raises(ProviderError, match="test_slow failed after"):
            await with_timeout_and_retry("test_slow", config, slow_operation)

        # Should have attempted initial + 1 retry = 2 times
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self) -> None:
        """Test retry on transient error succeeds on second attempt."""
        config = ResilienceConfig(max_retries=2)
        call_count = 0

        async def flaky_operation() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                # First attempt fails with transient error
                raise ConnectionError("transient network failure")

        await with_timeout_and_retry("flaky_op", config, flaky_operation)

        # Should have retried after first failure
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(self) -> None:
        """Test non-retryable validation errors fail immediately."""
        config = ResilienceConfig(max_retries=3)
        call_count = 0

        async def invalid_operation() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid parameter")

        with pytest.raises(ValueError, match="invalid parameter"):
            await with_timeout_and_retry("invalid_op", config, invalid_operation)

        # Should NOT retry on validation error
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """Test failure after max retries are exhausted."""
        config = ResilienceConfig(
            base_timeout_sec=0.01,
            max_retries=2,
        )
        call_count = 0

        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(1.0)  # Always timeout

        with pytest.raises(ProviderError, match="failed after 3 attempts"):
            await with_timeout_and_retry("always_fail", config, always_fails)

        # Should have attempted initial + 2 retries = 3 times
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_cancellation_not_retried(self) -> None:
        """Test that asyncio.CancelledError is not retried."""
        config = ResilienceConfig(max_retries=3)
        call_count = 0

        async def cancelled_operation() -> None:
            nonlocal call_count
            call_count += 1
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await with_timeout_and_retry("cancel_op", config, cancelled_operation)

        # Should NOT retry on cancellation
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_delay_between_retries(self) -> None:
        """Test that backoff delays increase between retries."""
        config = ResilienceConfig(
            base_timeout_sec=10.0,  # Long timeout to avoid timing out
            max_retries=2,
            backoff_factor=2.0,
            jitter_factor=0.0,  # No jitter for predictable timing
        )
        call_times: list[float] = []
        import time

        async def track_timing() -> None:
            call_times.append(time.time())
            if len(call_times) < 2:
                raise ConnectionError("transient failure")

        await with_timeout_and_retry("backoff_test", config, track_timing)

        # We should have a delay before the second attempt
        assert len(call_times) == 2
        # First backoff delay should be ~10.0 seconds (base_timeout_sec * (2.0 ** 0))
        delay = call_times[1] - call_times[0]
        # Allow some tolerance for test execution overhead
        assert delay >= 9.0

    @pytest.mark.asyncio
    async def test_operation_name_in_logs(self, caplog) -> None:
        """Test that operation name is logged correctly."""
        import logging

        caplog.set_level(logging.DEBUG)
        config = ResilienceConfig(max_retries=1)

        async def simple_op() -> None:
            pass

        await with_timeout_and_retry("my_custom_operation", config, simple_op)

        # Check that operation name appears in logs
        assert "my_custom_operation" in caplog.text
        assert "attempt 1" in caplog.text

    @pytest.mark.asyncio
    async def test_multiple_retries_succeed_eventually(self) -> None:
        """Test that multiple retries eventually succeed."""
        config = ResilienceConfig(
            base_timeout_sec=10.0,
            max_retries=5,
            backoff_factor=1.0,  # No exponential growth for faster test
        )
        call_count = 0

        async def eventually_succeeds() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                # Fail first 3 attempts
                raise ConnectionError("network unavailable")

        await with_timeout_and_retry("eventual_success", config, eventually_succeeds)

        # Should succeed on 4th attempt after 3 retries
        assert call_count == 4


class TestResilienceIntegration:
    """Integration tests for resilience behavior."""

    @pytest.mark.asyncio
    async def test_timeout_and_retry_combined(self) -> None:
        """Test timeout and retry working together."""
        config = ResilienceConfig(
            base_timeout_sec=0.05,
            max_retries=1,
            backoff_factor=2.0,
        )
        call_count = 0

        async def intermittent_slow() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt times out
                await asyncio.sleep(1.0)
            else:
                # Second attempt succeeds
                pass

        # Should retry after timeout and eventually succeed
        await with_timeout_and_retry("intermittent", config, intermittent_slow)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_retry_behavior(self) -> None:
        """Test rate limit errors are retried with backoff."""
        config = ResilienceConfig(
            base_timeout_sec=10.0,
            max_retries=2,
            backoff_factor=1.0,  # No exponential growth
            jitter_factor=0.0,
        )
        call_count = 0

        async def rate_limited_api() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ProviderError("rate_limit: too many requests")

        await with_timeout_and_retry("rate_limited", config, rate_limited_api)

        assert call_count == 2
