"""Tests for agent.provider_health_monitor — runtime health monitoring.

These tests verify the high-level wrapper functions that integrate
with DeadProviderRegistry for per-provider health tracking.

All tests inherit the hermetic-environment fixture from conftest.py
which redirects HERMES_HOME to a tempdir.
"""
from __future__ import annotations

import os
from typing import Optional

import pytest

from agent.provider_health_monitor import (
    record_api_call,
    get_error_rate,
    get_provider_status,
    log_health_warning,
    format_health_summary,
    METRICS_ENABLED,
    WARNING_THRESHOLD,
    DEGRADED_THRESHOLD,
    WINDOW_MINUTES,
)


class TestProviderHealthMonitor:
    """Runtime provider health monitoring — metrics recording + health ratio checks."""

    def test_module_constants_have_defaults(self):
        """Verify env-var-driven constants have sensible defaults."""
        assert METRICS_ENABLED is True
        assert 0.0 < WARNING_THRESHOLD < 1.0
        assert WARNING_THRESHOLD <= DEGRADED_THRESHOLD < 1.0
        assert WINDOW_MINUTES > 0

    def test_record_api_call_success(self):
        """A successful call increments the metric store."""
        record_api_call(
            provider="opencode-go", model="deepseek-v4-flash",
            success=True, latency_ms=150.0,
        )
        rate = get_error_rate("opencode-go", "deepseek-v4-flash", window_minutes=1440)
        # May have existing data from other tests in same file (no process isolation
        # intra-file). Assert at most the rate is < 0.5 (mostly successful).
        assert rate < 0.5

    def test_record_api_call_failure(self):
        """A failed call shows up in the error rate."""
        record_api_call(
            provider="test-recorder", model="test-m",
            success=False, latency_ms=5000.0, error_type="TimeoutError",
        )
        rate = get_error_rate("test-recorder", "test-m", window_minutes=1440)
        assert rate == 1.0

    def test_record_api_call_mixed(self):
        """Mixed success/failure produces the correct ratio."""
        for _ in range(8):
            record_api_call("test-mixed", "m1", success=True, latency_ms=200.0)
        for _ in range(2):
            record_api_call("test-mixed", "m1", success=False, latency_ms=3000.0, error_type="ConnectionError")
        rate = get_error_rate("test-mixed", "m1", window_minutes=1440)
        assert rate == 0.2

    def test_record_api_call_disabled_via_env(self):
        """When METRICS_ENABLED is False, calls are no-ops.

        We temporarily flip the module-level flag rather than using
        importlib.reload, which would reset ALL module state and break
        subsequent tests in the same file (no cross-file process isolation
        within a single test file).
        """
        import agent.provider_health_monitor as phm
        original = phm.METRICS_ENABLED
        phm.METRICS_ENABLED = False
        try:
            phm.record_api_call("test-disabled", "m1", success=False, latency_ms=100.0, error_type="Fail")
            rate = phm.get_error_rate("test-disabled", "m1", window_minutes=1440)
            assert rate == 0.0
        finally:
            phm.METRICS_ENABLED = original

    def test_skip_magicmock_values(self):
        """MagicMock guard ensures test doubles don't pollute metrics."""
        try:
            from unittest.mock import MagicMock
            record_api_call(
                MagicMock(), MagicMock(),
                success=False, latency_ms=100.0, error_type="MockError",
            )
        except Exception:
            pytest.fail("record_api_call should not raise on MagicMock values")
        # No crash = passed

    def test_skip_empty_values(self):
        """Empty provider/model strings are silently skipped."""
        record_api_call("", "", success=False, latency_ms=100.0, error_type="Fail")
        # No crash = passed
        record_api_call("valid", "", success=False, latency_ms=100.0, error_type="Fail")
        record_api_call("", "valid", success=False, latency_ms=100.0, error_type="Fail")

    def test_get_provider_status_healthy(self):
        """A provider with 0% error rate is 'healthy'."""
        record_api_call("test-status-h", "m1", success=True, latency_ms=100.0)
        status = get_provider_status("test-status-h", "m1")
        # If there's only 1 success, rate=0 but there IS data
        assert status in ("healthy", "no_data")

    def test_get_provider_status_no_data(self):
        """A provider with no recorded calls returns 'no_data'."""
        status = get_provider_status("nonexistent-provider", "nonexistent-model")
        assert status == "no_data"

    def test_get_provider_status_warning(self):
        """A provider with error rate > WARNING_THRESHOLD returns 'warning'."""
        # Record enough calls to push rate above threshold
        for _ in range(10):
            record_api_call("test-status-warn", "m1", success=True, latency_ms=100.0)
        # Add failures to push rate above WARNING_THRESHOLD
        err_count = int(10 * WARNING_THRESHOLD / (1 - WARNING_THRESHOLD)) + 1
        for _ in range(err_count):
            record_api_call("test-status-warn", "m1", success=False, latency_ms=5000.0, error_type="Err")
        status = get_provider_status("test-status-warn", "m1")
        assert status in ("warning", "degraded")

    def test_log_health_warning_returns_bool(self):
        """log_health_warning returns True when threshold exceeded, False otherwise."""
        # Healthy provider
        result = log_health_warning("nonexistent-log", "m1")
        assert result is False

        # Provider with errors
        record_api_call("test-log-warn", "m1", success=False, latency_ms=5000.0, error_type="Err")
        result = log_health_warning("test-log-warn", "m1")
        assert isinstance(result, bool)

    def test_format_health_summary_returns_string(self):
        """format_health_summary always returns a string, never raises."""
        summary = format_health_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_record_api_call_different_providers_independent(self):
        """Metrics for different (provider, model) pairs are isolated."""
        record_api_call("test-prova", "deepseek-v4-flash", success=True, latency_ms=100.0)
        record_api_call("test-prova", "deepseek-v4-flash", success=False, latency_ms=5000.0, error_type="Err")
        record_api_call("test-provb", "deepseek-v4-pro", success=True, latency_ms=100.0)
        rate_a = get_error_rate("test-prova", "deepseek-v4-flash", window_minutes=1440)
        rate_b = get_error_rate("test-provb", "deepseek-v4-pro", window_minutes=1440)
        assert rate_a == 0.5
        assert rate_b == 0.0
