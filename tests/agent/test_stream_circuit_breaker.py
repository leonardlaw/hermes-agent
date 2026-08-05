"""Tests for agent.stream_circuit_breaker — consecutive-failure circuit breaker.

All tests use a fresh module-level state (each test file gets its own
Python process per the conftest isolation policy).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.stream_circuit_breaker import (
    record_stream_failure,
    record_stream_success,
    reset_circuit_breaker,
    get_circuit_state,
    TRIP_THRESHOLD,
    TRIP_WINDOW_SECONDS,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear circuit-breaker state before each test."""
    from agent.stream_circuit_breaker import _state
    _state.clear()
    yield


@pytest.fixture
def agent():
    """A minimal agent mock with a dead_registry."""
    a = MagicMock()
    a._dead_registry = MagicMock()
    return a


class TestStreamCircuitBreaker:
    """Consecutive-failure circuit breaker for streaming API calls."""

    def test_single_failure_does_not_trip(self, agent):
        """A single failure within the window does not trip the breaker."""
        result = record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "Broken pipe")
        assert result is False

    def test_consecutive_failures_trip_breaker(self, agent):
        """After TRIP_THRESHOLD consecutive failures, the breaker trips."""
        for i in range(TRIP_THRESHOLD - 1):
            result = record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", f"error {i}")
            assert result is False, f"Should not trip at {i+1}/{TRIP_THRESHOLD}"

        # The Nth call should trip
        result = record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "final error")
        assert result is True, "Should trip at threshold"

    def test_tripped_breaker_marks_provider_dead(self, agent):
        """When the breaker trips, it marks the provider dead via dead_registry."""
        for _ in range(TRIP_THRESHOLD):
            record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "err")
        agent._dead_registry.mark_provider_dead.assert_called_once()

    def test_success_resets_counter(self, agent):
        """A successful call resets the consecutive-failure counter."""
        # Accumulate some failures
        for i in range(TRIP_THRESHOLD - 2):
            record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", f"err {i}")

        # Record a success
        record_stream_success(agent, "opencode-go", "deepseek-v4-flash")

        # The counter should be reset — need full threshold again to trip
        for i in range(TRIP_THRESHOLD - 1):
            result = record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", f"post-success {i}")
            assert result is False, f"Should not trip early at {i+1}/{TRIP_THRESHOLD}"

        result = record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "final")
        assert result is True

    def test_different_providers_independent(self, agent):
        """Failures on one provider don't affect another."""
        # Trip provider A — the 5th call returns True
        for i in range(TRIP_THRESHOLD - 1):
            result = record_stream_failure(agent, "provider-a", "model-a", "err")
            assert result is False
        result = record_stream_failure(agent, "provider-a", "model-a", "final err")
        assert result is True, "5th consecutive failure should trip"

        # Provider B should have clean state
        state_b = get_circuit_state("provider-b", "model-b")
        assert state_b is None

    def test_different_models_independent(self, agent):
        """Failures on one model don't affect another model on same provider."""
        for _ in range(TRIP_THRESHOLD):
            record_stream_failure(agent, "provider", "model-a", "err")

        state_b = get_circuit_state("provider", "model-b")
        assert state_b is None

    def test_reset_circuit_breaker_clears_state(self, agent):
        """Manual reset removes the state for a given provider/model."""
        for _ in range(TRIP_THRESHOLD):
            record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "err")
        assert get_circuit_state("opencode-go", "deepseek-v4-flash") is not None

        reset_circuit_breaker("opencode-go", "deepseek-v4-flash")
        assert get_circuit_state("opencode-go", "deepseek-v4-flash") is None

    def test_get_circuit_state_idle(self):
        """get_circuit_state returns None for untouched provider/model."""
        state = get_circuit_state("nonexistent", "nonexistent")
        assert state is None

    def test_get_circuit_state_active(self, agent):
        """get_circuit_state returns current failure count and window info."""
        for _ in range(2):
            record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "err")
        state = get_circuit_state("opencode-go", "deepseek-v4-flash")
        assert state is not None
        assert state["failure_count"] == 2
        assert state["tripped"] is False
        assert state["window_remaining_s"] > 0

    def test_get_circuit_state_tripped(self, agent):
        """get_circuit_state shows tripped=True after threshold exceeded."""
        for _ in range(TRIP_THRESHOLD):
            record_stream_failure(agent, "opencode-go", "deepseek-v4-flash", "err")
        state = get_circuit_state("opencode-go", "deepseek-v4-flash")
        assert state is not None
        # After trip, state is OPEN and cooldown is active
        assert state["state"] == "OPEN"
        assert state["tripped"] is True
        assert state["cooldown_remaining_s"] > 0
        assert state["cooldown_remaining_s"] <= 900.0

    def test_case_insensitive_provider(self, agent):
        """Provider name casing is normalized."""
        record_stream_failure(agent, "OpenCode-Go", "deepseek-v4-flash", "err")
        state = get_circuit_state("opencode-go", "deepseek-v4-flash")
        assert state is not None
        assert state["failure_count"] == 1
