"""Tests for agent.dead_provider_registry — SQLite-backed dead-provider
tracking and provider metrics collection.

These tests inherit the hermetic-environment fixture from conftest.py
which redirects HERMES_HOME to a tempdir, so the DB is created there.
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

from agent.dead_provider_registry import (
    DeadProviderRegistry,
    DEFAULT_TTL_SECONDS,
    get_default_registry,
)


class TestDeadProviderRegistry:
    """SQLite-backed dead-provider tracking with TTL expiry."""

    @pytest.fixture
    def registry(self) -> DeadProviderRegistry:
        return DeadProviderRegistry(db_path=":memory:")

    # ── Basic lifecycle ──

    def test_mark_and_check_dead(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "deepseek-v4-flash", reason="timeout")
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-flash") is True

    def test_live_provider_not_dead(self, registry: DeadProviderRegistry):
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-flash") is False

    def test_revive_brings_provider_back(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "deepseek-v4-flash", reason="timeout")
        assert registry.revive_provider("opencode-go", "deepseek-v4-flash") is True
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-flash") is False

    def test_revive_nonexistent_returns_false(self, registry: DeadProviderRegistry):
        assert registry.revive_provider("nonexistent", "model") is False

    def test_revive_all(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("p1", "m1")
        registry.mark_provider_dead("p2", "m2")
        assert registry.dead_count() == 2
        assert registry.revive_all() == 2
        assert registry.dead_count() == 0

    def test_list_dead_providers(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "deepseek-v4-flash", reason="timeout")
        records = registry.list_dead_providers()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "opencode-go"
        assert r.model == "deepseek-v4-flash"
        assert r.reason == "timeout"
        assert r.ttl_seconds == DEFAULT_TTL_SECONDS

    def test_case_insensitive_provider(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("OpenCode-Go", "deepseek-v4-flash")
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-flash") is True
        assert registry.is_provider_dead("OPENCODE-GO", "deepseek-v4-flash") is True

    # ── TTL expiry ──

    def test_provider_auto_expires_after_ttl(self, registry: DeadProviderRegistry):
        """Use a 0-second TTL so the entry expires immediately."""
        with registry._connect() as conn:
            conn.execute(
                "INSERT INTO dead_providers (provider, model, reason, marked_at, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?)",
                ("opencode-go", "deepseek-v4-flash", "timeout", time.monotonic() - 1, 0),
            )
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-flash") is False

    def test_evict_expired_cleans_up(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "keep-me", reason="fresh")
        with registry._connect() as conn:
            conn.execute(
                "INSERT INTO dead_providers (provider, model, reason, marked_at, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?)",
                ("opencode-go", "stale-me", "expired", time.monotonic() - 1, 0),
            )
        records = registry.list_dead_providers()
        assert len(records) == 1
        assert records[0].model == "keep-me"

    # ── Metrics ──

    def test_record_successful_call(self, registry: DeadProviderRegistry):
        registry.record_call("opencode-go", "deepseek-v4-flash", success=True, latency_ms=100)
        rate = registry.get_error_rate("opencode-go", "deepseek-v4-flash")
        assert rate == 0.0

    def test_record_failed_call(self, registry: DeadProviderRegistry):
        registry.record_call(
            "opencode-go", "deepseek-v4-flash",
            success=False, latency_ms=20000, error_type="timeout",
        )
        rate = registry.get_error_rate("opencode-go", "deepseek-v4-flash")
        assert rate == 1.0

    def test_error_rate_with_mixed_calls(self, registry: DeadProviderRegistry):
        for _ in range(8):
            registry.record_call("opencode-go", "deepseek-v4-flash", success=True, latency_ms=500)
        for _ in range(2):
            registry.record_call(
                "opencode-go", "deepseek-v4-flash",
                success=False, latency_ms=20000, error_type="timeout",
            )
        rate = registry.get_error_rate("opencode-go", "deepseek-v4-flash")
        assert rate == 0.2  # 2 failures out of 10

    def test_error_rate_zero_when_no_calls(self, registry: DeadProviderRegistry):
        rate = registry.get_error_rate("opencode-go", "deepseek-v4-flash")
        assert rate == 0.0

    def test_latency_percentiles(self, registry: DeadProviderRegistry):
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            registry.record_call("opencode-go", "deepseek-v4-flash", success=True, latency_ms=ms)
        p = registry.get_latency_percentiles("opencode-go", "deepseek-v4-flash")
        assert p["p50"] >= 500
        assert p["p90"] >= 900
        assert p["p99"] >= 990

    def test_latency_percentiles_zero_when_no_calls(self, registry: DeadProviderRegistry):
        p = registry.get_latency_percentiles("opencode-go", "deepseek-v4-flash")
        assert p == {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    # ── Cross-model isolation ──

    def test_different_models_are_independent(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "deepseek-v4-flash")
        assert registry.is_provider_dead("opencode-go", "deepseek-v4-pro") is False

    def test_different_providers_are_independent(self, registry: DeadProviderRegistry):
        registry.mark_provider_dead("opencode-go", "deepseek-v4-flash")
        assert registry.is_provider_dead("openrouter", "deepseek-v4-flash") is False


class TestDefaultRegistry:
    """The process-wide singleton registry."""

    def test_get_default_registry(self):
        reg = get_default_registry()
        assert isinstance(reg, DeadProviderRegistry)

    def test_default_registry_is_singleton(self):
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        assert reg1 is reg2
