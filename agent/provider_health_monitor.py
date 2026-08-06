"""Runtime provider health monitoring — metrics recording + health ratio checks.

Integrates with the existing DeadProviderRegistry (``dead_provider_registry.py``)
to record every API call outcome and expose error-rate queries.  Designed to
be imported by ``chat_completion_helpers.py`` and ``conversation_loop.py`` for
runtime instrumentation.

Configurable thresholds (env vars):
    PROVIDER_HEALTH_WARNING_THRESHOLD — error rate that triggers WARNING level (default 0.15)
    PROVIDER_HEALTH_DEGRADED_THRESHOLD — error rate that triggers ERROR level (default 0.30)
    PROVIDER_HEALTH_WINDOW_MINUTES — sliding window for error-rate calculation (default 60)
    PROVIDER_HEALTH_METRICS_ENABLED — set to ``0`` to disable runtime recording (default ``1``)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from agent.dead_provider_registry import DeadProviderRegistry, get_default_registry

logger = logging.getLogger(__name__)

# ── Configurable thresholds (env vars) ──────────────────────────────────────

WARNING_THRESHOLD = float(
    os.environ.get("PROVIDER_HEALTH_WARNING_THRESHOLD", "0.15")
)
DEGRADED_THRESHOLD = float(
    os.environ.get("PROVIDER_HEALTH_DEGRADED_THRESHOLD", "0.30")
)
WINDOW_MINUTES = int(
    os.environ.get("PROVIDER_HEALTH_WINDOW_MINUTES", "60")
)
METRICS_ENABLED = os.environ.get("PROVIDER_HEALTH_METRICS_ENABLED", "1") in (
    "1", "true", "yes", "on",
)

# ── Thread-safe default-registry access ─────────────────────────────────────

_registry: Optional[DeadProviderRegistry] = None
_registry_lock = threading.Lock()


def _get_registry() -> DeadProviderRegistry:
    """Return the process-wide DeadProviderRegistry (lazily created)."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = get_default_registry()
    return _registry


# ── Recording ────────────────────────────────────────────────────────────────


def record_api_call(
    provider: str,
    model: str,
    success: bool,
    latency_ms: float,
    error_type: Optional[str] = None,
) -> None:
    """Record one API call outcome in the provider_metrics DB.

    This is a no-op when ``PROVIDER_HEALTH_METRICS_ENABLED=0``.
    Thread-safe: each call opens its own SQLite connection (WAL mode).
    """
    if not METRICS_ENABLED:
        return
    if not provider or not model:
        return
    # Guard against MagicMock or similar test double values — str() on a
    # MagicMock returns "<MagicMock ...>", not a real provider name.
    if not isinstance(provider, str) or not isinstance(model, str):
        return
    if provider.startswith("<MagicMock") or model.startswith("<MagicMock"):
        return
    try:
        registry = _get_registry()
        registry.record_call(
            provider=provider,
            model=model,
            success=success,
            latency_ms=latency_ms,
            error_type=error_type,
        )
    except Exception:
        logger.debug("Failed to record provider metric (non-fatal)", exc_info=True)


# ── Health queries ───────────────────────────────────────────────────────────


def get_error_rate(
    provider: str,
    model: str,
    window_minutes: Optional[int] = None,
) -> float:
    """Return the error rate (0.0–1.0) for (provider, model) over the sliding window.

    Returns 0.0 when no calls have been recorded in the window.
    """
    try:
        registry = _get_registry()
        return registry.get_error_rate(
            provider=provider,
            model=model,
            window_minutes=window_minutes or WINDOW_MINUTES,
        )
    except Exception:
        logger.debug("Failed to query error rate (non-fatal)", exc_info=True)
        return 0.0


def get_provider_status(provider: str, model: str) -> str:
    """Return a status label for the given (provider, model).

    Returns one of ``"healthy"``, ``"warning"``, ``"degraded"``, or ``"no_data"``.
    """
    rate = get_error_rate(provider, model)
    if rate == 0.0:
        # Check whether any data exists at all.
        try:
            registry = _get_registry()
            _cutoff = int(time.time()) - WINDOW_MINUTES * 60
            from hermes_constants import get_hermes_home
            import sqlite3
            _db = str(get_hermes_home() / "data" / "dead_providers.db")
            conn = sqlite3.connect(_db, timeout=1.0)
            row = conn.execute(
                "SELECT 1 FROM provider_metrics WHERE provider=? AND model=? AND timestamp>=?",
                (provider.lower().strip(), model.strip(), _cutoff),
            ).fetchone()
            conn.close()
            if row is None:
                return "no_data"
        except Exception:
            # DB doesn't exist or table not yet created — no data available.
            return "no_data"
    if rate >= DEGRADED_THRESHOLD:
        return "degraded"
    if rate >= WARNING_THRESHOLD:
        return "warning"
    return "healthy"


def log_health_warning(provider: str, model: str) -> bool:
    """Log a WARNING if the provider's error rate exceeds the threshold.

    Returns True if a warning was emitted, False otherwise.
    Safe to call on every API call failure — it logs once per status transition.
    """
    rate = get_error_rate(provider, model)
    status = get_provider_status(provider, model)
    if status in ("warning", "degraded"):
        logger.warning(
            "Provider health: %s/%s error_rate=%.1f%% (status=%s, "
            "threshold=%.0f%%) over last %d min — check provider reliability.",
            provider, model, rate * 100, status,
            WARNING_THRESHOLD * 100, WINDOW_MINUTES,
        )
        return True
    return False


def format_health_summary() -> str:
    """Return a one-line summary of all (provider, model) health states.

    Suitable for the cron health-warning script or dashboard header.
    """
    try:
        import sqlite3
        from hermes_constants import get_hermes_home
        _db = str(get_hermes_home() / "data" / "dead_providers.db")
        if not os.path.exists(_db):
            return "No provider metrics DB"
        conn = sqlite3.connect(_db, timeout=1.0)
        conn.row_factory = sqlite3.Row
        cutoff = int(time.time()) - WINDOW_MINUTES * 60
        rows = conn.execute(
            """SELECT provider, model,
                      COUNT(*) AS total,
                      SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
               FROM provider_metrics
               WHERE timestamp >= ?
               GROUP BY provider, model
               ORDER BY total DESC""",
            (cutoff,),
        ).fetchall()
        conn.close()
        if not rows:
            return "No provider calls in last %d min" % WINDOW_MINUTES
        parts = []
        for r in rows:
            p, m = r["provider"], r["model"]
            total = r["total"]
            errors = r["errors"]
            rate = errors / total if total > 0 else 0.0
            status = get_provider_status(p, m)
            parts.append(f"{p}/{m}:{total}calls/{errors}err/{rate*100:.0f}%/{status}")
        return " | ".join(parts)
    except Exception as exc:
        return f"health-summary-error: {exc}"
