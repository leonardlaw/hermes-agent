"""
stream_circuit_breaker — State-machine circuit breaker for API calls.

Tracks failures per (provider, model) pair.  After N failures within a
configurable time window, transitions to OPEN state and blocks all requests
for that (provider, model) for 15 minutes.  After the cooldown, transitions
to HALF-OPEN and allows a single probe request.  A probe success returns to
CLOSED; a probe failure resets the 15-minute cooldown.

Memory-only state: resets on restart (acceptable per ADR).

States:
  CLOSED    — normal operation, tracking failures
  OPEN      — blocking all requests, cooldown timer running
  HALF-OPEN — allowing a single probe request

Designed for the opencode-go non-streaming API hang pattern where large
context requests stall indefinitely.  The breaker trips and falls back to
the configured fallback_providers chain.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# Number of consecutive failures within the window before tripping the breaker.
TRIP_THRESHOLD = 5

# Time window (seconds) for counting consecutive failures.
TRIP_WINDOW_SECONDS = 300  # 5 minutes

# Cooldown period (seconds) before allowing a probe (HALF-OPEN).
COOLDOWN_SECONDS = 900  # 15 minutes


# ── State machine ───────────────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@staticmethod
def _monotonic() -> float:
    return time.monotonic()


# ── Per-(provider, model) state ─────────────────────────────────────────────
# Maps (provider_lower, model) -> (failure_count, window_start, state, cooldown_until)
# Thread-safe via _lock.
# - failure_count: consecutive failures in the current window
# - window_start: monotonic time when the current window started
# - state: CircuitState enum
# - cooldown_until: monotonic time when the cooldown expires (OPEN -> HALF-OPEN)
# - cooldown_start: monotonic time when the cooldown started

_StateValue = Tuple[int, float, CircuitState, float, float]
#                                          ^       ^
#                                          |       cooldown_start (monotonic)
#                                          cooldown_until (monotonic)

_state: Dict[Tuple[str, str], _StateValue] = {}
_lock = threading.Lock()


def _key(provider: str, model: str) -> Tuple[str, str]:
    return (provider.lower().strip(), model.strip())


def _now() -> float:
    return time.monotonic()


def record_failure(
    agent,
    provider: str,
    model: str,
    error_hint: str = "",
) -> bool:
    """Record an API failure and return True if the circuit breaker tripped.

    Trips to OPEN and marks the provider dead when TRIP_THRESHOLD failures
    occur within TRIP_WINDOW_SECONDS.  On trip, calls
    ``dead_registry.mark_provider_dead()``.

    In HALF-OPEN state, a single failure resets the cooldown timer and
    returns the breaker to OPEN.

    Resets the counter when the window expires without reaching the
    threshold — an isolated spike doesn't permanently degrade the provider.
    """
    k = _key(provider, model)
    now = _now()

    with _lock:
        count, window_start, state, cooldown_until, cooldown_start = _state.get(
            k, (0, now, CircuitState.CLOSED, 0.0, 0.0)
        )

        # ── HALF-OPEN: probe failure → back to OPEN with new cooldown ──
        if state == CircuitState.HALF_OPEN:
            cooldown_until = now + COOLDOWN_SECONDS
            cooldown_start = now
            _state[k] = (count, window_start, CircuitState.OPEN, cooldown_until, cooldown_start)
            logger.warning(
                "Circuit breaker: HALF-OPEN probe FAILED for %s/%s — "
                "returning to OPEN for %.0fs. Error: %s",
                provider, model, COOLDOWN_SECONDS, error_hint,
            )
            _mark_provider_dead(agent, provider, model, error_hint)
            return True

        # ── OPEN: silently block ──────────────────────────────────────
        if state == CircuitState.OPEN:
            remaining = cooldown_until - now
            if remaining > 0:
                logger.debug(
                    "Circuit breaker: %s/%s OPEN (%.0fs remaining)",
                    provider, model, remaining,
                )
                return True
            # Cooldown expired → transition to HALF-OPEN
            _state[k] = (0, now, CircuitState.HALF_OPEN, 0.0, 0.0)
            logger.info(
                "Circuit breaker: %s/%s cooldown expired — "
                "transitioning to HALF-OPEN for next request",
                provider, model,
            )
            return False  # Don't block the probe request — it's allowed

        # ── CLOSED: count failures ────────────────────────────────────
        if now - window_start > TRIP_WINDOW_SECONDS:
            # Window expired — reset
            count = 0
            window_start = now
        count += 1
        _state[k] = (count, window_start, CircuitState.CLOSED, 0.0, 0.0)

    if count < TRIP_THRESHOLD:
        logger.debug(
            "Circuit breaker: %s/%s failure %d/%d (window=%.0fs) %s",
            provider, model, count, TRIP_WINDOW_SECONDS, error_hint,
        )
        return False

    # ── Trip the breaker: CLOSED → OPEN ───────────────────────────────
    cooldown_until = _now() + COOLDOWN_SECONDS
    with _lock:
        _state[k] = (0, _now(), CircuitState.OPEN, cooldown_until, _now())

    logger.warning(
        "Circuit breaker TRIPPED for %s/%s after %d consecutive failures "
        "within %.0fs — OPEN for %.0fs. Last error: %s",
        provider, model, count, TRIP_WINDOW_SECONDS, COOLDOWN_SECONDS, error_hint,
    )
    _mark_provider_dead(agent, provider, model, error_hint)
    return True


def record_success(agent, provider: str, model: str) -> None:
    """Record a successful API response.

    In CLOSED state: resets the consecutive-failure counter.
    In HALF-OPEN state: transitions to CLOSED (probe succeeded).
    In OPEN state: no-op (cooldown still running).
    """
    k = _key(provider, model)
    now = _now()

    with _lock:
        entry = _state.get(k)
        if entry is None:
            return
        count, window_start, state, cooldown_until, cooldown_start = entry

        if state == CircuitState.HALF_OPEN:
            # Probe succeeded — return to CLOSED
            _state[k] = (0, now, CircuitState.CLOSED, 0.0, 0.0)
            logger.info(
                "Circuit breaker: HALF-OPEN probe SUCCEEDED for %s/%s — "
                "returning to CLOSED",
                provider, model,
            )
            _revive_provider(agent, provider, model)
            return

        if state == CircuitState.OPEN:
            # Ignore successes during cooldown — the breaker was probably
            # bypassed.  Don't reset the CLOSED state until the probe.
            return

        # CLOSED state — reset counter
        if count > 0:
            logger.debug(
                "Circuit breaker: %s/%s success — resetting counter (was %d)",
                provider, model, count,
            )
        del _state[k]


def reset_circuit_breaker(provider: str, model: str) -> None:
    """Manually reset the circuit breaker for a specific provider/model."""
    k = _key(provider, model)
    with _lock:
        _state.pop(k, None)


def get_circuit_state(provider: str, model: str) -> Optional[Dict]:
    """Return the current circuit state for a provider/model, or None.

    Returns dict with keys:
      - state: str ("CLOSED", "OPEN", "HALF_OPEN")
      - failure_count: int
      - window_remaining_s: float (remaining in current failure window)
      - cooldown_remaining_s: float (remaining cooldown, 0 when not OPEN)
      - tripped: bool (True when OPEN or HALF_OPEN-from-cooldown)
    """
    k = _key(provider, model)
    with _lock:
        entry = _state.get(k)
        if entry is None:
            return None
        count, window_start, state, cooldown_until, cooldown_start = entry
        now = _now()
        elapsed = now - window_start
        window_remaining = max(0.0, TRIP_WINDOW_SECONDS - elapsed)
        cooldown_remaining = max(0.0, cooldown_until - now) if state == CircuitState.OPEN else 0.0
        return {
            "state": state.name,
            "failure_count": count,
            "window_remaining_s": window_remaining,
            "cooldown_remaining_s": cooldown_remaining,
            "tripped": state != CircuitState.CLOSED,
        }


# ── DeadProviderRegistry integration ────────────────────────────────────────


def _mark_provider_dead(agent, provider: str, model: str, error_hint: str) -> None:
    """Mark the provider dead via DeadProviderRegistry with 900s TTL."""
    _dead_reg = getattr(agent, "_dead_registry", None)
    if _dead_reg is not None:
        try:
            _dead_reg.mark_provider_dead(
                provider,
                model,
                reason=f"circuit_breaker: {error_hint}",
            )
        except Exception as exc:
            logger.error("Circuit breaker: failed to mark provider dead: %s", exc)
    else:
        logger.debug(
            "Circuit breaker: no dead_registry on agent — cannot mark %s/%s dead",
            provider, model,
        )


def _revive_provider(agent, provider: str, model: str) -> None:
    """Remove a dead-provider entry when the circuit recovers."""
    _dead_reg = getattr(agent, "_dead_registry", None)
    if _dead_reg is not None:
        try:
            _dead_reg.revive_provider(provider, model)
        except Exception as exc:
            logger.error("Circuit breaker: failed to revive provider: %s", exc)


# ── Legacy aliases (backward compatibility) ─────────────────────────────────
# The old names are preserved so existing call sites (chat_completion_helpers.py)
# continue to work unchanged.  New code should use record_failure / record_success.


def record_stream_failure(
    agent,
    provider: str,
    model: str,
    error_hint: str = "",
) -> bool:
    """Legacy alias for :func:`record_failure`.

    Retained for backward compatibility with existing call sites in
    ``chat_completion_helpers.py``.
    """
    return record_failure(agent, provider, model, error_hint)


def record_stream_success(agent, provider: str, model: str) -> None:
    """Legacy alias for :func:`record_success`."""
    return record_success(agent, provider, model)
