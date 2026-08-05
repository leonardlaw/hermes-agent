"""SQLite-backed dead-provider registry + provider metrics tracking.

DeadProviderRegistry
  Tracks providers/models that have been marked temporarily dead after
  repeated failures (TTL-based auto-expiry). Used by try_activate_fallback
  to skip known-bad entries.

ProviderMetricsTracker
  Records per-call success/failure, error type, and latency for every
  provider call. Supports error-rate and latency-percentile queries over
  a configurable window.

HealthCheckProbe
  Periodically probes dead providers to revive them when they recover.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default TTL for a dead provider (5 minutes).
DEFAULT_TTL_SECONDS = 300

# Metrics retention: prune entries older than this.
METRICS_RETENTION_SECONDS = 7 * 24 * 3600  # 7 days

# DB filename inside ~/.hermes/data/
DB_FILENAME = "dead_providers.db"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class DeadProviderRecord:
    provider: str
    model: str
    reason: str
    marked_at: float  # time.monotonic()
    ttl_seconds: int


@dataclass
class ProviderMetricsRecord:
    provider: str
    model: str
    success: bool
    latency_ms: float
    error_type: Optional[str] = None
    timestamp: int = 0  # unix seconds


# ── Helper: WAL connection ──────────────────────────────────────────────────


def _connect_db(db_path: str) -> sqlite3.Connection:
    """Open a thread-safe SQLite connection with WAL mode."""
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        timeout=1.0,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _get_db_path() -> str:
    """Return the path to the dead-providers SQLite database."""
    from hermes_constants import get_hermes_home
    data_dir = get_hermes_home() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / DB_FILENAME)


# ── Schema ──────────────────────────────────────────────────────────────────


DEAD_PROVIDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dead_providers (
    provider     TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    reason       TEXT    NOT NULL DEFAULT '',
    marked_at    REAL    NOT NULL,  -- time.monotonic()
    ttl_seconds  INTEGER NOT NULL DEFAULT 300,
    PRIMARY KEY (provider, model)
);
"""

PROVIDER_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    error_type  TEXT,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    timestamp   INTEGER NOT NULL,  -- unix seconds
    success     INTEGER NOT NULL DEFAULT 1  -- boolean
);
CREATE INDEX IF NOT EXISTS idx_provider_metrics_lookup
    ON provider_metrics(provider, model, timestamp);
"""


# ── DeadProviderRegistry ────────────────────────────────────────────────────


class DeadProviderRegistry:
    """SQLite-backed dead-provider registry with TTL expiry.

    Thread-safe: each operation opens its own connection (WAL mode allows
    concurrent readers/writers without blocking).
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _get_db_path()
        self._lock = threading.Lock()
        self._cached_conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(DEAD_PROVIDERS_SCHEMA)
            conn.executescript(PROVIDER_METRICS_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # For :memory: databases, cache the single connection. Each call to
        # _connect_db() opens a NEW independent in-memory DB, so without
        # caching every operation would see an empty database.
        if self._db_path == ":memory:":
            if self._cached_conn is None:
                self._cached_conn = _connect_db(self._db_path)
            return self._cached_conn
        return _connect_db(self._db_path)

    # ── Dead-provider operations ──────────────────────────────────────────

    def _evict_expired(self, conn: sqlite3.Connection) -> None:
        """Remove entries whose TTL has elapsed (monotonic clock)."""
        now = time.monotonic()
        conn.execute(
            "DELETE FROM dead_providers WHERE ? - marked_at >= ttl_seconds",
            (now,),
        )

    def mark_provider_dead(
        self, provider: str, model: str, reason: str = ""
    ) -> None:
        """Mark (provider, model) as dead with the given reason."""
        with self._connect() as conn:
            self._evict_expired(conn)
            conn.execute(
                """INSERT OR REPLACE INTO dead_providers
                   (provider, model, reason, marked_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    provider.lower().strip(),
                    model.strip(),
                    reason,
                    time.monotonic(),
                    DEFAULT_TTL_SECONDS,
                ),
            )

    def is_provider_dead(self, provider: str, model: str) -> bool:
        """Return True if (provider, model) is currently dead."""
        with self._connect() as conn:
            self._evict_expired(conn)
            row = conn.execute(
                "SELECT 1 FROM dead_providers WHERE provider = ? AND model = ?",
                (provider.lower().strip(), model.strip()),
            ).fetchone()
            return row is not None

    def list_dead_providers(self) -> List[DeadProviderRecord]:
        """Return all currently-dead provider records."""
        with self._connect() as conn:
            self._evict_expired(conn)
            rows = conn.execute(
                "SELECT provider, model, reason, marked_at, ttl_seconds "
                "FROM dead_providers ORDER BY provider, model"
            ).fetchall()
            return [
                DeadProviderRecord(
                    provider=row["provider"],
                    model=row["model"],
                    reason=row["reason"],
                    marked_at=row["marked_at"],
                    ttl_seconds=row["ttl_seconds"],
                )
                for row in rows
            ]

    def dead_count(self) -> int:
        """Return the number of currently-dead providers."""
        with self._connect() as conn:
            self._evict_expired(conn)
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM dead_providers"
            ).fetchone()
            return row["cnt"] if row else 0

    def revive_provider(self, provider: str, model: str) -> bool:
        """Remove a dead-provider entry. Returns True if it existed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM dead_providers WHERE provider = ? AND model = ?",
                (provider.lower().strip(), model.strip()),
            )
            return cursor.rowcount > 0

    def revive_all(self) -> int:
        """Revive all dead providers. Returns the count cleared."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM dead_providers")
            return cursor.rowcount

    # ── Metrics operations ────────────────────────────────────────────────

    def record_call(
        self,
        provider: str,
        model: str,
        success: bool,
        latency_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a single provider API call result."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO provider_metrics "
                "(provider, model, error_type, latency_ms, timestamp, success) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    provider.lower().strip(),
                    model.strip(),
                    error_type,
                    latency_ms,
                    int(time.time()),
                    1 if success else 0,
                ),
            )
            # Opportunistic prune: once every ~100 writes, clean old entries.
            self._prune_old_metrics(conn, probability=0.01)

    def _prune_old_metrics(
        self, conn: sqlite3.Connection, probability: float = 0.01
    ) -> None:
        """Prune metrics older than METRICS_RETENTION_SECONDS.

        Uses random sampling so we don't run a DELETE on every write.
        """
        if probability >= 1.0 or __import__("random").random() < probability:
            cutoff = int(time.time()) - METRICS_RETENTION_SECONDS
            conn.execute(
                "DELETE FROM provider_metrics WHERE timestamp < ?", (cutoff,)
            )

    def get_error_rate(
        self,
        provider: str,
        model: str,
        window_minutes: int = 1440,
    ) -> float:
        """Return the error rate (0.0–1.0) in the given time window.

        Returns 0.0 if no calls recorded in the window.
        """
        cutoff = int(time.time()) - window_minutes * 60
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                       COUNT(*) AS total,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
                   FROM provider_metrics
                   WHERE provider = ? AND model = ? AND timestamp >= ?""",
                (provider.lower().strip(), model.strip(), cutoff),
            ).fetchone()
            if row is None or row["total"] == 0:
                return 0.0
            return row["errors"] / row["total"]

    def get_latency_percentiles(
        self,
        provider: str,
        model: str,
        window_minutes: int = 1440,
    ) -> Dict[str, float]:
        """Return p50, p90, p99 latency in ms over the given window.

        Returns zeroed dict if no calls recorded in the window.
        """
        cutoff = int(time.time()) - window_minutes * 60
        with self._connect() as conn:
            # SQLite doesn't have PERCENTILE built-in. Fetch all latency
            # values and compute in Python. For large windows the dataset
            # is bounded (7 days of ~10 req/s ≈ 6M rows — still manageable).
            rows = conn.execute(
                "SELECT latency_ms FROM provider_metrics "
                "WHERE provider = ? AND model = ? AND timestamp >= ? "
                "ORDER BY latency_ms",
                (provider.lower().strip(), model.strip(), cutoff),
            ).fetchall()
            if not rows:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
            values = [r["latency_ms"] for r in rows]
            n = len(values)
            import math

            def _percentile(p: float) -> float:
                idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n) - 1)))
                return float(values[idx])

            return {
                "p50": _percentile(50),
                "p90": _percentile(90),
                "p99": _percentile(99),
            }


# ── HealthCheckProbe ───────────────────────────────────────────────────────


class HealthCheckProbe:
    """Periodic health-check probe for dead providers.

    On each tick, attempts a lightweight health check against each
    dead provider/model in the registry. If the check succeeds (the
    probe's ``check_fn`` returns True), the provider is revived.
    """

    def __init__(
        self,
        registry: DeadProviderRegistry,
        check_fn: Optional[callable] = None,  # type: ignore[valid-type]
        interval_seconds: float = 60.0,
    ):
        self._registry = registry
        self._check_fn = check_fn or self._default_check
        self._interval = interval_seconds
        self._last_run: float = 0.0

    @staticmethod
    def _default_check(provider: str, model: str) -> bool:
        """Default no-op check — always returns False (never auto-revive).

        External callers should set a real check_fn via the constructor
        or subclass.
        """
        return False

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def tick(self) -> int:
        """Run a health-check pass. Returns the number of providers revived."""
        now = time.monotonic()
        if now - self._last_run < self._interval:
            return 0
        self._last_run = now

        dead_list = self._registry.list_dead_providers()
        revived = 0
        for record in dead_list:
            try:
                if self._check_fn(record.provider, record.model):
                    self._registry.revive_provider(record.provider, record.model)
                    logger.info(
                        "HealthCheckProbe revived %s/%s",
                        record.provider, record.model,
                    )
                    revived += 1
            except Exception as exc:
                logger.debug(
                    "HealthCheckProbe check failed for %s/%s: %s",
                    record.provider, record.model, exc,
                )
        return revived


# ── Convenience factory ────────────────────────────────────────────────────


_default_registry: Optional[DeadProviderRegistry] = None
_registry_lock = threading.Lock()


def get_default_registry() -> DeadProviderRegistry:
    """Return the process-wide default DeadProviderRegistry (lazily created)."""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = DeadProviderRegistry()
    return _default_registry
