"""Provider health watch — cron script for periodic health reporting.

Queries the provider_health_monitor for all providers with recorded metrics
and reports any that are in warning or degraded state.  Designed to run as
a cron job (``hermes cron`` or system cron).

Output is suitable for Telegram or Slack delivery — a compact summary of
health status changes since the last check.

Exit code 0 when all providers are healthy, 1 when warnings exist.
"""
from __future__ import annotations

import logging
import sys
import time

# ── Configure logging (stderr for cron, stdout for the report) ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("provider_health_watch")

# ── Silence health-monitor's own logger during health checks ────────────
logging.getLogger("agent.provider_health_monitor").setLevel(logging.WARNING)


def main() -> int:
    """Run a health check pass and report results to stdout.

    Returns 0 if all providers are healthy, 1 if any warnings/degraded.
    """
    # Import here so the Python path is set up correctly in cron context.
    import os
    sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

    from agent.provider_health_monitor import (
        format_health_summary,
        get_provider_status,
        WINDOW_MINUTES,
    )
    from hermes_constants import get_hermes_home
    import sqlite3

    db_path = str(get_hermes_home() / "data" / "dead_providers.db")

    if not os.path.exists(db_path):
        print("ℹ️  Provider health: no metrics database yet (no API calls recorded).")
        return 0

    # Query all (provider, model) combos in the current window
    cutoff = int(time.time()) - WINDOW_MINUTES * 60
    conn = sqlite3.connect(db_path, timeout=1.0)
    conn.row_factory = sqlite3.Row
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
        print(f"ℹ️  Provider health: no API calls in the last {WINDOW_MINUTES} min.")
        return 0

    healthy_count = 0
    warning_count = 0
    degraded_count = 0
    lines = []

    for r in rows:
        p, m = r["provider"], r["model"]
        total = r["total"]
        errors = r["errors"]
        rate = errors / total if total > 0 else 0.0
        status = get_provider_status(p, m)

        icon = "✅" if status == "healthy" else ("⚠️" if status == "warning" else "🚨")
        lines.append(
            f"  {icon} {p}/{m}: {total}calls, {errors}err ({rate*100:.1f}%) — {status}"
        )

        if status == "warning":
            warning_count += 1
        elif status == "degraded":
            degraded_count += 1
        else:
            healthy_count += 1

    # Summary header
    total_providers = len(rows)
    if degraded_count > 0:
        header = f"🚨 Provider Health Report: {degraded_count} degraded, {warning_count} warning, {healthy_count} healthy"
    elif warning_count > 0:
        header = f"⚠️  Provider Health Report: {warning_count} warning, {healthy_count} healthy"
    else:
        header = f"✅ Provider Health Report: all {healthy_count} providers healthy"

    print(header)
    print(f"   (window: last {WINDOW_MINUTES} min)")
    for line in lines:
        print(line)

    if degraded_count > 0 or warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
