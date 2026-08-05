"""
Per-provider API metrics logger.

Thread-safe JSONL writer that records every API call result (success or failure)
with provider, model, latency, and error type.  Designed for cron-friendly
consumption — report views read the file and summarize.

File: HERMES_HOME/data/provider_metrics.jsonl
"""

import os
import json
import time
import atexit
import threading
from pathlib import Path

_lock = threading.Lock()
_buffer: list[str] = []
_flush_interval = 5.0  # seconds between flushes
_last_flush = time.monotonic()

_hermes_home = os.environ.get(
    "HERMES_HOME",
    os.path.expanduser("~/.hermes"),
)
_metrics_dir = Path(_hermes_home) / "data"
_metrics_path = _metrics_dir / "provider_metrics.jsonl"


def _ensure_dir():
    _metrics_dir.mkdir(parents=True, exist_ok=True)


def _flush():
    global _buffer, _last_flush
    if not _buffer:
        return
    _ensure_dir()
    try:
        with open(_metrics_path, "a") as f:
            for line in _buffer:
                f.write(line + "\n")
    except OSError:
        pass  # best-effort — don't crash the agent loop on disk errors
    _buffer = []
    _last_flush = time.monotonic()


def log_api_result(
    *,
    provider: str,
    model: str,
    latency_ms: float,
    success: bool = True,
    error_type: str = "",
    error_reason: str = "",
    retry_count: int = 0,
):
    """Record one API call result.

    Safe to call from any thread (locking at the buffer level, not the file
    level).  Flushes to disk every ``_flush_interval`` seconds — the final
    flush is guaranteed by the ``atexit`` handler.
    """
    record = {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "error_type": error_type,
        "error_reason": error_reason,
        "retry_count": retry_count,
    }
    line = json.dumps(record, sort_keys=True)
    with _lock:
        _buffer.append(line)
        if (time.monotonic() - _last_flush) >= _flush_interval:
            _flush()


def log_api_success(provider: str, model: str, latency_ms: float):
    """Shorthand for a successful API call."""
    log_api_result(
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        success=True,
    )


def log_api_error(
    provider: str,
    model: str,
    latency_ms: float,
    error_type: str,
    error_reason: str = "",
    retry_count: int = 0,
):
    """Shorthand for a failed API call."""
    log_api_result(
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        success=False,
        error_type=error_type,
        error_reason=error_reason,
        retry_count=retry_count,
    )


# ── Report view ────────────────────────────────────────────────────────────


def generate_report(duration_hours: float = 24) -> str:
    """Read the metrics file and return a plain-text summary table.

    Args:
        duration_hours: How far back to look (default 24h).

    Returns:
        A formatted multi-line string with per-provider error rates and latencies.
    """
    if not _metrics_path.exists():
        return "No metrics data found."

    cutoff = time.time() - (duration_hours * 3600)
    rows = []
    try:
        with open(_metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ts", 0) < cutoff:
                    continue
                rows.append(rec)
    except OSError as e:
        return f"Error reading metrics file: {e}"

    if not rows:
        return f"No metrics in the last {duration_hours:.0f}h."

    # Aggregate per (provider, model)
    from collections import defaultdict

    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": 0, "errors": 0, "latencies": []}
    )
    for r in rows:
        key = (r.get("provider", "?"), r.get("model", "?"))
        agg[key]["total"] += 1
        if not r.get("success", True):
            agg[key]["errors"] += 1
        agg[key]["latencies"].append(r.get("latency_ms", 0))

    lines = [
        f"Provider Metrics — last {duration_hours:.0f}h ({len(rows)} calls)",
        "",
        f"{'Provider':<20} {'Model':<25} {'Calls':>6} {'Errors':>6} {'Err%':>7} {'P50(ms)':>8} {'P90(ms)':>8} {'P99(ms)':>8}",
        "-" * 90,
    ]

    for (provider, model), data in sorted(agg.items()):
        total = data["total"]
        errors = data["errors"]
        error_pct = (errors / total * 100) if total else 0
        latencies = sorted(data["latencies"])
        p50 = _percentile(latencies, 50)
        p90 = _percentile(latencies, 90)
        p99 = _percentile(latencies, 99)
        lines.append(
            f"{provider:<20} {model:<25} {total:>6} {errors:>6} {error_pct:>6.1f}% {p50:>8.0f} {p90:>8.0f} {p99:>8.0f}"
        )

    # Aggregate error types
    error_types = defaultdict(int)
    for r in rows:
        if not r.get("success", True) and r.get("error_type"):
            error_types[r["error_type"]] += 1
    if error_types:
        lines.append("")
        lines.append("Error type breakdown:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {etype:<30} {count:>4}")

    return "\n".join(lines)


def _percentile(sorted_list: list[float], p: int) -> float:
    if not sorted_list:
        return 0
    idx = max(0, min(len(sorted_list) - 1, int(len(sorted_list) * p / 100)))
    return sorted_list[idx]


# ── CLI entry point ────────────────────────────────────────────────────────


def main():
    """CLI: print a metrics report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Provider metrics report",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Lookback window in hours (default: 24)",
    )
    args = parser.parse_args()
    print(generate_report(duration_hours=args.hours))


if __name__ == "__main__":
    main()

# ── Final flush at exit ────────────────────────────────────────────────────
atexit.register(_flush)
