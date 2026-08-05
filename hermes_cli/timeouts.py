from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def get_min_provider_stale_timeout_by_url(url: str) -> float | None:
    """Return the minimum stale_timeout_seconds across all models of the
    provider whose config matches *url*, or None if no provider matches.

    Matching strategy (first wins):
    1. **Explicit base_url in config** — if a provider entry has a ``base_url``
       field and it is a substring of *url* (case-insensitive), it's a match.
    2. **Known hostname mapping** — if the URL's hostname contains a key from
       ``_HOSTNAME_TO_PROVIDER``, that provider is used.  This handles providers
       whose base_url is determined dynamically at runtime (e.g. opencode-go).

    Once the provider is identified, the minimum stale timeout is computed
    across the provider-level ``stale_timeout_seconds`` and **every** model's
    ``stale_timeout_seconds``.  Using the tightest value is conservative: a
    single model with a very short timeout signals that the gateway closes idle
    connections aggressively, and disabling keepalive protects all models on
    that gateway.
    """
    if not url:
        return None

    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        hostname = ""

    if not hostname:
        return None

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None

    # ── Step 1: try matching provider configs by explicit base_url ──
    provider_id: str | None = None
    url_lower = url.lower()

    for pid, pconf in providers.items():
        if not isinstance(pconf, dict):
            continue
        pbase = pconf.get("base_url")
        if pbase and isinstance(pbase, str) and pbase.strip():
            if pbase.strip().lower() in url_lower:
                provider_id = pid
                break

    # ── Step 2: fall back to known hostname-to-provider mapping ──
    if provider_id is None:
        _HOSTNAME_TO_PROVIDER: dict[str, str] = {
            "opencode.ai": "opencode-go",
        }
        for host_pattern, pid in _HOSTNAME_TO_PROVIDER.items():
            if host_pattern in hostname:
                provider_id = pid
                break

    if provider_id is None:
        return None

    provider_config = providers[provider_id]
    if not isinstance(provider_config, dict):
        return None

    # ── Step 3: compute minimum stale timeout across provider + all models ──
    candidates: list[float] = []

    # Provider-level timeout
    pval = _coerce_timeout(provider_config.get("stale_timeout_seconds"))
    if pval is not None:
        candidates.append(pval)

    # Model-level timeouts (every model, since we don't know which is in use)
    models = provider_config.get("models", {})
    if isinstance(models, dict):
        for mname, mconf in models.items():
            if not isinstance(mconf, dict):
                continue
            mval = _coerce_timeout(mconf.get("stale_timeout_seconds"))
            if mval is not None:
                candidates.append(mval)

    if not candidates:
        return None

    return min(candidates)


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
