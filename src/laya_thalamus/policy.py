"""Timeout / degrade policy helpers (documented in POLICY.md).

**Unstable** module path; behavior is part of the ``AgentRouter`` contract.
"""

from __future__ import annotations

from laya_thalamus.config import FallbackConfig, ModelConfig, RouterConfig


def backend_matches(backend_name: str, patterns: list[str]) -> bool:
    """True if ``backend_name`` equals or is prefixed by any pattern (e.g. ``llm`` ->``llm:x``)."""
    name = backend_name or ""
    for p in patterns:
        if not p:
            continue
        if name == p or name.startswith(f"{p}:") or name.startswith(f"{p}_"):
            return True
    return False


def should_hard_fail_on_timeout(
    backend_name: str,
    latency_ms: float,
    model: ModelConfig,
) -> bool:
    """Whether to discard primary backend output after a latency overrun.

    Contract (see POLICY.md):
    - Only when ``model.hard_fail_on_timeout`` is true and ``timeout_ms > 0``.
    - Only for backends listed in ``model.timeout_hard_fail_backends``
      (default: ``["laya"]``). ``mock`` and ``llm*`` are excluded by default so
      slow chat models are not wiped by a System-1-oriented timeout.
    """
    if not model.hard_fail_on_timeout:
        return False
    if not model.timeout_ms or model.timeout_ms <= 0:
        return False
    if latency_ms <= model.timeout_ms:
        return False
    return backend_matches(backend_name, list(model.timeout_hard_fail_backends))


def should_hard_fail_on_exception(model: ModelConfig) -> bool:
    """Exceptions always hard-fail when enabled (default true)."""
    return bool(model.hard_fail_on_exception)


def fallback_allowed(fallback: FallbackConfig) -> bool:
    return bool(fallback.enabled and fallback.mode != "none")


def summarize_policy(cfg: RouterConfig) -> dict:
    """Stable dict for docs / health / tests."""
    return {
        "timeout_ms": cfg.model.timeout_ms,
        "hard_fail_on_timeout": cfg.model.hard_fail_on_timeout,
        "hard_fail_on_exception": cfg.model.hard_fail_on_exception,
        "timeout_hard_fail_backends": list(cfg.model.timeout_hard_fail_backends),
        "fallback_enabled": cfg.fallback.enabled,
        "fallback_mode": cfg.fallback.mode,
        "llm_timeout_s": cfg.fallback.llm.timeout_s,
        "cache_enabled": cfg.cache.enabled,
        "cache_ttl_s": cfg.cache.ttl_s,
    }
