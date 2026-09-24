"""OpenTelemetry helpers (optional ``[otel]`` extra).

**Unstable**. No-op when ``opentelemetry-api`` is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_TRACER_NAME = "laya_thalamus"


def _get_tracer():
    try:
        from opentelemetry import trace

        return trace.get_tracer(_TRACER_NAME)
    except ImportError:
        return None


@contextmanager
def route_span(
    enabled: bool,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Yield an OTel span around ``route`` when enabled and SDK is present."""
    if not enabled:
        yield None
        return
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span("laya_thalamus.route") as span:
        if attributes:
            for k, v in attributes.items():
                if v is None:
                    continue
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield span


def set_span_decision(span: Any, decision: Any) -> None:
    if span is None or decision is None:
        return
    try:
        span.set_attribute("laya.action", getattr(decision.action, "value", str(decision.action)))
        span.set_attribute("laya.selected_tool", decision.selected_tool or "")
        span.set_attribute("laya.decision_id", decision.decision_id or "")
        span.set_attribute("laya.backend", decision.backend or "")
        span.set_attribute("laya.degraded", bool(decision.degraded))
        span.set_attribute("laya.cache_hit", bool(getattr(decision, "cache_hit", False)))
        span.set_attribute("laya.latency_ms", float(decision.latency_ms or 0.0))
    except Exception:
        pass
