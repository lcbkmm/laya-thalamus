"""Structured JSON decision logging + Prometheus-oriented metrics.

**Unstable**. Prefer ``AgentRouter.metrics()`` / decision fields. See ``API.md``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from laya_thalamus.schemas import RouteDecision

logger = logging.getLogger("laya_thalamus")

# Stable structured log / record field names (do not rename lightly).
LOG_FIELDS = (
    "ts",
    "query",
    "action",
    "selected_tool",
    "confidence",
    "degraded",
    "degrade_reason",
    "latency_ms",
    "laya_latency_ms",
    "backend",
    "fallback_used",
    "session_id",
    "decision_id",
    "information_sufficient",
    "credibility_score",
    "reason",
    "cache_hit",
    "idempotent_replay",
)


@dataclass
class DecisionRecord:
    ts: float
    query: str
    action: str
    selected_tool: str | None
    confidence: float
    degraded: bool
    degrade_reason: str | None
    latency_ms: float
    laya_latency_ms: float
    backend: str
    fallback_used: str | None = None
    session_id: str | None = None
    decision_id: str | None = None
    information_sufficient: bool | None = None
    credibility_score: float | None = None
    reason: str = ""
    cache_hit: bool = False
    idempotent_replay: bool = False


@dataclass
class Metrics:
    total: int = 0
    degraded: int = 0
    cache_hits: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    tool_counts: Counter = field(default_factory=Counter)
    action_counts: Counter = field(default_factory=Counter)

    def snapshot(self) -> dict[str, Any]:
        """JSON metrics using Prometheus-style names (+ legacy aliases)."""
        count = self.total
        lat_sum = sum(self.latencies_ms)
        avg = (lat_sum / len(self.latencies_ms)) if self.latencies_ms else 0.0
        prom = {
            "thalamus_requests_total": self.total,
            "thalamus_degraded_total": self.degraded,
            "thalamus_cache_hits_total": self.cache_hits,
            "thalamus_latency_ms_sum": round(lat_sum, 2),
            "thalamus_latency_ms_count": len(self.latencies_ms),
            "thalamus_latency_ms_avg": round(avg, 2),
            "thalamus_tool_selected_total": dict(self.tool_counts),
            "thalamus_action_total": dict(self.action_counts),
        }
        # Legacy aliases (0.x compatibility)
        prom["total_requests"] = self.total
        prom["degrade_rate"] = (self.degraded / count) if count else 0.0
        prom["avg_latency_ms"] = round(avg, 2)
        prom["tool_distribution"] = dict(self.tool_counts)
        prom["action_distribution"] = dict(self.action_counts)
        return prom

    def prometheus_text(self) -> str:
        """OpenMetrics / Prometheus exposition format."""
        lines: list[str] = []
        snap = self.snapshot()

        def counter(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        def gauge(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        counter(
            "thalamus_requests_total",
            "Total route decisions",
            float(snap["thalamus_requests_total"]),
        )
        counter(
            "thalamus_degraded_total",
            "Decisions that used degrade / fallback path",
            float(snap["thalamus_degraded_total"]),
        )
        counter(
            "thalamus_cache_hits_total",
            "Decision-cache hits",
            float(snap["thalamus_cache_hits_total"]),
        )
        counter(
            "thalamus_latency_ms_sum",
            "Sum of end-to-end route latency in milliseconds",
            float(snap["thalamus_latency_ms_sum"]),
        )
        counter(
            "thalamus_latency_ms_count",
            "Count of latency samples",
            float(snap["thalamus_latency_ms_count"]),
        )
        gauge(
            "thalamus_latency_ms_avg",
            "Average route latency in milliseconds",
            float(snap["thalamus_latency_ms_avg"]),
        )
        lines.append(
            "# HELP thalamus_tool_selected_total Tool selections by name"
        )
        lines.append("# TYPE thalamus_tool_selected_total counter")
        for tool, n in sorted(self.tool_counts.items()):
            safe = str(tool).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'thalamus_tool_selected_total{{tool="{safe}"}} {n}')
        lines.append("# HELP thalamus_action_total Decisions by action")
        lines.append("# TYPE thalamus_action_total counter")
        for action, n in sorted(self.action_counts.items()):
            safe = str(action).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'thalamus_action_total{{action="{safe}"}} {n}')
        lines.append("")
        return "\n".join(lines)


class DecisionLogger:
    def __init__(
        self,
        enabled: bool = True,
        json_path: str | None = None,
        level: str = "INFO",
    ) -> None:
        self.enabled = enabled
        self.json_path = Path(json_path) if json_path else None
        self.metrics = Metrics()
        self._lock = threading.Lock()
        if self.json_path:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
        logging.getLogger("laya_thalamus").setLevel(
            getattr(logging, level.upper(), logging.INFO)
        )

    def log(
        self,
        query: str,
        decision: RouteDecision,
        session_id: str | None = None,
        decision_id: str | None = None,
    ) -> DecisionRecord:
        record = DecisionRecord(
            ts=time.time(),
            query=query[:500],
            action=decision.action.value,
            selected_tool=decision.selected_tool,
            confidence=decision.confidence,
            degraded=decision.degraded,
            degrade_reason=decision.degrade_reason,
            latency_ms=decision.latency_ms,
            laya_latency_ms=decision.laya_latency_ms,
            backend=decision.backend,
            fallback_used=decision.fallback_used,
            session_id=session_id or decision.session_id,
            decision_id=decision_id or decision.decision_id or None,
            information_sufficient=decision.information_sufficient,
            credibility_score=decision.credibility_score,
            reason=decision.reason,
            cache_hit=bool(decision.cache_hit),
            idempotent_replay=bool(decision.idempotent_replay),
        )
        if not self.enabled:
            return record

        with self._lock:
            self.metrics.total += 1
            if decision.degraded:
                self.metrics.degraded += 1
            if decision.cache_hit:
                self.metrics.cache_hits += 1
            self.metrics.latencies_ms.append(decision.latency_ms)
            self.metrics.action_counts[decision.action.value] += 1
            if decision.selected_tool:
                self.metrics.tool_counts[decision.selected_tool] += 1

            if self.json_path:
                with self.json_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        logger.info(
            "route action=%s tool=%s conf=%.2f latency_ms=%.1f backend=%s "
            "decision_id=%s session_id=%s degraded=%s cache_hit=%s fallback=%s",
            record.action,
            record.selected_tool or "-",
            record.confidence,
            record.latency_ms,
            record.backend,
            record.decision_id or "-",
            record.session_id or "-",
            record.degraded,
            record.cache_hit,
            record.fallback_used or "-",
        )
        return record
