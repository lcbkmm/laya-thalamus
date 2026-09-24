"""Pydantic schemas for route requests and decisions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionAction(str, Enum):
    """High-level action the Agent should take next."""

    ANSWER = "answer"  # enough info / no tool needed → LLM summarize
    CALL_TOOL = "call_tool"  # invoke selected tool
    RETRY_TOOL = "retry_tool"  # low score → retry / switch tool
    FALLBACK_LLM = "fallback_llm"  # hand routing decision to LLM
    TERMINATE = "terminate"  # circuit breaker / max rounds


class ToolSpec(BaseModel):
    name: str
    description: str
    group: str | None = None
    dangerous: bool = False
    # Optional JSON Schema for tool arguments (OpenAI ``parameters`` / LC args_schema).
    parameters: dict[str, Any] | None = None


class ToolCandidate(BaseModel):
    name: str
    confidence: float
    probability: float = 0.0


class RouteRequest(BaseModel):
    query: str
    context: str | dict[str, Any] | list[Any] | None = None
    tools: list[ToolSpec] | None = None
    tool_result: str | None = None  # latest tool / RAG snippet for score
    session_id: str | None = None
    history: list[str] = Field(default_factory=list)  # prior tool names
    config: dict[str, Any] | None = None  # per-request threshold overrides
    allowed_tools: list[str] | None = None  # session ACL
    # Same (session_id, idempotency_key) returns the cached decision (no re-infer).
    idempotency_key: str | None = None


class RouteDecision(BaseModel):
    action: DecisionAction
    selected_tool: str | None = None
    candidates: list[ToolCandidate] = Field(default_factory=list)
    information_sufficient: bool | None = None
    credibility_score: float | None = None  # 0-10
    confidence: float = 0.0
    reason: str = ""
    degraded: bool = False
    degrade_reason: str | None = None
    fallback_used: str | None = None  # llm | heuristic | None
    terminated: bool = False
    latency_ms: float = 0.0
    laya_latency_ms: float = 0.0
    backend: str = "mock"
    raw: dict[str, Any] = Field(default_factory=dict)
    # Observability / session
    decision_id: str = ""
    session_id: str | None = None
    idempotent_replay: bool = False
    cache_hit: bool = False

    def summary(self) -> str:
        parts = [f"action={self.action.value}"]
        if self.selected_tool:
            parts.append(f"tool={self.selected_tool}")
        if self.information_sufficient is not None:
            parts.append(f"sufficient={self.information_sufficient}")
        if self.credibility_score is not None:
            parts.append(f"score={self.credibility_score:.1f}")
        parts.append(f"conf={self.confidence:.2f}")
        if self.degraded:
            parts.append(f"degraded({self.degrade_reason})")
        if self.fallback_used:
            parts.append(f"via={self.fallback_used}")
        if self.cache_hit:
            parts.append("cache_hit")
        if self.decision_id:
            parts.append(f"id={self.decision_id[:8]}")
        return " | ".join(parts)
