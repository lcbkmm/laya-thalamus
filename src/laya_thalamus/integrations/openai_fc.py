"""OpenAI function-calling integration helpers."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

from laya_thalamus.router import AgentRouter
from laya_thalamus.schema_tools import tools_from_openai, tools_to_openai
from laya_thalamus.schemas import DecisionAction, RouteDecision, ToolSpec


def last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Extract the latest user message text from OpenAI-style ``messages``."""
    for msg in reversed(list(messages)):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(p for p in parts if p)
        if content is not None:
            return str(content)
    return ""


def decision_to_tool_calls(
    decision: RouteDecision,
    *,
    arguments: str = "{}",
    call_id: str | None = None,
) -> list[dict[str, Any]] | None:
    """Map a route decision to OpenAI ``tool_calls``, or ``None`` to answer.

    Returns ``None`` when the agent should produce a normal assistant message
    (ANSWER / TERMINATE / FALLBACK_LLM / none tool).
    """
    if decision.action in (
        DecisionAction.ANSWER,
        DecisionAction.TERMINATE,
        DecisionAction.FALLBACK_LLM,
    ):
        return None
    tool = decision.selected_tool
    if not tool or tool == "none":
        return None
    return [
        {
            "id": call_id or f"call_{decision.decision_id or uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": tool, "arguments": arguments},
        }
    ]


def route_openai_turn(
    router: AgentRouter,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | Sequence[ToolSpec] | None = None,
    *,
    session_id: str | None = None,
    tool_result: str | None = None,
    history: list[str] | None = None,
    idempotency_key: str | None = None,
    include_none: bool = True,
) -> RouteDecision:
    """One turn: OpenAI messages + tools ->:class:`RouteDecision`.

    Example::

        d = route_openai_turn(router, messages, openai_tools, session_id=sid)
        calls = decision_to_tool_calls(d)
        if calls is None:
            # call chat.completions without tools / force answer
            ...
        else:
            # execute calls[0]["function"]["name"]
            ...
    """
    query = last_user_text(messages)
    if not query:
        query = "(empty)"

    specs: list[ToolSpec] | None = None
    if tools is not None:
        if tools and isinstance(tools[0], ToolSpec):
            specs = list(tools)  # type: ignore[arg-type]
        else:
            specs = tools_from_openai(
                tools,  # type: ignore[arg-type]
                include_none=include_none,
            )

    return router.route(
        query=query,
        tools=specs,
        session_id=session_id,
        tool_result=tool_result,
        history=history,
        idempotency_key=idempotency_key,
    )


# Re-export converters for a single import path
__all__ = [
    "last_user_text",
    "decision_to_tool_calls",
    "route_openai_turn",
    "tools_from_openai",
    "tools_to_openai",
]
