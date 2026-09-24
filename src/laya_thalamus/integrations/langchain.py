"""LangChain-oriented adapter (no hard dependency on langchain)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from laya_thalamus.router import AgentRouter
from laya_thalamus.schema_tools import tools_from_langchain
from laya_thalamus.schemas import DecisionAction, RouteDecision, ToolSpec


def _steps_to_context(
    intermediate_steps: list | None,
) -> tuple[str | None, str | None, list[str]]:
    """Parse AgentExecutor-style ``[(AgentAction, observation), ...]``."""
    if not intermediate_steps:
        return None, None, []
    parts: list[str] = []
    history: list[str] = []
    for step in intermediate_steps:
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            action, obs = step[0], step[1]
            name = getattr(action, "tool", None) or str(action)
            history.append(str(name))
            parts.append(f"{name}: {obs}")
        else:
            parts.append(str(step))
    ctx = "\n".join(parts) if parts else None
    tool_result = parts[-1] if parts else None
    return ctx, tool_result, history


class LayaToolSelector:
    """Pick the next LangChain tool via Laya (drop-in for AgentExecutor thinking).

    Works with duck-typed tools (``name`` / ``description`` / optional
    ``args_schema``) -``langchain`` does not need to be installed.
    """

    def __init__(
        self,
        router: AgentRouter | None = None,
        tools: Sequence[Any] | Sequence[ToolSpec] | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        if tools is not None and len(tools) > 0 and isinstance(tools[0], ToolSpec):
            specs = list(tools)  # type: ignore[arg-type]
        elif tools:
            specs = tools_from_langchain(tools)
        else:
            specs = None
        self.router = router or AgentRouter(tools=specs)
        if specs is not None and router is not None:
            for t in specs:
                if t.name not in self.router.registry:
                    self.router.register_tool(t)
        self.session_id = session_id
        self._bound_tools = specs

    def decide(
        self,
        query: str,
        intermediate_steps: list | None = None,
        *,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RouteDecision:
        ctx, tool_result, history = _steps_to_context(intermediate_steps)
        return self.router.route(
            query=query,
            context=ctx,
            tool_result=tool_result,
            history=history,
            tools=self._bound_tools,
            session_id=session_id or self.session_id,
            idempotency_key=idempotency_key,
        )

    def pick(
        self,
        query: str,
        intermediate_steps: list | None = None,
        *,
        session_id: str | None = None,
    ) -> str | None:
        """Return tool name to call, or ``None`` to finish / let LLM answer."""
        d = self.decide(query, intermediate_steps, session_id=session_id)
        if d.action in (DecisionAction.ANSWER, DecisionAction.TERMINATE):
            return None
        tool = d.selected_tool
        return None if tool in (None, "none") else tool

    def as_runnable(self) -> Callable[[dict[str, Any]], RouteDecision]:
        """Minimal Runnable-shaped callable: ``{"input", "intermediate_steps?"}`` ->decision."""

        def _invoke(payload: dict[str, Any]) -> RouteDecision:
            query = str(payload.get("input") or payload.get("query") or "")
            steps = payload.get("intermediate_steps")
            sid = payload.get("session_id", self.session_id)
            key = payload.get("idempotency_key")
            return self.decide(
                query,
                steps,
                session_id=sid,
                idempotency_key=key,
            )

        return _invoke
