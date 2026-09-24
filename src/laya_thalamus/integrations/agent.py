"""5-line custom Agent loop helper with session + idempotency."""

from __future__ import annotations

import uuid
from typing import Any

from laya_thalamus.router import AgentRouter
from laya_thalamus.schemas import RouteDecision, ToolSpec


class AgentSession:
    """Thin session wrapper around :class:`AgentRouter`.

    Typical custom-agent loop::

        session = AgentSession(router)           # or AgentSession()
        d = session.next("计算 1+1")
        if d.action.value == "call_tool":
            result = run_tool(d.selected_tool)
            d = session.next("计算 1+1", tool_result=result)
        # d.decision_id / session.session_id for logs
    """

    def __init__(
        self,
        router: AgentRouter | None = None,
        *,
        session_id: str | None = None,
        tools: list[ToolSpec | dict] | None = None,
    ) -> None:
        self.router = router or AgentRouter(tools=tools)
        self.session_id = session_id or uuid.uuid4().hex

    def next(
        self,
        query: str,
        *,
        tool_result: str | None = None,
        context: str | dict | list | None = None,
        tools: list[ToolSpec | dict] | None = None,
        idempotency_key: str | None = None,
    ) -> RouteDecision:
        """Route one turn; circuit history is keyed by ``session_id``."""
        return self.router.route(
            query=query,
            context=context,
            tools=tools,
            tool_result=tool_result,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
        )

    def reset(self) -> None:
        self.router.reset_session(self.session_id)

    def snapshot(self) -> dict[str, Any]:
        return self.router.session_snapshot(self.session_id)
