"""Tests for schema conversion, integrations, session / idempotency."""

from __future__ import annotations

from laya_thalamus import AgentRouter, DecisionAction
from laya_thalamus.backend import MockBackend
from laya_thalamus.config import RouterConfig
from laya_thalamus.integrations import (
    AgentSession,
    LayaToolSelector,
    decision_to_tool_calls,
    last_user_text,
    route_openai_turn,
    tools_from_langchain,
    tools_from_openai,
    tools_to_openai,
)
from laya_thalamus.schema_tools import tools_from_json_schema, tools_to_json_schema
from laya_thalamus.schemas import ToolSpec


def _mock_router(tools: list[ToolSpec] | None = None) -> AgentRouter:
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.enabled = False
    return AgentRouter(config=cfg, backend=MockBackend(), tools=tools)


def test_openai_roundtrip():
    specs = [
        ToolSpec(
            name="calculator",
            description="math",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        )
    ]
    openai_tools = tools_to_openai(specs)
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "calculator"
    back = tools_from_openai(openai_tools, include_none=True)
    names = {t.name for t in back}
    assert "calculator" in names
    assert "none" in names
    assert back[[t.name for t in back].index("calculator")].parameters["required"] == [
        "expr"
    ]


def test_json_schema_roundtrip():
    raw = [
        {
            "name": "web_search",
            "description": "search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]
    specs = tools_from_json_schema(raw)
    assert specs[0].name == "web_search"
    out = tools_to_json_schema(specs)
    assert out[0]["parameters"]["properties"]["q"]["type"] == "string"


def test_tools_from_langchain_duck():
    class FakeTool:
        name = "Search"
        description = "Search the web"

        class args_schema:
            @staticmethod
            def model_json_schema():
                return {"type": "object", "properties": {"query": {"type": "string"}}}

    specs = tools_from_langchain([FakeTool()], include_none=False)
    assert specs[0].name == "Search"
    assert "query" in specs[0].parameters["properties"]


def test_decision_id_and_session():
    router = _mock_router()
    d = router.route("计算 2+2", session_id="sess-a")
    assert d.decision_id
    assert d.session_id == "sess-a"
    assert len(d.decision_id) >= 16
    snap = router.session_snapshot("sess-a")
    assert snap["last_decision_id"] == d.decision_id
    assert snap["route_count"] >= 1


def test_idempotency_replay():
    router = _mock_router()
    d1 = router.route(
        "计算 2+2",
        session_id="sess-b",
        idempotency_key="req-1",
    )
    d2 = router.route(
        "计算 2+2",
        session_id="sess-b",
        idempotency_key="req-1",
    )
    assert d1.decision_id == d2.decision_id
    assert d2.idempotent_replay is True
    assert d1.idempotent_replay is False
    d3 = router.route(
        "计算 2+2",
        session_id="sess-b",
        idempotency_key="req-2",
    )
    assert d3.decision_id != d1.decision_id


def test_reset_session_clears_idempotency():
    router = _mock_router()
    d1 = router.route("你好", session_id="sess-c", idempotency_key="k")
    router.reset_session("sess-c")
    d2 = router.route("你好", session_id="sess-c", idempotency_key="k")
    assert d2.idempotent_replay is False
    assert d2.decision_id != d1.decision_id


def test_agent_session_five_liner():
    session = AgentSession(_mock_router(), session_id="agent-1")
    d = session.next("计算 123*456")
    assert d.session_id == "agent-1"
    assert d.decision_id
    if d.action == DecisionAction.CALL_TOOL:
        d2 = session.next("计算 123*456", tool_result="56088")
        assert d2.decision_id
    snap = session.snapshot()
    assert snap["session_id"] == "agent-1"


def test_openai_turn_and_tool_calls():
    router = _mock_router()
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "计算 2+2"},
    ]
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "math",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert last_user_text(messages) == "计算 2+2"
    d = route_openai_turn(router, messages, openai_tools, session_id="oa-1")
    calls = decision_to_tool_calls(d)
    if d.action == DecisionAction.CALL_TOOL:
        assert calls is not None
        assert calls[0]["function"]["name"] == d.selected_tool
    else:
        assert calls is None


def test_langchain_selector():
    class T:
        name = "Calculator"
        description = "Useful for math expressions"

    sel = LayaToolSelector(tools=[T()], session_id="lc-1")
    tool = sel.pick("What is 2+2?")
    # mock keyword path may map to calculator-like names; just ensure API works
    assert tool is None or isinstance(tool, str)
    d = sel.as_runnable()({"input": "你好", "intermediate_steps": []})
    assert d.decision_id
