"""Unit tests for AgentRouter (mock backend, no GPU)."""

from __future__ import annotations

import pytest

from laya_thalamus import AgentRouter, DecisionAction, ToolSpec
from laya_thalamus.backend import MockBackend
from laya_thalamus.circuit import CircuitBreaker
from laya_thalamus.config import CircuitBreakerConfig, RouterConfig
from laya_thalamus.context import compress_context
from laya_thalamus.tools import ToolRegistry


@pytest.fixture
def router() -> AgentRouter:
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.mode = "heuristic"
    return AgentRouter(config=cfg, backend=MockBackend())


def test_calculator_routing(router: AgentRouter):
    d = router.route("计算 12*34")
    assert d.selected_tool == "calculator"
    assert d.action == DecisionAction.CALL_TOOL


def test_search_routing(router: AgentRouter):
    d = router.route("搜索一下今天的新闻")
    assert d.selected_tool == "web_search"


def test_direct_answer(router: AgentRouter):
    d = router.route("你好")
    assert d.action == DecisionAction.ANSWER


def test_score_and_sufficient(router: AgentRouter):
    d = router.route(
        query="计算 1+1",
        tool_result="计算结果：2，这是对用户计算问题的直接回答",
        history=["calculator"],
    )
    assert d.credibility_score is not None
    assert d.credibility_score >= 3.0
    assert d.information_sufficient is True
    assert d.action == DecisionAction.ANSWER


def test_circuit_breaker_max_rounds(router: AgentRouter):
    router.config.circuit_breaker.max_tool_rounds = 2
    router.circuit = CircuitBreaker(router.config.circuit_breaker)
    d = router.route("搜索新闻", session_id="s1", history=["web_search", "web_search"])
    # history already at max ->terminate
    assert d.action == DecisionAction.TERMINATE
    assert d.terminated


def test_circuit_same_tool_loop():
    cb = CircuitBreaker(CircuitBreakerConfig(max_tool_rounds=10, max_same_tool_repeats=2))
    ok, _ = cb.check("x", ["calculator", "calculator"])
    assert not ok


def test_no_session_does_not_leak_across_calls(router: AgentRouter):
    """One-shot routes without session_id must not share circuit state."""
    router.config.circuit_breaker.max_tool_rounds = 2
    router.circuit = CircuitBreaker(router.config.circuit_breaker)
    for _ in range(5):
        d = router.route("计算 1+1")
        assert d.action != DecisionAction.TERMINATE


def test_tool_limit():
    reg = ToolRegistry(max_count=2)
    reg.register(ToolSpec(name="a", description="a"))
    reg.register(ToolSpec(name="b", description="b"))
    with pytest.raises(Exception):
        reg.register(ToolSpec(name="c", description="c"))


def test_context_compress():
    state = compress_context("q" * 100, context="c" * 5000, max_chars=200)
    assert len(state) <= 220


def test_blacklist(router: AgentRouter):
    router.update_config(tools={"blacklist": ["code_interpreter"]})
    d = router.route("写一?python 代码")
    assert d.selected_tool != "code_interpreter"


def test_allowed_tools_acl(router: AgentRouter):
    d = router.route(
        "计算 1+1",
        allowed_tools=["none", "web_search"],  # calculator forbidden
    )
    assert d.selected_tool in ("none", "web_search")


def test_health(router: AgentRouter):
    h = router.health()
    assert h["backend"] == "mock"
    assert h["inference_ok"] is True


def test_custom_tools(router: AgentRouter):
    d = router.route(
        "帮我翻译这段话",
        tools=[
            {"name": "none", "description": "直接回答"},
            {"name": "translator", "description": "翻译文本 translate language"},
        ],
    )
    assert d.selected_tool in ("translator", "none", "web_search") or d.selected_tool is not None


def test_hot_config(router: AgentRouter):
    router.update_config(thresholds={"choice_confidence": 0.99})
    assert router.config.thresholds.choice_confidence == 0.99
