"""Tests for LLM client parsing and real fallback chain."""

from __future__ import annotations

from laya_thalamus import AgentRouter, DecisionAction, ToolSpec
from laya_thalamus.backend import MockBackend
from laya_thalamus.config import RouterConfig
from laya_thalamus.fallback import (
    FallbackChain,
    HeuristicFallback,
    LLMFallback,
    build_fallback,
)
from laya_thalamus.llm import parse_json_object


class FakeLLMClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def chat_json(self, system: str, user: str) -> dict:
        self.calls += 1
        return dict(self.payload)


def test_parse_json_object_fenced():
    data = parse_json_object('here\n```json\n{"tool": "calculator", "confidence": 0.9}\n```')
    assert data["tool"] == "calculator"


def test_llm_fallback_tool_pick():
    client = FakeLLMClient({"tool": "calculator", "confidence": 0.88})
    fb = LLMFallback(client)  # type: ignore[arg-type]
    tools = [
        ToolSpec(name="none", description="direct"),
        ToolSpec(name="calculator", description="math"),
    ]
    name, conf, reason = fb.decide_tool("User query: 1+1", tools)
    assert name == "calculator"
    assert conf == 0.88
    assert "llm" in reason


def test_fallback_chain_primary_then_secondary():
    class Boom:
        name = "boom"

        def decide_tool(self, state, tools):
            raise RuntimeError("offline")

        def decide_sufficient(self, state):
            raise RuntimeError("offline")

        def decide_score(self, state):
            raise RuntimeError("offline")

    chain = FallbackChain(Boom(), HeuristicFallback())  # type: ignore[arg-type]
    tools = [
        ToolSpec(name="none", description="direct"),
        ToolSpec(name="calculator", description="math arithmetic"),
    ]
    name, conf, _ = chain.decide_tool("User query: 计算 2+2", tools)
    assert name == "calculator"
    assert conf > 0


def test_build_fallback_heuristic_mode():
    chain = build_fallback(mode="heuristic", enabled=True)
    assert chain is not None
    assert chain.name == "heuristic"


def test_build_fallback_disabled():
    assert build_fallback(mode="none", enabled=True) is None
    assert build_fallback(mode="auto", enabled=False) is None


def test_router_uses_injected_llm_fallback_on_backend_error():
    class DeadBackend(MockBackend):
        def predict(self, *args, **kwargs):
            raise RuntimeError("boom")

    def llm_fn(state, tools):
        return "web_search", 0.91, "injected"

    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.enabled = True
    cfg.fallback.mode = "heuristic"
    router = AgentRouter(
        config=cfg,
        backend=DeadBackend(),
        llm_fallback=llm_fn,
    )
    d = router.route("搜索今天新闻")
    assert d.degraded is True
    assert d.fallback_used == "callable"
    assert d.selected_tool == "web_search"


def test_hard_timeout_discards_partial_and_falls_back():
    class SlowBackend(MockBackend):
        def predict(self, state, *, tools=None, need_choice=True, need_noul=True, need_score=False, score_max=10):
            batch = super().predict(
                state,
                tools=tools,
                need_choice=need_choice,
                need_noul=need_noul,
                need_score=need_score,
                score_max=score_max,
            )
            batch.latency_ms = 99999
            batch.backend = "laya"
            return batch

        @property
        def name(self) -> str:  # type: ignore[override]
            return "laya"

    called = {"n": 0}

    def llm_fn(state, tools):
        called["n"] += 1
        return "calculator", 0.8, "llm"

    cfg = RouterConfig()
    cfg.model.timeout_ms = 50
    cfg.fallback.enabled = True
    router = AgentRouter(
        config=cfg,
        backend=SlowBackend(),
        llm_fallback=llm_fn,
    )
    # Monkey-patch name via class already
    d = router.route("计算 3*3")
    assert d.degraded is True
    assert "timeout" in (d.degrade_reason or "")
    assert called["n"] >= 1
    assert d.selected_tool == "calculator"


def test_mock_does_not_self_fallback_on_low_confidence():
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.mode = "heuristic"
    cfg.thresholds.choice_confidence = 0.99  # force low-conf condition
    router = AgentRouter(config=cfg, backend=MockBackend())
    d = router.route("计算 1+1")
    # Should still pick calculator via mock itself, without marking heuristic fallback
    assert d.selected_tool == "calculator"
    assert d.fallback_used is None
