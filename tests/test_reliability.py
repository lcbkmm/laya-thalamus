"""Production reliability: async, cache, timeout policy, metrics."""

from __future__ import annotations

import asyncio

import pytest

from laya_thalamus import AgentRouter, DecisionAction
from laya_thalamus.backend import MockBackend
from laya_thalamus.config import ModelConfig, RouterConfig
from laya_thalamus.policy import should_hard_fail_on_timeout, summarize_policy
from laya_thalamus.schemas import ToolSpec


def _router(**cfg_kw) -> AgentRouter:
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.enabled = False
    for k, v in cfg_kw.items():
        if k == "cache":
            cfg.cache = cfg.cache.model_copy(update=v)
        elif k == "model":
            cfg.model = cfg.model.model_copy(update=v)
        elif k == "logging":
            cfg.logging = cfg.logging.model_copy(update=v)
    return AgentRouter(config=cfg, backend=MockBackend())


@pytest.mark.parametrize(
    "backend,latency,timeout,patterns,expect",
    [
        ("laya", 3000, 2000, ["laya"], True),
        ("laya", 1000, 2000, ["laya"], False),
        ("mock", 5000, 2000, ["laya"], False),
        ("llm", 5000, 2000, ["laya"], False),
        ("llm:deepseek-v3.1", 5000, 2000, ["laya"], False),
        ("llm:deepseek-v3.1", 5000, 2000, ["llm"], True),
        ("laya", 5000, 2000, ["laya"], True),
        ("laya", 5000, 0, ["laya"], False),
    ],
)
def test_timeout_hard_fail_matrix(backend, latency, timeout, patterns, expect):
    model = ModelConfig(
        timeout_ms=timeout,
        hard_fail_on_timeout=True,
        timeout_hard_fail_backends=list(patterns),
    )
    assert should_hard_fail_on_timeout(backend, latency, model) is expect


def test_hard_fail_disabled():
    model = ModelConfig(
        timeout_ms=100,
        hard_fail_on_timeout=False,
        timeout_hard_fail_backends=["laya"],
    )
    assert should_hard_fail_on_timeout("laya", 9999, model) is False


def test_decision_cache_hit():
    router = _router(cache={"enabled": True, "ttl_s": 60.0})
    d1 = router.route("计算 2+2")
    d2 = router.route("计算 2+2")
    assert d1.cache_hit is False
    assert d2.cache_hit is True
    assert d1.selected_tool == d2.selected_tool
    assert d1.decision_id != d2.decision_id
    m = router.metrics()
    assert m["thalamus_cache_hits_total"] >= 1


def test_decision_cache_miss_on_different_query():
    router = _router(cache={"enabled": True, "ttl_s": 60.0})
    router.route("计算 2+2")
    d = router.route("你好")
    assert d.cache_hit is False


def test_aroute():
    router = _router()

    async def _run():
        return await router.aroute("计算 1+1")

    d = asyncio.run(_run())
    assert d.decision_id
    assert d.action in (
        DecisionAction.CALL_TOOL,
        DecisionAction.ANSWER,
        DecisionAction.TERMINATE,
        DecisionAction.FALLBACK_LLM,
        DecisionAction.RETRY_TOOL,
    )


def test_prometheus_text_and_json_names():
    router = _router()
    router.route("计算 2+2")
    text = router.metrics_prometheus()
    assert "thalamus_requests_total" in text
    assert "# TYPE" in text
    snap = router.metrics()
    assert "thalamus_requests_total" in snap
    assert snap["total_requests"] >= 1  # legacy alias


def test_policy_in_health():
    router = _router()
    h = router.health()
    assert "policy" in h
    assert h["policy"]["timeout_hard_fail_backends"] == ["laya"]
    assert summarize_policy(router.config)["cache_enabled"] is False


def test_otel_noop_without_sdk():
    router = _router(logging={"otel": True})
    d = router.route("你好")
    assert d.decision_id
