"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from laya_thalamus.api import create_app
from laya_thalamus.backend import MockBackend
from laya_thalamus.config import RouterConfig
from laya_thalamus.router import AgentRouter


def _client() -> TestClient:
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    app = create_app(router=AgentRouter(config=cfg, backend=MockBackend()))
    return TestClient(app)


def test_health_endpoint():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["inference_ok"] is True


def test_route_endpoint():
    c = _client()
    r = c.post("/route", json={"query": "计算 2+2"})
    assert r.status_code == 200
    body = r.json()
    assert body["selected_tool"] == "calculator"
    assert "action" in body


def test_batch_endpoint():
    c = _client()
    r = c.post(
        "/batch",
        json={"items": [{"query": "你好"}, {"query": "搜索新闻"}]},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_config_get():
    c = _client()
    r = c.get("/config")
    assert r.status_code == 200
    assert "thresholds" in r.json()
