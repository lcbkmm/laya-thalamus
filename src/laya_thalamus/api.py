"""FastAPI HTTP surface: POST /route, /health, /models, /batch, /metrics.

Requires the ``api`` extra: ``pip install "laya-thalamus[api]"``.

HTTP request/response shapes are **unstable** in 0.x; library users should
prefer ``AgentRouter`` directly. See ``API.md``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from laya_thalamus.config import load_config
from laya_thalamus.extras import require_api
from laya_thalamus.router import AgentRouter
from laya_thalamus.schemas import RouteDecision, RouteRequest, ToolSpec

require_api()

from fastapi import FastAPI, HTTPException, Response  # noqa: E402

try:
    from laya_thalamus import __version__ as _VERSION
except Exception:  # pragma: no cover
    _VERSION = "0.1.0"


class ModelLoadBody(BaseModel):
    name: str | None = None
    backend: str | None = None


class ConfigUpdateBody(BaseModel):
    thresholds: dict[str, Any] | None = None
    circuit_breaker: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    tools: dict[str, Any] | None = None
    logging: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None


class BatchRouteBody(BaseModel):
    items: list[RouteRequest] = Field(default_factory=list)


def create_app(router: AgentRouter | None = None, config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path) if config_path else None
    agent = router or AgentRouter(config=cfg)

    app = FastAPI(
        title="Laya Thalamus",
        description="System-1 decision middleware: choice / noul / score",
        version=_VERSION,
    )
    app.state.agent = agent

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await asyncio_to_thread(agent.health)

    @app.get("/metrics")
    async def metrics_json() -> dict[str, Any]:
        """JSON metrics (Prometheus-style field names + legacy aliases)."""
        return agent.metrics()

    @app.get("/metrics/prometheus")
    async def metrics_prometheus() -> Response:
        """Prometheus / OpenMetrics text exposition."""
        return Response(
            content=agent.metrics_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/route", response_model=RouteDecision)
    async def route(body: RouteRequest) -> RouteDecision:
        try:
            return await agent.aroute_request(body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/batch")
    async def batch(body: BatchRouteBody) -> dict[str, Any]:
        results = []
        for item in body.items:
            try:
                d = await agent.aroute_request(item)
                results.append(d.model_dump())
            except Exception as e:
                results.append({"error": str(e)})
        return {"count": len(results), "results": results}

    @app.post("/models/load")
    async def load_model(body: ModelLoadBody) -> dict[str, Any]:
        if body.backend:
            agent.config.model.backend = body.backend
        try:
            await asyncio_to_thread(agent.load_model, body.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"status": "loaded", "backend": agent.backend.name, "model": agent.config.model.name}

    @app.post("/models/unload")
    async def unload_model() -> dict[str, str]:
        await asyncio_to_thread(agent.unload_model)
        return {"status": "unloaded"}

    @app.post("/config")
    async def update_config(body: ConfigUpdateBody) -> dict[str, Any]:
        overrides = {k: v for k, v in body.model_dump().items() if v is not None}
        agent.update_config(**overrides)
        return {"status": "updated", "config": agent.config.model_dump()}

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        return agent.config.model_dump()

    @app.get("/tools")
    async def list_tools() -> list[dict[str, Any]]:
        return [t.model_dump() for t in agent.registry.list()]

    @app.post("/tools")
    async def register_tool(tool: ToolSpec) -> dict[str, str]:
        agent.register_tool(tool)
        return {"status": "registered", "name": tool.name}

    return app


async def asyncio_to_thread(fn, *args, **kwargs):
    import asyncio

    if kwargs:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    return await asyncio.to_thread(fn, *args)


def run(host: str | None = None, port: int | None = None, config_path: str | None = None) -> None:
    require_api()
    import uvicorn

    cfg = load_config(config_path)
    app = create_app(config_path=config_path)
    uvicorn.run(
        app,
        host=host or cfg.server.host,
        port=port or cfg.server.port,
    )
