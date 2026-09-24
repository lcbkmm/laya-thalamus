"""Pre / post decision hooks.

**Unstable** -``AgentRouter.hooks`` may evolve. See ``API.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from laya_thalamus.schemas import RouteDecision, RouteRequest

PreHook = Callable[[RouteRequest], RouteRequest]
PostHook = Callable[[RouteRequest, RouteDecision], RouteDecision]
DegradeHook = Callable[[RouteRequest, RouteDecision, str], None]


class HookManager:
    def __init__(self) -> None:
        self._pre: list[PreHook] = []
        self._post: list[PostHook] = []
        self._degrade: list[DegradeHook] = []

    def on_before(self, fn: PreHook) -> PreHook:
        self._pre.append(fn)
        return fn

    def on_after(self, fn: PostHook) -> PostHook:
        self._post.append(fn)
        return fn

    def on_degrade(self, fn: DegradeHook) -> DegradeHook:
        self._degrade.append(fn)
        return fn

    def run_pre(self, request: RouteRequest) -> RouteRequest:
        for fn in self._pre:
            request = fn(request)
        return request

    def run_post(self, request: RouteRequest, decision: RouteDecision) -> RouteDecision:
        for fn in self._post:
            decision = fn(request, decision)
        return decision

    def run_degrade(
        self, request: RouteRequest, decision: RouteDecision, reason: str
    ) -> None:
        for fn in self._degrade:
            try:
                fn(request, decision, reason)
            except Exception:
                pass
