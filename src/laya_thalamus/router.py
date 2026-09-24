"""AgentRouter -one-line System-1 decision API for Agents."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable

from laya_thalamus.backend import DecisionBackend, create_backend
from laya_thalamus.cache import DecisionCache, fingerprint as decision_fingerprint
from laya_thalamus.circuit import CircuitBreaker
from laya_thalamus.config import RouterConfig, clone_config, load_config
from laya_thalamus.context import compress_context, sanitize_text
from laya_thalamus.exceptions import ToolLimitExceeded, ValidationError
from laya_thalamus.fallback import (
    CallableToolFallback,
    FallbackChain,
    HeuristicFallback,
    build_fallback,
)
from laya_thalamus.hooks import HookManager
from laya_thalamus.idempotency import IdempotencyCache
from laya_thalamus.logging_util import DecisionLogger
from laya_thalamus.policy import (
    should_hard_fail_on_exception,
    should_hard_fail_on_timeout,
    summarize_policy,
)
from laya_thalamus.schemas import (
    DecisionAction,
    RouteDecision,
    RouteRequest,
    ToolCandidate,
    ToolSpec,
)
from laya_thalamus.telemetry import route_span, set_span_decision
from laya_thalamus.tools import ToolRegistry

logger = logging.getLogger("laya_thalamus")

# Optional user-supplied LLM fallback: (state, tools) -> (tool_name, confidence, reason)
LLMFallbackFn = Callable[[str, list[ToolSpec]], tuple[str, float, str]]


class AgentRouter:
    """
    System-1 decision middleware.

    Typical usage::

        router = AgentRouter()                 # auto mock/laya
        decision = router.route(
            query="计算 123*456",
            tools=[{"name": "calculator", "description": "math"}],
            session_id="user-42",              # persist circuit across turns
        )
        print(decision.decision_id, decision.selected_tool)
    """

    def __init__(
        self,
        config: RouterConfig | str | None = None,
        *,
        backend: DecisionBackend | None = None,
        tools: list[ToolSpec | dict] | None = None,
        llm_fallback: LLMFallbackFn | None = None,
        idempotency_ttl_s: float | None = 600.0,
        idempotency_max_size: int = 1024,
    ) -> None:
        if isinstance(config, str):
            self.config = load_config(config)
        elif config is None:
            self.config = load_config()
        else:
            self.config = config

        self.registry = ToolRegistry(max_count=self.config.tools.max_count)
        if tools:
            self.registry.register_many(tools)
        else:
            self.registry.ensure_defaults()

        self.hooks = HookManager()
        self.circuit = CircuitBreaker(self.config.circuit_breaker)
        self.logger = DecisionLogger(
            enabled=self.config.logging.enabled,
            json_path=self.config.logging.json_path,
            level=self.config.logging.level,
        )
        self._idempotency: IdempotencyCache[RouteDecision] = IdempotencyCache(
            max_size=idempotency_max_size,
            ttl_s=idempotency_ttl_s,
        )
        self._decision_cache = DecisionCache(
            max_size=self.config.cache.max_size,
            ttl_s=self.config.cache.ttl_s,
        )
        fb_cfg = self.config.fallback
        self._fallback: FallbackChain | None
        if llm_fallback is not None:
            self._fallback = FallbackChain(
                CallableToolFallback(llm_fallback), HeuristicFallback()
            )
        else:
            self._fallback = build_fallback(
                mode=fb_cfg.mode,
                enabled=fb_cfg.enabled,
                llm_base_url=fb_cfg.llm.base_url,
                llm_api_key=fb_cfg.llm.api_key,
                llm_model=fb_cfg.llm.model,
                llm_timeout_s=fb_cfg.llm.timeout_s,
            )

        if backend is not None:
            self.backend = backend
            if not self.backend.is_ready():
                self.backend.load()
        else:
            self.backend = create_backend(
                backend=self.config.model.backend,
                model_name=self.config.model.name,
                device=self.config.model.device,
                timeout_ms=self.config.model.timeout_ms,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        context: str | dict | list | None = None,
        tools: list[ToolSpec | dict] | None = None,
        *,
        tool_result: str | None = None,
        session_id: str | None = None,
        history: list[str] | None = None,
        config: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> RouteDecision:
        """One-line routing entry. Returns a read-only decision (no answer text)."""
        req = RouteRequest(
            query=query,
            context=context,
            tools=[ToolSpec.model_validate(t) for t in tools] if tools else None,
            tool_result=tool_result,
            session_id=session_id,
            history=history or [],
            config=config,
            allowed_tools=allowed_tools,
            idempotency_key=idempotency_key,
        )
        return self.route_request(req)

    async def aroute(
        self,
        query: str,
        context: str | dict | list | None = None,
        tools: list[ToolSpec | dict] | None = None,
        *,
        tool_result: str | None = None,
        session_id: str | None = None,
        history: list[str] | None = None,
        config: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> RouteDecision:
        """Async wrapper -runs sync ``route`` in a worker thread."""
        return await asyncio.to_thread(
            self.route,
            query,
            context,
            tools,
            tool_result=tool_result,
            session_id=session_id,
            history=history,
            config=config,
            allowed_tools=allowed_tools,
            idempotency_key=idempotency_key,
        )

    async def aroute_request(self, request: RouteRequest) -> RouteDecision:
        return await asyncio.to_thread(self.route_request, request)

    def route_request(self, request: RouteRequest) -> RouteDecision:
        t0 = time.perf_counter()
        with route_span(
            bool(self.config.logging.otel),
            attributes={
                "laya.session_id": request.session_id or "",
                "laya.query_len": len(request.query or ""),
            },
        ) as span:
            decision = self._route_request_body(request, t0)
            set_span_decision(span, decision)
            return decision

    def _route_request_body(
        self, request: RouteRequest, t0: float
    ) -> RouteDecision:
        # Idempotent replay: same (session_id, key) ->identical decision_id
        if request.session_id and request.idempotency_key:
            cached = self._idempotency.get(request.session_id, request.idempotency_key)
            if cached is not None:
                replay = cached.model_copy(deep=True)
                replay.idempotent_replay = True
                replay.session_id = request.session_id
                return replay

        request = self.hooks.run_pre(request)
        cfg = self._effective_config(request.config)

        try:
            self._validate_query(request.query, cfg.context.max_chars)
        except ValidationError as e:
            decision = RouteDecision(
                action=DecisionAction.TERMINATE,
                reason=str(e),
                terminated=True,
                confidence=1.0,
                backend=self.backend.name,
            )
            return self._finish(request, decision, t0, cache_key=None)

        # Circuit breaker (persists only when session_id is set)
        if request.session_id:
            if request.history:
                self.circuit.sync_history(request.session_id, request.history)
            hist = (
                request.history
                if request.history
                else list(self.circuit.get_session(request.session_id).history)
            )
        else:
            hist = list(request.history or [])
        ok, trip_reason = self.circuit.check(request.session_id, hist)
        if not ok:
            decision = RouteDecision(
                action=DecisionAction.TERMINATE,
                reason=trip_reason or "circuit tripped",
                terminated=True,
                confidence=1.0,
                backend=self.backend.name,
            )
            return self._finish(request, decision, t0, cache_key=None)

        tool_list = self._resolve_tools(request, cfg)
        if len(tool_list) > cfg.tools.max_count:
            raise ToolLimitExceeded(
                f"{len(tool_list)} tools > limit {cfg.tools.max_count}; split groups"
            )

        cache_key: str | None = None
        if cfg.cache.enabled:
            cache_key = decision_fingerprint(
                request.query,
                tool_list,
                tool_result=request.tool_result,
                context=request.context,
            )
            hit = self._decision_cache.get(cache_key)
            if hit is not None:
                replay = hit.model_copy(deep=True)
                replay.cache_hit = True
                replay.session_id = request.session_id
                replay.decision_id = ""
                return self._finish(request, replay, t0, cache_key=None)

        state = compress_context(
            request.query,
            request.context,
            max_chars=cfg.context.max_chars,
            keep_tail=cfg.context.keep_tail,
            tool_result=request.tool_result,
        )

        need_score = bool(request.tool_result)
        degraded = False
        degrade_reason: str | None = None
        fallback_used: str | None = None
        raw: dict[str, Any] = {}
        laya_ms = 0.0
        hard_fail = False

        try:
            batch = self.backend.predict(
                state,
                tools=tool_list,
                need_choice=True,
                need_noul=True,
                need_score=need_score,
                score_max=10,
            )
            laya_ms = batch.latency_ms
            raw = batch.raw
            if should_hard_fail_on_timeout(
                self.backend.name, batch.latency_ms, cfg.model
            ):
                hard_fail = True
                degraded = True
                degrade_reason = (
                    f"{self.backend.name} timeout {batch.latency_ms:.0f}ms "
                    f"> {cfg.model.timeout_ms}ms"
                )
        except Exception as e:
            logger.exception("%s predict failed", self.backend.name)
            degraded = True
            if should_hard_fail_on_exception(cfg.model):
                hard_fail = True
            degrade_reason = f"{self.backend.name} error: {e}"
            batch = None

        selected = "none"
        probs: dict[str, float] = {}
        choice_conf = 0.0
        sufficient: bool | None = None
        noul_conf = 0.0
        cred: float | None = None
        score_conf = 0.0

        use_primary_choice = bool(batch and batch.choice and not hard_fail)
        if use_primary_choice:
            selected = batch.choice.choice  # type: ignore[union-attr]
            probs = batch.choice.probabilities  # type: ignore[union-attr]
            choice_conf = batch.choice.confidence  # type: ignore[union-attr]
            if self._should_degrade_confidence(
                choice_conf, cfg.thresholds.choice_confidence, cfg
            ):
                degraded = True
                degrade_reason = (
                    degrade_reason
                    or f"choice confidence {choice_conf:.2f} < {cfg.thresholds.choice_confidence}"
                )
                selected, choice_conf, fallback_used = self._run_tool_fallback(
                    state, tool_list
                )
        else:
            if cfg.fallback.enabled and self._fallback:
                degraded = True
                degrade_reason = degrade_reason or "choice unavailable"
                selected, choice_conf, fallback_used = self._run_tool_fallback(
                    state, tool_list
                )
            else:
                selected, choice_conf = "none", 0.0

        use_primary_noul = bool(batch and batch.noul and not hard_fail)
        if use_primary_noul:
            noul_p = batch.noul.noul  # type: ignore[union-attr]
            noul_conf = batch.noul.confidence  # type: ignore[union-attr]
            if self._should_degrade_confidence(
                noul_conf, cfg.thresholds.noul_confidence, cfg
            ):
                degraded = True
                degrade_reason = (
                    degrade_reason
                    or f"noul confidence {noul_conf:.2f} < {cfg.thresholds.noul_confidence}"
                )
                sufficient, noul_conf, src = self._run_noul_fallback(state)
                fallback_used = fallback_used or src
            else:
                sufficient = noul_p >= cfg.thresholds.noul_true_threshold
        else:
            if cfg.fallback.enabled and self._fallback:
                degraded = True
                degrade_reason = degrade_reason or "noul unavailable"
                sufficient, noul_conf, src = self._run_noul_fallback(state)
                fallback_used = fallback_used or src
            else:
                sufficient = False

        if need_score:
            if batch and batch.score and not hard_fail:
                cred = batch.score.score
                score_conf = batch.score.confidence
            elif cfg.fallback.enabled and self._fallback:
                degraded = True
                degrade_reason = degrade_reason or "score unavailable"
                cred, score_conf, src = self._run_score_fallback(state)
                fallback_used = fallback_used or src

        candidates = self._topk(probs, selected, choice_conf, cfg.thresholds.top_k)

        action, reason, overall_conf = self._decide_action(
            selected=selected,
            sufficient=sufficient,
            cred=cred,
            choice_conf=choice_conf,
            noul_conf=noul_conf,
            score_conf=score_conf,
            cfg=cfg,
            degraded=degraded,
            degrade_reason=degrade_reason,
            has_tool_result=need_score,
        )

        if action == DecisionAction.CALL_TOOL and selected and selected != "none":
            self.circuit.record(request.session_id, selected)

        decision = RouteDecision(
            action=action,
            selected_tool=selected if action in (
                DecisionAction.CALL_TOOL,
                DecisionAction.RETRY_TOOL,
                DecisionAction.ANSWER,
            ) else (selected if selected != "none" else None),
            candidates=candidates,
            information_sufficient=sufficient,
            credibility_score=cred,
            confidence=overall_conf,
            reason=reason,
            degraded=degraded,
            degrade_reason=degrade_reason,
            fallback_used=fallback_used,
            terminated=False,
            laya_latency_ms=laya_ms,
            backend=self.backend.name,
            raw=raw,
        )

        if degraded and degrade_reason:
            self.hooks.run_degrade(request, decision, degrade_reason)

        return self._finish(request, decision, t0, cache_key=cache_key)

    def choose_tool(
        self,
        query: str,
        tools: list[ToolSpec | dict] | None = None,
        context: str | None = None,
    ) -> RouteDecision:
        """Only run choice (tool routing)."""
        return self.route(query=query, context=context, tools=tools)

    def is_sufficient(
        self,
        query: str,
        context: str | None = None,
        tool_result: str | None = None,
    ) -> RouteDecision:
        """Only run noul (information sufficiency)."""
        return self.route(query=query, context=context, tool_result=tool_result)

    def score_result(
        self,
        query: str,
        tool_result: str,
        context: str | None = None,
    ) -> RouteDecision:
        """Only run score (credibility)."""
        return self.route(query=query, context=context, tool_result=tool_result)

    def update_config(self, **overrides: Any) -> None:
        """Hot-reload thresholds / flags without restarting the process."""
        self.config = self.config.update(**overrides)
        self.circuit = CircuitBreaker(self.config.circuit_breaker)
        self.registry.max_count = self.config.tools.max_count
        self._decision_cache = DecisionCache(
            max_size=self.config.cache.max_size,
            ttl_s=self.config.cache.ttl_s,
        )
        if "fallback" in overrides:
            fb_cfg = self.config.fallback
            self._fallback = build_fallback(
                mode=fb_cfg.mode,
                enabled=fb_cfg.enabled,
                llm_base_url=fb_cfg.llm.base_url,
                llm_api_key=fb_cfg.llm.api_key,
                llm_model=fb_cfg.llm.model,
                llm_timeout_s=fb_cfg.llm.timeout_s,
            )

    def load_model(self, name: str | None = None) -> None:
        if name:
            self.config.model.name = name
            self.config.model.backend = "laya"
        self.backend = create_backend(
            backend=self.config.model.backend,
            model_name=self.config.model.name,
            device=self.config.model.device,
            timeout_ms=self.config.model.timeout_ms,
        )

    def unload_model(self) -> None:
        self.backend.unload()

    def health(self) -> dict[str, Any]:
        ready = self.backend.is_ready()
        probe_ok = False
        latency = None
        if ready:
            try:
                t0 = time.perf_counter()
                self.backend.predict(
                    "User query: health check",
                    tools=[ToolSpec(name="none", description="direct answer")],
                    need_choice=True,
                    need_noul=False,
                    need_score=False,
                )
                latency = (time.perf_counter() - t0) * 1000
                probe_ok = True
            except Exception as e:
                return {
                    "status": "degraded",
                    "backend": self.backend.name,
                    "model_loaded": ready,
                    "inference_ok": False,
                    "error": str(e),
                    "policy": summarize_policy(self.config),
                }
        return {
            "status": "ok" if ready and probe_ok else "not_ready",
            "backend": self.backend.name,
            "is_mock": self.backend.name == "mock",
            "model_loaded": ready,
            "inference_ok": probe_ok,
            "latency_ms": latency,
            "fallback": None
            if self._fallback is None
            else {
                "enabled": True,
                "primary": self._fallback.name,
            },
            "policy": summarize_policy(self.config),
            "metrics": self.logger.metrics.snapshot(),
        }

    def metrics(self) -> dict[str, Any]:
        return self.logger.metrics.snapshot()

    def metrics_prometheus(self) -> str:
        return self.logger.metrics.prometheus_text()

    def register_tool(self, tool: ToolSpec | dict) -> None:
        self.registry.register(tool)

    def reset_session(self, session_id: str | None = None) -> None:
        self.circuit.reset(session_id)
        self._idempotency.clear(session_id)

    def clear_decision_cache(self) -> None:
        self._decision_cache.clear()

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        """Inspect circuit / history for a session (production debugging)."""
        return self.circuit.snapshot(session_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _finish(
        self,
        request: RouteRequest,
        decision: RouteDecision,
        t0: float,
        *,
        cache_key: str | None = None,
    ) -> RouteDecision:
        decision.latency_ms = (time.perf_counter() - t0) * 1000
        if not decision.decision_id:
            decision.decision_id = uuid.uuid4().hex
        decision.session_id = request.session_id
        decision = self.hooks.run_post(request, decision)
        self.circuit.note_decision(request.session_id, decision.decision_id)
        if (
            request.session_id
            and request.idempotency_key
            and not decision.idempotent_replay
            and not decision.cache_hit
        ):
            self._idempotency.put(
                request.session_id, request.idempotency_key, decision
            )
        if (
            cache_key
            and self.config.cache.enabled
            and not decision.terminated
            and not decision.cache_hit
            and not decision.idempotent_replay
        ):
            self._decision_cache.put(cache_key, decision)
        self.logger.log(
            request.query,
            decision,
            session_id=request.session_id,
            decision_id=decision.decision_id,
        )
        return decision

    def _effective_config(self, overrides: dict[str, Any] | None) -> RouterConfig:
        if not overrides:
            return self.config
        return clone_config(self.config).update(**overrides)

    def _validate_query(self, query: str, max_chars: int) -> None:
        q = sanitize_text(query)
        if not q:
            raise ValidationError("query is empty")
        if len(q) > max_chars * 2:
            raise ValidationError(f"query too long (>{max_chars * 2} chars)")

    def _resolve_tools(self, request: RouteRequest, cfg: RouterConfig) -> list[ToolSpec]:
        if request.tools:
            specs = [
                t if isinstance(t, ToolSpec) else ToolSpec.model_validate(t)
                for t in request.tools
            ]
            if not any(t.name == "none" for t in specs):
                specs = [
                    ToolSpec(
                        name="none",
                        description=(
                            "Do not call a tool. Answer now. Use when chitchat "
                            "or a usable Tool result is already present."
                        ),
                    )
                ] + specs
            bl = set(cfg.tools.blacklist)
            specs = [t for t in specs if t.name not in bl]
            if request.allowed_tools is not None:
                allow = set(request.allowed_tools) | {"none"}
                specs = [t for t in specs if t.name in allow]
            return specs

        return self.registry.filter(
            blacklist=cfg.tools.blacklist,
            whitelist=cfg.tools.whitelist or None,
            allowed=request.allowed_tools,
        )

    def _should_degrade_confidence(
        self, conf: float, threshold: float, cfg: RouterConfig
    ) -> bool:
        if not cfg.fallback.enabled or self._fallback is None:
            return False
        if conf >= threshold:
            return False
        if self.backend.name == "mock" and self._fallback.name == "heuristic":
            return False
        return True

    def _run_tool_fallback(
        self, state: str, tools: list[ToolSpec]
    ) -> tuple[str, float, str]:
        if not self._fallback:
            return "none", 0.0, "none"
        name, conf, _reason = self._fallback.decide_tool(state, tools)
        return name, conf, self._fallback.name

    def _run_noul_fallback(self, state: str) -> tuple[bool, float, str]:
        if not self._fallback:
            return False, 0.0, "none"
        ok, conf, _ = self._fallback.decide_sufficient(state)
        return ok, conf, self._fallback.name

    def _run_score_fallback(self, state: str) -> tuple[float, float, str]:
        if not self._fallback:
            return 0.0, 0.0, "none"
        score, conf, _ = self._fallback.decide_score(state)
        return score, conf, self._fallback.name

    def _fallback_tool(
        self, state: str, tools: list[ToolSpec]
    ) -> tuple[str, float, str]:
        return self._run_tool_fallback(state, tools)

    def _fallback_sufficient(self, state: str) -> tuple[bool, float, str]:
        return self._run_noul_fallback(state)

    def _fallback_score(self, state: str) -> tuple[float, float, str]:
        return self._run_score_fallback(state)

    def _topk(
        self,
        probs: dict[str, float],
        selected: str,
        choice_conf: float,
        k: int,
    ) -> list[ToolCandidate]:
        if not probs:
            return [
                ToolCandidate(
                    name=selected, confidence=choice_conf, probability=choice_conf
                )
            ]
        ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            ToolCandidate(name=n, confidence=p, probability=p) for n, p in ordered
        ]

    def _decide_action(
        self,
        *,
        selected: str,
        sufficient: bool | None,
        cred: float | None,
        choice_conf: float,
        noul_conf: float,
        score_conf: float,
        cfg: RouterConfig,
        degraded: bool,
        degrade_reason: str | None,
        has_tool_result: bool,
    ) -> tuple[DecisionAction, str, float]:
        if degraded and degrade_reason and (
            "error:" in degrade_reason or "timeout" in degrade_reason
        ):
            if not cfg.fallback.enabled:
                return DecisionAction.FALLBACK_LLM, degrade_reason, 0.0

        if has_tool_result and cred is not None:
            if cred < cfg.thresholds.score_low:
                return (
                    DecisionAction.RETRY_TOOL,
                    f"credibility {cred:.1f} < low threshold {cfg.thresholds.score_low}",
                    score_conf or choice_conf,
                )
            if cred >= cfg.thresholds.score_high or sufficient:
                return (
                    DecisionAction.ANSWER,
                    f"credibility {cred:.1f} high / info sufficient ->LLM summarize",
                    max(score_conf, noul_conf, choice_conf),
                )

        if sufficient and (selected == "none" or not has_tool_result):
            return (
                DecisionAction.ANSWER,
                "information sufficient ->LLM summarize",
                max(noul_conf, choice_conf),
            )

        if selected == "none":
            return (
                DecisionAction.ANSWER,
                "choice=none ->answer directly via LLM",
                choice_conf,
            )

        return (
            DecisionAction.CALL_TOOL,
            f"call tool '{selected}'",
            choice_conf,
        )
