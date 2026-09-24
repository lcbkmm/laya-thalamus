"""Fallback when Laya is low-confidence, times out, or errors.

**Unstable** internal chain. See ``API.md``.

Order: configured LLM (OpenAI-compatible) ->heuristic mock (offline last resort).
Heuristic is *not* a substitute for LLM in production -it only keeps CI/demos alive.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from laya_thalamus.backend import MockBackend
from laya_thalamus.llm import (
    LLMClientError,
    OpenAICompatibleClient,
    llm_is_configured,
)
from laya_thalamus.schemas import ToolSpec

logger = logging.getLogger("laya_thalamus")

_TOOL_SYS = (
    "You route an AI agent. Pick exactly one tool name from the provided list. "
    "Choose 'none' only for chitchat/thanks/opinions/translation, or when a usable "
    "tool result is already present. "
    'Reply with JSON only: {"tool": "<name>", "confidence": 0.0-1.0}'
)
_NOUL_SYS = (
    "Decide if the collected information is enough to answer the user correctly "
    "without calling more tools. "
    'Reply with JSON only: {"sufficient": true|false, "confidence": 0.0-1.0}'
)
_SCORE_SYS = (
    "Score how relevant and trustworthy the tool result is for the user query. "
    "0 = useless/wrong, 10 = perfect. "
    'Reply with JSON only: {"score": 0-10, "confidence": 0.0-1.0}'
)


class FallbackBackend(Protocol):
    name: str

    def decide_tool(
        self, state: str, tools: list[ToolSpec]
    ) -> tuple[str, float, str]: ...

    def decide_sufficient(self, state: str) -> tuple[bool, float, str]: ...

    def decide_score(self, state: str) -> tuple[float, float, str]: ...


class HeuristicFallback:
    """Keyword stand-in. Same family as MockBackend -not a real LLM."""

    name = "heuristic"

    def __init__(self) -> None:
        self._mock = MockBackend()

    def decide_tool(self, state: str, tools: list[ToolSpec]) -> tuple[str, float, str]:
        ans = self._mock.predict(state, tools=tools, need_choice=True, need_noul=False)
        assert ans.choice is not None
        return (
            ans.choice.choice,
            ans.choice.confidence,
            "heuristic fallback for tool choice",
        )

    def decide_sufficient(self, state: str) -> tuple[bool, float, str]:
        ans = self._mock.predict(state, need_choice=False, need_noul=True)
        assert ans.noul is not None
        sufficient = ans.noul.noul >= 0.5
        return sufficient, ans.noul.confidence, "heuristic fallback for noul"

    def decide_score(self, state: str) -> tuple[float, float, str]:
        ans = self._mock.predict(
            state, need_choice=False, need_noul=False, need_score=True
        )
        assert ans.score is not None
        return ans.score.score, ans.score.confidence, "heuristic fallback for score"


class LLMFallback:
    """Hand routing to a real chat model (OpenAI-compatible)."""

    name = "llm"

    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client or OpenAICompatibleClient()

    def decide_tool(self, state: str, tools: list[ToolSpec]) -> tuple[str, float, str]:
        names = [t.name for t in tools]
        listing = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        user = f"{state}\n\nAvailable tools:\n{listing}\n\nPick one tool name."
        data = self.client.chat_json(_TOOL_SYS, user)
        tool = str(data.get("tool") or data.get("choice") or "none")
        if tool not in names:
            # fuzzy: case-insensitive exact, else none
            lowered = {n.lower(): n for n in names}
            tool = lowered.get(tool.lower(), names[0] if names else "none")
        conf = _clip01(data.get("confidence", 0.7))
        return tool, conf, "llm fallback for tool choice"

    def decide_sufficient(self, state: str) -> tuple[bool, float, str]:
        data = self.client.chat_json(_NOUL_SYS, state)
        raw = data.get("sufficient", data.get("noul", False))
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            sufficient = float(raw) >= 0.5
        else:
            sufficient = bool(raw)
        conf = _clip01(data.get("confidence", 0.7))
        return sufficient, conf, "llm fallback for noul"

    def decide_score(self, state: str) -> tuple[float, float, str]:
        data = self.client.chat_json(_SCORE_SYS, state)
        score = float(data.get("score", 5.0))
        score = max(0.0, min(10.0, score))
        conf = _clip01(data.get("confidence", 0.7))
        return score, conf, "llm fallback for score"


class CallableToolFallback:
    """Adapter for a user-supplied ``(state, tools) -> (name, conf, reason)``."""

    name = "callable"

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self._h = HeuristicFallback()

    def decide_tool(self, state: str, tools: list[ToolSpec]) -> tuple[str, float, str]:
        out = self.fn(state, tools)
        if len(out) == 2:
            return out[0], float(out[1]), "callable fallback"
        return out[0], float(out[1]), str(out[2])

    def decide_sufficient(self, state: str) -> tuple[bool, float, str]:
        return self._h.decide_sufficient(state)

    def decide_score(self, state: str) -> tuple[float, float, str]:
        return self._h.decide_score(state)


class FallbackChain:
    """Try primary (usually LLM); on failure use secondary (heuristic)."""

    def __init__(
        self,
        primary: FallbackBackend | None,
        secondary: FallbackBackend | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary or HeuristicFallback()

    @property
    def name(self) -> str:
        if self.primary:
            return self.primary.name
        return self.secondary.name

    def decide_tool(self, state: str, tools: list[ToolSpec]) -> tuple[str, float, str]:
        return self._call("decide_tool", state, tools)

    def decide_sufficient(self, state: str) -> tuple[bool, float, str]:
        return self._call("decide_sufficient", state)

    def decide_score(self, state: str) -> tuple[float, float, str]:
        return self._call("decide_score", state)

    def _call(self, method: str, *args: Any) -> Any:
        if self.primary is not None:
            try:
                return getattr(self.primary, method)(*args)
            except Exception as e:
                logger.warning("primary fallback (%s) failed: %s", self.primary.name, e)
                if isinstance(e, LLMClientError):
                    pass
        return getattr(self.secondary, method)(*args)


def build_fallback(
    *,
    mode: str = "auto",
    enabled: bool = True,
    llm_client: OpenAICompatibleClient | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_timeout_s: float = 20.0,
    injected: FallbackBackend | None = None,
) -> FallbackChain | None:
    """Construct the production fallback chain from config.

    Returns None when fallback is disabled.
    """
    if not enabled or mode in ("none", "off", "disabled"):
        return None
    if injected is not None:
        return FallbackChain(injected, HeuristicFallback())

    want_llm = mode in ("auto", "llm")
    configured = llm_is_configured(llm_base_url, llm_api_key)
    primary: FallbackBackend | None = None
    if want_llm and (configured or mode == "llm" or llm_client is not None):
        client = llm_client or OpenAICompatibleClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            timeout_s=llm_timeout_s,
        )
        primary = LLMFallback(client)
        if mode == "llm" and not configured and llm_client is None:
            logger.warning(
                "fallback.mode=llm but no LAYA_LLM_BASE_URL / OPENAI_API_KEY; "
                "calls will fail until configured"
            )
    elif mode == "llm":
        logger.warning("fallback.mode=llm but LLM is not configured; using heuristic")

    if mode == "heuristic":
        primary = HeuristicFallback()

    if primary is None:
        primary = HeuristicFallback()

    secondary = HeuristicFallback() if primary.name != "heuristic" else HeuristicFallback()
    return FallbackChain(primary, secondary)


def looks_like_math(query: str) -> bool:
    return bool(re.search(r"[\d]+\s*[\+\-\*/×÷]\s*[\d]+", query))


def _clip01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, v))
