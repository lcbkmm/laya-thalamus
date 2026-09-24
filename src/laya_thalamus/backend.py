"""Decision backends: Laya model + heuristic mock for offline demos.

**Unstable** -prefer ``AgentRouter`` / ``RouterConfig``. See ``API.md``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from laya_thalamus.schemas import ToolSpec

logger = logging.getLogger("laya_thalamus")

# Bundled checkpoint folder names under a local hub root (convaiinnovations/laya layout).
_BUNDLE_SUBFOLDERS = ("multilingual", "typed-decisions")
_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.pt")


@dataclass
class ChoiceResult:
    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class NoulResult:
    noul: float  # P(true)
    confidence: float = 0.0


@dataclass
class ScoreResult:
    score: float  # expected level on rubric (we map to 0-0)
    confidence: float = 0.0
    distribution: list[float] = field(default_factory=list)


@dataclass
class BatchAnswers:
    choice: ChoiceResult | None = None
    noul: NoulResult | None = None
    score: ScoreResult | None = None
    latency_ms: float = 0.0
    backend: str = "mock"
    raw: dict[str, Any] = field(default_factory=dict)


class DecisionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def predict(
        self,
        state: str,
        *,
        tools: list[ToolSpec] | None = None,
        need_choice: bool = True,
        need_noul: bool = True,
        need_score: bool = False,
        score_max: int = 10,
    ) -> BatchAnswers: ...


# ---------------------------------------------------------------------------
# Heuristic mock (no GPU / no laya install) -good enough for demos & tests
# ---------------------------------------------------------------------------

_SEARCH_PAT = re.compile(
    r"(搜索|查一下|检索|最新|新闻|天气|谁是|什么是|how to|what is|who is|search|latest|news|today|internet)",
    re.I,
)
_CALC_PAT = re.compile(
    r"(计算|算一下|等于多少|[\d]+\s*[\+\-\*/×÷]\s*[\d]+|calculate|compute|\d+\s*%|平方|根号)",
    re.I,
)
_CODE_PAT = re.compile(
    r"(代码|python|脚本|写个函数|debug|编程|code|interpreter|pandas|排序算法|repl)",
    re.I,
)
_RAG_PAT = re.compile(
    r"(文档|知识库|手册|政策|根据资料|内部资料|内部文档|rag|retrieve|说明书|报销)",
    re.I,
)
_TRANSLATE_PAT = re.compile(r"(翻译|translate|译成)", re.I)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    exps = {k: math.exp(v - mx) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def _tool_intent_boosts(state: str) -> dict[str, float]:
    """Intent weights independent of tool naming (enables custom tool names)."""
    boosts: dict[str, float] = {}
    # Prefer search over calc when query is clearly informational
    search_hit = bool(_SEARCH_PAT.search(state))
    calc_hit = bool(_CALC_PAT.search(state))
    code_hit = bool(_CODE_PAT.search(state))
    rag_hit = bool(_RAG_PAT.search(state))
    # "量子计算" should not trigger calculator
    fake_calc = bool(re.search(r"(量子计算|云计算|边缘计算|计算科学|计算思维)", state))
    if calc_hit and not fake_calc and not (search_hit and not re.search(r"[\d\+\-\*/×÷]", state)):
        boosts["calc"] = 2.8
    if search_hit and not rag_hit:
        boosts["search"] = 2.5
    if search_hit and rag_hit:
        # both ->prefer rag for internal docs
        boosts["rag"] = 2.6
        boosts["search"] = 1.2
    elif rag_hit:
        boosts["rag"] = 2.4
    if code_hit:
        boosts["code"] = 2.6
    if _TRANSLATE_PAT.search(state):
        boosts["translate"] = 2.5
    return boosts


def _match_intent(tool_name: str, description: str, intent: str) -> bool:
    blob = f"{tool_name} {description}".lower()
    mapping = {
        "calc": ("calc", "math", "算术", "计算", "arithmetic"),
        "search": ("search", "web", "搜索", "检索", "internet", "bing", "google"),
        "code": ("code", "python", "repl", "interpreter", "代码"),
        "rag": ("rag", "retrieve", "知识", "kb", "document", "文档"),
        "translate": ("translat", "翻译"),
        "none": ("none", "direct", "answer", "直接"),
    }
    return any(k in blob for k in mapping.get(intent, ()))


class MockBackend(DecisionBackend):
    """Keyword / heuristic stand-in for Laya -enables zero-dependency demos."""

    name = "mock"

    def __init__(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self._ready = False

    def predict(
        self,
        state: str,
        *,
        tools: list[ToolSpec] | None = None,
        need_choice: bool = True,
        need_noul: bool = True,
        need_score: bool = False,
        score_max: int = 10,
    ) -> BatchAnswers:
        t0 = time.perf_counter()
        tools = tools or []

        choice_res = None
        if need_choice:
            choice_res = self._choose(state, tools)

        noul_res = None
        if need_noul:
            noul_res = self._noul(state)

        score_res = None
        if need_score:
            score_res = self._score(state, score_max)

        return BatchAnswers(
            choice=choice_res,
            noul=noul_res,
            score=score_res,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend=self.name,
        )

    def _choose(self, state: str, tools: list[ToolSpec]) -> ChoiceResult:
        if not tools:
            return ChoiceResult(choice="none", probabilities={"none": 1.0}, confidence=1.0)

        scores = {t.name: 0.1 for t in tools}
        for t in tools:
            if t.name == "none" or _match_intent(t.name, t.description, "none"):
                scores[t.name] = max(scores[t.name], 0.4)

        intent_boosts = _tool_intent_boosts(state)
        for t in tools:
            for intent, boost in intent_boosts.items():
                if _match_intent(t.name, t.description, intent):
                    scores[t.name] = max(scores[t.name], boost)

        # Exact built-in names (backward compatible)
        if "calculator" in scores and "calc" in intent_boosts:
            scores["calculator"] = max(scores["calculator"], intent_boosts["calc"])
        if "web_search" in scores and "search" in intent_boosts:
            scores["web_search"] = max(scores["web_search"], intent_boosts["search"])
        if "code_interpreter" in scores and "code" in intent_boosts:
            scores["code_interpreter"] = max(scores["code_interpreter"], intent_boosts["code"])
        if "rag_retrieve" in scores and "rag" in intent_boosts:
            scores["rag_retrieve"] = max(scores["rag_retrieve"], intent_boosts["rag"])

        # Any tool result ->prefer finishing (none) unless result looks like an error
        if "Tool result:" in state:
            result_tail = state.split("Tool result:")[-1]
            failed = bool(
                re.search(r"(?i)FAILED|\berror\b|失败|exception|quota", state)
            )
            if failed:
                pass
            else:
                for t in tools:
                    if t.name == "none" or _match_intent(t.name, t.description, "none"):
                        scores[t.name] = max(scores[t.name], 2.8)

        probs = _softmax(scores)
        best = max(probs, key=probs.get)
        conf = probs[best]
        return ChoiceResult(choice=best, probabilities=probs, confidence=conf)

    def _noul(self, state: str) -> NoulResult:
        result = ""
        if "Tool result:" in state:
            result = state.split("Tool result:")[-1].strip()
        has_result = bool(result)
        query_line = state.split("\n")[0].replace("User query: ", "")
        trivial = bool(
            re.search(
                r"^(你好|hello|hi|谢谢|thanks)[\s!->。]*$",
                query_line,
                re.I,
            )
        )
        needs_tool = bool(
            _SEARCH_PAT.search(state)
            or _CALC_PAT.search(state)
            or _CODE_PAT.search(state)
            or _RAG_PAT.search(state)
        )
        if has_result:
            if re.search(r"失败|error|exception", result, re.I):
                p = 0.2
            elif re.search(r"结果|result|\d|```|命中|摘要", result, re.I):
                p = 0.9
            else:
                p = 0.72
        elif trivial:
            p = 0.75
        elif needs_tool:
            p = 0.18
        else:
            p = 0.62
        return NoulResult(noul=p, confidence=0.75)

    def _score(self, state: str, score_max: int) -> ScoreResult:
        result = ""
        if "Tool result:" in state:
            result = state.split("Tool result:")[-1]
        query = state.split("\n")[0]
        if not result.strip():
            return ScoreResult(score=2.0, confidence=0.6, distribution=[])
        if re.search(r"失败|error|exception", result, re.I):
            return ScoreResult(score=2.0, confidence=0.7, distribution=[])

        # Strong signals for common mock / real tool outputs
        if _CALC_PAT.search(query) and re.search(r"\d", result):
            raw = 8.5
        elif _CODE_PAT.search(query) and ("```" in result or "def " in result):
            raw = 8.0
        elif (_SEARCH_PAT.search(query) or _RAG_PAT.search(query)) and len(result) > 20:
            raw = 7.5
        else:
            q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
            r_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", result.lower()))
            overlap = len(q_tokens & r_tokens) / max(len(q_tokens), 1)
            raw = 4.0 + overlap * 6.0

        score = max(0.0, min(float(score_max), raw))
        return ScoreResult(score=score, confidence=0.7, distribution=[])

# ---------------------------------------------------------------------------
# Real Laya backend
# ---------------------------------------------------------------------------


class LayaBackend(DecisionBackend):
    """Wraps the official `laya` package (Router or load).

    ``model_name`` may be:
    - a HuggingFace repo id (e.g. ``convaiinnovations/laya-multilingual``)
    - a local hub root that contains English weights and/or ``multilingual`` /
      ``typed-decisions`` subfolders (offline; no Hub download)
    - a local path to a single checkpoint directory
    """

    name = "laya"

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya-multilingual",
        device: str = "auto",
        timeout_ms: int = 2000,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.timeout_ms = timeout_ms
        self._agent: Any = None
        self._use_router = False
        self._local_root: Path | None = None

    def is_ready(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        try:
            import laya  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Package 'laya' is not installed. Run: pip install laya"
            ) from e

        local = resolve_local_model_path(self.model_name)
        if local is not None:
            self._load_local(laya, local)
            return

        # Remote / Hub id -prefer Router for multilingual auto-routing
        if hasattr(laya, "Router"):
            self._agent = _call_with_device(laya.Router, self.device)
            self._use_router = True
            logger.info("Loaded laya.Router() (hub id=%s)", self.model_name)
        else:
            self._agent = _laya_load(laya, self.model_name, device=self.device)
            self._use_router = False
            logger.info("Loaded laya model: %s", self.model_name)

    def _load_local(self, laya: Any, path: Path) -> None:
        """Load weights from a local directory (single checkpoint or hub bundle)."""
        kind = classify_local_checkpoint(path)

        if kind == "hub":
            self._load_local_hub(laya, path)
            return

        # Single checkpoint directory (root English, multilingual, or typed-decisions)
        if kind == "hub-child":
            # Pointed at .../LAYA/multilingual -load via parent + subfolder when possible
            parent = path.parent
            sub = path.name
            if classify_local_checkpoint(parent) == "hub":
                self._agent = _laya_load(
                    laya, str(parent), subfolder=sub, device=self.device
                )
            else:
                self._agent = _laya_load(laya, str(path), device=self.device)
        else:
            self._agent = _laya_load(laya, str(path), device=self.device)

        self._use_router = False
        self._local_root = path
        logger.info("Loaded local Laya checkpoint (%s): %s", kind, path)

    def _load_local_hub(self, laya: Any, root: Path) -> None:
        """Attach available local checkpoints to Router so Hub is never contacted."""
        self._local_root = root
        agents: dict[str, Any] = {}

        if _has_weights(root):
            agents["english"] = _laya_load(laya, str(root), device=self.device)

        for sub in _BUNDLE_SUBFOLDERS:
            sub_path = root / sub
            if sub_path.is_dir() and _has_weights(sub_path):
                agents[sub] = _laya_load(
                    laya, str(root), subfolder=sub, device=self.device
                )

        if not agents:
            raise RuntimeError(
                f"Local Laya path has no weights under {root}. "
                f"Expected model.safetensors (and optional multilingual/ typed-decisions/)."
            )

        # One checkpoint only ->skip Router
        if len(agents) == 1 or not hasattr(laya, "Router"):
            name, agent = next(iter(agents.items()))
            self._agent = agent
            self._use_router = False
            logger.info("Loaded local Laya single checkpoint (%s): %s", name, root)
            return

        router = _call_with_device(laya.Router, self.device, max_loaded=len(agents))
        for name, agent in agents.items():
            if hasattr(router, "attach"):
                router.attach(name, agent)
            else:
                break
        else:
            self._agent = router
            self._use_router = True
            logger.info(
                "Loaded local laya.Router with checkpoints %s from %s",
                list(agents),
                root,
            )
            return

        # attach unavailable -fall back to preferred single agent
        preferred = (
            agents.get("multilingual")
            or agents.get("typed-decisions")
            or agents.get("english")
        )
        self._agent = preferred
        self._use_router = False
        logger.info("Loaded local Laya checkpoint (no attach API): %s", root)

    def unload(self) -> None:
        self._agent = None
        self._local_root = None

    def predict(
        self,
        state: str,
        *,
        tools: list[ToolSpec] | None = None,
        need_choice: bool = True,
        need_noul: bool = True,
        need_score: bool = False,
        score_max: int = 10,
    ) -> BatchAnswers:
        if self._agent is None:
            raise RuntimeError("Laya model not loaded")

        tools = tools or []
        questions: dict[str, Any] = {}
        has_result = "Tool result" in state
        result_failed = bool(
            re.search(r"(?i)status:\s*FAILED|\berror\b|失败|exception|quota exceeded", state)
        )

        if need_choice and tools:
            criteria = {t.name: t.description for t in tools}
            # Strengthen none when evidence already exists
            if has_result and not result_failed and "none" in criteria:
                criteria["none"] = (
                    "STOP calling tools. A usable Tool result is already in the state. "
                    "Choose this to hand off to the LLM for the final answer."
                )
            if has_result and result_failed:
                # Prefer retrying a real tool, not none
                if "none" in criteria:
                    criteria["none"] = (
                        "Only if no tool can help. Prefer retrying a tool when "
                        "the previous Tool result failed with an error."
                    )
            if has_result and not result_failed:
                instructions = (
                    "A Tool result is ALREADY present in the state. "
                    "If it answers or substantially helps the user query, choose 'none'. "
                    "Only pick another tool if the result is missing, wrong, or an error."
                )
            else:
                instructions = (
                    "Pick the single next tool for an AI agent. "
                    "Use 'none' for greetings/thanks/simple chat/translation with no tool needed. "
                    "Use 'calculator' only for numeric arithmetic. "
                    "Use 'code_interpreter' to write/run Python or algorithms. "
                    "Use 'web_search' for public facts, news, weather, definitions (什么是/what is). "
                    "Use 'rag_retrieve' only for internal/company knowledge-base documents. "
                    "Never confuse code tasks with rag_retrieve."
                )
            questions["tool"] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }

        if need_noul:
            if has_result and not result_failed:
                noul_inst = (
                    "A Tool result is present and not marked FAILED. "
                    "Is it enough to answer the user query without more tools? "
                    "Return true if the result contains the needed number, facts, code, or passages."
                )
            elif has_result and result_failed:
                noul_inst = (
                    "The Tool result FAILED. Information is NOT sufficient; prefer false."
                )
            else:
                noul_inst = (
                    "No tool result yet. Is the user query answerable from the query alone "
                    "(greetings, thanks, simple translation)? If it needs calc/search/code/docs, false."
                )
            questions["sufficient"] = {
                "type": "noul",
                "instructions": noul_inst,
            }

        if need_score:
            levels = [
                "error/failure or completely irrelevant",
                "mostly irrelevant or broken output",
                "weakly related",
                "partially useful but major gaps",
                "somewhat relevant",
                "moderately relevant",
                "relevant with minor gaps",
                "mostly correct and useful for the query",
                "highly relevant and reliable",
                "excellent match to the query",
                "perfectly answers the user query",
            ]
            criteria = levels[: score_max + 1]
            while len(criteria) < score_max + 1:
                criteria.append(f"level {len(criteria)}")
            questions["credibility"] = {
                "type": "score",
                "instructions": (
                    "Score how well the Tool result answers the User query. "
                    "Errors/quota failures ->0-2. "
                    "Correct numeric result for a calc question ->8-10. "
                    "Useful search/RAG/code that addresses the ask ->7-10. "
                    "Off-topic content ->below 4."
                ),
                "criteria": criteria,
            }

        t0 = time.perf_counter()
        result = self._invoke(state, questions)
        latency = (time.perf_counter() - t0) * 1000
        if self.timeout_ms and latency > self.timeout_ms:
            raise TimeoutError(
                f"Laya inference {latency:.0f}ms exceeded timeout {self.timeout_ms}ms"
            )

        answers = _extract_answers(result)
        raw = result if isinstance(result, dict) else {"answers": answers}

        choice_res = None
        if need_choice:
            a = _find_answer(answers, "tool")
            if a:
                choice_res = _parse_choice(a, fallback="none")

        noul_res = None
        if need_noul:
            a = _find_answer(answers, "sufficient")
            if a:
                noul_res = _parse_noul(a)

        score_res = None
        if need_score:
            a = _find_answer(answers, "credibility")
            if a:
                score_res = _parse_score(a)

        return BatchAnswers(
            choice=choice_res,
            noul=noul_res,
            score=score_res,
            latency_ms=latency,
            backend=self.name,
            raw=raw,
        )

    def _invoke(self, state: str, questions: dict[str, Any]) -> Any:
        if self._use_router:
            model_kw = _checkpoint_alias(self.model_name, self._local_root)
            try:
                if model_kw:
                    return self._agent.predict(state, questions, model=model_kw)
                return self._agent.predict(state, questions)
            except TypeError:
                return self._agent.predict(state, questions)
        return self._agent.predict(state, questions)


class LLMRouterBackend(DecisionBackend):
    """Use a chat LLM as the *primary* router -for latency/accuracy comparison."""

    name = "llm"

    def __init__(self, client: Any | None = None) -> None:
        from laya_thalamus.fallback import LLMFallback
        from laya_thalamus.llm import OpenAICompatibleClient

        self._client = client or OpenAICompatibleClient()
        self._fb = LLMFallback(self._client)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._ready = True

    def unload(self) -> None:
        self._ready = False

    def predict(
        self,
        state: str,
        *,
        tools: list[ToolSpec] | None = None,
        need_choice: bool = True,
        need_noul: bool = True,
        need_score: bool = False,
        score_max: int = 10,
    ) -> BatchAnswers:
        t0 = time.perf_counter()
        tools = tools or []
        choice_res = None
        noul_res = None
        score_res = None
        errors: list[str] = []
        if need_choice:
            try:
                name, conf, _ = self._fb.decide_tool(state, tools)
                choice_res = ChoiceResult(
                    choice=name,
                    probabilities={name: conf},
                    confidence=conf,
                )
            except Exception as e:
                errors.append(f"choice:{e}")
                logger.warning("LLMRouterBackend choice failed: %s", e)
        if need_noul:
            try:
                ok, conf, _ = self._fb.decide_sufficient(state)
                noul_res = NoulResult(noul=1.0 if ok else 0.0, confidence=conf)
            except Exception as e:
                errors.append(f"noul:{e}")
                logger.warning("LLMRouterBackend noul failed: %s", e)
        if need_score:
            try:
                score, conf, _ = self._fb.decide_score(state)
                score_res = ScoreResult(score=score, confidence=conf)
            except Exception as e:
                errors.append(f"score:{e}")
                logger.warning("LLMRouterBackend score failed: %s", e)
        if need_choice and choice_res is None:
            # Soft default -do not raise (would collapse eval to all-none under
            # fallback-disabled compare runs).
            choice_res = ChoiceResult(
                choice="none", probabilities={"none": 0.0}, confidence=0.0
            )
        return BatchAnswers(
            choice=choice_res,
            noul=noul_res,
            score=score_res,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend=self.name,
            raw={"errors": errors} if errors else {},
        )


def _has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_NAMES)


def resolve_local_model_path(model_name: str) -> Path | None:
    """Return a resolved directory if ``model_name`` is an existing local path.

    HuggingFace repo ids like ``org/name`` are left alone (they do not exist
    as filesystem paths in normal setups).
    """
    if not model_name or not str(model_name).strip():
        return None
    raw = str(model_name).strip().strip("\"'")
    path = Path(raw).expanduser()
    try:
        if path.exists() and path.is_dir():
            return path.resolve()
    except OSError:
        return None
    return None


def classify_local_checkpoint(path: Path) -> str:
    """Classify a local Laya directory.

    Returns:
        ``hub`` -bundle root (english ± multilingual / typed-decisions)
        ``hub-child`` -a known subfolder (``multilingual`` / ``typed-decisions``)
        ``single`` -standalone checkpoint directory
    """
    name = path.name.lower().replace("_", "-")
    if name in _BUNDLE_SUBFOLDERS and _has_weights(path):
        return "hub-child"

    has_bundle_child = any(
        (path / sub).is_dir() and _has_weights(path / sub) for sub in _BUNDLE_SUBFOLDERS
    )
    if has_bundle_child or (
        _has_weights(path) and any((path / sub).is_dir() for sub in _BUNDLE_SUBFOLDERS)
    ):
        return "hub"

    return "single"


def _device_kwargs(device: str) -> dict[str, Any]:
    if not device or device == "auto":
        return {}
    return {"device": device}


def _call_with_device(factory: Any, device: str, **extra: Any) -> Any:
    kwargs = {**extra, **_device_kwargs(device)}
    try:
        return factory(**kwargs) if kwargs else factory()
    except TypeError:
        # Older laya builds may not accept device=/max_loaded=
        try:
            return factory(**extra) if extra else factory()
        except TypeError:
            return factory()


def _laya_load(
    laya: Any,
    repo_or_path: str,
    *,
    subfolder: str | None = None,
    device: str = "auto",
) -> Any:
    kwargs: dict[str, Any] = {}
    if subfolder:
        kwargs["subfolder"] = subfolder
    kwargs.update(_device_kwargs(device))
    try:
        return laya.load(repo_or_path, **kwargs) if kwargs else laya.load(repo_or_path)
    except TypeError:
        # Retry without device; keep subfolder if supported
        if subfolder:
            try:
                return laya.load(repo_or_path, subfolder=subfolder)
            except TypeError:
                # Last resort: load the subfolder path directly
                sub_path = Path(repo_or_path) / subfolder
                if sub_path.is_dir():
                    return laya.load(str(sub_path))
                raise
        return laya.load(repo_or_path)


def _checkpoint_alias(
    model_name: str,
    local_root: Path | None = None,
) -> str | None:
    """Map config model name / local path to a Router checkpoint key."""
    if local_root is not None:
        kind = classify_local_checkpoint(local_root)
        if kind == "hub-child":
            name = local_root.name.lower().replace("_", "-")
            if name in _BUNDLE_SUBFOLDERS:
                return name
            return None
        if kind == "hub":
            # Auto language routing across attached local checkpoints
            return None
        # single english-style root
        return "english" if _has_weights(local_root) else None

    if not model_name:
        return None
    lowered = model_name.replace("\\", "/").lower()
    if "typed" in lowered:
        return "typed-decisions"
    if "multilingual" in lowered:
        return "multilingual"
    if lowered.endswith("/laya") or lowered.rstrip("/").endswith("/laya") or model_name == "laya":
        return "english"
    # bare "english"
    if lowered.rstrip("/").endswith("english") or model_name == "english":
        return "english"
    return None


def _as_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())  # type: ignore[arg-type]
        except Exception:
            pass
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not str(k).startswith("_")}
    return {}


def _extract_answers(result: Any) -> dict[str, Any]:
    mapping = _as_mapping(result)
    if "answers" in mapping:
        return _as_mapping(mapping["answers"])
    return mapping


def _find_answer(answers: dict[str, Any], key: str) -> dict[str, Any] | None:
    if key in answers:
        return _as_mapping(answers[key])
    return None


def _parse_choice(a: dict[str, Any], fallback: str = "none") -> ChoiceResult:
    choice = a.get("choice") or a.get("selected") or a.get("label") or fallback
    probs = a.get("probabilities") or a.get("probs") or a.get("distribution") or {}
    if not isinstance(probs, dict):
        probs = {}
    conf = a.get("confidence")
    if conf is None and probs:
        conf = max(probs.values())
    return ChoiceResult(
        choice=str(choice),
        probabilities={str(k): float(v) for k, v in probs.items()},
        confidence=float(conf or 0.0),
    )


def _parse_noul(a: dict[str, Any]) -> NoulResult:
    p = a.get("noul", a.get("probability", a.get("p_true")))
    if p is None and "true" in a:
        p = a["true"]
    p = float(p if p is not None else 0.0)
    conf = a.get("confidence")
    if conf is None:
        conf = abs(2 * p - 1)
    return NoulResult(noul=p, confidence=float(conf))


def _parse_score(a: dict[str, Any]) -> ScoreResult:
    dist = a.get("distribution") or []
    if not isinstance(dist, list):
        dist = []
    return ScoreResult(
        score=float(a.get("score", a.get("expected", 0.0)) or 0.0),
        confidence=float(a.get("confidence", 0.0) or 0.0),
        distribution=[float(x) for x in dist],
    )


def laya_importable() -> bool:
    try:
        import laya  # noqa: F401

        return True
    except ImportError:
        return False


def create_backend(
    backend: str = "auto",
    model_name: str = "convaiinnovations/laya-multilingual",
    device: str = "auto",
    timeout_ms: int = 2000,
    llm_client: Any | None = None,
) -> DecisionBackend:
    """Factory: auto ->laya if importable else mock. ``llm`` is the comparison backend."""
    want = backend.lower()
    if want == "mock":
        logger.warning(
            "Using MockBackend (keyword heuristic). Not production -install laya "
            "and set model.backend=laya for System-1 decisions."
        )
        return MockBackend()

    if want == "llm":
        return LLMRouterBackend(client=llm_client)

    if want in ("laya", "auto"):
        if not laya_importable():
            if want == "laya":
                raise RuntimeError(
                    "backend=laya requested but package 'laya' is not installed. "
                    "Run: pip install laya"
                )
            logger.warning(
                "laya not installed -using MockBackend. "
                "This is a demo stand-in, not calibrated System-1 probabilities."
            )
            return MockBackend()

        be = LayaBackend(model_name=model_name, device=device, timeout_ms=timeout_ms)
        try:
            be.load()
            return be
        except Exception as e:
            if want == "laya":
                raise
            logger.warning("Failed to load Laya (%s) -using MockBackend", e)
            return MockBackend()

    raise ValueError(f"Unknown backend: {backend}")
