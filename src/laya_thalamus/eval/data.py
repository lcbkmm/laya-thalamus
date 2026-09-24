"""Locate / load packaged gold traces (bilingual).

Files:
- ``traces.zh.json`` — Chinese gold set (n=100)
- ``traces.en.json`` — English gold set (n=100)

Each item is **JEV-like** (see ``eval/temple.txt``):

- ``state`` / ``questions`` (choice · noul · score) / ``answers``
- Flat ``query`` / ``expected_*`` for ``evaluate_items``
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Literal

Lang = Literal["zh", "en"]
DEFAULT_LANG: Lang = "zh"


def _traces_ref(lang: Lang = DEFAULT_LANG):
    name = f"traces.{lang}.json"
    return resources.files("laya_thalamus").joinpath("resources", name)


def default_traces_path(lang: Lang = DEFAULT_LANG) -> Path:
    """Filesystem path to packaged traces when available on disk."""
    ref = _traces_ref(lang)
    with resources.as_file(ref) as path:
        resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"packaged {ref.name} not found; reinstall laya-thalamus "
            "or pass --dataset explicitly"
        )
    return resolved


def _query_from_state(state: str) -> str:
    for prefix in ("User query:", "User:", "Query:"):
        if prefix in state:
            rest = state.split(prefix, 1)[1].strip()
            return rest.split("\n", 1)[0].strip()
    return state.strip().split("\n", 1)[0].strip()


def normalize_trace(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw JSON object into the eval flat schema."""
    row = dict(item)

    answers = row.get("answers") or row.get("expected") or {}
    if isinstance(answers, dict):
        tool = answers.get("tool")
        if "expected_tool" not in row and tool is not None:
            if isinstance(tool, dict):
                row["expected_tool"] = tool.get("choice") or tool.get("tool")
            else:
                row["expected_tool"] = tool

        suff = answers.get("sufficient")
        if "expected_sufficient" not in row and suff is not None:
            if isinstance(suff, dict):
                if "label" in suff:
                    row["expected_sufficient"] = bool(suff["label"])
                elif "noul" in suff:
                    row["expected_sufficient"] = float(suff["noul"]) >= 0.5
                else:
                    row["expected_sufficient"] = bool(suff.get("true", False))
            else:
                row["expected_sufficient"] = bool(suff)

        cred = answers.get("credibility") or answers.get("score")
        if isinstance(cred, dict):
            if "expected_score_min" not in row and "score_min" in cred:
                row["expected_score_min"] = cred["score_min"]
            if "expected_score_max" not in row and "score_max" in cred:
                row["expected_score_max"] = cred["score_max"]
            if (
                "expected_score_min" not in row
                and "score" in cred
                and isinstance(cred["score"], (int, float))
            ):
                s = float(cred["score"])
                row["expected_score_min"] = max(0.0, s - 1.0)
                row.setdefault("expected_score_max", min(10.0, s + 1.0))

        action = answers.get("action")
        if "expected_action" not in row and action is not None:
            row["expected_action"] = (
                action.get("choice") if isinstance(action, dict) else action
            )

    if "query" not in row or not row["query"]:
        if row.get("state"):
            row["query"] = _query_from_state(str(row["state"]))
        else:
            raise ValueError(f"trace {row.get('id')!r} missing query/state")

    if "expected_tool" not in row:
        raise ValueError(f"trace {row.get('id')!r} missing expected_tool/answers.tool")

    if "expected_action" not in row:
        if row["expected_tool"] in (None, "none"):
            row["expected_action"] = "answer"
        else:
            row["expected_action"] = "call_tool"

    return row


def resolve_lang(lang: str | None = None) -> Lang:
    raw = (lang or os.environ.get("LAYA_TRACES_LANG") or DEFAULT_LANG).lower()
    if raw not in ("zh", "en"):
        raise ValueError(f"lang must be 'zh' or 'en', got {lang!r}")
    return raw  # type: ignore[return-value]


def load_default_traces(lang: str | None = None) -> list[dict[str, Any]]:
    resolved = resolve_lang(lang)
    raw = json.loads(_traces_ref(resolved).read_text(encoding="utf-8"))
    return [normalize_trace(x) for x in raw]


def load_traces(
    path: str | Path | None = None,
    *,
    lang: str | None = None,
) -> list[dict[str, Any]]:
    """Load gold traces from a path, or packaged ``traces.{zh|en}.json``."""
    if path is None:
        return load_default_traces(lang=lang)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [normalize_trace(x) for x in raw]
