"""Convert JEV-like traces into Laya training sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from laya.common import QTYPES, build_sequence, render_options


def load_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    return raw


def soft_target(
    qtype: str, criteria: Any, answer: dict[str, Any]
) -> tuple[list[float], int] | None:
    """Build a soft probability target + argmax label from a gold answer."""
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            return None
        keys = list(criteria.keys())
        if "probabilities" in answer and isinstance(answer["probabilities"], dict):
            target = [float(answer["probabilities"].get(k, 0.0)) for k in keys]
        else:
            choice = answer.get("choice")
            if choice is None:
                return None
            target = [1.0 if k == choice else 0.0 for k in keys]
    elif qtype == "noul":
        if "probabilities" in answer and isinstance(answer["probabilities"], dict):
            p = answer["probabilities"]
            target = [float(p.get("false", 0.5)), float(p.get("true", 0.5))]
        elif "noul" in answer:
            p_true = float(answer["noul"])
            target = [1.0 - p_true, p_true]
        elif "label" in answer:
            p_true = 1.0 if bool(answer["label"]) else 0.0
            target = [1.0 - p_true, p_true]
        else:
            return None
    elif qtype == "score":
        if isinstance(criteria, list):
            n_levels = len(criteria)
        else:
            n_levels = int(answer.get("n_levels") or 10)
        if n_levels < 2:
            return None
        if "probabilities" in answer and isinstance(answer["probabilities"], dict):
            target = [
                float(answer["probabilities"].get(str(i), 0.0)) for i in range(n_levels)
            ]
        else:
            if "score" in answer:
                lvl = int(round(float(answer["score"])))
            elif "score_min" in answer:
                lo = float(answer["score_min"])
                hi = float(answer.get("score_max", lo))
                lvl = int(round((lo + hi) / 2.0))
            elif "label" in answer:
                lvl = int(answer["label"])
            else:
                return None
            lvl = max(0, min(n_levels - 1, lvl))
            target = [0.0] * n_levels
            target[lvl] = 1.0
    else:
        return None

    s = sum(target)
    if s <= 0:
        target = [1.0 / len(target)] * len(target)
    else:
        target = [v / s for v in target]
    label = int(max(range(len(target)), key=lambda i: target[i]))
    return target, label


def build_training_item(
    tok: Any,
    state: Any,
    question: dict[str, Any],
    answer: dict[str, Any],
    *,
    max_len: int,
    head_max_len: int,
) -> dict[str, Any] | None:
    qtype = str(question.get("type") or "").lower()
    if qtype not in QTYPES:
        return None
    criteria = question.get("criteria", {} if qtype != "score" else [])
    packed = soft_target(qtype, criteria, answer)
    if packed is None:
        return None
    target, label = packed
    q = {
        "t": qtype,
        "ins": question.get("instructions") or "",
        "crit": criteria,
    }
    k = len(render_options(q))
    if k < 1:
        return None
    seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
    if len(markers) != k:
        return None
    if len(target) != k:
        # pad / trim soft target to marker count
        if len(target) < k:
            target = target + [0.0] * (k - len(target))
        else:
            target = target[:k]
        s = sum(target) or 1.0
        target = [v / s for v in target]
        label = int(max(range(len(target)), key=lambda i: target[i]))
    return {
        "ids": seq,
        "markers": markers,
        "qtype": int(QTYPES[qtype]),
        "target": target,
        "label": label,
    }


def traces_to_items(
    rows: list[dict[str, Any]],
    tok: Any,
    *,
    max_len: int,
    head_max_len: int,
    limit: int = 0,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        state = row.get("state")
        questions = row.get("questions") or {}
        answers = row.get("answers") or row.get("gold") or {}
        if state is None or not isinstance(questions, dict) or not isinstance(answers, dict):
            continue
        for qid, qdef in questions.items():
            if qid not in answers:
                continue
            ans = answers[qid]
            if not isinstance(ans, dict) or not isinstance(qdef, dict):
                continue
            it = build_training_item(
                tok,
                state,
                qdef,
                ans,
                max_len=max_len,
                head_max_len=head_max_len,
            )
            if it:
                items.append(it)
            if limit and len(items) >= limit:
                return items
    return items
