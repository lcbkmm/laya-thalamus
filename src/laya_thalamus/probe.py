"""Probe real Laya weights -usable after pip install (no source-tree paths).

Prefer ``thalamus probe-laya`` (stable CLI). Module layout is **unstable**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from laya_thalamus.backend import LayaBackend, laya_importable
from laya_thalamus.schemas import ToolSpec


def probe_laya(
    *,
    model: str = "convaiinnovations/laya-multilingual",
    timeout_ms: int = 30_000,
    device: str = "auto",
) -> int:
    """Load Laya and run a tiny choice/noul/score smoke. Returns process exit code."""
    if not laya_importable():
        print("FAIL: package `laya` is not installed. Run: pip install \"laya-thalamus[laya]\"")
        return 1

    tools = [
        ToolSpec(name="none", description="Answer directly without tools"),
        ToolSpec(name="web_search", description="Search the web for current facts"),
        ToolSpec(name="calculator", description="Evaluate arithmetic expressions"),
        ToolSpec(name="code_interpreter", description="Run Python code"),
        ToolSpec(name="rag_retrieve", description="Retrieve internal knowledge-base passages"),
    ]

    model_name = model
    local = Path(model_name).expanduser()
    if local.exists():
        model_name = str(local.resolve())
        print(f"Using local weights: {model_name}")

    be = LayaBackend(
        model_name=model_name,
        device=device,
        timeout_ms=timeout_ms,
    )
    try:
        be.load()
    except Exception as e:
        print(f"FAIL: could not load Laya weights: {e}")
        return 2

    state = (
        "User query: 计算 123*456 等于多少\n"
        "Context: (empty)\n"
    )
    batch = be.predict(state, tools=tools, need_choice=True, need_noul=True, need_score=False)
    print("backend:", batch.backend)
    print("latency_ms:", round(batch.latency_ms, 2))
    if batch.choice:
        print("choice:", batch.choice.choice, "conf=", round(batch.choice.confidence, 3))
        print("probs:", {k: round(v, 3) for k, v in batch.choice.probabilities.items()})
    if batch.noul:
        print(
            "noul P(sufficient)=",
            round(batch.noul.noul, 3),
            "conf=",
            round(batch.noul.confidence, 3),
        )

    state2 = state + "Tool result: 计算结果->6088\n"
    batch2 = be.predict(state2, tools=tools, need_choice=True, need_noul=True, need_score=True)
    print("\n--- after tool ---")
    print("latency_ms:", round(batch2.latency_ms, 2))
    if batch2.choice:
        print("choice:", batch2.choice.choice, "conf=", round(batch2.choice.confidence, 3))
    if batch2.noul:
        print("noul P(sufficient)=", round(batch2.noul.noul, 3))
    if batch2.score:
        print("score:", round(batch2.score.score, 2), "/10")

    print("\nOK: real Laya path works.")
    print(json.dumps({"raw_keys": list((batch.raw or {}).keys())}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Laya backend (HF id or local path)")
    parser.add_argument(
        "--model",
        default="convaiinnovations/laya-multilingual",
        help="HF repo id or local path to hub root / checkpoint",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return probe_laya(model=args.model, timeout_ms=args.timeout_ms, device=args.device)


if __name__ == "__main__":
    raise SystemExit(main())
