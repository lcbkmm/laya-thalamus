"""
Compare backends on the packaged (or custom) gold traces.

Usage::
    thalamus compare
    thalamus compare --dataset path/to/traces.json --skip-llm
    thalamus compare --device cuda --with-fallback --skip-llm
    python -m laya_thalamus.eval.compare --llm-models deepseek-v3.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from laya_thalamus import AgentRouter
from laya_thalamus.backend import LLMRouterBackend, laya_importable
from laya_thalamus.config import load_config
from laya_thalamus.eval.common import evaluate_items
from laya_thalamus.eval.data import load_traces
from laya_thalamus.llm import OpenAICompatibleClient, llm_is_configured

DEFAULT_LLM_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "deepseek-v3.1",
    "qwen3.8-max",
]


def _try_router(
    backend: str,
    *,
    config_path: str | None = None,
    model_name: str | None = None,
    timeout_ms: int | None = None,
    device: str | None = None,
    llm_model: str | None = None,
    fallback: bool = False,
) -> AgentRouter | None:
    cfg = load_config(config_path) if config_path else load_config()
    cfg.model.backend = backend
    cfg.fallback.enabled = fallback
    if fallback:
        cfg.fallback.mode = "auto"
    if model_name:
        cfg.model.name = model_name
    if timeout_ms is not None:
        cfg.model.timeout_ms = timeout_ms
    if device:
        cfg.model.device = device

    try:
        if backend == "llm":
            client = OpenAICompatibleClient(
                model=llm_model or os.environ.get("LAYA_LLM_MODEL"),
                enable_thinking=False,
                timeout_s=90.0,
            )
            be = LLMRouterBackend(client=client)
            label = llm_model or client.model or "llm"
            be.name = f"llm:{label}"
            return AgentRouter(config=cfg, backend=be)
        return AgentRouter(config=cfg)
    except Exception as e:
        tag = f"llm:{llm_model}" if backend == "llm" else backend
        print(f"[skip] backend={tag}: {e}")
        return None


def _random_baseline(data: list[dict]) -> dict:
    import random

    tools = sorted({item["expected_tool"] for item in data})
    rng = random.Random(0)
    y_true = [item["expected_tool"] for item in data]
    y_pred = [rng.choice(tools) for _ in data]
    n = len(data)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0
    return {
        "backend": "random",
        "n": n,
        "tool_accuracy": round(acc, 4),
        "avg_latency_ms": 0.0,
        "note": "uniform random over gold label set - lower bound",
    }


def _compact(report: dict) -> dict:
    report = dict(report)
    report.pop("errors", None)
    report.pop("confusion_matrix", None)
    return report


def _print_table(rows: list[dict]) -> None:
    print(
        "\n{:<32} {:>8} {:>10} {:>12} {:>8} {:>8}".format(
            "backend", "tool_acc", "noul_acc", "latency_ms", "errors", "fb%"
        )
    )
    print("-" * 84)
    for row in rows:
        fb = row.get("fallback_rate")
        fb_s = f"{100 * fb:.0f}%" if isinstance(fb, (int, float)) else "-"
        print(
            "{:<32} {:>8} {:>10} {:>12} {:>8} {:>8}".format(
                str(row.get("backend", "?"))[:32],
                row.get("tool_accuracy", "-"),
                row.get("noul_accuracy")
                if row.get("noul_accuracy") is not None
                else "-",
                row.get("avg_latency_ms", "-"),
                row.get("error_count", "-"),
                fb_s,
            )
        )


def _run_laya_column(
    data: list[dict],
    *,
    label: str,
    config_path: str | None,
    model_name: str | None,
    timeout_ms: int | None,
    device: str | None,
    fallback: bool,
) -> dict | None:
    router = _try_router(
        "laya",
        config_path=config_path,
        model_name=model_name,
        timeout_ms=timeout_ms,
        device=device,
        fallback=fallback,
    )
    if router is None:
        return None
    resolved_device = router.config.model.device
    print(
        f"[laya] label={label!r} model={router.config.model.name!r} "
        f"device={resolved_device} fallback={fallback}"
    )
    report = _compact(evaluate_items(router, data))
    report["backend"] = label
    report["device"] = resolved_device
    report["fallback"] = fallback
    print(
        f"[done] {label} tool_acc={report['tool_accuracy']} "
        f"latency={report['avg_latency_ms']}ms "
        f"fallback_rate={report.get('fallback_rate', 0)}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="thalamus compare")
    ap.add_argument(
        "--dataset",
        default=None,
        help="Gold JSON path (default: packaged traces.json)",
    )
    ap.add_argument(
        "--out",
        default="compare_report.json",
        help="Report output path (default: ./compare_report.json)",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="Local/HF Laya weights path")
    ap.add_argument("--timeout-ms", type=int, default=None)
    ap.add_argument(
        "--device",
        default=None,
        help="Laya device: auto | cpu | cuda | mps (also labels the Laya column)",
    )
    ap.add_argument(
        "--llm-models",
        default=",".join(DEFAULT_LLM_MODELS),
        help="Comma-separated chat models for the LLM router columns",
    )
    ap.add_argument("--skip-llm", action="store_true", help="Only run mock + laya")
    ap.add_argument(
        "--skip-laya",
        action="store_true",
        help="Skip Laya column (LLM-only / mock-only sweeps)",
    )
    ap.add_argument(
        "--with-fallback",
        action="store_true",
        help="Also run Laya with fallback enabled (product path: LLM if configured, else heuristic)",
    )
    ap.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip bare Laya; only run Laya+fallback (implies --with-fallback)",
    )
    args = ap.parse_args(argv)
    if args.fallback_only:
        args.with_fallback = True

    data = load_traces(args.dataset)
    dataset_label = Path(args.dataset).name if args.dataset else "traces.json (packaged)"
    print(f"[dataset] {dataset_label} n={len(data)}")

    device_tag = (args.device or "auto").lower()
    rows: list[dict] = [_random_baseline(data)]

    router = _try_router(
        "mock",
        config_path=args.config,
        timeout_ms=args.timeout_ms,
        device=args.device,
        fallback=False,
    )
    if router:
        report = _compact(evaluate_items(router, data))
        rows.append(report)
        print(f"[done] mock tool_acc={report['tool_accuracy']}")

    if not args.skip_laya:
        if laya_importable():
            if not args.fallback_only:
                bare = _run_laya_column(
                    data,
                    label=f"laya:{device_tag}",
                    config_path=args.config,
                    model_name=args.model,
                    timeout_ms=args.timeout_ms,
                    device=args.device,
                    fallback=False,
                )
                if bare:
                    rows.append(bare)
            if args.with_fallback:
                fb = _run_laya_column(
                    data,
                    label=f"laya+fallback:{device_tag}",
                    config_path=args.config,
                    model_name=args.model,
                    timeout_ms=args.timeout_ms,
                    device=args.device,
                    fallback=True,
                )
                if fb:
                    fb["fallback_note"] = (
                        "LLM if LAYA_LLM_* / OPENAI_* configured; else heuristic"
                    )
                    rows.append(fb)
        else:
            print(
                '[skip] laya: package not installed (pip install "laya-thalamus[laya]")'
            )

    llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
    if args.skip_llm:
        print("[skip] llm: --skip-llm")
    elif not llm_is_configured():
        print(
            "[skip] llm: set LAYA_LLM_BASE_URL + LAYA_LLM_API_KEY "
            "(or OPENAI_* / DASHSCOPE_*)"
        )
    else:
        for mid in llm_models:
            print(f"[llm] evaluating {mid} ...")
            router = _try_router(
                "llm",
                config_path=args.config,
                llm_model=mid,
                fallback=False,
            )
            if router is None:
                continue
            report = _compact(evaluate_items(router, data))
            report["backend"] = f"llm:{mid}"
            report["llm_model"] = mid
            rows.append(report)
            print(
                f"[done] llm:{mid} tool_acc={report['tool_accuracy']} "
                f"latency={report['avg_latency_ms']}ms"
            )

    summary = {
        "dataset": dataset_label,
        "n": len(data),
        "laya_model": args.model or "<configured>",
        "device": args.device or "auto",
        "with_fallback": bool(args.with_fallback),
        "llm_models": llm_models if not args.skip_llm else [],
        "backends": rows,
        "how_to_read": {
            "mock": "Keyword heuristic - not a product metric.",
            "laya:<device>": "Bare System-1 (fallback off).",
            "laya+fallback:<device>": (
                "Product path: low-confidence / hard-fail -> LLM if configured, "
                "else heuristic."
            ),
            "llm:*": "Chat model as primary router (cost/latency baseline).",
            "random": "Chance-level lower bound.",
            "gpu": "Pass --device cuda (or mps) to fill GPU latency; omitted if unavailable.",
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_table(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
