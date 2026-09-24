"""
评测：对人工标注集跑柝一?backend?
默认关闭 fallback，靿兝「mock 评测冝被 heuristic 救回来〝造戝虚高?
用法::
    python eval/run_eval.py --backend mock
    python eval/run_eval.py --dataset eval/traces.json --backend mock
    thalamus compare --skip-llm   # pip 安装坎的对比入坣
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from laya_thalamus import AgentRouter
from laya_thalamus.config import load_config
from laya_thalamus.eval.common import evaluate_items
from laya_thalamus.eval.data import load_traces


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=None,
        help="Gold JSON path (default: packaged traces.json)",
    )
    ap.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "laya", "llm", "auto"],
    )
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--errors", default="errors.jsonl")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--model", default=None, help="HF id or local Laya path")
    ap.add_argument("--timeout-ms", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        default=True,
        help="Disable fallback (default). Use --fallback to re-enable.",
    )
    ap.add_argument(
        "--fallback",
        dest="no_fallback",
        action="store_false",
        help="Keep fallback enabled (not recommended for measuring a backend).",
    )
    args = ap.parse_args(argv)

    data = load_traces(args.dataset)
    dataset_label = Path(args.dataset).name if args.dataset else "traces.json (packaged)"
    cfg = load_config(args.config) if args.config else load_config()
    cfg.model.backend = args.backend
    if args.model:
        cfg.model.name = args.model
    if args.timeout_ms is not None:
        cfg.model.timeout_ms = args.timeout_ms
    if args.device:
        cfg.model.device = args.device
    if args.no_fallback:
        cfg.fallback.enabled = False
    router = AgentRouter(config=cfg)

    report = evaluate_items(router, data)
    errors = report.pop("errors")
    report["dataset"] = dataset_label
    report["fallback_enabled"] = not args.no_fallback
    report["model"] = cfg.model.name
    report["metrics"] = router.metrics()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    err_path = Path(args.errors)
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote report ->{out}")
    print(f"Wrote errors ->{err_path} ({len(errors)} cases)")
    if router.backend.name == "mock":
        print(
            "\nNOTE: backend=mock is a keyword heuristic. "
            "Install `laya-thalamus[laya]` and rerun with --backend laya "
            "for System-1 numbers."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
