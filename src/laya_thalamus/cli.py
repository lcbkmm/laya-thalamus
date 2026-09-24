"""CLI: thalamus serve | demo | health | probe-laya | compare."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thalamus", description="Laya Thalamus CLI")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start HTTP API server (needs [api] extra)")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    p_demo = sub.add_parser("demo", help="Run a few sample routes")
    p_demo.add_argument("--backend", default="mock", choices=["mock", "laya", "llm", "auto"])

    p_health = sub.add_parser("health", help="Probe backend health")
    p_health.add_argument("--backend", default="auto", choices=["mock", "laya", "llm", "auto"])

    p_probe = sub.add_parser("probe-laya", help="Fail loudly if real Laya cannot load")
    p_probe.add_argument(
        "--model",
        default="convaiinnovations/laya-multilingual",
        help="HF repo id or local weights path",
    )
    p_probe.add_argument("--timeout-ms", type=int, default=30_000)
    p_probe.add_argument("--device", default="auto")

    p_compare = sub.add_parser("compare", help="Compare backends on gold traces")
    p_compare.add_argument("--dataset", default=None, help="Gold JSON (default: packaged)")
    p_compare.add_argument("--out", default="compare_report.json")
    p_compare.add_argument("--config", default=None)
    p_compare.add_argument("--model", default=None, help="Local/HF Laya weights")
    p_compare.add_argument("--timeout-ms", type=int, default=None)
    p_compare.add_argument("--device", default=None)
    p_compare.add_argument("--llm-models", default=None)
    p_compare.add_argument("--skip-llm", action="store_true")
    p_compare.add_argument("--skip-laya", action="store_true")
    p_compare.add_argument(
        "--with-fallback",
        action="store_true",
        help="Also evaluate Laya with fallback on (product path)",
    )
    p_compare.add_argument(
        "--fallback-only",
        action="store_true",
        help="Only Laya+fallback (skip bare Laya)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from laya_thalamus.api import run

        run(host=args.host, port=args.port, config_path=args.config)
        return 0

    if args.cmd == "probe-laya":
        from laya_thalamus.probe import probe_laya

        return probe_laya(
            model=args.model,
            timeout_ms=args.timeout_ms,
            device=args.device,
        )

    if args.cmd == "compare":
        from laya_thalamus.eval.compare import main as compare_main

        argv2: list[str] = []
        if args.dataset:
            argv2 += ["--dataset", args.dataset]
        if args.out:
            argv2 += ["--out", args.out]
        if args.config:
            argv2 += ["--config", args.config]
        if args.model:
            argv2 += ["--model", args.model]
        if args.timeout_ms is not None:
            argv2 += ["--timeout-ms", str(args.timeout_ms)]
        if args.device:
            argv2 += ["--device", args.device]
        if args.llm_models:
            argv2 += ["--llm-models", args.llm_models]
        if args.skip_llm:
            argv2.append("--skip-llm")
        if args.skip_laya:
            argv2.append("--skip-laya")
        if getattr(args, "with_fallback", False):
            argv2.append("--with-fallback")
        if getattr(args, "fallback_only", False):
            argv2.append("--fallback-only")
        return compare_main(argv2)

    from laya_thalamus.config import load_config
    from laya_thalamus.router import AgentRouter

    cfg = load_config(args.config)
    if hasattr(args, "backend") and args.backend:
        cfg.model.backend = args.backend

    router = AgentRouter(config=cfg)

    if args.cmd == "health":
        print(json.dumps(router.health(), ensure_ascii=False, indent=2))
        if router.health().get("is_mock"):
            print(
                "\nNOTE: is_mock=true -install `laya-thalamus[laya]` and set "
                "model.backend=laya for production System-1 routing.",
                file=sys.stderr,
            )
        return 0

    if args.cmd == "demo":
        if router.backend.name == "mock":
            print(
                "NOTE: running on MockBackend (keyword heuristic), not Laya.\n",
                file=sys.stderr,
            )
        samples = [
            "你好",
            "计算 123 * 456 等于多少",
            "搜索一下今天北京的天气",
            "用 Python 写一个快速排序",
            "根据知识库文档，公司年假政策是什么",
        ]
        for q in samples:
            d = router.route(q)
            print(f"Q: {q}")
            print(f"   ->{d.summary()}")
            print(f"   reason: {d.reason}")
            print()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
