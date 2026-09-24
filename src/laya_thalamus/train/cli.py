"""CLI: thalamus finetune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="thalamus finetune",
        description="Fine-tune a Laya checkpoint on JEV-like gold traces",
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Gold JSON list (state + questions + answers), e.g. eval/traces.zh.json",
    )
    ap.add_argument(
        "--base-model",
        default="convaiinnovations/laya-multilingual",
        help="HF id or local Laya checkpoint directory",
    )
    ap.add_argument(
        "--subfolder",
        default=None,
        help="Optional subfolder inside a hub bundle (multilingual / typed-decisions)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="./checkpoints/finetuned",
        help="Output directory for the fine-tuned checkpoint",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--max-items", type=int, default=0, help="Cap training items (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    args = ap.parse_args(argv)

    try:
        import laya  # noqa: F401
    except ImportError:
        print(
            'ERROR: install training deps: pip install "laya-thalamus[laya]"',
            file=sys.stderr,
        )
        return 2

    from laya_thalamus.train.loop import run_finetune

    ds = Path(args.dataset)
    if not ds.exists():
        print(f"ERROR: dataset not found: {ds}", file=sys.stderr)
        return 2

    meta = run_finetune(
        dataset=ds,
        base_model=args.base_model,
        output_dir=args.output,
        subfolder=args.subfolder,
        device=args.device,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        group_size=args.group_size,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        max_items=args.max_items,
        seed=args.seed,
        ce_weight=args.ce_weight,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(
        "\nNext:\n"
        f"  thalamus probe-laya --model {args.output}\n"
        f"  thalamus compare --model {args.output} --lang zh --skip-llm\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
