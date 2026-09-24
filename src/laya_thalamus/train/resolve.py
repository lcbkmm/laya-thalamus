"""Resolve a Laya base checkpoint (HF id or local path) to an on-disk directory."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_checkpoint_dir(
    model: str,
    *,
    subfolder: str | None = None,
    token: str | None = None,
) -> Path:
    """Return the directory that contains ``rl_agent_config.json`` + weights."""
    model = (model or "").strip()
    if not model:
        raise ValueError("base model is empty")

    # Heuristic: HF multilingual id maps to subfolder on the bundle repo.
    if subfolder is None and model.rstrip("/").endswith("laya-multilingual"):
        # Prefer dedicated multilingual repo if local/cache; else bundle subfolder.
        # Keep as-is when path exists.
        pass

    path = Path(model)
    if path.exists():
        root = path
        if subfolder:
            root = root / subfolder
        cfg = root / "rl_agent_config.json"
        if not cfg.exists():
            raise FileNotFoundError(
                f"Not a Laya checkpoint (missing rl_agent_config.json): {root}"
            )
        return root.resolve()

    from huggingface_hub import snapshot_download

    # Map convenience aliases to bundle + subfolder.
    repo = model
    sub = subfolder
    if model == "convaiinnovations/laya-multilingual" and sub is None:
        repo, sub = "convaiinnovations/laya", "multilingual"
    elif model == "convaiinnovations/laya-typed-decisions" and sub is None:
        repo, sub = "convaiinnovations/laya", "typed-decisions"

    prefix = f"{sub}/" if sub else ""
    allow = [
        prefix + name
        for name in (
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        )
    ]
    model_dir = snapshot_download(
        repo,
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=allow,
    )
    root = Path(model_dir)
    if sub:
        root = root / sub
    if not (root / "rl_agent_config.json").exists():
        raise FileNotFoundError(
            f"Downloaded checkpoint incomplete: {root} (repo={repo!r} sub={sub!r})"
        )
    return root.resolve()
