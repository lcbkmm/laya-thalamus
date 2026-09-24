"""Locate / load the packaged gold traces dataset."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def _traces_ref():
    return resources.files("laya_thalamus").joinpath("resources", "traces.json")


def default_traces_path() -> Path:
    """Filesystem path to packaged ``traces.json`` when available on disk.

    Prefer :func:`load_default_traces` for reading -that works from wheels too.
    """
    ref = _traces_ref()
    # Directory installs expose a real path; zip installs should use load_*.
    with resources.as_file(ref) as path:
        resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            "packaged traces.json not found; reinstall laya-thalamus "
            "or pass --dataset explicitly"
        )
    return resolved


def load_default_traces() -> list[dict[str, Any]]:
    return json.loads(_traces_ref().read_text(encoding="utf-8"))


def load_traces(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return load_default_traces()
    return json.loads(Path(path).read_text(encoding="utf-8"))
