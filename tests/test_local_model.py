"""Tests for local Laya path resolution (no GPU / no real weights required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from laya_thalamus.backend import (
    _checkpoint_alias,
    classify_local_checkpoint,
    resolve_local_model_path,
)


def _touch_weights(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "model.safetensors").write_bytes(b"fake")


def test_resolve_ignores_hf_repo_id(tmp_path: Path):
    assert resolve_local_model_path("convaiinnovations/laya-multilingual") is None
    assert resolve_local_model_path("convaiinnovations/laya") is None


def test_load_config_resolves_relative_model_path(tmp_path: Path):
    from laya_thalamus.config import load_config

    weights = tmp_path / "weights" / "LAYA"
    weights.mkdir(parents=True)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "local.yaml"
    cfg_file.write_text(
        "model:\n  backend: mock\n  name: ../weights/LAYA\n  timeout_ms: 30000\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert Path(cfg.model.name) == weights.resolve()


def test_resolve_local_directory(tmp_path: Path):
    hub = tmp_path / "LAYA"
    hub.mkdir()
    resolved = resolve_local_model_path(str(hub))
    assert resolved == hub.resolve()


def test_classify_hub_bundle(tmp_path: Path):
    hub = tmp_path / "LAYA"
    _touch_weights(hub)
    _touch_weights(hub / "multilingual")
    _touch_weights(hub / "typed-decisions")
    assert classify_local_checkpoint(hub) == "hub"
    assert classify_local_checkpoint(hub / "multilingual") == "hub-child"
    assert classify_local_checkpoint(hub / "typed-decisions") == "hub-child"


def test_classify_single_checkpoint(tmp_path: Path):
    only = tmp_path / "my-laya"
    _touch_weights(only)
    assert classify_local_checkpoint(only) == "single"


def test_checkpoint_alias_hf_and_local(tmp_path: Path):
    assert _checkpoint_alias("convaiinnovations/laya-multilingual") == "multilingual"
    assert _checkpoint_alias("convaiinnovations/laya-typed-decisions") == "typed-decisions"
    assert _checkpoint_alias("convaiinnovations/laya") == "english"

    hub = tmp_path / "LAYA"
    _touch_weights(hub)
    _touch_weights(hub / "multilingual")
    assert _checkpoint_alias(str(hub), local_root=hub) is None  # auto-route
    assert (
        _checkpoint_alias(str(hub / "multilingual"), local_root=hub / "multilingual")
        == "multilingual"
    )


def test_laya_backend_load_local_hub_with_fake_laya(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise Local hub attach path without downloading weights."""
    hub = tmp_path / "LAYA"
    _touch_weights(hub)
    _touch_weights(hub / "multilingual")

    loaded: list[tuple[str, str | None]] = []

    class FakeAgent:
        def predict(self, state, questions, model=None):
            return {
                "answers": {
                    "tool": {"choice": "none", "confidence": 0.9, "probabilities": {"none": 0.9}},
                    "sufficient": {"noul": 0.8, "confidence": 0.7},
                }
            }

    class FakeRouter:
        def __init__(self, **kwargs):
            self.attached: dict = {}
            self.kwargs = kwargs

        def attach(self, name, agent):
            self.attached[name] = agent

        def predict(self, state, questions, model=None):
            return FakeAgent().predict(state, questions, model=model)

    class FakeLaya:
        Router = FakeRouter

        @staticmethod
        def load(repo, subfolder=None, **kwargs):
            loaded.append((repo, subfolder))
            return FakeAgent()

    import laya_thalamus.backend as be_mod
    from laya_thalamus.backend import LayaBackend
    from laya_thalamus.schemas import ToolSpec

    monkeypatch.setitem(__import__("sys").modules, "laya", FakeLaya)

    backend = LayaBackend(model_name=str(hub), timeout_ms=10_000)
    backend.load()

    assert backend.is_ready()
    assert backend._use_router is True
    assert ("english", None) in [(p, s) for p, s in loaded] or any(
        Path(p).name == "LAYA" and s is None for p, s in loaded
    )
    assert any(s == "multilingual" for _, s in loaded)

    batch = backend.predict(
        "User query: hi",
        tools=[ToolSpec(name="none", description="direct")],
        need_choice=True,
        need_noul=True,
    )
    assert batch.choice is not None
    assert batch.choice.choice == "none"
