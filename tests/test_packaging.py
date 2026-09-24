"""Packaging / CLI path independence."""

from __future__ import annotations

import json
from pathlib import Path

from laya_thalamus.eval.data import load_default_traces
from laya_thalamus.cli import main as cli_main


def test_packaged_traces_loadable():
    data = load_default_traces()
    assert len(data) >= 1
    assert "query" in data[0]
    assert "expected_tool" in data[0]


def test_repo_traces_match_packaged_when_present():
    """Keep checkout eval/traces.json in sync with the packaged copy."""
    repo = Path(__file__).resolve().parents[1] / "eval" / "traces.json"
    if not repo.exists():
        return
    packaged = load_default_traces()
    checkout = json.loads(repo.read_text(encoding="utf-8"))
    assert checkout == packaged


def test_cli_compare_mock_only(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    code = cli_main(
        ["compare", "--skip-llm", "--skip-laya", "--out", str(out)]
    )
    assert code == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    backends = {b["backend"] for b in report["backends"]}
    assert "mock" in backends
    assert "random" in backends


def test_core_import_without_fastapi():
    """Library import path must not require the [api] extra."""
    import laya_thalamus
    from laya_thalamus import AgentRouter

    assert laya_thalamus.__version__
    assert AgentRouter is not None


def test_version_is_pep440ish():
    from laya_thalamus import __version__

    assert __version__
    assert __version__[0].isdigit()
