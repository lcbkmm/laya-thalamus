"""Packaging / CLI path independence."""

from __future__ import annotations

import json
from pathlib import Path

from laya_thalamus.eval.data import load_default_traces, load_traces
from laya_thalamus.cli import main as cli_main


def test_packaged_traces_bilingual():
    for lang in ("zh", "en"):
        data = load_default_traces(lang=lang)
        assert len(data) == 100
        assert data[0]["lang"] == lang
        assert "query" in data[0]
        assert "expected_tool" in data[0]
        assert "state" in data[0]
        assert "questions" in data[0]
        assert "answers" in data[0]
        assert data[0]["questions"]["tool"]["type"] == "choice"
        assert data[0]["questions"]["sufficient"]["type"] == "noul"


def test_repo_traces_match_packaged_when_present():
    """Keep checkout eval/traces.*.json in sync with packaged copies."""
    root = Path(__file__).resolve().parents[1]
    for lang in ("zh", "en"):
        name = f"traces.{lang}.json"
        repo = root / "eval" / name
        packaged_path = root / "src" / "laya_thalamus" / "resources" / name
        if not repo.exists() or not packaged_path.exists():
            continue
        checkout = json.loads(repo.read_text(encoding="utf-8"))
        packaged = json.loads(packaged_path.read_text(encoding="utf-8"))
        assert checkout == packaged
        assert len(checkout) == 100


def test_load_traces_lang_env(monkeypatch):
    monkeypatch.setenv("LAYA_TRACES_LANG", "en")
    data = load_traces()
    assert len(data) == 100
    assert data[0]["lang"] == "en"


def test_cli_compare_mock_only(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    code = cli_main(
        ["compare", "--lang", "en", "--skip-llm", "--skip-laya", "--out", str(out)]
    )
    assert code == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    backends = {b["backend"] for b in report["backends"]}
    assert "mock" in backends
    assert "random" in backends
    assert "traces.en.json" in report["dataset"]


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
