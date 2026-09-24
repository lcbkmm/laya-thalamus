"""Unit tests for fine-tune data conversion (no GPU / no weights)."""

from __future__ import annotations

from laya_thalamus.train.data import soft_target


def test_soft_target_choice_one_hot():
    t, label = soft_target(
        "choice",
        {"none": "chat", "calculator": "math"},
        {"choice": "calculator"},
    )
    assert label == 1
    assert t == [0.0, 1.0]


def test_soft_target_noul_from_label():
    t, label = soft_target("noul", {"true": "y", "false": "n"}, {"label": True})
    assert abs(t[1] - 1.0) < 1e-6
    assert label == 1


def test_soft_target_score_band_midpoint():
    crit = [f"l{i}" for i in range(10)]
    t, label = soft_target(
        "score", crit, {"score_min": 6, "score_max": 8}
    )
    assert label == 7
    assert abs(sum(t) - 1.0) < 1e-6
    assert t[7] == 1.0
