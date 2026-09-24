"""Compare OpenRouter JEV vs local Laya on gold traces (zh / en).

Uses each item's ``state`` + ``questions`` (JEV typed-decision protocol).

  set OPENROUTER_API_KEY=...
  python eval/compare_jev_laya.py
  python eval/compare_jev_laya.py --lang zh --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus import AgentRouter  # noqa: E402
from laya_thalamus.backend import LayaBackend, _extract_answers, _find_answer  # noqa: E402
from laya_thalamus.config import load_config  # noqa: E402
from laya_thalamus.eval.data import load_traces  # noqa: E402

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


def _score_item(
    item: dict[str, Any],
    *,
    tool: str | None,
    noul: float | None,
    score: float | None,
    latency_ms: float,
    err: str | None = None,
) -> dict[str, Any]:
    exp_tool = item["expected_tool"]
    tool_ok = tool == exp_tool if tool is not None else False
    noul_ok = None
    if "expected_sufficient" in item and noul is not None:
        pred_suff = noul >= 0.5
        noul_ok = pred_suff == bool(item["expected_sufficient"])
    score_ok = None
    if "expected_score_min" in item and score is not None:
        lo = float(item["expected_score_min"])
        hi = float(item.get("expected_score_max", 10))
        score_ok = lo <= score <= hi
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "expected_tool": exp_tool,
        "pred_tool": tool,
        "tool_ok": tool_ok,
        "noul": noul,
        "noul_ok": noul_ok,
        "score": score,
        "score_ok": score_ok,
        "latency_ms": round(latency_ms, 2),
        "error": err,
    }


def _summarize(rows: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    n = len(rows)
    tool_ok = [r for r in rows if r.get("tool_ok")]
    noul_rows = [r for r in rows if r.get("noul_ok") is not None]
    score_rows = [r for r in rows if r.get("score_ok") is not None]
    errs = [r for r in rows if r.get("error")]
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[str(r.get("category") or "other")].append(bool(r.get("tool_ok")))
    lats = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    return {
        "backend": backend,
        "n": n,
        "tool_accuracy": round(len(tool_ok) / n, 4) if n else 0.0,
        "noul_accuracy": round(
            sum(1 for r in noul_rows if r["noul_ok"]) / len(noul_rows), 4
        )
        if noul_rows
        else None,
        "score_band_accuracy": round(
            sum(1 for r in score_rows if r["score_ok"]) / len(score_rows), 4
        )
        if score_rows
        else None,
        "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0.0,
        "error_count": len(errs),
        "per_category_accuracy": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())
        },
        "pred_tool_dist": dict(Counter(r.get("pred_tool") or "ERROR" for r in rows)),
        "errors": [
            {
                "id": r["id"],
                "expected": r["expected_tool"],
                "predicted": r["pred_tool"],
                "category": r.get("category"),
                "error": r.get("error"),
            }
            for r in rows
            if (not r.get("tool_ok")) or r.get("error")
        ][:40],
    }


def _parse_typed_answers(answers: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    tool = None
    noul = None
    score = None
    t = answers.get("tool") or {}
    if isinstance(t, dict):
        tool = t.get("choice")
    s = answers.get("sufficient") or {}
    if isinstance(s, dict) and "noul" in s:
        noul = float(s["noul"])
    c = answers.get("credibility") or {}
    if isinstance(c, dict) and "score" in c:
        score = float(c["score"])
    return tool, noul, score


def call_jev(
    state: str,
    questions: dict[str, Any],
    *,
    api_key: str,
    model: str = JEV_MODEL,
    timeout_s: float = 90.0,
    retries: int = 3,
) -> tuple[dict[str, Any], float]:
    payload = {"model": model, "state": state, "questions": questions}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lcbkmm/laya-thalamus",
        "X-OpenRouter-Title": "laya-thalamus-eval",
    }
    # Bypass broken local proxies (e.g. 127.0.0.1:789x refused).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_err: Exception | None = None
    total_lat = 0.0
    for attempt in range(retries):
        req = urllib.request.Request(JEV_URL, data=data, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with opener.open(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            total_lat += (time.perf_counter() - t0) * 1000
            return body, total_lat
        except Exception as e:
            total_lat += (time.perf_counter() - t0) * 1000
            last_err = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 400:
                raise
            time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def eval_jev(data: list[dict[str, Any]], api_key: str, *, model: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        try:
            body, lat = call_jev(item["state"], item["questions"], api_key=api_key, model=model)
            answers = body.get("answers") or {}
            tool, noul, score = _parse_typed_answers(answers)
            rows.append(
                _score_item(item, tool=tool, noul=noul, score=score, latency_ms=lat)
            )
            flag = "OK" if rows[-1]["tool_ok"] else "MISS"
            print(
                f"  [jev {i}/{len(data)}] {item.get('id')} tool={tool} "
                f"exp={item['expected_tool']} {flag} {lat:.0f}ms"
            )
        except Exception as e:
            msg = str(e)
            if isinstance(e, urllib.error.HTTPError):
                try:
                    msg = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
            print(f"  [jev {i}/{len(data)}] {item.get('id')} ERROR {msg[:120]}")
            rows.append(
                _score_item(
                    item, tool=None, noul=None, score=None, latency_ms=0.0, err=msg[:300]
                )
            )
            time.sleep(1.0)
    return _summarize(rows, f"jev:{model}")


def eval_laya(data: list[dict[str, Any]], *, device: str | None, timeout_ms: int) -> dict[str, Any]:
    cfg = load_config()
    cfg.model.backend = "laya"
    cfg.fallback.enabled = False
    cfg.model.timeout_ms = timeout_ms
    cfg.model.hard_fail_on_timeout = False
    cfg.model.hard_fail_on_exception = False
    if device:
        cfg.model.device = device
    router = AgentRouter(config=cfg)
    be = router.backend
    if not isinstance(be, LayaBackend):
        raise RuntimeError(f"expected LayaBackend, got {type(be)}")
    # warm load
    print(f"[laya] warming model={cfg.model.name!r} device={cfg.model.device}")
    _ = router.route("ping")

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        try:
            t0 = time.perf_counter()
            raw = be._invoke(item["state"], item["questions"])
            lat = (time.perf_counter() - t0) * 1000
            answers = _extract_answers(raw)
            # prefer same keys as gold
            tool_a = _find_answer(answers, "tool") or {}
            suff_a = _find_answer(answers, "sufficient") or {}
            cred_a = _find_answer(answers, "credibility") or {}
            tool = tool_a.get("choice") if isinstance(tool_a, dict) else None
            noul = float(suff_a["noul"]) if isinstance(suff_a, dict) and "noul" in suff_a else None
            score = float(cred_a["score"]) if isinstance(cred_a, dict) and "score" in cred_a else None
            rows.append(
                _score_item(item, tool=tool, noul=noul, score=score, latency_ms=lat)
            )
            flag = "OK" if rows[-1]["tool_ok"] else "MISS"
            print(
                f"  [laya {i}/{len(data)}] {item.get('id')} tool={tool} "
                f"exp={item['expected_tool']} {flag} {lat:.0f}ms"
            )
        except Exception as e:
            print(f"  [laya {i}/{len(data)}] {item.get('id')} ERROR {e}")
            rows.append(
                _score_item(
                    item, tool=None, noul=None, score=None, latency_ms=0.0, err=str(e)[:300]
                )
            )
    return _summarize(rows, f"laya:{cfg.model.name}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "\n{:<40} {:>8} {:>10} {:>10} {:>12} {:>6}".format(
            "backend", "tool_acc", "noul_acc", "score_acc", "latency_ms", "errs"
        )
    )
    print("-" * 92)
    for r in rows:
        print(
            "{:<40} {:>8} {:>10} {:>10} {:>12} {:>6}".format(
                str(r.get("backend", "?"))[:40],
                r.get("tool_accuracy", "-"),
                r.get("noul_accuracy") if r.get("noul_accuracy") is not None else "-",
                r.get("score_band_accuracy")
                if r.get("score_band_accuracy") is not None
                else "-",
                r.get("avg_latency_ms", "-"),
                r.get("error_count", "-"),
            )
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="optional per-lang cap")
    ap.add_argument("--skip-jev", action="store_true")
    ap.add_argument("--skip-laya", action="store_true")
    ap.add_argument("--jev-model", default=JEV_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--timeout-ms", type=int, default=60_000)
    ap.add_argument("--out", default=str(ROOT / "eval" / "compare_jev_laya_report.json"))
    args = ap.parse_args(argv)

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LAYA_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not args.skip_jev and not api_key:
        print("ERROR: set OPENROUTER_API_KEY for JEV", file=sys.stderr)
        return 2

    langs = ["zh", "en"] if args.lang == "both" else [args.lang]
    report: dict[str, Any] = {
        "protocol": "gold state + questions (typed decisions)",
        "jev_model": args.jev_model,
        "langs": {},
    }

    for lang in langs:
        data = load_traces(None, lang=lang)
        if args.limit and args.limit > 0:
            data = data[: args.limit]
        print(f"\n=== lang={lang} n={len(data)} ===")
        block: dict[str, Any] = {"n": len(data), "backends": []}
        if not args.skip_jev:
            print("[jev] evaluating ...")
            block["backends"].append(eval_jev(data, api_key or "", model=args.jev_model))
        if not args.skip_laya:
            print("[laya] evaluating ...")
            block["backends"].append(
                eval_laya(data, device=args.device, timeout_ms=args.timeout_ms)
            )
        report["langs"][lang] = block
        _print_table(block["backends"])

    out = Path(args.out)
    # drop long error lists from disk summary? keep them for analysis
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
