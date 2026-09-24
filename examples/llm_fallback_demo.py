"""
演示：低置信 / 超时 / 异常时走真实 OpenAI-compatible LLM 降级?

未配?LLM 时，会用 Fake 客户端演示链路（不发起网络请求）?

真调?:
    set LAYA_LLM_BASE_URL=http://127.0.0.1:8000/v1
    set LAYA_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    python examples/llm_fallback_demo.py --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus import AgentRouter
from laya_thalamus.backend import MockBackend
from laya_thalamus.config import RouterConfig
from laya_thalamus.fallback import FallbackChain, HeuristicFallback, LLMFallback
from laya_thalamus.llm import llm_is_configured


class DeadBackend(MockBackend):
    name = "laya"

    def predict(self, *args, **kwargs):
        raise RuntimeError("simulated Laya crash")


class ScriptedClient:
    def chat_json(self, system: str, user: str) -> dict:
        # Only look at the query line, not the tool catalog (which contains "calculator").
        query_line = user.split("\n", 1)[0]
        if "Pick exactly one tool" in system or "Pick one tool" in system:
            if any(k in query_line for k in ("计算", "算一下")) or "*" in query_line:
                return {"tool": "calculator", "confidence": 0.93}
            if any(k in query_line for k in ("搜索", "查一下")) or "search" in query_line.lower():
                return {"tool": "web_search", "confidence": 0.9}
            return {"tool": "none", "confidence": 0.7}
        if "enough" in system.lower() or "sufficient" in system.lower():
            return {"sufficient": False, "confidence": 0.8}
        return {"score": 4.0, "confidence": 0.7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Call real LLM endpoint from env")
    args = ap.parse_args()

    cfg = RouterConfig()
    cfg.model.backend = "mock"
    cfg.fallback.enabled = True
    cfg.fallback.mode = "llm"

    if args.live:
        if not llm_is_configured():
            print("No LAYA_LLM_BASE_URL / OPENAI_API_KEY configured.")
            sys.exit(1)
        router = AgentRouter(config=cfg, backend=DeadBackend())
        # rebuild fallback from env via config
        router._fallback = FallbackChain(
            LLMFallback(), HeuristicFallback()
        )
        print("Using LIVE LLM fallback")
    else:
        router = AgentRouter(config=cfg, backend=DeadBackend())
        router._fallback = FallbackChain(
            LLMFallback(ScriptedClient()), HeuristicFallback()  # type: ignore[arg-type]
        )
        print("Using scripted LLM fallback (offline demo)")

    for q in ["计算 99*88", "搜索一下今天的科技新闻"]:
        d = router.route(q)
        print(f"\nQ: {q}")
        print(f"   {d.summary()}")
        print(f"   degrade={d.degrade_reason}")


if __name__ == "__main__":
    main()
