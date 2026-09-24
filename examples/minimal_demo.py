"""
最小独?Demo：纯 Laya Router（默?mock backend），打印路由决策?

运行::
    python examples/minimal_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus import AgentRouter, ToolSpec


def main() -> None:
    router = AgentRouter()  # backend=auto ->mock if laya not installed

    @router.hooks.on_after
    def _print_latency(req, decision):
        print(f"   [hook] latency={decision.latency_ms:.1f}ms backend={decision.backend}")
        return decision

    queries = [
        "你好，介绍一下你自己",
        "计算 (17+29)*3 等于多少？",
        "帮我搜索一下 2026 年诺贝尔奖得主",
        "写一段 Python 代码统计列表中位数",
        "根据内部知识库，报销流程是什么？",
    ]

    print("=" * 60)
    print("Laya Thalamus -Minimal Demo")
    print("=" * 60)

    for q in queries:
        decision = router.route(query=q)
        print(f"\nQuery : {q}")
        print(f"Action: {decision.action.value}")
        print(f"Tool  : {decision.selected_tool}")
        print(f"Suff  : {decision.information_sufficient}")
        print(f"Conf  : {decision.confidence:.2f}")
        print(f"Reason: {decision.reason}")
        if decision.candidates:
            tops = ", ".join(f"{c.name}={c.probability:.2f}" for c in decision.candidates)
            print(f"Top-K : {tops}")

    # Simulate tool result ->score + noul
    print("\n" + "-" * 60)
    print("After calculator tool returns...")
    d2 = router.route(
        query="计算 (17+29)*3 等于多少？",
        tool_result="计算结果：138",
        history=["calculator"],
        session_id="demo-1",
    )
    print(f"Action: {d2.action.value} | score={d2.credibility_score} | suff={d2.information_sufficient}")
    print(f"Reason: {d2.reason}")

    print("\nMetrics:", router.metrics())


if __name__ == "__main__":
    main()
