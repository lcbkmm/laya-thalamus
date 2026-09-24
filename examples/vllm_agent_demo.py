"""
vLLM + Laya Router 分层 Agent 示例?

架构->
  User Query
      ->Laya AgentRouter（System1：选工?/ 是否够信?/ 打分->
      ->工具（calculator / web_search mock / rag mock->
      ->vLLM / OpenAI-compatible LLM（System2：总结回答->

未配?vLLM 时，自动用本?stub LLM，方便离线演示?

运行::
    python examples/vllm_agent_demo.py
    VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
        python examples/vllm_agent_demo.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus import AgentRouter, DecisionAction


# ---------- mock tools ----------

def tool_calculator(query: str) -> str:
    expr = re.sub(r"[^\d\+\-\*/\(\)\.\s]", "", query)
    expr = expr.strip() or re.findall(r"[\d\+\-\*/\(\)\.]+", query)
    if isinstance(expr, list):
        expr = expr[0] if expr else ""
    try:
        # safe-ish eval for demo only
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return f"计算结果：{val}"
    except Exception as e:
        return f"计算失败：{e}"


def tool_web_search(query: str) -> str:
    return (
        f"[mock search] 关于「{query}」的检索摘要："
        "根据公开资料，相关信息显示该话题近期有多篇报道，"
        "核心结论为--（此处为演示占位内容）"
    )


def tool_rag(query: str) -> str:
    return (
        f"[mock RAG] 知识库命中片段：与「{query}」相关的内部文档写明："
        "标准流程为提交申请 -> 主管审批 -> 财务复核。"
    )


def tool_code(query: str) -> str:
    return (
        "```python\n"
        "def median(xs):\n"
        "    s = sorted(xs)\n"
        "    n = len(s)\n"
        "    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2\n"
        "```"
    )


TOOLS = {
    "calculator": tool_calculator,
    "web_search": tool_web_search,
    "rag_retrieve": tool_rag,
    "code_interpreter": tool_code,
}


# ---------- LLM (vLLM OpenAI-compatible or stub) ----------

def call_llm(prompt: str) -> str:
    base = os.environ.get("VLLM_BASE_URL")
    model = os.environ.get("VLLM_MODEL", "default")
    if not base:
        # stub summary
        return f"[stub-LLM 总结] 根据已收集信息：{prompt[-300:]}"

    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是助手。根据用户问题与工具结果，给出简洁准确的中文回答。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, TimeoutError) as e:
        return f"[LLM 调用失败，回退 stub] {e}\n材料：{prompt[-200:]}"


# ---------- Agent loop ----------

def run_agent(query: str, router: AgentRouter, max_rounds: int = 5) -> str:
    session = "vllm-demo"
    router.reset_session(session)
    history: list[str] = []
    context_parts: list[str] = []
    tool_result = None

    print(f"\n{'='*60}\nUser: {query}\n{'='*60}")

    for round_i in range(max_rounds):
        decision = router.route(
            query=query,
            context="\n".join(context_parts) if context_parts else None,
            tool_result=tool_result,
            session_id=session,
            history=history,
        )
        print(f"[round {round_i+1}] {decision.summary()}")
        print(f"         reason: {decision.reason}")

        if decision.action == DecisionAction.TERMINATE:
            break

        if decision.action in (DecisionAction.ANSWER, DecisionAction.FALLBACK_LLM):
            materials = "\n".join(context_parts) or "(无工具结?"
            prompt = f"用户问题：{query}\n\n已收集信息：\n{materials}"
            answer = call_llm(prompt)
            print(f"LLM ->{answer}")
            return answer

        if decision.action in (DecisionAction.CALL_TOOL, DecisionAction.RETRY_TOOL):
            tool = decision.selected_tool or "web_search"
            if tool == "none":
                materials = "\n".join(context_parts) or "(无工具结?"
                return call_llm(f"用户问题：{query}\n\n已收集信息：\n{materials}")
            fn = TOOLS.get(tool)
            if not fn:
                print(f"  unknown tool {tool}, stop")
                break
            tool_result = fn(query)
            context_parts.append(f"[{tool}] {tool_result}")
            history.append(tool)
            print(f"  tool[{tool}] ->{tool_result[:120]}...")
            continue

    return call_llm(f"用户问题：{query}\n材料：\n" + "\n".join(context_parts))


def main() -> None:
    router = AgentRouter()
    for q in [
        "计算 123*456",
        "搜索一下量子计算最新进展",
        "根据知识库，报销需要哪些材料？",
    ]:
        run_agent(q, router)


if __name__ == "__main__":
    main()
