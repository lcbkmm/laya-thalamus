"""
OpenAI function-calling 接入样板-> 行核心循环）?
违行::
    python examples/openai_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus import AgentRouter
from laya_thalamus.config import RouterConfig
from laya_thalamus.backend import MockBackend
from laya_thalamus.integrations import (
    decision_to_tool_calls,
    route_openai_turn,
    tools_to_openai,
)
from laya_thalamus.schemas import ToolSpec


def main() -> None:
    cfg = RouterConfig()
    cfg.model.backend = "mock"
    specs = [
        ToolSpec(name="calculator", description="math", parameters={"type": "object", "properties": {}}),
        ToolSpec(name="web_search", description="search the web"),
    ]
    router = AgentRouter(config=cfg, backend=MockBackend(), tools=specs)
    openai_tools = tools_to_openai(specs)

    messages = [{"role": "user", "content": "计算 17*19"}]
    d = route_openai_turn(router, messages, openai_tools, session_id="demo-oa")
    calls = decision_to_tool_calls(d)
    print("decision_id:", d.decision_id)
    print("action:", d.action.value, "tool:", d.selected_tool)
    print("tool_calls:", calls)


if __name__ == "__main__":
    main()
