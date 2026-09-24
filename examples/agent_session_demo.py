"""
自研 Agent 5 行样板：AgentSession ?session_id / decision_id / 熔断?
违行::
    python examples/agent_session_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus.integrations import AgentSession
from laya_thalamus.schemas import DecisionAction


def main() -> None:
    session = AgentSession()  # mock if laya missing
    query = "计算 123*456"
    d = session.next(query)
    print(d.decision_id, d.summary())
    if d.action == DecisionAction.CALL_TOOL:
        d = session.next(query, tool_result="56088", idempotency_key="after-calc")
        print(d.decision_id, d.summary())
    print("snapshot:", session.snapshot())


if __name__ == "__main__":
    main()
