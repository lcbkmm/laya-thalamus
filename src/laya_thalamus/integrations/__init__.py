"""First-class adapters for plugging Laya into existing Agent stacks.

* :mod:`laya_thalamus.integrations.openai_fc` -OpenAI function-calling
* :mod:`laya_thalamus.integrations.langchain` -LangChain tools / Runnable-style
* :mod:`laya_thalamus.integrations.agent` -5-line custom Agent loop helper

Schema conversion also lives in :mod:`laya_thalamus.schema_tools`.
"""

from __future__ import annotations

from laya_thalamus.integrations.agent import AgentSession
from laya_thalamus.integrations.langchain import LayaToolSelector
from laya_thalamus.integrations.openai_fc import (
    decision_to_tool_calls,
    last_user_text,
    route_openai_turn,
)
from laya_thalamus.schema_tools import (
    tools_from_json_schema,
    tools_from_langchain,
    tools_from_openai,
    tools_to_json_schema,
    tools_to_openai,
)

__all__ = [
    "AgentSession",
    "LayaToolSelector",
    "decision_to_tool_calls",
    "last_user_text",
    "route_openai_turn",
    "tools_from_json_schema",
    "tools_from_langchain",
    "tools_from_openai",
    "tools_to_json_schema",
    "tools_to_openai",
]
