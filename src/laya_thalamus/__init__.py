"""Laya Thalamus -- System-1 decision middleware for Agents.

Stable public surface is documented in ``API.md`` and listed in ``__all__``.
Other submodules may change without notice in 0.x (see module docstrings).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from laya_thalamus.config import RouterConfig, load_config
from laya_thalamus.router import AgentRouter
from laya_thalamus.schema_tools import (
    tools_from_json_schema,
    tools_from_openai,
    tools_to_json_schema,
    tools_to_openai,
)
from laya_thalamus.schemas import (
    DecisionAction,
    RouteDecision,
    RouteRequest,
    ToolSpec,
)
from laya_thalamus.tools import ToolRegistry

try:
    __version__ = _pkg_version("laya-thalamus")
except PackageNotFoundError:  # pragma: no cover - editable / bare PYTHONPATH
    __version__ = "0.1.0"

__all__ = [
    "AgentRouter",
    "RouterConfig",
    "load_config",
    "RouteDecision",
    "RouteRequest",
    "ToolSpec",
    "DecisionAction",
    "ToolRegistry",
    "tools_from_openai",
    "tools_to_openai",
    "tools_from_json_schema",
    "tools_to_json_schema",
    "__version__",
]
