"""Plugin-style tool registry."""

from __future__ import annotations

from laya_thalamus.exceptions import ToolLimitExceeded
from laya_thalamus.schemas import ToolSpec

# Built-in starter tools -criteria text is the main signal for Laya choice.
DEFAULT_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="none",
        description=(
            "Finish without tools: greetings, thanks, opinions, translation, "
            "or when a good tool result is already available."
        ),
    ),
    ToolSpec(
        name="web_search",
        description=(
            "Public internet lookup: news, weather, people, definitions "
            "(什么是 / what is / who is), current world facts."
        ),
    ),
    ToolSpec(
        name="calculator",
        description=(
            "Numeric arithmetic only: +, -, *, /, %, parentheses, percentages."
        ),
    ),
    ToolSpec(
        name="code_interpreter",
        description=(
            "Python programming: write functions, algorithms, sorting, "
            "median, pandas, data transforms, execute scripts."
        ),
        dangerous=True,
    ),
    ToolSpec(
        name="rag_retrieve",
        description=(
            "Internal company knowledge base / employee handbook / "
            "policy docs: 年假, 报销, 考勤, 内部资料, 员工手册."
        ),
    ),
]


class ToolRegistry:
    """Dynamic tool registration. Laya chooses among descriptions -no retraining."""

    def __init__(self, max_count: int = 20) -> None:
        self.max_count = max_count
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec | dict) -> None:
        if not isinstance(tool, ToolSpec):
            tool = ToolSpec.model_validate(tool)
        if tool.name not in self._tools and len(self._tools) >= self.max_count:
            raise ToolLimitExceeded(
                f"Tool count exceeds Laya choice limit ({self.max_count}). "
                "Split tools into groups and route in stages."
            )
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolSpec | dict]) -> None:
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear(self) -> None:
        self._tools.clear()

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def as_criteria(self) -> dict[str, str]:
        """Map tool name ->short description for Laya choice criteria."""
        return {t.name: t.description for t in self._tools.values()}

    def filter(
        self,
        *,
        blacklist: list[str] | None = None,
        whitelist: list[str] | None = None,
        allowed: list[str] | None = None,
    ) -> list[ToolSpec]:
        tools = self.list()
        bl = set(blacklist or [])
        tools = [t for t in tools if t.name not in bl]
        if whitelist:
            wl = set(whitelist)
            tools = [t for t in tools if t.name in wl]
        if allowed is not None:
            al = set(allowed)
            tools = [t for t in tools if t.name in al]
        return tools

    def ensure_defaults(self) -> None:
        if not self._tools:
            self.register_many(DEFAULT_TOOLS)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
