"""
LangChain 接入：用包内 ``LayaToolSelector``（丝强制安装 langchain）?
违行::
    python examples/langchain_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_thalamus.integrations import LayaToolSelector


class Search:
    name = "Search"
    description = "Search the internet for current information"


class Calculator:
    name = "Calculator"
    description = "Useful for math expressions"


class PythonREPL:
    name = "PythonREPL"
    description = "Execute python code"


def main() -> None:
    picker = LayaToolSelector(tools=[Search(), Calculator(), PythonREPL()])
    for q in ["What is 2+2?", "Who won the latest World Cup?", "你好"]:
        tool = picker.pick(q)
        print(f"{q!r} ->next_tool={tool}")


if __name__ == "__main__":
    main()
