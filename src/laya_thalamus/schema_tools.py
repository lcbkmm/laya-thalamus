"""OpenAI function-calling / JSON Schema ->ToolSpec conversion."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from laya_thalamus.schemas import ToolSpec

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}


def tool_to_json_schema(tool: ToolSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Single tool ->JSON Schema object (name/description/parameters)."""
    spec = tool if isinstance(tool, ToolSpec) else ToolSpec.model_validate(tool)
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters or dict(_EMPTY_OBJECT_SCHEMA),
    }


def tools_to_json_schema(
    tools: Sequence[ToolSpec | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [tool_to_json_schema(t) for t in tools if _name_of(t) != "none"]


def tools_from_json_schema(
    schemas: Sequence[Mapping[str, Any]],
    *,
    include_none: bool = False,
) -> list[ToolSpec]:
    """JSON Schema tool defs (name + description + parameters) ->ToolSpec list."""
    out: list[ToolSpec] = []
    for raw in schemas:
        name = str(raw.get("name") or raw.get("title") or "").strip()
        if not name:
            continue
        desc = str(raw.get("description") or name)
        params = raw.get("parameters") or raw.get("schema")
        if params is not None and not isinstance(params, dict):
            params = None
        out.append(ToolSpec(name=name, description=desc, parameters=params))
    if include_none and not any(t.name == "none" for t in out):
        out.insert(
            0,
            ToolSpec(
                name="none",
                description="Do not call a tool; answer the user directly.",
            ),
        )
    return out


def tools_to_openai(
    tools: Sequence[ToolSpec | Mapping[str, Any]],
    *,
    style: str = "tools",
) -> list[dict[str, Any]]:
    """ToolSpec ->OpenAI Chat Completions ``tools`` (or legacy ``functions``).

    ``style="tools"`` (default)::

        {"type": "function", "function": {"name", "description", "parameters"}}

    ``style="functions"`` (legacy)::

        {"name", "description", "parameters"}
    """
    items: list[dict[str, Any]] = []
    for t in tools:
        if _name_of(t) == "none":
            continue
        js = tool_to_json_schema(t)
        if style == "functions":
            items.append(js)
        else:
            items.append({"type": "function", "function": js})
    return items


def tools_from_openai(
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    include_none: bool = True,
) -> list[ToolSpec]:
    """OpenAI ``tools`` / ``functions`` / bare function dicts ->ToolSpec list."""
    if not tools:
        return tools_from_json_schema([], include_none=include_none)

    normalized: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), Mapping):
            normalized.append(dict(item["function"]))
        elif "function" in item and isinstance(item["function"], Mapping):
            normalized.append(dict(item["function"]))
        elif "name" in item:
            normalized.append(dict(item))
    return tools_from_json_schema(normalized, include_none=include_none)


def tools_from_langchain(
    tools: Iterable[Any],
    *,
    include_none: bool = True,
) -> list[ToolSpec]:
    """Duck-typed LangChain ``BaseTool`` (or anything with name/description) ->ToolSpec.

    Does **not** require ``langchain`` to be installed.
    """
    schemas: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if not name:
            continue
        description = getattr(t, "description", None) or str(name)
        parameters = _langchain_parameters(t)
        schemas.append(
            {"name": str(name), "description": str(description), "parameters": parameters}
        )
    return tools_from_json_schema(schemas, include_none=include_none)


def _langchain_parameters(tool: Any) -> dict[str, Any] | None:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return None
    if isinstance(args_schema, dict):
        return args_schema
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass
    if hasattr(args_schema, "schema"):
        try:
            return args_schema.schema()
        except Exception:
            pass
    return None


def _name_of(tool: ToolSpec | Mapping[str, Any]) -> str:
    if isinstance(tool, ToolSpec):
        return tool.name
    return str(tool.get("name") or "")
