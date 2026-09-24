"""Optional dependency helpers for extras: api / laya / llm / -
**Unstable**. See ``API.md``.
"""

from __future__ import annotations

INSTALL_HINTS = {
    "api": 'pip install "laya-thalamus[api]"',
    "laya": 'pip install "laya-thalamus[laya]"',
    "llm": 'pip install "laya-thalamus[llm]"',
    "otel": 'pip install "laya-thalamus[otel]"',
    "dev": 'pip install "laya-thalamus[dev]"',
}


def missing_extra(name: str, feature: str, *, underlying: BaseException | None = None) -> ImportError:
    hint = INSTALL_HINTS.get(name, f'pip install "laya-thalamus[{name}]"')
    msg = f"{feature} requires the `{name}` extra. Install with: {hint}"
    err = ImportError(msg)
    if underlying is not None:
        err.__cause__ = underlying
    return err


def require_api() -> None:
    """Raise ImportError with install hint if FastAPI/uvicorn are missing."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        raise missing_extra("api", "HTTP API / `thalamus serve`", underlying=e) from e
