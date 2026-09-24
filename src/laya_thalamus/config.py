"""Configuration loading and runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    backend: str = "auto"  # auto | mock | laya | llm
    # HF repo id OR local path (hub root / single checkpoint / .../multilingual)
    name: str = "convaiinnovations/laya-multilingual"
    device: str = "auto"
    timeout_ms: int = 2000  # System-1 budget; see POLICY.md
    hard_fail_on_timeout: bool = True
    hard_fail_on_exception: bool = True
    # Discard primary output when over timeout - default only real Laya, not mock/llm*
    timeout_hard_fail_backends: list[str] = Field(default_factory=lambda: ["laya"])


class ThresholdConfig(BaseModel):
    choice_confidence: float = 0.55
    noul_confidence: float = 0.60
    noul_true_threshold: float = 0.5
    score_low: float = 3.0
    score_high: float = 7.0
    top_k: int = 3


class CircuitBreakerConfig(BaseModel):
    max_tool_rounds: int = 5
    max_same_tool_repeats: int = 2


class ContextConfig(BaseModel):
    max_chars: int = 1500
    keep_tail: bool = True


class ToolsConfig(BaseModel):
    max_count: int = 20
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)


class LLMFallbackConfig(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_s: float = 20.0


class FallbackConfig(BaseModel):
    enabled: bool = True
    # auto: LLM if env/config present, else heuristic
    # llm | heuristic | none
    mode: str = "auto"
    llm: LLMFallbackConfig = Field(default_factory=LLMFallbackConfig)


class CacheConfig(BaseModel):
    """Short-TTL decision cache keyed by query+tools fingerprint."""

    enabled: bool = False
    ttl_s: float = 30.0
    max_size: int = 1024


class LoggingConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    json_path: str | None = None
    # Emit OpenTelemetry spans when opentelemetry-api is installed ([otel] extra)
    otel: bool = False


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class RouterConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def update(self, **overrides: Any) -> "RouterConfig":
        """Return a new config with nested dict/kwargs overrides applied."""
        data = self.model_dump()
        _deep_merge(data, overrides)
        return RouterConfig.model_validate(data)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _default_yaml_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "default.yaml"


def load_config(path: str | Path | None = None) -> RouterConfig:
    """Load RouterConfig from YAML. Defaults to packaged default.yaml.

    Relative ``model.name`` filesystem paths are resolved against the config
    file's parent directory (so ``config/local.yaml`` can use ``../models/LAYA``).
    """
    cfg_path = Path(path) if path else _default_yaml_path()
    if not cfg_path.exists():
        return RouterConfig()
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = RouterConfig.model_validate(raw)
    return _resolve_model_path_against_config(cfg, cfg_path.parent)


def _resolve_model_path_against_config(
    cfg: RouterConfig, base_dir: Path
) -> RouterConfig:
    name = (cfg.model.name or "").strip().strip("\"'")
    if not name:
        return cfg
    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        return cfg
    # HF repo ids look like "org/name" and usually don't exist on disk
    rooted = (base_dir / candidate).resolve()
    if rooted.is_dir():
        return cfg.update(model={"name": str(rooted)})
    # Also try CWD-relative (unchanged name); leave as-is for Hub ids
    cwd_try = Path(name).expanduser()
    try:
        if cwd_try.exists() and cwd_try.is_dir():
            return cfg.update(model={"name": str(cwd_try.resolve())})
    except OSError:
        pass
    return cfg


def dump_config(config: RouterConfig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.model_dump(),
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def clone_config(config: RouterConfig) -> RouterConfig:
    return RouterConfig.model_validate(deepcopy(config.model_dump()))
