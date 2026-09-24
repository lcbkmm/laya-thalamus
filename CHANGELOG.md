# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the **stable public API** listed in [API.md](API.md).

## [Unreleased]

### Changed

- **Rename**: package `laya-agent-router` → **`laya-thalamus`** (import `laya_thalamus`, CLI `thalamus` / `laya-thalamus`)
- Prometheus metric prefix `laya_router_*` → `thalamus_*`

### Added

- Async `AgentRouter.aroute` / `aroute_request` and async FastAPI endpoints
- Decision cache (`cache.enabled` / TTL fingerprint of query+tools)
- [POLICY.md](POLICY.md) timeout/degrade contract + unit test matrix
- Prometheus-style metrics (`thalamus_*`) and `GET /metrics/prometheus`
- Optional OpenTelemetry spans (`logging.otel`, `[otel]` extra)

### Added (prior)

- Optional `[llm]` extra (`httpx`) for custom HTTP clients; built-in LLM client stays on stdlib `urllib`
- GitHub Actions CI (pytest on Python 3.10鈥?.12) and PyPI publish workflow
- [API.md](API.md) 鈥?locked stable surface vs unstable internals

## [0.1.0] 鈥?2026-09-23

### Added

- Core `AgentRouter` with Laya `choice` / `noul` / `score`, mock backend, LLM fallback
- Extras: `[api]`, `[laya]`, `[dev]`, `[all]`; core depends only on `pydantic` + `pyyaml`
- Packaged gold traces + `thalamus compare` / `probe-laya` (no source-tree paths)
- Integrations: OpenAI function-calling, LangChain duck-typed selector, `AgentSession`
- Schema converters: OpenAI / JSON Schema 鈫?`ToolSpec`
- Session semantics: `session_id`, `decision_id`, idempotency cache
