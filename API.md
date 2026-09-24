# Stable public API

This document defines the **supported** surface of `laya-thalamus` for 0.x.
Anything not listed here is **unstable**: it may change or be removed without a
deprecation period. Prefer imports from the top-level package or
`laya_thalamus.integrations`.

SemVer note: before 1.0, minor bumps may still tighten behavior; we will not
break the signatures below without a CHANGELOG entry and a minor version bump.

---

## Stable (import freely)

| Symbol | Module | Notes |
|--------|--------|--------|
| `AgentRouter` | `laya_thalamus` | Primary entry: `route` / `aroute`, `route_request` / `aroute_request`, `reset_session`, `session_snapshot`, `clear_decision_cache`, `register_tool`, `health`, `metrics`, `metrics_prometheus` |
| `RouterConfig` / `load_config` | `laya_thalamus` | YAML / env-backed config |
| `RouteRequest` / `RouteDecision` / `ToolSpec` / `DecisionAction` | `laya_thalamus` | Pydantic schemas |
| `ToolRegistry` | `laya_thalamus` | Dynamic tool registration |
| `tools_from_openai` / `tools_to_openai` | `laya_thalamus` | OpenAI tools 鈫?`ToolSpec` |
| `tools_from_json_schema` / `tools_to_json_schema` | `laya_thalamus` | JSON Schema 鈫?`ToolSpec` |
| `__version__` | `laya_thalamus` | Matches installed package metadata |

### Integrations (stable helpers)

Import from `laya_thalamus.integrations`:

| Symbol | Role |
|--------|------|
| `AgentSession` | Custom agent loop + `session_id` |
| `LayaToolSelector` | LangChain-style `pick` / `as_runnable` |
| `route_openai_turn` / `decision_to_tool_calls` / `last_user_text` | OpenAI chat + tools turn |

### CLI (stable commands)

```text
thalamus serve | demo | health | probe-laya | compare
```

`serve` requires the `[api]` extra. `probe-laya` / `compare` use packaged code
and do **not** require a git checkout.

---

## Unstable (do not rely on)

These modules are for internal use or advanced customization. Treat names,
signatures, and layout as private:

- `laya_thalamus.backend` (except constructing backends via `AgentRouter` / config)
- `laya_thalamus.fallback`, `llm`, `circuit`, `context`, `hooks`
- `laya_thalamus.idempotency`, `extras`, `logging_util`
- `laya_thalamus.eval.*` (eval helpers may move; CLI `compare` is the supported entry)
- `laya_thalamus.api` (HTTP shapes may evolve; install `[api]` knowing this)
- `laya_thalamus.probe` (prefer `thalamus probe-laya`)
- Private helpers prefixed with `_`

Repo-only paths (`examples/`, top-level `eval/` scripts, `config/*.yaml` samples)
are convenience for contributors 鈥?not part of the install contract beyond what
is packaged under `laya_thalamus/resources/`.

---

## Compatibility promises

1. **`AgentRouter.route(...)`** keeps keyword args: `query`, `context`, `tools`,
   `tool_result`, `session_id`, `history`, `config`, `allowed_tools`,
   `idempotency_key`. Return type remains `RouteDecision`.
2. **`RouteDecision`** keeps fields used by integrators: `action`,
   `selected_tool`, `confidence`, `decision_id`, `session_id`,
   `idempotent_replay`, `cache_hit`, `information_sufficient`, `credibility_score`,
   `fallback_used`, `backend`, `summary()`.
3. **`ToolSpec`** keeps `name`, `description`, optional `parameters` / `group` /
   `dangerous`.
4. Extras names `[api]`, `[laya]`, `[llm]`, `[otel]`, `[dev]`, `[all]` stay reserved.
5. Timeout / degrade / cache contracts: see [POLICY.md](POLICY.md).
