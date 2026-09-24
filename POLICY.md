# Timeout / degrade / cache policy

Contract for production routing behavior. Implemented in
`laya_thalamus.policy` and enforced by `AgentRouter`.

## Timeout hard-fail

| Knob | Default | Meaning |
|------|---------|---------|
| `model.timeout_ms` | `2000` | Latency budget for primary System-1 backends |
| `model.hard_fail_on_timeout` | `true` | When over budget, discard primary choice/noul/score |
| `model.timeout_hard_fail_backends` | `["laya"]` | Only these backends (exact or prefix, e.g. `llm` 鈫?`llm:x`) |
| `model.hard_fail_on_exception` | `true` | Treat predict exceptions as hard-fail |

**Why `llm*` is excluded by default:** chat models used as routers often exceed
2s; applying the Laya wall-clock budget wiped all LLM columns (all-`none`
collapse). Use `fallback.llm.timeout_s` for LLM HTTP timeouts instead.

Matrix (unit-tested):

| backend | latency > timeout_ms | hard_fail? |
|---------|----------------------|------------|
| `laya` | yes | yes |
| `mock` | yes | no |
| `llm` / `llm:foo` | yes | no |
| any | exception | yes (if `hard_fail_on_exception`) |

After hard-fail, if `fallback.enabled`, the fallback chain fills tool / noul / score.

## Fallback

| `fallback.mode` | Behavior |
|-----------------|----------|
| `auto` | LLM if env/config present, else heuristic |
| `llm` | LLM only (then heuristic if chain secondary) |
| `heuristic` | keyword heuristic |
| `none` / `enabled: false` | no fallback |

## Decision cache vs idempotency

| Feature | Key | Purpose |
|---------|-----|---------|
| Decision cache | hash(query + tools + tool_result + context) | Short TTL; cut latency/cost |
| Idempotency | `(session_id, idempotency_key)` | Exact replay of the same client request |

Enable cache: `cache.enabled: true`, `cache.ttl_s`, `cache.max_size`.

## Observability

- Structured log fields: see `logging_util.LOG_FIELDS`
- JSON metrics: Prometheus-style names (`thalamus_*`) + legacy aliases
- Scrape: `GET /metrics/prometheus`
- Spans: `logging.otel: true` + `pip install "laya-thalamus[otel]"`
