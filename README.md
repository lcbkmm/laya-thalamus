# Thalamus | Laya-powered System-1 Router for AI Agents

**English** | [中文](README.zh-CN.md)

> **Thalamus** — the brain’s fast relay for sensory signals. Here it is the System-1 decision layer for Agents.

**Pull “which tool / is info enough / can I trust the result” out of the LLM loop — sub-second routing; leave hard reasoning to the big model.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pytest%203.10--3.12-informational.svg)](.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip-laya--thalamus-blue.svg)](https://pypi.org/project/laya-thalamus/)

One-line drop-in · OpenAI / LangChain / custom Agents · real LLM fallback · comparable evals

```bash
pip install "laya-thalamus[laya]"
```

```python
from laya_thalamus import AgentRouter
```

---

## Benchmarks (numbers first)

Human gold set **n=56** (incl. post-tool turns). Same dataset.

| Backend | Tool Acc | Latency (CPU) | Latency (GPU) | Notes |
|---------|---------:|--------------:|--------------:|-------|
| **Laya + fallback** | **64.3%** | **~0.30 s** | — | **Product path** · heuristic fallback here (no LLM keys on measure host) |
| Laya bare (System-1) | 55.4% | ~0.47 s | — | Fallback **off** · CPU weights |
| deepseek-v3.1 (LLM router) | 87.5% | ~2.0 s | — | Accurate but costly/slow — comparison baseline |
| Random | 21.4% | — | — | Chance lower bound |
| Keyword mock | 89.3% | \<1 ms | — | **Heuristic, not model skill** — CI / Demo only |

> Raw report: [`eval/compare_report.json`](eval/compare_report.json).  
> Reproduce: `thalamus compare --with-fallback --skip-llm` · GPU: add `--device cuda`.  
> **Do not treat mock accuracy as a product metric.** See [How to read the eval](#how-to-read-the-eval).

**Takeaway:** Bare Laya is the fast local prior; **turn on fallback** for the real product path (LLM when configured, else heuristic). GPU latency is empty on the publish host (no CUDA) — fill it with `--device cuda`.

---

## Pain points

| Pain | Without Thalamus | With Thalamus |
|------|------------------|---------------|
| Ask a big model every “call a tool?” | High latency & token cost | `choice` in ms–seconds |
| Enough info? Trust the tool result? | Brittle prompt rules | `noul` / `score` + thresholds |
| Tool loops / same-tool spam | DIY per framework | `session_id` circuit + `decision_id` |
| LangChain / OpenAI function-calling | Wrap the model yourself | `integrations.*` in ~5 lines |
| Timeout returns half-baked as truth | Silent mis-routes | [POLICY.md](POLICY.md): hard-fail → LLM → heuristic |
| “It’s fast” with no proof | No apples-to-apples table | Packaged gold + `thalamus compare` |

---

## vs upstream Laya

| | [Laya](https://github.com/NandhaKishorM/laya) runtime | **Thalamus (`laya-thalamus`)** |
|--|--|--|
| Role | System-1 **inference core** | Agent **decision middleware** (thalamic relay) |
| You get | `predict(...)` | `router.route(...)` + HTTP / CLI / adapters |
| Production gaps | You own fallback, circuit, eval | Built-in fallback, circuit, cache, idempotency, Prometheus, OTel |
| Integration | Call-site locked | OpenAI tools / LangChain / custom Session |

**Relationship:** Laya is the engine; Thalamus is the steering wheel and seatbelts. `pip install laya` for the engine; `laya-thalamus` for the middleware.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
  User / Agent ──►  │         AgentRouter.route()         │
                    │  session · cache · circuit · otel   │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
         LayaBackend         LLMRouterBackend      MockBackend
         (System-1)          (compare / optional)  (CI only)
              │
              └──── hard-fail / low-conf ──► Fallback (LLM → heuristic)
                                   │
                                   ▼
                          RouteDecision
              action · tool · noul · score · decision_id
```

Primitives: **choice** · **noul** · **score**. Routing decisions only — no final answer text.

---

## 30-second start

```bash
pip install "laya-thalamus[laya]"   # real System-1 (strongly recommended)
# library only: pip install laya-thalamus
# HTTP:         pip install "laya-thalamus[api]"
```

```python
from laya_thalamus import AgentRouter
from laya_thalamus.config import RouterConfig

cfg = RouterConfig()
cfg.model.backend = "laya"       # auto | laya | mock | llm
cfg.model.timeout_ms = 30_000    # raise for local CPU cold start
router = AgentRouter(config=cfg)

d = router.route("What is 123*456?", session_id="user-42")
print(d.decision_id, d.selected_tool, d.summary())
```

**Drop into existing stacks:**

```python
from laya_thalamus.integrations import (
    AgentSession, LayaToolSelector, route_openai_turn, decision_to_tool_calls,
)

d = route_openai_turn(router, messages, openai_tools, session_id="s1")
calls = decision_to_tool_calls(d)   # None → let the LLM answer

tool = LayaToolSelector(tools=lc_tools).pick(query, intermediate_steps)

session = AgentSession(router)
d = session.next(query)
```

```bash
thalamus demo --backend mock
thalamus probe-laya --model /path/to/LAYA
thalamus compare --skip-llm
thalamus serve --port 8080          # needs [api]
```

Docs: [API.md](API.md) · [POLICY.md](POLICY.md) · [`examples/`](examples/)

---

## How to read the eval

| Row | Meaning |
|-----|---------|
| **laya+fallback** | **Product path.** Low-confidence / hard-fail → LLM if `LAYA_LLM_*` set, else heuristic. +9pp tool acc vs bare on this set. |
| **laya bare** | System-1 only (fallback off). Honest ceiling of the local model alone. |
| **Latency (CPU)** | Measured on publish host (~0.5 s warm; colder runs can approach ~1 s). |
| **Latency (GPU)** | Not measured here (no CUDA). Run `thalamus compare --device cuda --with-fallback --skip-llm`. |
| **llm:\*** | Chat model as *primary* router — cost/latency baseline, not the Thalamus hot path. |
| **mock** | Keyword heuristic. Pretty ≠ production. |
| **random** | Chance floor. |

The goal is not “always beat the LLM”, but: **System-1 on the hot path; fallback on edges / low confidence**.

---

## FAQ

**Q: Is Thalamus another Agent framework?**  
A: No. Decision middleware only: next action is `call_tool` / `answer` / `terminate`.

**Q: Why “Thalamus”?**  
A: The thalamus is the brain’s fast signal relay; this project is that relay for Agent System-1 routing.

**Q: Why not ask GPT every time to pick a tool?**  
A: Often more accurate, also expensive and slow. Thalamus holds System-1 on the hot path (~0.5 s CPU here) and falls back when needed.

**Q: Why is bare Laya only ~55%?**  
A: Weak on post-tool “stop calling tools” turns. Enable fallback (`laya+fallback` → **64.3%** here; higher when an LLM is configured). See the benchmark table.

**Q: Is mock 89% inflated?**  
A: Yes — heuristic. **Not a product metric.** Called out in the table.

**Q: Relation to the official `laya` package?**  
A: `laya` is the model runtime; this repo is engineering middleware on optional extra `[laya]`.

**Q: Do I need FastAPI?**  
A: No. Core deps are `pydantic` + `pyyaml` only.

**Q: What are the names?**  
A: PyPI `laya-thalamus`, import `laya_thalamus`, CLI `thalamus` / `laya-thalamus`.

**Q: Stable API?**  
A: See [API.md](API.md).

---

## Roadmap

**Shipped (0.1)** — choice/noul/score · fallback · circuit · session/idempotency/cache · adapters · async · Prometheus/OTel · CI · packaged eval  

**Next** — larger gold set · Redis cache · decision UI · multi-tenant · 1.0 API freeze  

See [FEATURES.md](FEATURES.md) · [CHANGELOG.md](CHANGELOG.md).

---

## Contribute

1. Read [API.md](API.md) and [POLICY.md](POLICY.md)  
2. `pip install -e ".[dev]"` → `pytest -q`  
3. Keep `eval/traces.json` in sync with `src/laya_thalamus/resources/traces.json`  
4. Don’t break `AgentRouter.route` signatures  

Welcome: gold-set growth, adapters, docs, profiling.

---

## License

[Apache-2.0](LICENSE). System-1 weights: [Laya](https://github.com/NandhaKishorM/laya) / [laya.convaiinnovations.com](https://laya.convaiinnovations.com/).

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lcbkmm/laya-thalamus&type=Date)](https://www.star-history.com/#lcbkmm/laya-thalamus&Date)

If Thalamus saved you a pile of routing prompts — give it a **Star**.
