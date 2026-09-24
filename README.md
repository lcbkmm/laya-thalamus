<div align="center">

# Thalamus

**System-1 decision middleware for AI Agents**  
Powered by [Laya](https://github.com/NandhaKishorM/laya) · Fast local routing · LLM fallback when unsure

[English](#thalamus) · [中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-2ea44f)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pytest%203.10--3.12-informational)](.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip-laya--thalamus-blue)](https://pypi.org/project/laya-thalamus/)

</div>

---

**Thalamus** (named after the brain’s fast sensory relay) pulls *which tool / is info enough / can I trust the result* out of the LLM loop — local System-1 for the hot path, big models for hard cases.

| | |
|:--|:--|
| **Drop-in** | OpenAI tools · LangChain · custom Agents |
| **Primitives** | `choice` · `noul` · `score` |
| **Production** | Fallback · circuit breaker · cache · idempotency · Prometheus / OTel |

## Install

> Not on PyPI yet — install from GitHub:

```bash
pip install "laya-thalamus[laya] @ git+https://github.com/lcbkmm/laya-thalamus.git"
```

```python
from laya_thalamus import AgentRouter
router = AgentRouter()
d = router.route("What is 123 * 456?")
print(d.selected_tool, d.action, d.summary())
```

---

## Benchmarks

Gold set **n=56** (incl. post-tool turns). Same data for every row.

| Backend | Tool Acc | CPU latency | GPU | Role |
|---------|---------:|------------:|:---:|------|
| **Laya + fallback** | **64.3%** | **~0.30 s** | — | **Product path** |
| Laya bare | 55.4% | ~0.47 s | — | Fallback off |
| deepseek-v3.1 | 87.5% | ~2.0 s | — | LLM-as-router baseline |
| Random | 21.4% | — | — | Chance floor |
| Keyword mock | 89.3% | &lt;1 ms | — | CI / demo only — **not** a product metric |

<details>
<summary><b>How to read these numbers</b></summary>

| Row | Meaning |
|-----|---------|
| **Laya + fallback** | Low-confidence / hard-fail → LLM if `LAYA_LLM_*` set, else heuristic. **+9 pp** vs bare on this set. |
| **Laya bare** | System-1 alone — honest local ceiling. |
| **CPU / GPU** | CPU measured on publish host (~0.5 s warm; cold can approach ~1 s). GPU blank (no CUDA) — fill with `--device cuda`. |
| **mock** | Keyword heuristic tuned to this set. Do not cite as model skill. |

Reproduce: [`eval/compare_report.json`](eval/compare_report.json) · `thalamus compare --with-fallback --skip-llm`

</details>

> Goal: **System-1 on the hot path**, fallback on edges — not “always beat the LLM”.

---

## Why Thalamus

| Pain | Without | With Thalamus |
|------|---------|---------------|
| Ask a big model every “call a tool?” | Latency + tokens | `choice` in ms–seconds |
| Enough info? Trust the result? | Brittle prompts | `noul` / `score` + thresholds |
| Tool loops / same-tool spam | DIY per stack | `session_id` circuit + `decision_id` |
| OpenAI / LangChain wiring | Hand-roll adapters | `integrations.*` in ~5 lines |
| Timeout half-answers as truth | Silent mis-routes | [POLICY.md](POLICY.md) hard-fail → LLM → heuristic |

---

## Laya vs Thalamus

| | [Laya](https://github.com/NandhaKishorM/laya) | **Thalamus** (`laya-thalamus`) |
|--|--|--|
| Role | System-1 **inference core** | Agent **decision middleware** |
| API | `predict(...)` | `router.route(...)` + HTTP / CLI / adapters |
| Gaps you own | Fallback, circuit, eval | Built-in |
| Fit | Call-site locked | OpenAI · LangChain · custom Session |

Laya is the engine. Thalamus is the steering wheel and seatbelts.

---

## Architecture

```mermaid
flowchart TB
  A[User / Agent] --> R["AgentRouter.route()"]
  R --> S[session · cache · circuit · otel]
  S --> L[LayaBackend<br/>System-1]
  S --> M[LLMRouterBackend<br/>compare / optional]
  S --> K[MockBackend<br/>CI only]
  L -->|hard-fail / low-conf| F[Fallback<br/>LLM → heuristic]
  F --> D[RouteDecision]
  L --> D
  M --> D
  K --> D
  D --> O["action · tool · noul · score · decision_id"]
```

Outputs **routing decisions only** — never the final answer text.

---

## Quick start

```bash
pip install "laya-thalamus[laya] @ git+https://github.com/lcbkmm/laya-thalamus.git"   # real System-1 (recommended)
# pip install "laya-thalamus @ git+https://github.com/lcbkmm/laya-thalamus.git"        # library only
# pip install "laya-thalamus[api] @ git+https://github.com/lcbkmm/laya-thalamus.git"   # HTTP server
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

**Plug into existing stacks**

```python
from laya_thalamus.integrations import (
    AgentSession, LayaToolSelector, route_openai_turn, decision_to_tool_calls,
)

d = route_openai_turn(router, messages, openai_tools, session_id="s1")
calls = decision_to_tool_calls(d)          # None → let the LLM answer

tool = LayaToolSelector(tools=lc_tools).pick(query, intermediate_steps)
d = AgentSession(router).next(query)
```

```bash
thalamus demo --backend mock
thalamus probe-laya --model /path/to/LAYA
thalamus compare --with-fallback --skip-llm
thalamus serve --port 8080                 # needs [api]
```

**Docs:** [API.md](API.md) · [POLICY.md](POLICY.md) · [FEATURES.md](FEATURES.md) · [`examples/`](examples/)

---

## FAQ

<details>
<summary><b>Is this another Agent framework?</b></summary>

No. Decision middleware only: next action is `call_tool` / `answer` / `terminate`.
</details>

<details>
<summary><b>Why “Thalamus”?</b></summary>

The thalamus is the brain’s fast signal relay — this project is that relay for Agent System-1 routing.
</details>

<details>
<summary><b>Why not ask GPT every time?</b></summary>

Often more accurate, also expensive and slow. Thalamus keeps System-1 on the hot path (~0.5 s CPU here) and falls back when needed.
</details>

<details>
<summary><b>Why is bare Laya ~55%?</b></summary>

Weak on post-tool “stop calling tools” turns. Enable fallback → **64.3%** here (higher with a real LLM configured).
</details>

<details>
<summary><b>Is mock 89% inflated?</b></summary>

Yes — heuristic. **Not a product metric.**
</details>

<details>
<summary><b>Relation to the official <code>laya</code> package?</b></summary>

`laya` = model runtime. This repo = engineering middleware via optional extra `[laya]`.
</details>

<details>
<summary><b>Package / import / CLI names?</b></summary>

PyPI `laya-thalamus` · import `laya_thalamus` · CLI `thalamus` / `laya-thalamus`. Core deps: `pydantic` + `pyyaml` only (FastAPI optional).
</details>

---

## Roadmap & contribute

**0.1 shipped** — choice / noul / score · fallback · circuit · session / idempotency / cache · adapters · async · Prometheus / OTel · CI · packaged eval  

**Next** — larger gold set · Redis cache · decision UI · multi-tenant · 1.0 API freeze  

See [CHANGELOG.md](CHANGELOG.md).

1. Read [API.md](API.md) + [POLICY.md](POLICY.md)  
2. `pip install -e ".[dev]"` → `pytest -q`  
3. Keep `eval/traces.json` ↔ `src/laya_thalamus/resources/traces.json` in sync  
4. Don’t break `AgentRouter.route` signatures  

---

## License

[Apache-2.0](LICENSE) · Weights: [Laya](https://github.com/NandhaKishorM/laya) · [laya.convaiinnovations.com](https://laya.convaiinnovations.com/)

<p align="center">
  <a href="https://www.star-history.com/#lcbkmm/laya-thalamus&Date">
    <img src="https://api.star-history.com/svg?repos=lcbkmm/laya-thalamus&type=Date" alt="Star History" width="520" />
  </a>
</p>

<p align="center"><i>If Thalamus saved you a pile of routing prompts — give it a star.</i></p>
