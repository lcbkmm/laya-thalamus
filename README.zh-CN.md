<div align="center">

# Thalamus

**面向 AI Agent 的 System-1 决策中间件**  
基于 [Laya](https://github.com/NandhaKishorM/laya) · 本地快速路由 · 拿不准再降级 LLM

[English](README.md) · [中文](#thalamus)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-2ea44f)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pytest%203.10--3.12-informational)](.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip-laya--thalamus-blue)](https://pypi.org/project/laya-thalamus/)

</div>

---

**Thalamus（丘脑）**：大脑的快速信号中转站。这里把 Agent 里「选哪个工具 / 信息够不够 / 结果信不信」从 LLM 里拆出来——热路径走本地 System-1，难题留给大模型。

| | |
|:--|:--|
| **一行接入** | OpenAI tools · LangChain · 自研 Agent |
| **三种原语** | `choice` · `noul` · `score` |
| **可上线** | 降级 · 熔断 · 缓存 · 幂等 · Prometheus / OTel |

## 安装使用

```bash
pip install "laya-thalamus[laya]"
```

```python
from laya_thalamus import AgentRouter
router = AgentRouter()
d = router.route("计算 123 * 456 等于多少")
print(d.selected_tool, d.action, d.summary())
```

---

## 实测评测

人工金标 **n=56**（含工具执行后多轮）。同一数据集对比。

| Backend | Tool Acc | CPU 延迟 | GPU | 角色 |
|---------|---------:|---------:|:---:|------|
| **Laya + fallback** | **64.3%** | **~0.30 s** | — | **产品主路径** |
| Laya bare | 55.4% | ~0.47 s | — | 关闭 fallback |
| deepseek-v3.1 | 87.5% | ~2.0 s | — | LLM 当路由器的基线 |
| Random | 21.4% | — | — | 机会水平下限 |
| Keyword mock | 89.3% | &lt;1 ms | — | 仅 CI / Demo — **不是**产品指标 |

<details>
<summary><b>评测怎么读</b></summary>

| 行 | 含义 |
|----|------|
| **Laya + fallback** | 低置信 / 硬失败 → 有 `LAYA_LLM_*` 走 LLM，否则 heuristic。本集比裸 Laya **+9 pp**。 |
| **Laya bare** | 仅 System-1——看清本地模型单独上限。 |
| **CPU / GPU** | CPU 为发布机实测（热机 ~0.5 s；冷机可接近 ~1 s）。GPU 未测（无 CUDA）——用 `--device cuda` 补齐。 |
| **mock** | 贴着本集写的关键词启发式，禁止当模型能力。 |

原始报告：[`eval/compare_report.json`](eval/compare_report.json) · 复现：`thalamus compare --with-fallback --skip-llm`

</details>

> 目标是 **热路径用 System-1、边缘走 fallback**，不是「永远打败 LLM」。

---

## 解决什么痛点

| 痛点 | 没有 Thalamus | 有 Thalamus |
|------|---------------|-------------|
| 每次「要不要调工具」都问大模型 | 延迟高、Token 贵 | `choice` 毫秒～秒级 |
| 信息够不够、结果信不信 | Prompt 硬编码 | `noul` / `score` + 阈值 |
| 工具死循环 / 同工具狂调 | 各框架各写一套 | `session_id` 熔断 + `decision_id` |
| 接 LangChain / OpenAI FC | 自己包装 | `integrations.*` 约五行 |
| 超时半截结果当真理 | 静默错路由 | [POLICY.md](POLICY.md) 硬失败 → LLM → heuristic |

---

## 和原生 Laya

| | [Laya](https://github.com/NandhaKishorM/laya) | **Thalamus** (`laya-thalamus`) |
|--|--|--|
| 定位 | System-1 **推理内核** | Agent **决策中间件** |
| 你拿到的 | `predict(...)` | `router.route(...)` + HTTP / CLI / 适配器 |
| 生产缺口 | 自写降级、熔断、评测 | 开箱即有 |
| 接入 | 绑死调用方式 | OpenAI · LangChain · 自研 Session |

Laya 是引擎；Thalamus 是方向盘和保险杠。

---

## 架构

```mermaid
flowchart TB
  A[User / Agent] --> R["AgentRouter.route()"]
  R --> S[session · cache · circuit · otel]
  S --> L[LayaBackend<br/>System-1]
  S --> M[LLMRouterBackend<br/>对比 / 可选]
  S --> K[MockBackend<br/>仅 CI]
  L -->|hard-fail / low-conf| F[Fallback<br/>LLM → heuristic]
  F --> D[RouteDecision]
  L --> D
  M --> D
  K --> D
  D --> O["action · tool · noul · score · decision_id"]
```

只输出**路由决策**，不生成最终回答。

---

## 30 秒上手

```bash
pip install "laya-thalamus[laya]"    # 真 System-1（推荐）
# pip install laya-thalamus          # 只要库
# pip install "laya-thalamus[api]"   # HTTP
```

```python
from laya_thalamus import AgentRouter
from laya_thalamus.config import RouterConfig

cfg = RouterConfig()
cfg.model.backend = "laya"       # auto | laya | mock | llm
cfg.model.timeout_ms = 30_000    # 本地 CPU 冷启动建议加大
router = AgentRouter(config=cfg)

d = router.route("计算 123*456 等于多少", session_id="user-42")
print(d.decision_id, d.selected_tool, d.summary())
```

**插进现有栈**

```python
from laya_thalamus.integrations import (
    AgentSession, LayaToolSelector, route_openai_turn, decision_to_tool_calls,
)

d = route_openai_turn(router, messages, openai_tools, session_id="s1")
calls = decision_to_tool_calls(d)          # None → 直接让 LLM 回答

tool = LayaToolSelector(tools=lc_tools).pick(query, intermediate_steps)
d = AgentSession(router).next(query)
```

```bash
thalamus demo --backend mock
thalamus probe-laya --model /path/to/LAYA
thalamus compare --with-fallback --skip-llm
thalamus serve --port 8080                 # 需 [api]
```

**文档：** [API.md](API.md) · [POLICY.md](POLICY.md) · [FEATURES.md](FEATURES.md) · [`examples/`](examples/)

---

## FAQ

<details>
<summary><b>这是又一个 Agent 框架吗？</b></summary>

不是。只做决策中间件：下一步是 `call_tool` / `answer` / `terminate`。
</details>

<details>
<summary><b>为什么叫 Thalamus？</b></summary>

丘脑是大脑的快速信号中转站；本项目做 Agent 的 System-1 路由中转。
</details>

<details>
<summary><b>为什么不每次都问 GPT？</b></summary>

可以更准，但贵且慢。Thalamus 用 System-1 扛热路径（本机 CPU ~0.5 s），低置信再降级。
</details>

<details>
<summary><b>为什么裸 Laya 只有约 55%？</b></summary>

弱在「工具结果已够用该停」的多轮题。打开 fallback → 本表 **64.3%**（配置真 LLM 通常更高）。
</details>

<details>
<summary><b>mock 89% 是水分吗？</b></summary>

是启发式，**禁止当产品指标**。
</details>

<details>
<summary><b>和官方 <code>laya</code> 包什么关系？</b></summary>

`laya` = 模型运行时；本仓库 = 可选 extras `[laya]` 之上的工程化中间件。
</details>

<details>
<summary><b>包名 / import / CLI？</b></summary>

PyPI `laya-thalamus` · import `laya_thalamus` · CLI `thalamus` / `laya-thalamus`。核心依赖只有 `pydantic` + `pyyaml`（FastAPI 可选）。
</details>

---

## 路线图与贡献

**0.1 已交付** — choice / noul / score · 降级 · 熔断 · 会话 / 幂等 / 缓存 · 适配器 · 异步 · Prometheus / OTel · CI · 打包评测  

**下一步** — 更大金标 · Redis 缓存 · 决策可视化 · 多租户 · 1.0 冻结 API  

详见 [CHANGELOG.md](CHANGELOG.md)。

1. 读 [API.md](API.md) + [POLICY.md](POLICY.md)  
2. `pip install -e ".[dev]"` → `pytest -q`  
3. 同步 `eval/traces.json` ↔ `src/laya_thalamus/resources/traces.json`  
4. 不破坏 `AgentRouter.route` 签名  

---

## License

[Apache-2.0](LICENSE) · 权重：[Laya](https://github.com/NandhaKishorM/laya) · [laya.convaiinnovations.com](https://laya.convaiinnovations.com/)

<p align="center">
  <a href="https://www.star-history.com/#lcbkmm/laya-thalamus&Date">
    <img src="https://api.star-history.com/svg?repos=lcbkmm/laya-thalamus&type=Date" alt="Star History" width="520" />
  </a>
</p>

<p align="center"><i>如果 Thalamus 帮你少写了一坨路由 Prompt——点个 Star。</i></p>
