# Thalamus｜Laya-powered System-1 Router for AI Agents

[English](README.md) | **中文**

> **Thalamus（丘脑）**：大脑里负责快速中转与信号路由的器官——对应本项目的 System-1 决策层。

**把 Agent 里「选哪个工具 / 信息够不够 / 结果信不信」从 LLM 里拆出来：亚秒级路由，复杂推理仍交给大模型。**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pytest%203.10--3.12-informational.svg)](.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip-laya--thalamus-blue.svg)](https://pypi.org/project/laya-thalamus/)

一行接入 · OpenAI / LangChain / 自研 Agent · 真降级 · 可对比评测

```bash
pip install "laya-thalamus[laya]"
```

```python
from laya_thalamus import AgentRouter
```

---

## 实测评测（先看数）

人工金标 **n=56**（含工具执行后多轮状态）。同一数据集。

| Backend | Tool Acc | 延迟 (CPU) | 延迟 (GPU) | 说明 |
|---------|---------:|----------:|----------:|------|
| **Laya + fallback** | **64.3%** | **~0.30 s** | — | **产品主路径** · 本机无 LLM key，走 heuristic 降级 |
| Laya bare (System-1) | 55.4% | ~0.47 s | — | 关闭 fallback · CPU 权重 |
| deepseek-v3.1 (LLM router) | 87.5% | ~2.0 s | — | 准，但贵、慢，适合对比基线 |
| Random | 21.4% | — | — | 机会水平下限 |
| Keyword mock | 89.3% | \<1 ms | — | **启发式，不是模型能力**，仅 CI / Demo |

> 原始报告：[`eval/compare_report.json`](eval/compare_report.json)。  
> 复现：`thalamus compare --with-fallback --skip-llm` · GPU：加 `--device cuda`。  
> **不要拿 mock 准确率当产品指标。** 解读见 [评测怎么读](#评测怎么读)。

**结论：** 裸 Laya 是本地先验；**打开 fallback** 才是产品路径（配置了 LLM 就降级到大模型，否则 heuristic）。发布机无 CUDA，GPU 列为空——用 `--device cuda` 自行补齐。

---

## 它解决什么痛点

| 痛点 | 没有 Thalamus | 有 Thalamus |
|------|---------------|-------------|
| 每次「要不要调工具」都问大模型 | 延迟高、Token 贵 | `choice` 毫秒～秒级出决策 |
| 工具结果够不够、能不能信 | Prompt 硬编码难维护 | `noul` / `score` + 阈值 |
| 工具死循环 / 同工具狂调 | 各框架各写一套 | `session_id` 熔断 + `decision_id` |
| 想接 LangChain / OpenAI FC | 自己包装模型 API | `integrations.*` 五行走通 |
| 超时半截结果当真理 | 静默错路由 | [POLICY.md](POLICY.md) 硬失败 → LLM → heuristic |
| 口头说「很快」 | 无法对比 | 打包金标 + `thalamus compare` |

---

## 和原生 Laya 的区别

| | [Laya](https://github.com/NandhaKishorM/laya) 模型库 | **Thalamus (`laya-thalamus`)** |
|--|--|--|
| 定位 | System-1 **推理内核** | Agent **决策中间件**（丘脑式中转） |
| 你拿到的 | `predict(...)` | `router.route(...)` + HTTP / CLI / 适配器 |
| 生产缺口 | 需自写降级、熔断、评测 | 开箱：降级、熔断、缓存、幂等、Prometheus、OTel |
| 接入 | 绑死调用方式 | OpenAI tools / LangChain / 自研 Session |

**关系：** Laya 是引擎；Thalamus 是方向盘与保险杠。`pip install laya` 装引擎，`laya-thalamus` 装中间层。

---

## 架构

```
                    ┌─────────────────────────────────────┐
  User / Agent ──►  │         AgentRouter.route()         │
                    │  session · cache · circuit · otel   │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
         LayaBackend         LLMRouterBackend      MockBackend
         (System-1)          (对比 / 可选主路由)    (CI only)
              │
              └──── hard-fail / low-conf ──► Fallback (LLM → heuristic)
                                   │
                                   ▼
                          RouteDecision
              action · tool · noul · score · decision_id
```

三种原语：**choice** · **noul** · **score**。只输出路由判断，不生成最终回答。

---

## 30 秒上手

```bash
pip install "laya-thalamus[laya]"   # 真 System-1（强烈推荐）
# 只要库：pip install laya-thalamus
# HTTP：  pip install "laya-thalamus[api]"
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

**插进现有栈：**

```python
from laya_thalamus.integrations import (
    AgentSession, LayaToolSelector, route_openai_turn, decision_to_tool_calls,
)

d = route_openai_turn(router, messages, openai_tools, session_id="s1")
calls = decision_to_tool_calls(d)   # None → 直接让 LLM 回答

tool = LayaToolSelector(tools=lc_tools).pick(query, intermediate_steps)

session = AgentSession(router)
d = session.next(query)
```

```bash
thalamus demo --backend mock
thalamus probe-laya --model /path/to/LAYA
thalamus compare --skip-llm
thalamus serve --port 8080          # 需 [api]
```

文档：[API.md](API.md) · [POLICY.md](POLICY.md) · [`examples/`](examples/)

---

## 评测怎么读

| 行 | 正确理解 |
|----|----------|
| **laya+fallback** | **产品主路径。** 低置信 / 硬失败 → 有 `LAYA_LLM_*` 走 LLM，否则 heuristic。本集比裸 Laya **+9pp**。 |
| **laya bare** | 仅 System-1（关 fallback）。看清本地模型单独上限。 |
| **延迟 (CPU)** | 发布机实测（热机约 ~0.5 s；冷机可接近 ~1 s）。 |
| **延迟 (GPU)** | 本机无 CUDA，未测。复现：`thalamus compare --device cuda --with-fallback --skip-llm`。 |
| **llm:\*** | 聊天模型当*主*路由器——成本/延迟基线，不是 Thalamus 热路径。 |
| **mock** | 关键词启发式。好看 ≠ 能上线。 |
| **random** | 随机下限。 |

意图不是「永远打败 LLM」，而是：**高频路径用 System-1；边界与低置信走 fallback**。

---

## FAQ

**Q: Thalamus 是又一个 Agent 框架吗？**  
A: 不是。只做决策中间件：告诉你下一步 `call_tool` / `answer` / `terminate`。

**Q: 为什么叫 Thalamus？**  
A: 丘脑是大脑的快速信号中转站；本项目做 Agent 的 System-1 路由中转——词源一句话就够。

**Q: 为什么不直接每次问 GPT 选工具？**  
A: 可以更准，但贵且慢。Thalamus 用 System-1 扛热路径（本机 CPU ~0.5 s），低置信再降级。

**Q: 为什么裸 Laya 只有约 55%？**  
A: 弱在「工具结果已够用该停」的多轮题。打开 fallback（本表 **64.3%**；配置 LLM 后通常更高）。见上方评测表。

**Q: mock 89% 是不是水分？**  
A: 是启发式，**禁止当产品指标**。表格已单独标注。

**Q: 和官方 `laya` 包什么关系？**  
A: `laya` 是模型运行时；本仓库是可选 extras `[laya]` 之上的工程化中间件。

**Q: 必须装 FastAPI 吗？**  
A: 否。核心只有 `pydantic` + `pyyaml`。

**Q: import 名是什么？**  
A: 包名 `laya-thalamus`，import `laya_thalamus`，CLI `thalamus` / `laya-thalamus`。

**Q: 稳定 API？**  
A: 见 [API.md](API.md)。

---

## 路线图

**已交付（0.1）** — choice/noul/score · 降级 · 熔断 · 会话/幂等/缓存 · 适配器 · 异步 · Prometheus/OTel · CI · 打包评测  

**下一步** — 更大金标 · Redis 缓存 · 决策可视化 · 多租户 · 1.0 冻结 API  

详见 [FEATURES.md](FEATURES.md) · [CHANGELOG.md](CHANGELOG.md)。

---

## Contribute

1. 读 [API.md](API.md) 与 [POLICY.md](POLICY.md)  
2. `pip install -e ".[dev]"` → `pytest -q`  
3. 改金标时同步 `eval/traces.json` 与 `src/laya_thalamus/resources/traces.json`  
4. 不破坏 `AgentRouter.route` 签名  

欢迎：金标扩充、适配器、文档样例、性能剖析。

---

## License

[Apache-2.0](LICENSE). System-1 权重见 [Laya](https://github.com/NandhaKishorM/laya) / [laya.convaiinnovations.com](https://laya.convaiinnovations.com/).

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lcbkmm/laya-thalamus&type=Date)](https://www.star-history.com/#lcbkmm/laya-thalamus&Date)

如果 Thalamus 帮你少写了一坨路由 Prompt——点个 **Star**。
