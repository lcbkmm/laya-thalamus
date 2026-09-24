# V1 功能清单（产品规格）

产品名：**Thalamus**（PyPI: `laya-thalamus`，import: `laya_thalamus`，CLI: `thalamus`）。  
完整说明以代码与根目录 README 为准。

## V1 必做（已实现）

1. Laya `choice`：工具路由，支持自定义工具列表 / 动态注册
2. Laya `noul`：信息是否充足
3. Laya `score`：结果可信度 0-10
4. 阈值配置 + **真 LLM 降级**（OpenAI 兼容；失败再 heuristic）
5. 防循环调用熔断（最大轮次 + 同工具重复）
6. 同步 / 异步 HTTP API（`/route` `/health` `/batch` `/config` `/tools` `/models/*` `/metrics/prometheus`）
7. 结构化 JSON 决策日志 + Prometheus 风格 metrics
8. Demo：minimal / vLLM / LangChain / OpenAI / AgentSession / LLM 降级 / probe
9. 评测：人工金标 `traces.zh.json` / `traces.en.json`（各 n=100，JEV-like）+ `thalamus compare --lang zh|en`
10. YAML 配置（含 `fallback.llm` / `cache` / `logging.otel`）
11. Extras：`[api]` / `[llm]` / `[laya]` / `[otel]` / `[dev]` / `[all]`
12. 集成：`integrations` + OpenAI/JSON Schema ↔ ToolSpec
13. 会话：`session_id`、`decision_id`、幂等、决策缓存
14. 发布成熟度：版本单一来源、`API.md`、CHANGELOG、CI 3.10-3.12
15. 生产可靠性：`aroute`、POLICY 超时降级契约、OTel

## V2（未做）

- Web 可视化监控面板
- 多实例业务隔离
- 跨进程决策缓存（进程内 TTL 已有）
- 微调脚本 / 多语言专项
