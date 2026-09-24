# Gold traces schema (JEV-like)

Aligned with the typed-decision style in [`temple.txt`](temple.txt)
(OpenRouter / typesafe JEV `state` + `questions` with `choice` / `noul` / `score`).

Each item in [`traces.json`](traces.json):

| Field | Meaning |
|-------|---------|
| `state` | Full decision context (user query + optional tool result / context) |
| `questions` | Typed questions: `tool` (choice), `sufficient` (noul), optional `credibility` (score) |
| `answers` | Gold labels (`tool.choice`, `sufficient`, optional score band) |
| `query` / `expected_*` | Flat mirrors for `thalamus compare` / `evaluate_items` |
| `category` | `direct` · `calculator` · `search` · `code` · `rag` · `after_tool` · `after_tool_fail` · `mixed` |

Regenerate (maintainers):

```bash
python eval/_gen_traces_jev.py
```

Keep `eval/traces.json` and `src/laya_thalamus/resources/traces.json` identical.
