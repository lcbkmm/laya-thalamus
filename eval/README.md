# Gold traces schema (JEV-like, bilingual)

Aligned with [`temple.txt`](temple.txt) (OpenRouter / JEV `state` + `questions`).

| File | Language | Size |
|------|----------|------|
| [`traces.zh.json`](traces.zh.json) | Chinese | n=100 |
| [`traces.en.json`](traces.en.json) | English | n=100 |

Packaged copies live under `src/laya_thalamus/resources/` (must stay identical).

### Item fields

| Field | Meaning |
|-------|---------|
| `lang` | `zh` or `en` |
| `state` | Full decision context |
| `questions` | `tool` (choice) · `sufficient` (noul) · optional `credibility` (score) |
| `answers` | Gold labels |
| `query` / `expected_*` | Flat mirrors for `evaluate_items` |
| `category` | `direct` · `calculator` · `search` · `code` · `rag` · `after_tool` · `after_tool_fail` · `edge` |

### Usage

```bash
# JEV vs Laya on gold state+questions (needs OPENROUTER_API_KEY + [laya])
python eval/compare_jev_laya.py --lang both --out eval/compare_jev_laya_report.json

thalamus compare --lang zh --with-fallback --skip-llm
thalamus compare --lang en --with-fallback --skip-llm
python eval/run_eval.py --lang en --backend mock
```

Default packaged language is **zh** (`LAYA_TRACES_LANG=en` to override).

Latest JEV / Laya numbers: [`compare_jev_laya_report.json`](compare_jev_laya_report.json) (also summarized in root README).

Regenerate:

```bash
python eval/_gen_traces_jev.py
```
