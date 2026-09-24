"""Generate JEV-aligned gold traces (n=100) and write both repo copies."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOOL_CRITERIA = {
    "none": "Greetings, thanks, chitchat, translation, or opinion — no tool needed",
    "calculator": "Numeric arithmetic / percentages / simple math expressions",
    "web_search": "Public facts, news, weather, definitions, who-is, live web info",
    "code_interpreter": "Write/run Python, algorithms, pandas, plots, debugging code",
    "rag_retrieve": "Internal/company knowledge-base, handbook, HR/policy documents",
}

SCORE_CRITERIA = [
    "error/failure or completely irrelevant",
    "mostly irrelevant or broken output",
    "weakly related",
    "partially useful but major gaps",
    "somewhat relevant",
    "moderately relevant",
    "relevant with minor gaps",
    "mostly correct and useful for the query",
    "highly relevant and reliable",
    "excellent match to the query",
    "perfectly answers the user query",
]


def _questions(
    *,
    has_result: bool,
    result_failed: bool,
    need_score: bool,
) -> dict:
    if has_result and not result_failed:
        tool_inst = (
            "A Tool result is ALREADY present in the state. "
            "If it answers or substantially helps the user query, choose 'none'. "
            "Only pick another tool if the result is missing, wrong, or an error."
        )
        noul_inst = (
            "A Tool result is present and not marked FAILED. "
            "Is it enough to answer the user query without more tools?"
        )
        criteria = dict(TOOL_CRITERIA)
        criteria["none"] = (
            "STOP calling tools. A usable Tool result is already in the state."
        )
    elif has_result and result_failed:
        tool_inst = (
            "The previous Tool result FAILED. Prefer retrying an appropriate tool; "
            "choose 'none' only if no tool can help."
        )
        noul_inst = "The Tool result FAILED. Information is NOT sufficient; prefer false."
        criteria = dict(TOOL_CRITERIA)
        criteria["none"] = "Only if no tool can help after a failed tool result."
    else:
        tool_inst = (
            "Pick the single next tool for an AI agent. "
            "Use 'none' for greetings/thanks/simple chat/translation with no tool needed. "
            "Use 'calculator' only for numeric arithmetic. "
            "Use 'code_interpreter' to write/run Python or algorithms. "
            "Use 'web_search' for public facts, news, weather, definitions. "
            "Use 'rag_retrieve' only for internal/company knowledge-base documents."
        )
        noul_inst = (
            "No tool result yet. Is the user query answerable from the query alone "
            "(greetings, thanks, simple translation)? If it needs calc/search/code/docs, false."
        )
        criteria = dict(TOOL_CRITERIA)

    qs: dict = {
        "tool": {
            "type": "choice",
            "instructions": tool_inst,
            "criteria": criteria,
        },
        "sufficient": {
            "type": "noul",
            "instructions": noul_inst,
            "criteria": {
                "true": "Enough information to answer without more tools",
                "false": "Needs a tool call or more information",
            },
        },
    }
    if need_score:
        qs["credibility"] = {
            "type": "score",
            "instructions": (
                "Score how well the Tool result answers the User query. "
                "Errors/quota failures -> 0-2. Correct useful results -> 7-10."
            ),
            "criteria": SCORE_CRITERIA,
        }
    return qs


def _state(query: str, *, tool_result: str | None = None, context: str | None = None) -> str:
    parts = [f"User query: {query}"]
    if context:
        parts.append(f"Context: {context}")
    if tool_result:
        failed = any(
            x in tool_result.lower()
            for x in ("error", "failed", "exception", "quota exceeded", "失败")
        )
        status = "FAILED" if failed else "OK"
        parts.append(f"Tool result (status: {status}):\n{tool_result}")
    return "\n".join(parts)


def item(
    id_: str,
    query: str,
    category: str,
    *,
    expected_tool: str,
    expected_action: str,
    expected_sufficient: bool,
    tool_result: str | None = None,
    history: list[str] | None = None,
    context: str | None = None,
    expected_score_min: float | None = None,
    expected_score_max: float | None = None,
) -> dict:
    has_result = tool_result is not None
    result_failed = bool(
        tool_result
        and any(
            x in tool_result.lower()
            for x in ("error", "failed", "exception", "quota exceeded", "失败")
        )
    )
    need_score = expected_score_min is not None or has_result
    answers: dict = {
        "tool": {"choice": expected_tool},
        "sufficient": {
            "noul": 1.0 if expected_sufficient else 0.0,
            "label": bool(expected_sufficient),
        },
    }
    if expected_score_min is not None:
        answers["credibility"] = {
            "score_min": expected_score_min,
            "score_max": expected_score_max if expected_score_max is not None else 10,
        }

    row: dict = {
        "id": id_,
        "category": category,
        "schema": "jev-like/v1",
        "state": _state(query, tool_result=tool_result, context=context),
        "query": query,
        "questions": _questions(
            has_result=has_result,
            result_failed=result_failed,
            need_score=need_score,
        ),
        "answers": answers,
        # Flat fields for thalamus evaluate_items (derived / mirrored)
        "expected_tool": expected_tool,
        "expected_action": expected_action,
        "expected_sufficient": expected_sufficient,
    }
    if tool_result is not None:
        row["tool_result"] = tool_result
    if history:
        row["history"] = history
    if context:
        row["context"] = context
    if expected_score_min is not None:
        row["expected_score_min"] = expected_score_min
    if expected_score_max is not None:
        row["expected_score_max"] = expected_score_max
    return row


def build() -> list[dict]:
    rows: list[dict] = []

    # ---- direct (14) ----
    direct = [
        ("greet", "你好，今天过得怎么样"),
        ("thanks", "谢谢你的帮助"),
        ("bye", "再见，下次聊"),
        ("translate_intent", "把下面这句话翻译成英文：今天会议取消了"),
        ("opinion", "你觉得春天好还是秋天好？"),
        ("hello_en", "Hi, how are you?"),
        ("joke", "讲个简短的冷笑话吧"),
        ("summarize_given", "请用一句话总结：苹果是一种常见水果，富含维生素。"),
        ("ambiguous_math_word", "人生的意义是什么"),
        ("encourage", "我有点沮丧，说两句鼓励的话吧"),
        ("roleplay_light", "用温柔的语气跟我说晚安"),
        ("rewrite_tone", "把这句话改得更礼貌：把文件马上发给我"),
        ("brainstorm_name", "给一个读书会起三个好听的中文名字"),
        ("simple_compare", "猫和狗哪个更适合公寓养？给个简短看法即可"),
    ]
    for i, q in direct:
        rows.append(
            item(
                i,
                q,
                "direct",
                expected_tool="none",
                expected_action="answer",
                expected_sufficient=True,
            )
        )

    # ---- calculator (12) ----
    calcs = [
        ("calc_mul", "请计算 384 * 27 的结果"),
        ("percent", "85 的 20% 是多少，帮我算一下"),
        ("calc_expr", "算一下 (120-45)/5+8"),
        ("calc_sqrt", "计算 144 的平方根"),
        ("calc_mix", "3.5 乘以 2.2 再加 10 等于多少"),
        ("calc_div", "用计算器算 98765 / 5"),
        ("calc_pow", "计算 2 的 10 次方"),
        ("calc_avg", "求 12、18、25、30 的平均数"),
        ("calc_tax", "税前 10000，税率 10%，税后多少"),
        ("calc_discount", "原价 899，打 85 折是多少"),
        ("calc_compound", "本金 5000，年利率 3%，一年后多少钱（单利）"),
        ("calc_unit_arith", "3.2 公里等于多少米，直接算"),
    ]
    for i, q in calcs:
        rows.append(
            item(
                i,
                q,
                "calculator",
                expected_tool="calculator",
                expected_action="call_tool",
                expected_sufficient=False,
            )
        )

    # ---- search (16) ----
    searches = [
        ("search_weather", "帮我查一下今天上海会不会下雨"),
        ("quantum_progress", "搜索一下量子计算最近有什么突破"),
        ("factoid", "什么是 CRISPR 基因编辑"),
        ("factoid2", "什么是变压器架构 Transformer"),
        ("news", "最新的全球科技新闻头条有哪些"),
        ("who_is", "谁是图灵奖得主 Geoffrey Hinton"),
        ("stock", "查一下今天上证指数收盘大概多少点"),
        ("weather_bj", "北京明天最高气温多少度"),
        ("quantum_not_calc", "量子计算和经典计算有什么区别"),
        ("cloud_not_calc", "云计算有哪些主流服务商"),
        ("how_to_public", "如何申请护照需要哪些材料（公开信息）"),
        ("unit_convert_search", "1 英里等于多少公里（查权威换算）"),
        ("olympics", "最近一届夏季奥运会在哪举办的"),
        ("llm_release", "OpenAI 最近发布了哪些主要模型（公开报道）"),
        ("exchange_rate", "查一下美元兑人民币今日大概汇率"),
        ("wiki_person", "搜索一下图灵是谁，给个简介"),
    ]
    for i, q in searches:
        rows.append(
            item(
                i,
                q,
                "search",
                expected_tool="web_search",
                expected_action="call_tool",
                expected_sufficient=False,
            )
        )

    # ---- code (12) ----
    codes = [
        ("code_median", "用 Python 写一段代码求列表中位数"),
        ("code_sort", "写一个 Python 快速排序函数"),
        ("code_pandas", "用 pandas 把 CSV 按列求和并输出"),
        ("code_vs_rag", "帮我用代码实现二叉树层序遍历"),
        ("code_debug", "这段 Python 报错怎么修：list index out of range"),
        ("code_regex", "用 Python 正则提取邮箱地址写个函数"),
        ("plot_code", "用 matplotlib 画一条正弦曲线的代码"),
        ("sql_code", "写一段 Python 连接 SQLite 并查询的示例"),
        ("code_bfs", "用 Python 实现图的 BFS"),
        ("code_json", "写个函数把嵌套 dict 展平成点号路径 key"),
        ("code_thread", "给一个 Python 多线程下载文件的最小示例"),
        ("code_test", "为加法函数写一个 pytest 用例示例"),
    ]
    for i, q in codes:
        rows.append(
            item(
                i,
                q,
                "code",
                expected_tool="code_interpreter",
                expected_action="call_tool",
                expected_sufficient=False,
            )
        )

    # ---- rag (12) ----
    rags = [
        ("rag_leave", "根据公司内部知识库，年假最多可以请几天？"),
        ("rag_expense", "查一下内部资料里的差旅报销流程"),
        ("rag_policy", "员工手册里加班调休怎么算？"),
        ("rag_vs_search", "我们公司知识库里的考勤制度是什么"),
        ("rag_onboard", "内部入职须知里电脑领用流程是什么"),
        ("rag_severance", "根据公司政策文档，离职交接清单有哪些"),
        ("internal_vpn", "公司内网 VPN 开通流程在知识库哪一章"),
        ("rag_benefit", "内部福利手册里餐补标准是多少"),
        ("rag_security", "公司信息安全规范对 U 盘使用有什么要求"),
        ("rag_meeting", "会议室预约制度在内部文档哪一节"),
        ("rag_probation", "知识库里试用期考核流程怎么走"),
        ("rag_remote", "内部远程办公政策允许每周居家几天"),
    ]
    for i, q in rags:
        rows.append(
            item(
                i,
                q,
                "rag",
                expected_tool="rag_retrieve",
                expected_action="call_tool",
                expected_sufficient=False,
            )
        )

    # ---- after_tool success (20) ----
    after_ok = [
        (
            "calc_after_ok",
            "请计算 384 * 27 的结果",
            "计算结果：10368",
            ["calculator"],
            None,
            6,
        ),
        (
            "percent_after_ok",
            "85 的 20% 是多少，帮我算一下",
            "计算结果：17",
            ["calculator"],
            None,
            6,
        ),
        (
            "calc_expr_after_ok",
            "算一下 (120-45)/5+8",
            "计算结果：23",
            ["calculator"],
            None,
            6,
        ),
        (
            "calc_pow_after_ok",
            "计算 2 的 10 次方",
            "计算结果：1024",
            ["calculator"],
            None,
            6,
        ),
        (
            "search_after_ok",
            "帮我查一下今天上海会不会下雨",
            "气象台摘要：上海今天多云转阴，降水概率 10%，基本不会下雨。",
            ["web_search"],
            None,
            6,
        ),
        (
            "factoid_after_ok",
            "什么是 CRISPR 基因编辑",
            "百科摘要：CRISPR 是一种细菌免疫系统衍生的基因编辑技术，可精确切割 DNA。",
            ["web_search"],
            None,
            6,
        ),
        (
            "mixed_after_search",
            "搜索一下北京今天气温",
            "北京今天白天最高 22℃，夜间最低 12℃，空气质量良。",
            ["web_search"],
            None,
            6,
        ),
        (
            "news_after_ok",
            "最新的全球科技新闻头条有哪些",
            "检索摘要：1) 新一代开源模型发布；2) 芯片制程突破报道；3) 卫星互联网商用进展。",
            ["web_search"],
            None,
            6,
        ),
        (
            "code_after_ok",
            "用 Python 写一段代码求列表中位数",
            "```python\ndef median(xs):\n    s=sorted(xs); n=len(s)\n    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "code_sort_after_ok",
            "写一个 Python 快速排序函数",
            "```python\ndef qsort(a):\n    if len(a)<=1: return a\n    p=a[0]\n    return qsort([x for x in a[1:] if x<p])+[p]+qsort([x for x in a[1:] if x>=p])\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "plot_after_ok",
            "用 matplotlib 画一条正弦曲线的代码",
            "```python\nimport numpy as np, matplotlib.pyplot as plt\nx=np.linspace(0,2*np.pi,200)\nplt.plot(x,np.sin(x)); plt.show()\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "rag_after_ok",
            "根据公司内部知识库，年假最多可以请几天？",
            "知识库命中：《员工手册》第 4.2 条：司龄满 1 年享有 5 天年假，满 10 年享有 15 天，单次申请不超过当年剩余额度。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "rag_expense_after_ok",
            "查一下内部资料里的差旅报销流程",
            "知识库：差旅报销需提交申请单→发票→行程单，财务 3 个工作日内审核。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "rag_vpn_after_ok",
            "公司内网 VPN 开通流程在知识库哪一章",
            "知识库：《IT 服务手册》第 6 章：提交工单→主管审批→IT 开通，通常 1 个工作日。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "multi_turn_enough",
            "刚才算出来的结果再解释一下含义",
            "计算结果：56088",
            ["calculator"],
            "上一轮计算器结果：56088",
            5,
        ),
        (
            "calc_discount_after_ok",
            "原价 899，打 85 折是多少",
            "计算结果：764.15",
            ["calculator"],
            None,
            6,
        ),
        (
            "weather_bj_after_ok",
            "北京明天最高气温多少度",
            "预报：北京明天最高气温 26℃，最低 15℃，晴转多云。",
            ["web_search"],
            None,
            6,
        ),
        (
            "code_regex_after_ok",
            "用 Python 正则提取邮箱地址写个函数",
            "```python\nimport re\ndef emails(s): return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', s)\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "rag_remote_after_ok",
            "内部远程办公政策允许每周居家几天",
            "知识库：《远程办公指引》：经主管同意，每周最多居家 2 天，核心协作日需到岗。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "search_exchange_after_ok",
            "查一下美元兑人民币今日大概汇率",
            "行情摘要：USD/CNY 约 7.24（仅供参考，非交易报价）。",
            ["web_search"],
            None,
            6,
        ),
    ]
    for i, q, tr, hist, ctx, smin in after_ok:
        rows.append(
            item(
                i,
                q,
                "after_tool",
                expected_tool="none",
                expected_action="answer",
                expected_sufficient=True,
                tool_result=tr,
                history=hist,
                context=ctx,
                expected_score_min=smin,
            )
        )

    # ---- after_tool_fail (8) ----
    fails = [
        (
            "search_after_fail",
            "帮我查一下今天上海会不会下雨",
            "error: search quota exceeded",
            ["web_search"],
            "web_search",
        ),
        (
            "calc_fail_retry",
            "计算 999/0",
            "error: division by zero",
            ["calculator"],
            "calculator",
        ),
        (
            "code_fail_retry",
            "用 Python 写快速排序",
            "error: interpreter timeout",
            ["code_interpreter"],
            "code_interpreter",
        ),
        (
            "rag_fail_retry",
            "根据公司知识库查福利补贴",
            "error: retrieval index unavailable",
            ["rag_retrieve"],
            "rag_retrieve",
        ),
        (
            "search_fail_timeout",
            "搜索一下量子计算最近有什么突破",
            "error: upstream timeout",
            ["web_search"],
            "web_search",
        ),
        (
            "calc_fail_overflow",
            "计算 10 ** 100000",
            "error: numeric overflow",
            ["calculator"],
            "calculator",
        ),
        (
            "code_fail_syntax",
            "用 pandas 把 CSV 按列求和并输出",
            "error: SyntaxError: unexpected EOF",
            ["code_interpreter"],
            "code_interpreter",
        ),
        (
            "rag_fail_empty",
            "员工手册里加班调休怎么算？",
            "error: no documents matched",
            ["rag_retrieve"],
            "rag_retrieve",
        ),
    ]
    for i, q, tr, hist, tool in fails:
        rows.append(
            item(
                i,
                q,
                "after_tool_fail",
                expected_tool=tool,
                expected_action="retry_tool",
                expected_sufficient=False,
                tool_result=tr,
                history=hist,
                expected_score_min=0,
                expected_score_max=4,
            )
        )

    # ---- mixed / bilingual / edge (6) ----
    mixed = [
        (
            "mixed_en_calc",
            "Please compute 45 * 18 for me",
            "calculator",
            "call_tool",
            False,
            "mixed",
        ),
        (
            "mixed_en_search",
            "What is the capital of Australia? Look it up if needed.",
            "web_search",
            "call_tool",
            False,
            "mixed",
        ),
        (
            "mixed_code_en",
            "Write a Python function to reverse a linked list",
            "code_interpreter",
            "call_tool",
            False,
            "mixed",
        ),
        (
            "mixed_direct_en",
            "Thanks a lot, that was helpful!",
            "none",
            "answer",
            True,
            "mixed",
        ),
        (
            "mixed_rag_zh_en",
            "Check our internal KB: what is the probation review process?",
            "rag_retrieve",
            "call_tool",
            False,
            "mixed",
        ),
        (
            "mixed_prefer_search_not_rag",
            "公开资料里，世界卫生组织总部在哪个城市",
            "web_search",
            "call_tool",
            False,
            "mixed",
        ),
    ]
    for i, q, tool, action, suff, cat in mixed:
        rows.append(
            item(
                i,
                q,
                cat,
                expected_tool=tool,
                expected_action=action,
                expected_sufficient=suff,
            )
        )

    return rows


def main() -> None:
    rows = build()
    assert len(rows) == 100, len(rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    from collections import Counter

    print("n=", len(rows), "by_cat=", dict(Counter(r["category"] for r in rows)))

    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    paths = [
        ROOT / "eval" / "traces.json",
        ROOT / "src" / "laya_thalamus" / "resources" / "traces.json",
    ]
    for p in paths:
        p.write_text(text, encoding="utf-8")
        print("wrote", p)


if __name__ == "__main__":
    main()
