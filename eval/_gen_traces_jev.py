"""Generate bilingual JEV-aligned gold sets: traces.zh.json + traces.en.json (n=100 each)."""

from __future__ import annotations

import json
from collections import Counter
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


def _questions(*, has_result: bool, result_failed: bool, need_score: bool) -> dict:
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
        noul_inst = (
            "The Tool result FAILED. Information is NOT sufficient; prefer false."
        )
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


def _state(
    query: str, *, tool_result: str | None = None, context: str | None = None
) -> str:
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
    lang: str,
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
        "lang": lang,
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


def _pack_simple(
    lang: str,
    category: str,
    pairs: list[tuple[str, str]],
    *,
    tool: str,
    action: str,
    sufficient: bool,
) -> list[dict]:
    return [
        item(
            i,
            q,
            category,
            lang,
            expected_tool=tool,
            expected_action=action,
            expected_sufficient=sufficient,
        )
        for i, q in pairs
    ]


def build_zh() -> list[dict]:
    lang = "zh"
    rows: list[dict] = []

    rows += _pack_simple(
        lang,
        "direct",
        [
            ("zh_greet", "你好，今天过得怎么样"),
            ("zh_thanks", "谢谢你的帮助"),
            ("zh_bye", "再见，下次聊"),
            ("zh_translate", "把下面这句话翻译成英文：今天会议取消了"),
            ("zh_opinion", "你觉得春天好还是秋天好？"),
            ("zh_joke", "讲个简短的冷笑话吧"),
            ("zh_summarize", "请用一句话总结：苹果是一种常见水果，富含维生素。"),
            ("zh_meaning", "人生的意义是什么"),
            ("zh_encourage", "我有点沮丧，说两句鼓励的话吧"),
            ("zh_goodnight", "用温柔的语气跟我说晚安"),
            ("zh_rewrite", "把这句话改得更礼貌：把文件马上发给我"),
            ("zh_brainstorm", "给一个读书会起三个好听的中文名字"),
            ("zh_pets", "猫和狗哪个更适合公寓养？给个简短看法即可"),
            ("zh_poem", "写两句关于秋天的短句就行"),
        ],
        tool="none",
        action="answer",
        sufficient=True,
    )

    rows += _pack_simple(
        lang,
        "calculator",
        [
            ("zh_calc_mul", "请计算 384 * 27 的结果"),
            ("zh_percent", "85 的 20% 是多少，帮我算一下"),
            ("zh_calc_expr", "算一下 (120-45)/5+8"),
            ("zh_calc_sqrt", "计算 144 的平方根"),
            ("zh_calc_mix", "3.5 乘以 2.2 再加 10 等于多少"),
            ("zh_calc_div", "用计算器算 98765 / 5"),
            ("zh_calc_pow", "计算 2 的 10 次方"),
            ("zh_calc_avg", "求 12、18、25、30 的平均数"),
            ("zh_calc_tax", "税前 10000，税率 10%，税后多少"),
            ("zh_calc_discount", "原价 899，打 85 折是多少"),
            ("zh_calc_interest", "本金 5000，年利率 3%，一年后多少钱（单利）"),
            ("zh_calc_meter", "3.2 公里等于多少米，直接算"),
        ],
        tool="calculator",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "search",
        [
            ("zh_search_weather", "帮我查一下今天上海会不会下雨"),
            ("zh_quantum", "搜索一下量子计算最近有什么突破"),
            ("zh_crispr", "什么是 CRISPR 基因编辑"),
            ("zh_transformer", "什么是变压器架构 Transformer"),
            ("zh_news", "最新的全球科技新闻头条有哪些"),
            ("zh_hinton", "谁是图灵奖得主 Geoffrey Hinton"),
            ("zh_stock", "查一下今天上证指数收盘大概多少点"),
            ("zh_weather_bj", "北京明天最高气温多少度"),
            ("zh_quantum_diff", "量子计算和经典计算有什么区别"),
            ("zh_cloud", "云计算有哪些主流服务商"),
            ("zh_passport", "如何申请护照需要哪些材料（公开信息）"),
            ("zh_mile", "1 英里等于多少公里（查权威换算）"),
            ("zh_olympics", "最近一届夏季奥运会在哪举办的"),
            ("zh_openai", "OpenAI 最近发布了哪些主要模型（公开报道）"),
            ("zh_fx", "查一下美元兑人民币今日大概汇率"),
            ("zh_turing", "搜索一下图灵是谁，给个简介"),
        ],
        tool="web_search",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "code",
        [
            ("zh_code_median", "用 Python 写一段代码求列表中位数"),
            ("zh_code_sort", "写一个 Python 快速排序函数"),
            ("zh_code_pandas", "用 pandas 把 CSV 按列求和并输出"),
            ("zh_code_bfs_tree", "帮我用代码实现二叉树层序遍历"),
            ("zh_code_debug", "这段 Python 报错怎么修：list index out of range"),
            ("zh_code_regex", "用 Python 正则提取邮箱地址写个函数"),
            ("zh_code_plot", "用 matplotlib 画一条正弦曲线的代码"),
            ("zh_code_sqlite", "写一段 Python 连接 SQLite 并查询的示例"),
            ("zh_code_bfs", "用 Python 实现图的 BFS"),
            ("zh_code_flatten", "写个函数把嵌套 dict 展平成点号路径 key"),
            ("zh_code_thread", "给一个 Python 多线程下载文件的最小示例"),
            ("zh_code_pytest", "为加法函数写一个 pytest 用例示例"),
        ],
        tool="code_interpreter",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "rag",
        [
            ("zh_rag_leave", "根据公司内部知识库，年假最多可以请几天？"),
            ("zh_rag_expense", "查一下内部资料里的差旅报销流程"),
            ("zh_rag_ot", "员工手册里加班调休怎么算？"),
            ("zh_rag_attend", "我们公司知识库里的考勤制度是什么"),
            ("zh_rag_laptop", "内部入职须知里电脑领用流程是什么"),
            ("zh_rag_exit", "根据公司政策文档，离职交接清单有哪些"),
            ("zh_rag_vpn", "公司内网 VPN 开通流程在知识库哪一章"),
            ("zh_rag_meal", "内部福利手册里餐补标准是多少"),
            ("zh_rag_usb", "公司信息安全规范对 U 盘使用有什么要求"),
            ("zh_rag_room", "会议室预约制度在内部文档哪一节"),
            ("zh_rag_probation", "知识库里试用期考核流程怎么走"),
            ("zh_rag_remote", "内部远程办公政策允许每周居家几天"),
        ],
        tool="rag_retrieve",
        action="call_tool",
        sufficient=False,
    )

    after_ok = [
        ("zh_calc_after_ok", "请计算 384 * 27 的结果", "计算结果：10368", ["calculator"], None, 6),
        ("zh_percent_after_ok", "85 的 20% 是多少，帮我算一下", "计算结果：17", ["calculator"], None, 6),
        ("zh_expr_after_ok", "算一下 (120-45)/5+8", "计算结果：23", ["calculator"], None, 6),
        ("zh_pow_after_ok", "计算 2 的 10 次方", "计算结果：1024", ["calculator"], None, 6),
        ("zh_discount_after_ok", "原价 899，打 85 折是多少", "计算结果：764.15", ["calculator"], None, 6),
        (
            "zh_search_after_ok",
            "帮我查一下今天上海会不会下雨",
            "气象台摘要：上海今天多云转阴，降水概率 10%，基本不会下雨。",
            ["web_search"],
            None,
            6,
        ),
        (
            "zh_crispr_after_ok",
            "什么是 CRISPR 基因编辑",
            "百科摘要：CRISPR 是一种细菌免疫系统衍生的基因编辑技术，可精确切割 DNA。",
            ["web_search"],
            None,
            6,
        ),
        (
            "zh_bj_after_ok",
            "北京明天最高气温多少度",
            "预报：北京明天最高气温 26℃，最低 15℃，晴转多云。",
            ["web_search"],
            None,
            6,
        ),
        (
            "zh_news_after_ok",
            "最新的全球科技新闻头条有哪些",
            "检索摘要：1) 新一代开源模型发布；2) 芯片制程突破报道；3) 卫星互联网商用进展。",
            ["web_search"],
            None,
            6,
        ),
        (
            "zh_fx_after_ok",
            "查一下美元兑人民币今日大概汇率",
            "行情摘要：USD/CNY 约 7.24（仅供参考，非交易报价）。",
            ["web_search"],
            None,
            6,
        ),
        (
            "zh_code_after_ok",
            "用 Python 写一段代码求列表中位数",
            "```python\ndef median(xs):\n    s=sorted(xs); n=len(s)\n    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "zh_sort_after_ok",
            "写一个 Python 快速排序函数",
            "```python\ndef qsort(a):\n    if len(a)<=1: return a\n    p=a[0]\n    return qsort([x for x in a[1:] if x<p])+[p]+qsort([x for x in a[1:] if x>=p])\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "zh_plot_after_ok",
            "用 matplotlib 画一条正弦曲线的代码",
            "```python\nimport numpy as np, matplotlib.pyplot as plt\nx=np.linspace(0,2*np.pi,200)\nplt.plot(x,np.sin(x)); plt.show()\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "zh_regex_after_ok",
            "用 Python 正则提取邮箱地址写个函数",
            "```python\nimport re\ndef emails(s): return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', s)\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "zh_rag_after_ok",
            "根据公司内部知识库，年假最多可以请几天？",
            "知识库命中：《员工手册》第 4.2 条：司龄满 1 年享有 5 天年假，满 10 年享有 15 天。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "zh_expense_after_ok",
            "查一下内部资料里的差旅报销流程",
            "知识库：差旅报销需提交申请单→发票→行程单，财务 3 个工作日内审核。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "zh_vpn_after_ok",
            "公司内网 VPN 开通流程在知识库哪一章",
            "知识库：《IT 服务手册》第 6 章：提交工单→主管审批→IT 开通，通常 1 个工作日。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "zh_remote_after_ok",
            "内部远程办公政策允许每周居家几天",
            "知识库：《远程办公指引》：经主管同意，每周最多居家 2 天。",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "zh_multi_turn",
            "刚才算出来的结果再解释一下含义",
            "计算结果：56088",
            ["calculator"],
            "上一轮计算器结果：56088",
            5,
        ),
        (
            "zh_temp_after_ok",
            "搜索一下北京今天气温",
            "北京今天白天最高 22℃，夜间最低 12℃，空气质量良。",
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
                lang,
                expected_tool="none",
                expected_action="answer",
                expected_sufficient=True,
                tool_result=tr,
                history=hist,
                context=ctx,
                expected_score_min=smin,
            )
        )

    fails = [
        ("zh_search_fail", "帮我查一下今天上海会不会下雨", "error: search quota exceeded", ["web_search"], "web_search"),
        ("zh_calc_fail", "计算 999/0", "error: division by zero", ["calculator"], "calculator"),
        ("zh_code_fail", "用 Python 写快速排序", "error: interpreter timeout", ["code_interpreter"], "code_interpreter"),
        ("zh_rag_fail", "根据公司知识库查福利补贴", "error: retrieval index unavailable", ["rag_retrieve"], "rag_retrieve"),
        ("zh_search_timeout", "搜索一下量子计算最近有什么突破", "error: upstream timeout", ["web_search"], "web_search"),
        ("zh_calc_overflow", "计算 10 ** 100000", "error: numeric overflow", ["calculator"], "calculator"),
        ("zh_code_syntax", "用 pandas 把 CSV 按列求和并输出", "error: SyntaxError: unexpected EOF", ["code_interpreter"], "code_interpreter"),
        ("zh_rag_empty", "员工手册里加班调休怎么算？", "error: no documents matched", ["rag_retrieve"], "rag_retrieve"),
    ]
    for i, q, tr, hist, tool in fails:
        rows.append(
            item(
                i,
                q,
                "after_tool_fail",
                lang,
                expected_tool=tool,
                expected_action="retry_tool",
                expected_sufficient=False,
                tool_result=tr,
                history=hist,
                expected_score_min=0,
                expected_score_max=4,
            )
        )

    # edge cases within Chinese (ambiguous routing)
    edges = [
        ("zh_edge_philosophy", "时间到底是什么", "none", "answer", True),
        ("zh_edge_public_not_rag", "公开资料里，世界卫生组织总部在哪个城市", "web_search", "call_tool", False),
        ("zh_edge_calc_word", "把三乘以七点五算一下", "calculator", "call_tool", False),
        ("zh_edge_code_not_search", "写个冒泡排序的 Python 实现", "code_interpreter", "call_tool", False),
        ("zh_edge_internal", "查内部制度：试用期能不能提前转正", "rag_retrieve", "call_tool", False),
        ("zh_edge_thanks", "太感谢了，问题解决了", "none", "answer", True),
    ]
    for i, q, tool, action, suff in edges:
        rows.append(
            item(
                i,
                q,
                "edge",
                lang,
                expected_tool=tool,
                expected_action=action,
                expected_sufficient=suff,
            )
        )

    return rows


def build_en() -> list[dict]:
    lang = "en"
    rows: list[dict] = []

    rows += _pack_simple(
        lang,
        "direct",
        [
            ("en_greet", "Hi, how are you today?"),
            ("en_thanks", "Thanks a lot for your help"),
            ("en_bye", "Goodbye, talk next time"),
            ("en_translate", "Translate to Chinese: The meeting is canceled today."),
            ("en_opinion", "Do you prefer spring or autumn? Just your take."),
            ("en_joke", "Tell me a short clean joke"),
            ("en_summarize", "Summarize in one sentence: Apples are common fruits rich in vitamins."),
            ("en_meaning", "What is the meaning of life?"),
            ("en_encourage", "I'm feeling down — say two encouraging sentences"),
            ("en_goodnight", "Wish me good night in a gentle tone"),
            ("en_rewrite", "Make this more polite: Send me the file right now"),
            ("en_brainstorm", "Suggest three nice English names for a book club"),
            ("en_pets", "Cats or dogs for an apartment — brief opinion only"),
            ("en_poem", "Write two short lines about autumn"),
        ],
        tool="none",
        action="answer",
        sufficient=True,
    )

    rows += _pack_simple(
        lang,
        "calculator",
        [
            ("en_calc_mul", "Please compute 384 * 27"),
            ("en_percent", "What is 20% of 85? Calculate it"),
            ("en_calc_expr", "Calculate (120-45)/5+8"),
            ("en_calc_sqrt", "What is the square root of 144?"),
            ("en_calc_mix", "What is 3.5 times 2.2 plus 10?"),
            ("en_calc_div", "Divide 98765 by 5 with the calculator"),
            ("en_calc_pow", "Compute 2 to the power of 10"),
            ("en_calc_avg", "Average of 12, 18, 25, and 30"),
            ("en_calc_tax", "Pre-tax 10000 with 10% tax — after-tax amount?"),
            ("en_calc_discount", "Price 899 with 15% off — final price?"),
            ("en_calc_interest", "Principal 5000 at 3% simple interest for one year"),
            ("en_calc_meter", "Convert 3.2 kilometers to meters (just compute)"),
        ],
        tool="calculator",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "search",
        [
            ("en_search_weather", "Will it rain in Shanghai today? Look it up"),
            ("en_quantum", "Search recent breakthroughs in quantum computing"),
            ("en_crispr", "What is CRISPR gene editing?"),
            ("en_transformer", "What is the Transformer architecture?"),
            ("en_news", "What are today's top global tech headlines?"),
            ("en_hinton", "Who is Turing Award winner Geoffrey Hinton?"),
            ("en_stock", "What is roughly today's close for the S&P 500?"),
            ("en_weather_nyc", "What's the high temperature in New York tomorrow?"),
            ("en_quantum_diff", "How does quantum computing differ from classical?"),
            ("en_cloud", "Who are the major cloud service providers?"),
            ("en_passport", "What documents are needed to apply for a US passport (public info)?"),
            ("en_mile", "How many kilometers in one mile? Cite an authority"),
            ("en_olympics", "Where was the most recent Summer Olympics held?"),
            ("en_openai", "Which major models has OpenAI released recently (public reports)?"),
            ("en_fx", "What's today's approximate USD to EUR rate?"),
            ("en_turing", "Who was Alan Turing? Give a short public bio"),
        ],
        tool="web_search",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "code",
        [
            ("en_code_median", "Write Python code to compute the median of a list"),
            ("en_code_sort", "Write a Python quicksort function"),
            ("en_code_pandas", "Use pandas to sum columns of a CSV and print them"),
            ("en_code_bfs_tree", "Implement level-order traversal of a binary tree in code"),
            ("en_code_debug", "How do I fix this Python error: list index out of range"),
            ("en_code_regex", "Write a Python function to extract emails with regex"),
            ("en_code_plot", "Matplotlib code to plot a sine wave"),
            ("en_code_sqlite", "Python example connecting to SQLite and running a query"),
            ("en_code_bfs", "Implement BFS on a graph in Python"),
            ("en_code_flatten", "Write a function to flatten a nested dict with dotted keys"),
            ("en_code_thread", "Minimal Python multithread file download example"),
            ("en_code_pytest", "Write a pytest case for an add(a, b) function"),
        ],
        tool="code_interpreter",
        action="call_tool",
        sufficient=False,
    )

    rows += _pack_simple(
        lang,
        "rag",
        [
            ("en_rag_leave", "Per our internal knowledge base, what's the max annual leave days?"),
            ("en_rag_expense", "Look up the internal travel reimbursement process"),
            ("en_rag_ot", "In the employee handbook, how is overtime time-off calculated?"),
            ("en_rag_attend", "What is our company attendance policy in the knowledge base?"),
            ("en_rag_laptop", "Internal onboarding: how do I request a laptop?"),
            ("en_rag_exit", "From company policy docs, what is on the offboarding checklist?"),
            ("en_rag_vpn", "Which chapter in the KB covers internal VPN setup?"),
            ("en_rag_meal", "What is the meal allowance in the internal benefits handbook?"),
            ("en_rag_usb", "What does our info-sec policy say about USB drives?"),
            ("en_rag_room", "Where in internal docs is the meeting room booking policy?"),
            ("en_rag_probation", "KB: what is the probation review process?"),
            ("en_rag_remote", "Internal remote-work policy: how many WFH days per week?"),
        ],
        tool="rag_retrieve",
        action="call_tool",
        sufficient=False,
    )

    after_ok = [
        ("en_calc_after_ok", "Please compute 384 * 27", "Result: 10368", ["calculator"], None, 6),
        ("en_percent_after_ok", "What is 20% of 85? Calculate it", "Result: 17", ["calculator"], None, 6),
        ("en_expr_after_ok", "Calculate (120-45)/5+8", "Result: 23", ["calculator"], None, 6),
        ("en_pow_after_ok", "Compute 2 to the power of 10", "Result: 1024", ["calculator"], None, 6),
        ("en_discount_after_ok", "Price 899 with 15% off — final price?", "Result: 764.15", ["calculator"], None, 6),
        (
            "en_search_after_ok",
            "Will it rain in Shanghai today? Look it up",
            "Weather brief: Shanghai cloudy, rain chance 10%, unlikely to rain.",
            ["web_search"],
            None,
            6,
        ),
        (
            "en_crispr_after_ok",
            "What is CRISPR gene editing?",
            "Encyclopedia: CRISPR is a gene-editing method derived from bacterial immune systems.",
            ["web_search"],
            None,
            6,
        ),
        (
            "en_nyc_after_ok",
            "What's the high temperature in New York tomorrow?",
            "Forecast: NYC tomorrow high 78°F / 26°C, partly cloudy.",
            ["web_search"],
            None,
            6,
        ),
        (
            "en_news_after_ok",
            "What are today's top global tech headlines?",
            "Headlines: 1) New open model release 2) Chip process news 3) Satellite internet update.",
            ["web_search"],
            None,
            6,
        ),
        (
            "en_fx_after_ok",
            "What's today's approximate USD to EUR rate?",
            "FX brief: USD/EUR ≈ 0.92 (indicative only).",
            ["web_search"],
            None,
            6,
        ),
        (
            "en_code_after_ok",
            "Write Python code to compute the median of a list",
            "```python\ndef median(xs):\n    s=sorted(xs); n=len(s)\n    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "en_sort_after_ok",
            "Write a Python quicksort function",
            "```python\ndef qsort(a):\n    if len(a)<=1: return a\n    p=a[0]\n    return qsort([x for x in a[1:] if x<p])+[p]+qsort([x for x in a[1:] if x>=p])\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "en_plot_after_ok",
            "Matplotlib code to plot a sine wave",
            "```python\nimport numpy as np, matplotlib.pyplot as plt\nx=np.linspace(0,2*np.pi,200)\nplt.plot(x,np.sin(x)); plt.show()\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "en_regex_after_ok",
            "Write a Python function to extract emails with regex",
            "```python\nimport re\ndef emails(s): return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', s)\n```",
            ["code_interpreter"],
            None,
            6,
        ),
        (
            "en_rag_after_ok",
            "Per our internal knowledge base, what's the max annual leave days?",
            "KB hit: Employee Handbook §4.2 — 5 days after 1 year; 15 days after 10 years.",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "en_expense_after_ok",
            "Look up the internal travel reimbursement process",
            "KB: submit form → receipts → itinerary; finance reviews within 3 business days.",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "en_vpn_after_ok",
            "Which chapter in the KB covers internal VPN setup?",
            "KB: IT Service Manual ch.6 — ticket → manager approve → IT provision (~1 day).",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "en_remote_after_ok",
            "Internal remote-work policy: how many WFH days per week?",
            "KB: Remote Work Guide — up to 2 WFH days/week with manager approval.",
            ["rag_retrieve"],
            None,
            6,
        ),
        (
            "en_multi_turn",
            "Explain what that calculated number means",
            "Result: 56088",
            ["calculator"],
            "Previous calculator result: 56088",
            5,
        ),
        (
            "en_temp_after_ok",
            "Search today's temperature in Beijing",
            "Beijing today: high 22°C, low 12°C, air quality good.",
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
                lang,
                expected_tool="none",
                expected_action="answer",
                expected_sufficient=True,
                tool_result=tr,
                history=hist,
                context=ctx,
                expected_score_min=smin,
            )
        )

    fails = [
        ("en_search_fail", "Will it rain in Shanghai today? Look it up", "error: search quota exceeded", ["web_search"], "web_search"),
        ("en_calc_fail", "Compute 999/0", "error: division by zero", ["calculator"], "calculator"),
        ("en_code_fail", "Write a Python quicksort", "error: interpreter timeout", ["code_interpreter"], "code_interpreter"),
        ("en_rag_fail", "Look up benefits in the company knowledge base", "error: retrieval index unavailable", ["rag_retrieve"], "rag_retrieve"),
        ("en_search_timeout", "Search recent breakthroughs in quantum computing", "error: upstream timeout", ["web_search"], "web_search"),
        ("en_calc_overflow", "Compute 10 ** 100000", "error: numeric overflow", ["calculator"], "calculator"),
        ("en_code_syntax", "Use pandas to sum columns of a CSV and print them", "error: SyntaxError: unexpected EOF", ["code_interpreter"], "code_interpreter"),
        ("en_rag_empty", "In the employee handbook, how is overtime time-off calculated?", "error: no documents matched", ["rag_retrieve"], "rag_retrieve"),
    ]
    for i, q, tr, hist, tool in fails:
        rows.append(
            item(
                i,
                q,
                "after_tool_fail",
                lang,
                expected_tool=tool,
                expected_action="retry_tool",
                expected_sufficient=False,
                tool_result=tr,
                history=hist,
                expected_score_min=0,
                expected_score_max=4,
            )
        )

    edges = [
        ("en_edge_philosophy", "What really is time?", "none", "answer", True),
        ("en_edge_public_not_rag", "Where is WHO headquarters? Use public sources.", "web_search", "call_tool", False),
        ("en_edge_calc_word", "Please multiply three by seven point five", "calculator", "call_tool", False),
        ("en_edge_code_not_search", "Implement bubble sort in Python", "code_interpreter", "call_tool", False),
        ("en_edge_internal", "Internal policy: can probation end early?", "rag_retrieve", "call_tool", False),
        ("en_edge_thanks", "Really appreciate it — that fixed my issue", "none", "answer", True),
    ]
    for i, q, tool, action, suff in edges:
        rows.append(
            item(
                i,
                q,
                "edge",
                lang,
                expected_tool=tool,
                expected_action=action,
                expected_sufficient=suff,
            )
        )

    return rows


def _write(lang: str, rows: list[dict]) -> None:
    assert len(rows) == 100, (lang, len(rows))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate ids in {lang}"
    print(lang, "n=", len(rows), "by_cat=", dict(Counter(r["category"] for r in rows)))
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    name = f"traces.{lang}.json"
    for base in (ROOT / "eval", ROOT / "src" / "laya_thalamus" / "resources"):
        path = base / name
        path.write_text(text, encoding="utf-8")
        print("wrote", path)


def main() -> None:
    _write("zh", build_zh())
    _write("en", build_en())
    # remove legacy combined traces.json if present
    for base in (ROOT / "eval", ROOT / "src" / "laya_thalamus" / "resources"):
        legacy = base / "traces.json"
        if legacy.exists():
            legacy.unlink()
            print("removed", legacy)


if __name__ == "__main__":
    main()
