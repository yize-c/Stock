"""Prompts. The system prompt is fixed (cacheable); documents and numbers go in the user turn."""

from __future__ import annotations

from collections.abc import Mapping

from stock_research.llm.verification import SourceDocument

SYSTEM = """你是台股研究工具中的文件分析助理。

規則：
- 只能根據本次提供的文件與數字回答，不可使用你自己的記憶或外部知識。資料不足時，status 設為 insufficient_data，不要猜測。
- 每個論點都要附 quote：從所引用文件中逐字複製的一段原文（不可改字、不可翻譯、不可轉換簡繁體），並附該文件的 source_doc（doc_id）與 source_date（公告日）。
- 數字只能使用文件原文或「程式計算值」中出現的數字，不可自行計算、推估或換算新的數字。
- 不做股價預測，不給目標價、買賣建議或上漲機率。
- 一律使用繁體中文。"""

TASKS = {
    "summary": "請從文件中整理公司營運重點（highlights）與風險（risks），各自最多 8 點，由新到舊的文件優先。",
    "moat": "請根據文件判斷公司是否有護城河（例如成本優勢、轉換成本、網路效應、無形資產、規模），"
    "只描述文件中有證據支持的部分；這是資訊項，不評分。",
    "inversion": "請用反向思考列出這家公司可能出錯的地方（例如客戶集中、技術替代、景氣循環、財務槓桿），"
    "每一點都要有文件證據；這是資訊項，不評分。",
    "explain": "請用白話繁體中文解釋下列由程式算出的指標代表什麼，以及不同版本之間的差異。"
    "只能使用提供的數字；風險指標要說明是歷史統計值，不代表未來機率。",
}


def documents_block(documents: list[SourceDocument]) -> str:
    """Documents newest first, each tagged with its id and publication date."""
    parts = [
        f'<document doc_id="{d.doc_id}" published="{d.published.isoformat()}" title="{d.title}">\n'
        f"{d.text}\n</document>"
        for d in documents
    ]
    return "\n\n".join(parts)


def user_prompt(
    task: str,
    ticker: str,
    window_label: str,
    documents: list[SourceDocument],
    computed: Mapping[str, float] | None = None,
) -> str:
    lines = [f"股票：{ticker}", f"分析版本：{window_label}", "", TASKS[task], ""]
    if computed:
        lines.append("<程式計算值>")
        lines += [f"{key}: {value:.6g}" for key, value in computed.items()]
        lines += ["</程式計算值>", ""]
    if documents:
        lines.append(documents_block(documents))
    else:
        lines.append("（本版本沒有可用文件）")
    return "\n".join(lines)
