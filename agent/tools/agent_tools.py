"""Agent 可调用的工具集（7 个）。

约定：
- 每个工具的 docstring 就是模型看到的 description，直接决定调用准确性，必须写清「什么时候用」。
- 工具内部不打印、不 print 调试信息，统一走 logger。
"""

import csv
import json
import random
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# ---------------------------------------------------------------- RAG

_rag_service = None


def _get_rag_service():
    """延迟初始化：避免导入时就建 Chroma 连接、读提示词。"""
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagSummarizeService

        _rag_service = RagSummarizeService()
    return _rag_service


@tool
def rag_summarize(query: str) -> str:
    """检索扫地机器人知识库并作答。适用于产品功能、故障排查、清洁保养、耗材更换、选购建议等问题。
    凡涉及产品事实的问题都应先调用本工具，不要凭记忆回答。"""
    return _get_rag_service().rag_summarize(query)


# ---------------------------------------------------------------- 模拟数据工具

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟数据）。用于回答与天气相关的闲聊或使用建议。"""
    return random.choice(
        [f"{city}今天晴，气温 24~31℃，适合开窗通风。",
         f"{city}今天多云转小雨，气温 21~27℃，湿度较高。",
         f"{city}今天阴，气温 19~25℃，空气干燥，注意补水。"]
    )


@tool
def get_user_location() -> str:
    """获取当前用户所在城市。回答与本地相关的任何问题前应先调用。"""
    return random.choice(["深圳", "合肥", "杭州"])


@tool
def get_user_id() -> str:
    """获取当前用户ID。查询用户使用数据前必须先调用。"""
    return str(random.randint(1001, 1010))


@tool
def get_current_month() -> str:
    """获取当前统计月份，格式 YYYY-MM。查询用户使用数据前必须先调用。
    注意：本工具返回的是系统统计月份，若用户明确指定了月份，以用户指定的为准。"""
    return f"2025-{random.randint(1, 12):02d}"


# ---------------------------------------------------------------- 外部数据

@lru_cache(maxsize=1)
def _load_records() -> dict[tuple[str, str], dict[str, str]]:
    """载入 records.csv，键为 (用户ID, 时间)。

    字段内是**字面 \\n**（反斜杠+n），这里统一换成真换行。
    """
    path = Path(get_abs_path(agent_conf["external_data_path"]))
    index: dict[tuple[str, str], dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cleaned = {k: (v or "").replace("\\n", "\n").strip() for k, v in row.items()}
            index[(cleaned["用户ID"], cleaned["时间"])] = cleaned
    logger.info("外部数据载入完成 | %d 条记录", len(index))
    return index


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """按用户ID与月份查询真实使用记录（清洁效率、耗材、同类对比）。
    参数必须来自 get_user_id 与 get_current_month 的返回值，不要自行编造。查不到返回空字符串。"""
    row = _load_records().get((user_id.strip(), month.strip()))
    if not row:
        logger.warning("外部数据未命中 | user_id=%s month=%s", user_id, month)
        return ""
    return json.dumps(row, ensure_ascii=False)


# ---------------------------------------------------------------- 场景信号

@tool
def fill_context_for_report() -> str:
    """切换到报告模式。当用户要求生成使用报告、使用总结、月度报告时，必须首先调用本工具。
    报告需要用户ID、城市、月份，建议在同一次工具调用中一并取齐。"""
    return "已切换到报告模式"
