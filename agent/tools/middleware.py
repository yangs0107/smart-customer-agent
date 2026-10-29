"""Agent 中间件（3 个）+ 运行期上下文定义。

数据流：
    wrap_tool_call(monitor_tool) 在工具执行前写 context["report"]
        ↓（同一次运行的共享上下文）
    dynamic_prompt(report_prompt_switch) 在下一次模型调用前读取它
"""

from typing import TypedDict

from langchain.agents.middleware import before_model, dynamic_prompt, wrap_tool_call

from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class AgentContext(TypedDict, total=False):
    """跨中间件共享的运行期上下文。用 TypedDict 以支持 dict 式读写。"""

    report: bool


def _load_prompt(key: str) -> str:
    with open(get_abs_path(prompts_conf[key]), "r", encoding="utf-8") as f:
        return f.read()


MAIN_PROMPT = _load_prompt("main_prompt")
REPORT_PROMPT = _load_prompt("report_prompt")


@before_model
def log_before_model(state, runtime):
    """记录每轮模型调用的消息条数，兼作轮次计数器。"""
    logger.info("before_model | 消息数=%d", len(state["messages"]))
    return None


@wrap_tool_call
def monitor_tool(request, handler):
    """记录工具名与参数；fill_context_for_report 触发时翻转报告开关。"""
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    logger.info("tool_call | %s | %s", name, args)

    if name == "fill_context_for_report":
        request.runtime.context["report"] = True
        logger.info("context 切换 | report=True")

    return handler(request)


@dynamic_prompt
def report_prompt_switch(request):
    """每次模型调用前实时求值，按上下文返回对应提示词。"""
    if request.runtime.context.get("report"):
        logger.info("prompt 选择 | 报告版")
        return REPORT_PROMPT
    return MAIN_PROMPT
