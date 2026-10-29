"""ReAct Agent 装配与两个对外出口。

- execute_stream(query)：命令行调试用，直接打印文本
- stream_events(question, thread_id)：API 用，产出结构化事件
    {"event": "token"|"tool_call"|"tool_result"|"done"|"error", "data": {...}}
"""

import sqlite3
from collections.abc import Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.session_store import SessionStore
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    AgentContext,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from model.factory import chat_model
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

TOOLS = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]


class ReactAgent:
    def __init__(self) -> None:
        db_path = get_abs_path(agent_conf["memory_db_path"])
        # check_same_thread=False：SQLite 连接会被 FastAPI 的线程池跨线程使用
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self._agent = create_agent(
            model=chat_model,
            tools=TOOLS,
            middleware=[
                log_before_model,
                monitor_tool,
                report_prompt_switch,
                SummarizationMiddleware(
                    model=chat_model,
                    trigger=("messages", agent_conf["summary_trigger_messages"]),
                    keep=("messages", agent_conf["summary_keep_messages"]),
                ),
            ],
            checkpointer=SqliteSaver(conn),
            context_schema=AgentContext,
        )
        self._sessions = SessionStore(db_path)
        logger.info("ReactAgent 初始化完成 | 工具 %d 个", len(TOOLS))

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _events_from_message(message) -> Iterator[dict]:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                yield {"event": "tool_call", "data": {"name": call["name"], "args": call["args"]}}
        elif isinstance(message, ToolMessage):
            yield {
                "event": "tool_result",
                "data": {"name": message.name, "content": str(message.content)[:500]},
            }

    # ------------------------------------------------------------ 出口 1：CLI

    def execute_stream(self, query: str, thread_id: str = "cli") -> Iterator[str]:
        for event in self.stream_events(query, thread_id):
            if event["event"] == "token":
                yield event["data"]["content"]
            elif event["event"] == "tool_call":
                print(f"\n[工具] {event['data']['name']}({event['data']['args']})")
            elif event["event"] == "error":
                yield f"\n[错误] {event['data']['message']}"

    # ------------------------------------------------------------ 出口 2：API

    def stream_events(self, question: str, thread_id: str) -> Iterator[dict]:
        logger.info("收到提问 | thread=%s | %s", thread_id, question)
        self._sessions.touch(thread_id, question)
        try:
            for mode, payload in self._agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=self._config(thread_id),
                context=AgentContext(report=False),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, _meta = payload
                    if isinstance(chunk, AIMessageChunk):
                        text = chunk.content
                        # 只转发正文；reasoning token 不进入对话流
                        if isinstance(text, str) and text:
                            yield {"event": "token", "data": {"content": text}}
                elif mode == "updates":
                    for update in payload.values():
                        if not isinstance(update, dict):
                            continue
                        for message in update.get("messages", []) or []:
                            yield from self._events_from_message(message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream_events 执行失败")
            yield {"event": "error", "data": {"message": str(exc)}}
            return

        yield {"event": "done", "data": {"thread_id": thread_id}}

    # ------------------------------------------------------------ 历史

    def get_history(self, thread_id: str) -> list[dict]:
        """返回可渲染的对话历史（只取正文，跳过工具消息）。"""
        state = self._agent.get_state(self._config(thread_id))
        messages = state.values.get("messages", []) if state.values else []

        history = []
        for message in messages:
            if message.type == "human" and isinstance(message.content, str):
                history.append({"role": "user", "content": message.content})
            elif message.type == "ai" and isinstance(message.content, str) and message.content:
                history.append({"role": "assistant", "content": message.content})
        return history

    # ------------------------------------------------------------ 会话管理

    def list_sessions(self) -> list[dict]:
        return self._sessions.list_all()

    def delete_session(self, thread_id: str) -> None:
        self._sessions.delete(thread_id)

    def rename_session(self, thread_id: str, title: str) -> bool:
        return self._sessions.rename(thread_id, title)


def main() -> None:
    agent = ReactAgent()
    print("已进入命令行对话，输入 exit 退出。")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"exit", "quit"}:
            break
        if not question:
            continue

        print("助手: ", end="", flush=True)
        for piece in agent.execute_stream(question):
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
