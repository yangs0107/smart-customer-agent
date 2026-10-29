"""Streamlit 前端。

硬约束：只依赖 streamlit / httpx / 标准库，不得 import agent、model、rag。
前后端唯一通道是 HTTP。

启动：streamlit run ui/app.py
"""

import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="扫地机器人客服 Agent", page_icon="🤖", layout="centered")


# ---------------------------------------------------------------- HTTP 封装

def _health() -> bool:
    try:
        return httpx.get(f"{API_BASE}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _new_thread() -> str | None:
    try:
        resp = httpx.post(f"{API_BASE}/chat/new", timeout=5.0)
        resp.raise_for_status()
        return resp.json()["thread_id"]
    except httpx.HTTPError:
        return None


def _history(thread_id: str) -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE}/chat/history", params={"thread_id": thread_id}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()["messages"]
    except httpx.HTTPError:
        return []


def _sessions() -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE}/chat/sessions", timeout=5.0)
        resp.raise_for_status()
        return resp.json()["sessions"]
    except httpx.HTTPError:
        return []


def _delete_session(thread_id: str) -> bool:
    try:
        resp = httpx.delete(f"{API_BASE}/chat/sessions/{thread_id}", timeout=10.0)
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _rename_session(thread_id: str, title: str) -> bool:
    try:
        resp = httpx.patch(
            f"{API_BASE}/chat/sessions/{thread_id}", json={"title": title}, timeout=10.0
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _iter_sse(question: str, thread_id: str):
    """消费后端 SSE，逐帧产出 (事件名, 数据字典)。

    必须用 httpx.stream：requests 会读完整个响应才返回，无法逐字渲染。
    """
    payload = {"question": question, "thread_id": thread_id}
    timeout = httpx.Timeout(300.0, connect=5.0)
    with httpx.stream("POST", f"{API_BASE}/chat/stream", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        event_name: str | None = None
        data_lines: list[str] = []

        for line in resp.iter_lines():
            if line == "":
                if event_name:
                    raw = "".join(data_lines)
                    try:
                        yield event_name, json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        yield event_name, {"raw": raw}
                event_name, data_lines = None, []
                continue

            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())


# ---------------------------------------------------------------- 会话状态

def _init_state() -> None:
    # thread_id 落到 URL：浏览器刷新会重建 session_state，只有 URL 能跨刷新保留会话
    thread_id = st.query_params.get("tid") or st.session_state.get("thread_id")

    if not thread_id:
        thread_id = _new_thread()
        if thread_id:
            st.query_params["tid"] = thread_id

    st.session_state.thread_id = thread_id
    st.session_state.setdefault("renaming", None)

    if "messages" not in st.session_state:
        st.session_state.messages = _history(thread_id) if thread_id else []


def _switch_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.messages = _history(thread_id)
    st.session_state.renaming = None
    st.query_params["tid"] = thread_id


def _reset_thread() -> None:
    thread_id = _new_thread()
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    st.session_state.renaming = None
    if thread_id:
        st.query_params["tid"] = thread_id


def _short(text: str, limit: int = 16) -> str:
    text = (text or "未命名").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------- 侧栏

def _sidebar(online: bool) -> None:
    with st.sidebar:
        if st.button("＋ 新建对话", use_container_width=True, type="primary"):
            _reset_thread()
            st.rerun()

        st.write("")
        sessions = _sessions() if online else []
        st.caption(f"历史对话（{len(sessions)}）")

        if not sessions:
            st.caption("暂无历史对话")

        for item in sessions:
            thread_id = item["thread_id"]
            is_active = thread_id == st.session_state.get("thread_id")
            renaming = st.session_state.get("renaming") == thread_id

            col_label, col_edit, col_del = st.columns([0.6, 0.2, 0.2], gap="small")
            label = ("▸ " if is_active else "") + _short(item.get("title", ""))
            if col_label.button(
                label,
                key=f"sel_{thread_id}",
                use_container_width=True,
                disabled=is_active,
                help=f"{item.get('updated_at', '')}｜{thread_id[:12]}",
            ):
                _switch_thread(thread_id)
                st.rerun()

            if col_edit.button("✏️", key=f"ren_{thread_id}", help="重命名"):
                st.session_state.renaming = None if renaming else thread_id
                st.rerun()

            if col_del.button("🗑", key=f"del_{thread_id}", help="删除该对话"):
                _delete_session(thread_id)
                if is_active:
                    _reset_thread()
                st.rerun()

            if renaming:
                new_title = st.text_input(
                    "对话名称",
                    value=item.get("title", ""),
                    key=f"title_{thread_id}",
                    label_visibility="collapsed",
                    placeholder="输入新的对话名称",
                )
                col_save, col_cancel = st.columns(2, gap="small")
                if col_save.button("保存", key=f"save_{thread_id}", type="primary", use_container_width=True):
                    if _rename_session(thread_id, new_title):
                        st.session_state.renaming = None
                    else:
                        st.toast("重命名失败，请检查后端", icon="⚠️")
                    st.rerun()
                if col_cancel.button("取消", key=f"cancel_{thread_id}", use_container_width=True):
                    st.session_state.renaming = None
                    st.rerun()

        st.divider()
        if online:
            st.caption("🟢 后端在线")
        else:
            st.caption("🔴 后端未连接")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    st.title("🤖 扫地机器人客服 Agent")
    st.caption("RAG 检索 + 精排 ｜ 多轮记忆 ｜ 报告生成")

    online = _health()
    _sidebar(online)

    if not online:
        st.error(
            "**无法连接后端服务。**\n\n"
            f"请确认已启动 `uvicorn api.main:app --port 8000`，当前地址 `{API_BASE}`。"
        )
        st.stop()

    if not st.session_state.get("thread_id"):
        st.warning("会话创建失败，请检查后端日志后点击侧栏「新建对话」。")
        st.stop()

    if not st.session_state.messages:
        st.info("这是一个新对话。可以问我产品功能、故障排查、清洁保养，或让我生成使用报告。")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("请输入你的问题…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("正在思考…", expanded=True)
        placeholder = st.empty()
        answer = ""

        try:
            for event, data in _iter_sse(question, st.session_state.thread_id):
                if os.getenv("UI_DEBUG"):
                    print(f"[UI_DEBUG] {event} | {str(data)[:120]}", flush=True)

                if event == "token":
                    answer += data.get("content", "")
                    placeholder.markdown(answer)
                elif event == "tool_call":
                    status.write(f"🔧 调用 `{data.get('name')}` ｜ 参数 `{data.get('args')}`")
                elif event == "tool_result":
                    snippet = str(data.get("content", ""))[:120].replace("\n", " ")
                    status.write(f"📄 `{data.get('name')}` 返回：{snippet}…")
                elif event == "done":
                    status.update(label="完成", state="complete", expanded=False)
                elif event == "error":
                    status.update(label="出错", state="error", expanded=True)
                    st.error(data.get("message", "未知错误"))
        except httpx.HTTPError as exc:
            status.update(label="连接失败", state="error", expanded=True)
            st.error(f"与后端通信失败：{type(exc).__name__} — {exc}")
            return

        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})


_init_state()
main()
