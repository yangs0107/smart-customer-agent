"""对话相关接口：SSE 流式、历史查询、新建会话。"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    DeleteResponse,
    HistoryResponse,
    NewThreadResponse,
    RenameRequest,
    RenameResponse,
    SessionListResponse,
)
from utils.logger_handler import logger

router = APIRouter(prefix="/chat", tags=["chat"])

# SSE 必须禁用缓冲，否则前端拿到的是攒满一整块才吐一次
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _frame(event: str, data: dict) -> str:
    """把事件字典拼成 SSE 帧：event 行 + data 行 + 空行结尾。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    agent = request.app.state.agent
    thread_id = payload.thread_id or uuid.uuid4().hex

    def event_source():
        try:
            for event in agent.stream_events(payload.question, thread_id):
                yield _frame(event["event"], event["data"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE 推流异常")
            yield _frame("error", {"message": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/history", response_model=HistoryResponse)
def chat_history(request: Request, thread_id: str = Query(..., min_length=1)) -> HistoryResponse:
    agent = request.app.state.agent
    return HistoryResponse(thread_id=thread_id, messages=agent.get_history(thread_id))


@router.post("/new", response_model=NewThreadResponse)
def new_thread() -> NewThreadResponse:
    return NewThreadResponse(thread_id=uuid.uuid4().hex)


# ---------------------------------------------------------------- 会话管理

@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(request: Request) -> SessionListResponse:
    return SessionListResponse(sessions=request.app.state.agent.list_sessions())


@router.delete("/sessions/{thread_id}", response_model=DeleteResponse)
def delete_session(thread_id: str, request: Request) -> DeleteResponse:
    request.app.state.agent.delete_session(thread_id)
    return DeleteResponse(deleted=thread_id)


@router.patch("/sessions/{thread_id}", response_model=RenameResponse)
def rename_session(thread_id: str, payload: RenameRequest, request: Request) -> RenameResponse:
    title = payload.title.strip()
    if not request.app.state.agent.rename_session(thread_id, title):
        raise HTTPException(status_code=404, detail="会话不存在或名称为空")
    return RenameResponse(thread_id=thread_id, title=title)
