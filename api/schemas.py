"""接口层的数据契约。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    thread_id: str | None = Field(None, description="会话ID，缺省则新建")


class NewThreadResponse(BaseModel):
    thread_id: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class SessionItem(BaseModel):
    thread_id: str
    title: str = ""
    updated_at: str = ""


class SessionListResponse(BaseModel):
    sessions: list[SessionItem]


class DeleteResponse(BaseModel):
    deleted: str


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=40)


class RenameResponse(BaseModel):
    thread_id: str
    title: str
