"""FastAPI 入口。

启动：uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.react_agent import ReactAgent
from api.routes import chat
from utils.logger_handler import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Agent 只建一次：Checkpointer 连接、Chroma 连接都在构造里完成
    app.state.agent = ReactAgent()
    logger.info("服务启动完成 | agent 已就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(title="扫地机器人客服 Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}
