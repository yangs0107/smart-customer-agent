"""FastAPI 入口。

启动：uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from utils.path_tool import get_abs_path

# 必须在导入 model/agent 之前加载 .env：model.factory 在导入时就会读取密钥
load_dotenv(get_abs_path(".env"))

from agent.react_agent import ReactAgent  # noqa: E402
from api.routes import chat  # noqa: E402
from utils.logger_handler import logger  # noqa: E402


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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """单端口模式：同一个端口同时提供页面与接口，便于对外分享。"""
    return FileResponse(get_abs_path("api/static/index.html"), media_type="text/html")
