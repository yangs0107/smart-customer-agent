"""精排客户端：走 DashScope 原生端点，不属于 OpenAI 兼容协议。"""

import os

import httpx

from utils.config_handler import rag_conf
from utils.logger_handler import logger


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """对候选切片重新打分。

    Returns:
        [{"index": 原始下标, "relevance_score": 分数}, ...]，按分数降序。
    """
    if not documents:
        return []

    top_n = top_n or rag_conf["rerank_top_n"]

    resp = httpx.post(
        rag_conf["rerank_url"],
        headers={
            "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
            "Content-Type": "application/json",
            # 缺这个 header 会被网关按错误服务路由
            "X-DashScope-Service": "text-rerank",
        },
        json={
            "model": rag_conf["rerank_model_name"],
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": top_n},
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["output"]["results"]
    logger.info("精排完成 | 候选 %d 条 → 保留 %d 条", len(documents), len(results))
    return results
