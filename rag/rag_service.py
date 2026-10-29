"""RAG 检索服务：粗排召回 → 精排 → 拼接 → LLM 总结。

对外只暴露 rag_summarize(query) -> str，
Agent 层不感知候选数、精排、提示词模板等细节。
"""

from langchain_core.documents import Document

from model.factory import chat_model
from model.rerank import rerank
from rag.vector_store import get_vector_store
from utils.config_handler import chroma_conf, prompts_conf, rag_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class RagSummarizeService:
    def __init__(self) -> None:
        self._store = get_vector_store()
        prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt"])
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._prompt = f.read()

    def recall(self, query: str, k: int | None = None) -> list[Document]:
        """纯向量召回，供调试用。"""
        return self._store.similarity_search(query, k=k or chroma_conf["recall_k"])

    def _refine(self, query: str, candidates: list[Document]) -> list[Document]:
        """精排；接口异常时降级为召回顺序，保证链路不断。"""
        top_n = rag_conf["rerank_top_n"]
        try:
            ranked = rerank(query, [d.page_content for d in candidates], top_n=top_n)
            return [candidates[r["index"]] for r in ranked]
        except Exception as exc:  # noqa: BLE001
            logger.warning("精排失败，降级为向量顺序 | %s", exc)
            return candidates[:top_n]

    def rag_summarize(self, query: str) -> str:
        logger.info("RAG 检索 | query=%s", query)
        candidates = self.recall(query)
        if not candidates:
            logger.warning("RAG 未召回任何切片 | query=%s", query)
            return "知识库中没有检索到相关内容。"

        picked = self._refine(query, candidates)
        for i, doc in enumerate(picked):
            logger.info("命中 %d | %s | %s", i + 1, doc.metadata.get("source"),
                        doc.page_content[:60].replace("\n", " "))

        context = "\n\n".join(
            f"【片段{i + 1}｜来源：{d.metadata.get('source', '未知')}】\n{d.page_content}"
            for i, d in enumerate(picked)
        )
        answer = chat_model.invoke(self._prompt.format(context=context, question=query)).content
        logger.info("RAG 作答完成 | 长度=%d", len(answer))
        return answer


if __name__ == "__main__":
    service = RagSummarizeService()
    for question in ["扫地机器人滤网多久清理一次", "边刷缠绕头发怎么处理"]:
        print("=" * 70)
        print("Q:", question)
        print(service.rag_summarize(question))
        print()
