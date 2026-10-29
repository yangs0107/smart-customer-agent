"""模型工厂：全项目唯一的模型接入点。

对外暴露两个模块级对象：
    chat_model  —— 对话模型，支持 bind_tools / 流式输出
    embed_model —— 向量化模型，供 RAG 召回使用

参数来自 config/rag.yml，本文件只负责组装。
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "环境变量 DASHSCOPE_API_KEY 未设置，无法初始化模型。"
        "请在启动后端的同一个终端里先设置该变量。"
    )

chat_model = ChatOpenAI(
    model=rag_conf["chat_model_name"],
    base_url=rag_conf["base_url"],
    api_key=API_KEY,
    temperature=0,
)

embed_model = OpenAIEmbeddings(
    model=rag_conf["embedding_model_name"],
    base_url=rag_conf["base_url"],
    api_key=API_KEY,
    # 不可省略：默认先把文本切成 token id 数组再发送，
    # 该服务端只接受字符串数组，否则报 400 input must be an array of strings。
    check_embedding_ctx_length=False,
    # 该平台单次批量上限 10，默认 1000 会报 batch size is invalid
    chunk_size=rag_conf["embedding_batch_size"],
)
