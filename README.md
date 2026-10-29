# 扫地机器人客服 Agent

基于 **LangChain v1 / LangGraph** 的 ReAct Agent 智能客服系统，**FastAPI + Streamlit 前后端分离**。

支持 RAG 检索（召回 + 精排）、7 个工具调用、多轮会话记忆、双场景提示词切换（客服问答 / 报告生成）。

---

## 功能特性

| 能力 | 实现 |
|---|---|
| ReAct Agent | `create_agent` + 原生 function calling，模型可并行调用多个工具 |
| RAG 检索 | Chroma 向量库 → 粗排召回 20 条 → `qwen3.7-text-rerank` 精排保留 3 条 → LLM 总结 |
| 知识库构建 | txt/pdf 解析，MD5 去重，`RecursiveCharacterTextSplitter` 切分（200/20） |
| 多工具调用 | 7 个工具：知识库检索、天气、位置、用户 ID、月份、外部数据、报告模式开关 |
| 动态提示词 | 中间件 `dynamic_prompt` 在每次模型调用前按运行期上下文切换系统提示词 |
| 多轮记忆 | LangGraph `SqliteSaver` 按 `thread_id` 持久化，附上下文摘要压缩 |
| 会话管理 | 新建 / 列表 / 重命名 / 删除 |
| 流式输出 | 后端 SSE，前端逐字渲染并展示工具调用链路 |

---

## 架构

```
浏览器
  └─ Streamlit (8501)          ← 只负责渲染与 SSE 解析，不含任何模型代码
       │  httpx.stream + iter_lines
       ▼
    FastAPI (8000)             ← /chat/stream 返回 SSE 事件流
       │
       ▼
    ReactAgent (LangGraph)     ← 7 工具 + 3 中间件 + SqliteSaver
       ├─ RAG：Chroma 召回 → 精排 → 总结
       ├─ 外部数据：records.csv（120 行）
       └─ 会话：agent_memory.db
```

SSE 事件格式：

```
event: token        data: {"content": "扫"}
event: tool_call    data: {"name": "rag_summarize", "args": {...}}
event: tool_result  data: {"name": "rag_summarize", "content": "..."}
event: done         data: {"thread_id": "..."}
event: error        data: {"message": "..."}
```

---

## 目录结构

```
config/      rag.yml  chroma.yml  agent.yml  prompts.yml      # 参数集中管理
utils/       path_tool  config_handler  logger_handler  file_handler
model/       factory.py（对话 + 向量）  rerank.py（精排）
rag/         vector_store.py（建库）  rag_service.py（召回→精排→总结）
prompts/     main_prompt  rag_summarize  report_prompt
agent/       react_agent.py（装配与出口）
             session_store.py（会话元数据）
             tools/agent_tools.py（7 个工具）  tools/middleware.py（3 个中间件）
api/         main.py  schemas.py  routes/chat.py
ui/          app.py
data/        知识库文档 + external/records.csv
SOP.md       从 0 到 1 的开发记录与排错手册
```

---

## 快速开始

### 1. 环境

Python 3.12+。建议 conda / venv 独立环境。

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env       # 填入真实密钥
```

或直接设置环境变量（**必须与启动服务的终端一致**）：

```bash
# bash
export DASHSCOPE_API_KEY=sk-xxx
# PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxx"
```

### 3. 构建知识库

```bash
python -m rag.vector_store
```

首次运行会将 `data/` 下的 txt/pdf 切分入库，生成 `chroma_db/` 与 `md5.txt`；重复运行会按 MD5 跳过已入库文件。

> 重建时 `chroma_db/` 与 `md5.txt` 要一起删，只删其一会导致重复入库或全部跳过。

### 4. 启动后端

```bash
uvicorn api.main:app --port 8000 --reload
```

### 5. 启动前端

```bash
streamlit run ui/app.py
```

打开 http://localhost:8501 。后端地址可通过环境变量覆盖：`API_BASE=http://127.0.0.1:8001`。

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat/stream` | body `{thread_id?, question}` → SSE 事件流 |
| GET | `/chat/history?thread_id=` | 会话历史消息 |
| POST | `/chat/new` | 生成新 thread_id |
| GET | `/chat/sessions` | 会话列表（标题 / 更新时间） |
| PATCH | `/chat/sessions/{thread_id}` | 重命名会话 |
| DELETE | `/chat/sessions/{thread_id}` | 删除会话 |
| GET | `/health` | 健康检查 |

交互式文档：http://127.0.0.1:8000/docs

---

## 关键设计说明

**模型接入走 OpenAI 兼容协议。** 项目使用的第三方 MaaS 平台不支持 DashScope 原生 `text-generation`，只提供 `/compatible-mode/v1`。因此统一使用 `ChatOpenAI` / `OpenAIEmbeddings`，而非 `ChatTongyi`。

**两个必须显式设置的参数：**

```python
OpenAIEmbeddings(
    check_embedding_ctx_length=False,   # 否则报 400 input must be an array of strings
    chunk_size=10,                      # 平台单次批量上限 10，默认 1000 会报 batch size is invalid
)
```

**精排不属于 OpenAI 兼容协议。** `/compatible-mode/v1/rerank` 返回 404，必须走 DashScope 原生端点 `/api/v1/services/rerank/text-rerank/text-rerank`，报文用 `input.query` / `input.documents` 结构。实现见 `model/rerank.py`。

**提示词切换靠工具触发，不靠关键词匹配。** 模型调用 `fill_context_for_report` → 中间件写入 `runtime.context["report"]=True` → 下一次模型调用时 `dynamic_prompt` 返回报告版提示词。

**前端禁止 import 后端模块。** `ui/app.py` 只依赖 `streamlit` / `httpx` / 标准库，前后端唯一通道是 HTTP。

---

## 已知限制

- 知识库仅 6 个文档、约 320 个切片，覆盖面有限。提问用词与文档措辞差异较大时（例如问「边刷缠绕」而文档只讲「主刷」），会返回「现有资料无法回答」而非强行作答。
- `get_user_id` / `get_current_month` / `get_user_location` 返回随机模拟值，用于演示报告场景，不代表真实用户。
- 无用户鉴权、无多租户隔离，会话以 `thread_id` 区分，适合本地演示。
- 客户端中途断开连接时，本轮回答不会写入 checkpointer（节点执行完后才落库）。

---

## 技术栈

Python 3.12 · LangChain 1.4 · LangGraph 1.2 · Chroma 1.5 · FastAPI · Streamlit · SQLite

模型：`qwen3.8-flash`（对话）· `text-embedding-v4`（向量，1024 维）· `qwen3.7-text-rerank`（精排）

---

## 数据来源

`data/` 下的知识库文档（扫地机器人问答、故障排除、维护保养、选购指南等）与
`data/external/records.csv` 均为演示用公开语料，来源为参考项目
[lhh737/LangChain-ReAct-Agent](https://github.com/lhh737/LangChain-ReAct-Agent)，
版权归原作者所有，仅用于学习与功能演示。如需商用请自行替换为自有语料。

