# 项目开发 SOP · 从 0 到 1

> **给新窗口的 AI**：用户要求自己写代码，不接受可直接粘贴的完整实现。你的角色是讲原理、给模块任务书、审阅代码、排查错误。
>
> **给用户**：本文档是从 0 到 1 开发本项目的执行流程。每步都有验收命令，通过后再进入下一步。

---

## 0. 项目定义

**复现目标**：`lhh737/LangChain-ReAct-Agent`，基于 LangChain v1 / LangGraph 的 ReAct Agent 智能客服系统。

**功能范围**：RAG 检索 + 7 个工具调用 + 双场景（客服问答 / 报告生成）+ 多轮会话记忆。

**唯一改动**：原项目是 Streamlit 单体，本项目改为 **FastAPI + Streamlit 前后端分离**。

**不做**：跨会话长期记忆、多用户鉴权、知识库管理界面。

---

## 1. 已完成的前置工作

新窗口不要重复做这些。

| 项 | 状态 |
|---|---|
| conda 环境 | `agent312`，Python 3.12.16，路径 `D:\Python\CondaEnvs\agent312` |
| 依赖 | 已全部安装（见下表） |
| 数据 | `data/` 下 6 个知识文档 + `external/records.csv` |
| Git | 已初始化，分支 `main`，有 4 次提交 |
| 模型 | 已实测打通对话 / 工具调用 / 流式 / Embedding |
| 目录 | **已清空**，按开发顺序逐个创建 |

**已装依赖**

```
langchain 1.4.7          langgraph 1.2.18         langgraph-checkpoint-sqlite
langchain-community      langchain-chroma         langchain-openai 1.7.3
chromadb 1.5.9           dashscope 2.0.2          fastapi 0.162.0
uvicorn 0.54.0           streamlit 1.66.1         pypdf / PyYAML / httpx / python-dotenv
```

装包命令：

```bash
uv pip install --python "D:\Python\CondaEnvs\agent312\python.exe" <包名>
```

---

## 2. 关键技术事实（务必遵守）

### 2.1 模型接入：必须走 OpenAI 兼容协议

用户使用的是**第三方 MaaS 平台**，不是阿里云官方百炼。原项目的 `ChatTongyi` / `DashScopeEmbeddings` **在本环境不可用**。

| 接口 | 结果 |
|---|---|
| DashScope `text-generation` | ❌ 400 `url error` |
| DashScope `multimodal-generation` | ✅ 200（只能跑多模态） |
| OpenAI 兼容 `chat/completions` | ✅ 200 |
| OpenAI 兼容 `embeddings` | ✅ 200 |

**正确写法**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE = "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1"

chat_model = ChatOpenAI(model="qwen3.8-flash", base_url=BASE,
                        api_key=KEY, temperature=0)

embed_model = OpenAIEmbeddings(model="text-embedding-v4", base_url=BASE,
                               api_key=KEY,
                               check_embedding_ctx_length=False)
```

**`check_embedding_ctx_length=False` 不可省略。** 默认行为会把文本转成 token id 数组发送，该服务端要求字符串数组，否则报 `400 input must be an array of strings`——这个报错完全指不到真实原因。

**已验证能力**：普通对话 ✅ / `bind_tools` 工具调用 ✅ / 流式 ✅ / Embedding 1024 维 ✅

可用模型共 371 个，除 `qwen3.8-flash` 外还有 `qwen3.8-max`、`deepseek-v4.1-flash`、`kimi-k3.1-pro` 等。

### 2.2 数据规格

`data/external/records.csv`：表头 `"用户ID","特征","清洁效率","耗材","对比","时间"`，120 行 = 10 用户 × 12 月。

三个约束：
1. 字段内用**字面 `\n`** 表示换行，不是真换行符
2. 随机取值必须覆盖全部组合：`get_user_id` 取 `1001~1010`，`get_current_month` 取 `2025-01~2025-12`。**不要改成真实当前月份**，否则报告场景取不到数据
3. 建议 `fetch_external_data` 用 `json.dumps(..., ensure_ascii=False)` 返回字符串

### 2.3 Chroma 版本

实际装的是 `chromadb 1.5.9`，原项目锁的是 `0.5.15`。**统一走 `langchain_chroma.Chroma` 封装，不要直连 `chromadb.PersistentClient`**，否则会遇到 1.x 的 API 变更。

---

## 3. 开发顺序

每个阶段只创建**当前需要**的文件和目录。不要提前建空骨架。

### 阶段 1 · 让模型能说话

**创建**：`model/` 目录 + `model/factory.py`

**内容**：两个模块级变量 `chat_model`、`embed_model`，参数硬编码（先用 `2.1` 的写法）。

**验收**

```bash
python -c "from model.factory import chat_model, embed_model; print(chat_model.invoke('你好').content[:30]); print(len(embed_model.embed_query('测试')))"
```

> 这一步先不接配置，目的是先拿到真实反馈。配置在下一步抽。

---

### 阶段 2 · 配置层 + 路径工具

**创建**：`config/` 目录、`utils/` 目录

**文件**：
- `config/rag.yml`：`base_url` / `chat_model_name` / `embedding_model_name`
- `config/chroma.yml`：`collection_name` `persist_directory` `k` `data_path` `md5_hex_store` `allow_knowledge_file_type` `chunk_size` `chunk_overlap` `separators`
- `config/agent.yml`：`external_data_path`
- `config/prompts.yml`：三个提示词路径
- `utils/path_tool.py`：`get_abs_path(relative_path) -> str`
- `utils/config_handler.py`：导入时加载 YAML，导出 `rag_conf` / `chroma_conf` / `agent_conf` / `prompts_conf`

**关键点**

- `path_tool` 必须基于 `__file__` 推导根目录，**禁止 `os.getcwd()`**；推导前先 `resolve()`
- 配置加载发生在模块导入时，其他模块直接 `from utils.config_handler import rag_conf`
- 完成后把 `factory.py` 里的硬编码改成读 `rag_conf`

**验收**

```bash
# 在根目录和 utils/ 子目录各跑一次，结果必须一致
python -c "from utils.path_tool import get_abs_path; print(get_abs_path('data'))"
python -c "from utils.config_handler import rag_conf, chroma_conf; print(rag_conf); print(chroma_conf)"
```

---

### 阶段 3 · 日志

**创建**：`utils/logger_handler.py`、`logs/` 目录（运行时自动建也可）

**内容**：导出 `logger`，控制台 + 文件双输出，文件写 `logs/`。

**为什么现在做**：后面排查 Agent 的工具调用全靠它，越早建越好。

**验收**

```bash
python -c "from utils.logger_handler import logger; logger.info('test')"
```
控制台有输出且 `logs/` 下生成文件。

---

### 阶段 4 · 文档解析与提示词

**创建**：`prompts/` 目录 + 三个 txt、`utils/file_handler.py`、`utils/prompt_loader.py`

**提示词文件**
- `main_prompt.txt`：客服问答 System Prompt
- `rag_summarize.txt`：RAG 总结 Prompt。**必须明确「仅依据参考资料作答，资料不足则说明」**，否则模型会抛开检索结果自由发挥
- `report_prompt.txt`：报告生成 System Prompt

**file_handler**：PDF 用 `pypdf`，TXT 直读，统一返回字符串。

**验收**

```bash
python -c "from utils.file_handler import read_file; print(read_file('data/故障排除.txt')[:80])"
python -c "from utils.prompt_loader import load_system_prompts; print(load_system_prompts()[:50])"
```

---

### 阶段 5 · RAG 链路（必须独立验证）

**创建**：`rag/` 目录 + `vector_store.py`、`rag_service.py`

**vector_store 职责**
- 遍历 `data/` 下 txt + pdf
- MD5 去重，已入库文件记到 `md5.txt`
- `RecursiveCharacterTextSplitter` 按配置切分（`chunk_size=200`、`chunk_overlap=20`）
- 写入 Chroma（走 `langchain_chroma.Chroma`）
- 提供 `get_retriever()`

**rag_service 职责**
- `rag_summarize(query) -> str`：检索 Top-K → 拼接 → LLM 按 `rag_summarize.txt` 总结

**验收（不要接 Agent）**

```bash
python -c "from rag.rag_service import RagSummarizeService; print(RagSummarizeService().rag_summarize('扫地机器人有哪些功能'))"
```

输出必须是基于知识库的中文总结。若失败，**先打印命中的切片确认召回正常**，再怀疑模型。

同时检查：首次运行生成 `chroma_db/` 与 `md5.txt`；再次运行跳过全部文件。

---

### 阶段 6 · 工具集与中间件

**创建**：`agent/` + `agent/tools/` 目录 + `agent_tools.py`、`middleware.py`

**7 个工具**

| 工具 | 签名 | 行为 |
|---|---|---|
| `rag_summarize` | `(query: str) -> str` | 调 `RagSummarizeService` |
| `get_weather` | `(city: str) -> str` | 固定返回模拟数据 |
| `get_user_location` | `() -> str` | 深圳/合肥/杭州 随机 |
| `get_user_id` | `() -> str` | `1001`~`1010` 随机 |
| `get_current_month` | `() -> str` | `2025-01`~`2025-12` 随机 |
| `fetch_external_data` | `(user_id: str, month: str) -> str` | 读 `records.csv`，未命中返回 `""` |
| `fill_context_for_report` | `() -> str` | 无逻辑，仅作中间件触发信号 |

**3 个中间件**

| 中间件 | 装饰器 | 行为 |
|---|---|---|
| `monitor_tool` | `@wrap_tool_call` | 记录工具名与参数；工具名为 `fill_context_for_report` 时置 `request.runtime.context["report"] = True` |
| `log_before_model` | `@before_model` | 记录消息条数 |
| `report_prompt_switch` | `@dynamic_prompt` | `context["report"]` 为 True 返回报告提示词，否则返回客服提示词 |

**验收**

```bash
python -c "from agent.tools.agent_tools import *; from agent.tools.middleware import *; print('ok')"
```

---

### 阶段 7 · Agent 主体与会话记忆

**创建**：`agent/react_agent.py`

**内容**
- `create_agent(model=chat_model, system_prompt=..., tools=[7个], middleware=[3个])`
- Checkpointer：`SqliteSaver`，库文件 `agent_memory.db`
- 上下文压缩：`SummarizationMiddleware`（按 token 或消息条数触发）
- 两个出口方法：
  - `execute_stream(query) -> Iterator[str]`：命令行调试用
  - `stream_events(question, thread_id) -> Iterator[dict]`：API 用，结构化事件
- `stream_events` 用 `stream_mode="messages"`

**验收（命令行，不接 HTTP）**

```bash
python -m agent.react_agent
```

连续输入三轮，确认：
1. 第二轮能引用第一轮内容 → 记忆生效
2. 输入「帮我生成使用报告」→ 日志出现 `fill_context_for_report`，随后提示词切换为报告版
3. 知识库问题能触发 `rag_summarize`

---

### 阶段 8 · FastAPI 接口层

**创建**：`api/` + `api/routes/` 目录 + `main.py`、`schemas.py`、`routes/chat.py`

**内容**
- `lifespan` 中创建 `ReactAgent` 单例，存 `app.state`
- CORS 允许 `localhost:8501`
- SSE 用 `StreamingResponse(media_type="text/event-stream")`

**接口**

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat/stream` | body `{thread_id?, question}` → SSE |
| GET | `/chat/history?thread_id=` | 历史消息 |
| POST | `/chat/new` | 生成新 thread_id |
| GET | `/health` | 健康检查 |

**SSE 事件格式**

```
event: token        data: {"content": "扫"}
event: tool_call    data: {"name": "...", "args": {...}}
event: tool_result  data: {"name": "...", "content": "..."}
event: done         data: {"thread_id": "..."}
event: error        data: {"message": "..."}
```

**验收**

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "{\"question\":\"扫地机器人有哪些功能\"}"
```

---

### 阶段 9 · Streamlit 前端

**创建**：`ui/` 目录 + `app.py`

**硬约束**：**禁止 import `agent` / `model` / `rag`**，只允许 `httpx` + `streamlit`。

**内容**
- `session_state` 只存 `thread_id` 与渲染用消息
- `httpx.stream` + `iter_lines()` 解析 SSE（**不要用 `requests`**）
- 工具调用事件用 `st.status()` 或 `st.expander()` 展示

**验收**
1. 后端未启动时给出明确错误提示，不白屏
2. 逐字流式输出
3. 触发 RAG 时能看到工具名与参数
4. 报告类问题触发提示词切换
5. 刷新页面能通过 `/chat/history` 恢复历史

---

## 4. 常用命令

```bash
# 激活环境
conda activate agent312

# 启动后端
uvicorn api.main:app --reload --port 8000

# 启动前端
streamlit run ui/app.py

# 重建知识库（重复运行会跳过已入库文件）
python -m rag.vector_store
```

---

## 5. 排错优先级

| 症状 | 先查什么 |
|---|---|
| 模型调不通 | `base_url` 是否带了 `/compatible-mode/v1`；是否误用 `ChatTongyi` |
| Embedding 报 400 | 是否漏了 `check_embedding_ctx_length=False` |
| RAG 答非所问 | 先打印命中切片，确认召回正常，再怀疑提示词 |
| Agent 该查知识库却不查 | 先改工具的 `description`，再怀疑模型 |
| 多轮对话重复回答 | `thread_id` 是否传进 `config["configurable"]` |
| SSE 前端不逐字 | 是否用了 `requests`；必须 `httpx.stream` |
| 报告场景不切换 | `runtime.context["report"]` 写入时机 |

---

## 6. 开发纪律

1. 代码自己写，不复制原项目实现
2. 每个文件写完立即验证，不攒到最后
3. 每阶段验收通过后单独提交
4. 配置项写进 YAML，不散落在代码里
5. 路径操作一律走 `get_abs_path`
6. `ui/` 只能通过 HTTP 与后端通信

---

## 7. 进度

```
[ ] 阶段 1  模型能说话        model/factory.py
[ ] 阶段 2  配置与路径        config/  utils/path_tool  utils/config_handler
[ ] 阶段 3  日志              utils/logger_handler
[ ] 阶段 4  文档与提示词      utils/file_handler  utils/prompt_loader  prompts/
[ ] 阶段 5  RAG 链路          rag/vector_store  rag/rag_service
[ ] 阶段 6  工具与中间件      agent/tools/
[ ] 阶段 7  Agent 与记忆      agent/react_agent.py
[ ] 阶段 8  FastAPI 接口      api/
[ ] 阶段 9  Streamlit 前端    ui/app.py
```
