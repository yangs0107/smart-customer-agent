# SOP · 开发执行版

> 复现 `lhh737/LangChain-ReAct-Agent`，唯一改动是 FastAPI + Streamlit 前后端分离。代码自己写，不 clone。
> 详细设计理由见 git 历史中的初版 SOP（commit `3ee4dd7`）。本文档只保留「做什么 / 怎么验」。

---

## 1. 环境（已完成）

| 项 | 值 |
|---|---|
| 环境 | conda `agent312`，路径 `D:\Python\CondaEnvs\agent312` |
| Python | 3.12.16 |
| 模型 | 阿里云百炼 `qwen3-max` + `text-embedding-v4` |
| Key | `.env` → `DASHSCOPE_API_KEY=sk-xxx` |

激活：`conda activate agent312`
装包：`uv pip install --python "D:\Python\CondaEnvs\agent312\python.exe" <包名>`

已验证版本：langchain 1.4.7 / langgraph 1.2.18 / chromadb 1.5.9 / dashscope 2.0.2 / fastapi 0.162.0 / streamlit 1.66.1

---

## 2. 目录结构

```
Lanchain/
├── config/        agent.yml  chroma.yml  prompts.yml  rag.yml
├── prompts/       main_prompt.txt  rag_summarize.txt  report_prompt.txt
├── utils/         path_tool  config_handler  logger_handler  prompt_loader  file_handler
├── model/         factory.py
├── rag/           vector_store.py  rag_service.py
├── agent/         react_agent.py
│   └── tools/     agent_tools.py  middleware.py
├── api/           main.py  schemas.py
│   └── routes/    chat.py
├── ui/            app.py
├── data/          6 个知识文档 + external/records.csv（已就绪）
└── logs/  chroma_db/  md5.txt   （运行时生成）
```

依赖方向单向：`utils ← model ← rag ← agent ← api ← ui`

---

## 3. 阶段清单

### 阶段 1 · utils

- [ ] **`path_tool.py`**
  - `get_abs_path(relative_path: str) -> str`
  - 基于 `__file__` 推导根目录，**禁止 `os.getcwd()`**
  - 推导前先 `resolve()` 转绝对路径
  - 验：`python -c "from utils.path_tool import get_abs_path; print(get_abs_path('data'))"` → 输出项目下的 `data`；切到 `utils/` 子目录再跑一次，结果必须相同

- [ ] **`logger_handler.py`**
  - 导出 `logger`，控制台 + 文件双输出，文件写到 `logs/`
  - 验：`python -c "from utils.logger_handler import logger; logger.info('test')"` → 控制台有输出且 `logs/` 生成文件

- [ ] **`config/*.yml` + `config_handler.py`**
  - 四个配置文件，键名如下：

    | 文件 | 键 |
    |---|---|
    | `rag.yml` | `chat_model_name` / `embedding_model_name` |
    | `chroma.yml` | `collection_name` `persist_directory` `k` `data_path` `md5_hex_store` `allow_knowledge_file_type` `chunk_size` `chunk_overlap` `separators` |
    | `agent.yml` | `external_data_path` |
    | `prompts.yml` | 三个提示词路径 |

  - 模块导入时加载，导出 `rag_conf` / `chroma_conf` / `agent_conf` / `prompts_conf`
  - 验：`python -c "from utils.config_handler import rag_conf, chroma_conf; print(rag_conf); print(chroma_conf)"`

- [ ] **`prompts/*.txt` + `prompt_loader.py`**
  - 三个提示词：客服问答 / RAG 总结 / 报告生成
  - `rag_summarize.txt` 必须明确「仅依据参考资料作答，资料不足则说明」
  - 验：`python -c "from utils.prompt_loader import load_system_prompts; print(load_system_prompts()[:50])"`

- [ ] **`file_handler.py`**
  - PDF 用 `pypdf`，TXT 直读，统一返回字符串
  - 验：分别解析 `data/故障排除.txt` 与 `data/扫地机器人100问.pdf`，打印前 100 字

**提交**：`feat: 实现 utils 基础设施层`

---

### 阶段 2 · model

- [ ] **`factory.py`**
  - 抽象基类 + 两个工厂：`ChatModelFactory` / `EmbeddingsFactory`
  - 模块级导出 `chat_model`、`embed_model`
  - `ChatTongyi(model=rag_conf["chat_model_name"])`
  - `DashScopeEmbeddings(model=rag_conf["embedding_model_name"])`
  - 验：`python -c "from dotenv import load_dotenv; load_dotenv(); from model.factory import chat_model; print(chat_model.invoke('你好').content)"`

**提交**：`feat: 实现模型工厂`

---

### 阶段 3 · rag

- [ ] **`vector_store.py`**
  - 遍历 `data/` 下 txt + pdf
  - MD5 去重，已入库记录写入 `md5.txt`
  - `RecursiveCharacterTextSplitter` 按配置切分
  - 写入 Chroma（用 `langchain_chroma.Chroma`，**不要直连 `chromadb.PersistentClient`**，版本差异大）
  - 提供 `get_retriever()`
  - 验：首次运行生成 `chroma_db/` 与 `md5.txt`；再次运行应跳过全部文件

- [ ] **`rag_service.py`**
  - `rag_summarize(query) -> str`：检索 Top-K → 拼接 → LLM 按 `rag_summarize.txt` 总结
  - 验（**必须独立于 Agent**）：
    `python -c "from rag.rag_service import RagSummarizeService; print(RagSummarizeService().rag_summarize('扫地机器人有哪些功能'))"`
  - 输出必须是基于知识库的中文总结，不是「不知道」
  - 若失败，先打印命中的切片确认召回正常，再怀疑模型

**提交**：`feat: 实现 RAG 检索链路`

---

### 阶段 4 · tools

- [ ] **`agent_tools.py`** — 7 个 `@tool`

  | 工具 | 签名 | 说明 |
  |---|---|---|
  | `rag_summarize` | `(query: str) -> str` | 调 `RagSummarizeService` |
  | `get_weather` | `(city: str) -> str` | 固定返回（模拟） |
  | `get_user_location` | `() -> str` | 深圳/合肥/杭州 随机 |
  | `get_user_id` | `() -> str` | `1001`~`1010` 随机 |
  | `get_current_month` | `() -> str` | `2025-01`~`2025-12` 随机 |
  | `fetch_external_data` | `(user_id: str, month: str) -> str` | 读 `records.csv`，未命中返回 `""` |
  | `fill_context_for_report` | `() -> str` | 无逻辑，仅作中间件触发信号 |

  **约束**：`get_user_id` 与 `get_current_month` 的随机范围必须覆盖 CSV 全部组合，不要改成真实当前月份。
  **建议**：`fetch_external_data` 返回值用 `json.dumps(..., ensure_ascii=False)` 显式转字符串。

- [ ] **`middleware.py`** — 3 个中间件

  | 中间件 | 装饰器 | 行为 |
  |---|---|---|
  | `monitor_tool` | `@wrap_tool_call` | 记录工具名与参数；工具名为 `fill_context_for_report` 时置 `request.runtime.context["report"] = True` |
  | `log_before_model` | `@before_model` | 记录消息条数 |
  | `report_prompt_switch` | `@dynamic_prompt` | `context["report"]` 为 True 返回报告提示词，否则返回客服提示词 |

  - 验：`python -c "from agent.tools.agent_tools import *; from agent.tools.middleware import *; print('ok')"`

**提交**：`feat: 实现工具集与中间件`

---

### 阶段 5 · agent

- [ ] **`react_agent.py`**
  - `create_agent(model, system_prompt, tools, middleware)`
  - Checkpointer：`SqliteSaver`，库文件如 `agent_memory.db`
  - 上下文压缩：`SummarizationMiddleware`（按 token 或消息条数触发）
  - 两个出口：
    - `execute_stream(query) -> Iterator[str]`：命令行调试，纯文本
    - `stream_events(question, thread_id) -> Iterator[dict]`：API 用，结构化事件
  - `stream_events` 用 `stream_mode="messages"`
  - 验（**不接 HTTP**）：`python -m agent.react_agent`，连续输入三轮
    1. 第二轮能引用第一轮内容 → 记忆生效
    2. 「帮我生成使用报告」→ 日志出现 `fill_context_for_report`，随后提示词切换
    3. 知识库问题能触发 `rag_summarize`

**提交**：`feat: 实现 ReAct Agent 与会话记忆`

---

### 阶段 6 · api

- [ ] **`main.py`**
  - `lifespan` 中创建 `ReactAgent` 单例，存 `app.state`
  - CORS 允许 `localhost:8501`
  - SSE 用 `StreamingResponse(media_type="text/event-stream")`

- [ ] **`schemas.py`** + **`routes/chat.py`**

  | 方法 | 路径 | 说明 |
  |---|---|---|
  | POST | `/chat/stream` | body `{thread_id?, question}` → SSE |
  | GET | `/chat/history?thread_id=` | 历史消息 |
  | POST | `/chat/new` | 生成新 thread_id |
  | GET | `/health` | 健康检查 |

  SSE 事件格式：

  ```
  event: token        data: {"content": "扫"}
  event: tool_call    data: {"name": "...", "args": {...}}
  event: tool_result  data: {"name": "...", "content": "..."}
  event: done         data: {"thread_id": "..."}
  event: error        data: {"message": "..."}
  ```

  - 验：`curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "{\"question\":\"扫地机器人有哪些功能\"}"`

**提交**：`feat: 实现 FastAPI 接口层与 SSE 流式输出`

---

### 阶段 7 · ui

- [ ] **`ui/app.py`**
  - **禁止 import `agent` / `model` / `rag`**，只允许 `httpx` + `streamlit`
  - `session_state` 只存 `thread_id` 与渲染用消息
  - `httpx.stream` + `iter_lines()` 解析 SSE（**不要用 requests**）
  - 工具调用事件用 `st.status()` 或 `st.expander()` 展示

  - 验：
    1. 后端未启动时给出明确错误提示，不白屏
    2. 逐字流式输出
    3. 触发 RAG 时能看到工具名与参数
    4. 报告类问题触发提示词切换
    5. 刷新页面能通过 `/chat/history` 恢复历史

**提交**：`feat: 实现 Streamlit 前端`

---

## 4. 硬约束

1. 代码自己写，不复制原项目实现
2. 每完成一个模块立即验证，不攒到最后
3. `ui/` 只能通过 HTTP 与后端通信
4. 路径操作一律走 `get_abs_path`，不用 `getcwd`
5. 配置项写进 YAML，不散落在代码里
6. `get_user_id` / `get_current_month` 的随机范围不可改
7. Chroma 操作走 `langchain_chroma.Chroma` 封装
8. 每阶段验证通过后单独提交

## 5. 进度

```
[✅] 阶段 0  环境、数据、Git、目录骨架
[  ] 阶段 1  utils
[  ] 阶段 2  model
[  ] 阶段 3  rag
[  ] 阶段 4  tools
[  ] 阶段 5  agent
[  ] 阶段 6  api
[  ] 阶段 7  ui
```
