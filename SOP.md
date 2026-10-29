# 项目 SOP · LangChain ReAct Agent 智能客服

> 本文档是项目的唯一事实来源。新开会话时，先让 AI 阅读本文件再开始。
> 最后更新：2026-10-28

---

## 1. 项目定义

**复现目标**：从 0 到 1 复现 [lhh737/LangChain-ReAct-Agent](https://github.com/lhh737/LangChain-ReAct-Agent) 的全部功能，唯一改动是前后端分离（FastAPI + Streamlit）。

**一句话定位**：基于 LangChain v1 / LangGraph 的 ReAct Agent，集成 RAG 检索与多工具调用，通过中间件动态切换提示词以支持「智能客服」与「使用报告生成」两种场景。

### 1.1 硬性约束

| 约束 | 说明 |
|---|---|
| 代码自己写 | 不 clone 原仓库、不复制粘贴。原仓库只作架构参考（目录结构、模块职责、配置项设计） |
| 功能对齐 | 除前后端分离外，行为与原项目一致 |
| 不做的事 | 不做跨会话长期记忆、不做多用户鉴权、不做知识库管理界面 |

### 1.2 改动边界

| 模块 | 与原项目关系 |
|---|---|
| `utils/` `model/` `rag/` `agent/` `config/` `prompts/` `data/` | 行为一致 |
| `api/` | **全新**（原项目没有） |
| `ui/` | **重写**（原 `app.py` 直接 import Agent，改为 HTTP 调用） |

---

## 2. 技术栈

### 2.1 依赖（版本不锁定，让 pip 解析）

```
langchain>=1.0
langgraph>=1.0
langgraph-checkpoint-sqlite   # SqliteSaver，会话持久化必需（不在 langgraph 主包里）
langchain-community
langchain-chroma
dashscope
pyyaml
pypdf
fastapi
uvicorn[standard]
streamlit
httpx
python-dotenv
sse-starlette        # 可选，也可手写 text/event-stream
```

**关键风险**：原项目 `requirements.txt` 锁 `langchain==0.3.7`，但代码用的是 v1 的 `create_agent` API，二者不兼容。任何情况下都不要用 0.x 版本。

### 2.2 模型

**平台**：阿里云百炼（官方），控制台 `bailian.console.aliyun.com`。

> 注意：`platform.qianwenai.com` 是第三方多模型中转站（同时代理 Qwen / Kimi / DeepSeek），不是阿里官方。本项目不使用。

| 用途 | 模型 | 环境变量 |
|---|---|---|
| 对话 | `qwen3-max` | `DASHSCOPE_API_KEY` |
| Embedding | `text-embedding-v4` | 同上 |

**接入要点**

- SDK 内置端点 `dashscope.aliyuncs.com`，无需手工配置 `base_url`
- `ChatTongyi(model=...)` + `DashScopeEmbeddings(model=...)` 都依赖 `dashscope` 包
- 百炼另有 OpenAI 兼容端点 `compatible-mode/v1`，但**仅覆盖对话模型**，embedding 不在其中，不要混用

**开工前必须确认**：登录控制台，确认账号已开通 `qwen3-max` 与 `text-embedding-v4` 的调用权限。免费额度未必包含全部模型，若不可用则替换为：

| 备选 | 说明 |
|---|---|
| `qwen-max` / `qwen-plus` | 权限更宽松，效果略降 |
| `text-embedding-v3` | v4 不可用时的替代 |

模型名不可用时的报错是 `ModelNotFound` 或 `InvalidParameter`，容易被误判为代码问题。**先确认权限，再写 `factory.py`。**

### 2.3 运行环境

- Python 3.13.12（本机唯一版本，无 py launcher）
- 若 `chromadb` 在 3.13 装不上，改用 3.11 虚拟环境，不要降级 Chroma 到 0.5.x

---

## 3. 已定架构决策

| # | 决策项 | 结论 | 理由 |
|---|---|---|---|
| 1 | 会话持久化 | LangGraph Checkpointer 落 SQLite，按 `thread_id` | 重启不丢；内存版不适合展示 |
| 2 | 上下文控制 | `SummarizationMiddleware` 自动摘要 + 保留最近 K 轮 | 存储全量、注入裁剪，token 成本可控 |
| 3 | RAG 触发 | 封装为 `@tool`，由 Agent 自主决定是否调用 | 保留原项目设计；闲聊不浪费检索开销 |
| 4 | 长期记忆 | 不做 | 第一版范围控制，后续可加 |
| 5 | 流式方式 | `stream_mode="messages"` 实现 token 级流式 | 原项目 `values` 模式是伪流式，体验差 |
| 6 | 事件结构 | SSE 发结构化事件（token / tool_call / tool_result / done / error） | 前端要展示工具调用链路 |
| 7 | 前端职责 | 只持有 `thread_id` 与渲染状态，不持有对话历史 | 裁剪权必须在后端 |
| 8 | Agent 实例 | FastAPI lifespan 创建全局单例 | 避免每会话重复初始化 Chroma/模型客户端 |

### 3.1 关于决策 5 的兼容处理

保留原 `execute_stream(query) -> Iterator[str]` 方法不动（用于命令行调试），**另外新增** `stream_events(question, thread_id) -> Iterator[dict]` 供 API 层使用。两套出口，互不干扰。

---

## 4. 目录结构

```
Lanchain/
├── config/
│   ├── agent.yml
│   ├── chroma.yml
│   ├── prompts.yml
│   └── rag.yml
├── prompts/
│   ├── main_prompt.txt
│   ├── rag_summarize.txt
│   └── report_prompt.txt
├── utils/
│   ├── path_tool.py
│   ├── config_handler.py
│   ├── logger_handler.py
│   ├── prompt_loader.py
│   └── file_handler.py
├── model/
│   └── factory.py
├── rag/
│   ├── vector_store.py
│   └── rag_service.py
├── agent/
│   ├── react_agent.py
│   └── tools/
│       ├── agent_tools.py
│       └── middleware.py
├── api/
│   ├── main.py          # FastAPI 应用入口
│   ├── schemas.py       # 请求响应模型
│   ├── deps.py          # 单例依赖
│   └── routes/
│       └── chat.py
├── ui/
│   └── app.py           # Streamlit 入口
├── data/
│   ├── *.txt / *.pdf    # 知识库文档（扫地机器人）
│   └── external/
│       └── records.csv  # 用户使用记录
├── chroma_db/           # 运行时生成
├── md5.txt              # 运行时生成
├── .env
└── SOP.md
```

---

## 5. 分阶段实施

每个阶段都必须**先跑通验收标准**再进入下一阶段。前置未通就写后续代码，出问题时无法定位。

### 阶段 0 · 环境与依赖

**环境**：conda 环境 `agent312`（路径 `D:\Python\CondaEnvs\agent312`，Python 3.12.16）。**不使用 venv**，conda 已提供等价的依赖隔离，且 3.12 对 chromadb 的 wheel 支持优于 3.13。

激活：`conda activate agent312`。此后所有 `pip` 安装一律用 `python -m pip`，避免误装到 base 环境。

**动作**
1. 删除根目录 `main.py`（PyCharm 模板）— 已完成
2. 按 2.1 补齐依赖
3. 配置 `.env`：`DASHSCOPE_API_KEY=sk-xxx`
4. 获取 `data/` 数据 — 已完成（6 个知识文档 + `external/records.csv`）

**实测状态（2026-10-28，已完成）**

依赖用 uv 装入 `agent312`：

```bash
uv pip install --python "D:\Python\CondaEnvs\agent312\python.exe" \
  langchain-community langgraph-checkpoint-sqlite langchain-chroma dashscope \
  fastapi "uvicorn[standard]" streamlit python-dotenv
```

| 包 | 实测版本 | 验证结果 |
|---|---|---|
| `langchain` | 1.4.7 | `create_agent` + 三中间件导入通过 |
| `langgraph` | 1.2.18 | 通过 |
| `langgraph-checkpoint-sqlite` | 已装 | `SqliteSaver` 导入通过 |
| `langchain-community` | 已装 | `ChatTongyi` 导入通过（有弃用警告） |
| `chromadb` | 1.5.9 | 通过 |
| `langchain-chroma` | 已装 | `Chroma` 导入通过 |
| `dashscope` | 2.0.2 | 通过 |
| `fastapi` | 0.162.0 | 通过 |
| `uvicorn` | 0.54.0 | 通过 |
| `streamlit` | 1.66.1 | 通过 |
| `pypdf` / `PyYAML` / `httpx` / `python-dotenv` | 6.21.2 / 6.0.3 / 0.28.1 / 1.2.4 | 通过 |

**两个已知差异**

1. **`langchain-community` 已进入 sunset 状态**。导入时输出 `DeprecationWarning: langchain-community is being sunset and is no longer actively maintained`。官方指向独立集成包（`langchain-dashscope` 存在但仅 0.1.8，对 v1 支持不明）。
   → **第一版继续用 `ChatTongyi`**（与原项目一致且已验证可用）。若后续需要迁移，改 `model/factory.py` 一处即可，不影响其他模块。

2. **`chromadb` 为 1.5.9，原项目锁的是 0.5.15**，跨两个大版本。
   → 写 `rag/vector_store.py` 时**优先用 `langchain_chroma.Chroma` 封装，不要直接调裸 `chromadb.PersistentClient`**。封装层能屏蔽大部分版本差异，直连客户端则会遇到 API 变更。

**验收**

```bash
python -c "from langchain.agents import create_agent; from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt; print('v1 core ok')"

python -c "from langchain_community.chat_models.tongyi import ChatTongyi; from langgraph.checkpoint.sqlite import SqliteSaver; print('integrations ok')"
```

两条都打印 ok 才算通过。若 `ChatTongyi` 仍失败，改用 `init_chat_model("dashscope:qwen3-max")` 或安装 `langchain-dashscope`。

**已就绪的数据集**（2026-10-28 下载完成，位于 `data/`）

| 文件 | 大小 | 类型 |
|---|---|---|
| `扫地机器人100问.pdf` | 192 KB | PDF，12 页级 |
| `扫地机器人100问2.txt` | 18 KB | TXT |
| `扫拖一体机器人100问.txt` | 21 KB | TXT |
| `故障排除.txt` | 42 KB | TXT，200 条故障条目 |
| `维护保养.txt` | 24 KB | TXT |
| `选购指南.txt` | 22 KB | TXT |
| `external/records.csv` | 24 KB | 120 行数据 |

`records.csv` 结构：表头 `"用户ID","特征","清洁效率","耗材","对比","时间"`，共 **120 条 = 10 个用户 × 12 个月**。

三个实现时必须注意的点：

1. **字段内用字面 `\n` 表示换行，不是真换行**。所以一行 CSV 不会被 `readline()` 截断。
2. **字段内不含逗号**，所以 `line.split(",")` 勉强可用；但更稳妥的是 `csv` 模块，取值后再 `strip('"')` 并把 `\n` 替换为真实换行。
3. **随机取值必须保证命中**：`get_user_id` 从 `1001~1010` 随机、`get_current_month` 从 `2025-01~2025-12` 随机，正好覆盖 `records.csv` 的全部组合。这是原项目的刻意设计——保证报告生成场景一定能取到数据。**不要改成真实当前月份**，否则会取不到记录。

**原实现的一处瑕疵**：`fetch_external_data` 声明返回 `str`，实际返回的是嵌套 dict。复现时建议显式 `json.dumps(..., ensure_ascii=False)` 或格式化成可读文本，否则工具返回值的序列化行为依赖框架实现，不易调试。

---

### 阶段 1 · utils（基础设施）

**文件**：`path_tool.py`、`config_handler.py`、`logger_handler.py`、`prompt_loader.py`、`file_handler.py`

**要点**
- `path_tool.py`：基于 `__file__` 推导项目根目录，**禁止用 `os.getcwd()`**。所有配置中的相对路径都通过它转绝对路径
- `config_handler.py`：模块导入时加载 YAML 为 dict，导出 `agent_conf` / `chroma_conf` / `rag_conf` / `prompts_conf`
- `logger_handler.py`：控制台 + 文件双输出，供工具调用追踪使用
- `prompt_loader.py`：读取 `prompts/*.txt`，返回字符串
- `file_handler.py`：PDF 用 `PyPDFLoader`，TXT 直接读；统一返回文本内容

**验收**
```bash
python -c "from utils.config_handler import rag_conf, chroma_conf; print(rag_conf); print(chroma_conf)"
```
从**项目根目录和任意子目录**各执行一次都成功，才说明路径逻辑正确。

---

### 阶段 2 · model（模型工厂）

**文件**：`model/factory.py`

**要点**
- 用抽象基类 + 工厂模式，产出 `chat_model` 与 `embed_model` 两个模块级单例
- `ChatTongyi(model=rag_conf["chat_model_name"])`
- `DashScopeEmbeddings(model=rag_conf["embedding_model_name"])`

**验收**
```python
from model.factory import chat_model
print(chat_model.invoke("你好").content)
```

---

### 阶段 3 · rag（检索引擎）

**文件**：`vector_store.py`、`rag_service.py`

**`vector_store.py` 职责**
- 遍历 `data/` 下允许的文件类型（txt / pdf）
- MD5 去重：把已入库文件的 MD5 记录到 `md5.txt`，重复文件跳过
- `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, separators=[...])`
- 写入 Chroma（`collection_name=agent`，`persist_directory=chroma_db`）
- 提供 `get_retriever()`

**`rag_service.py` 职责**
- `rag_summarize(query)`：检索 Top-3 切片 → 拼接 → 用 `rag_summarize.txt` 提示词让 LLM 总结成一段答案
- 供 `agent_tools.rag_summarize` 工具调用

**验收（必须先独立跑通，不接 Agent）**
```bash
python -c "from rag.rag_service import RagSummarizeService; print(RagSummarizeService().rag_summarize('扫地机器人有哪些功能'))"
```
输出应当是基于知识库内容的**中文总结**，而不是「我不知道」。

> 这一步失败的典型原因是检索结果为空或 LLM 没用上上下文。先打印命中的切片确认召回正常。

---

### 阶段 4 · agent/tools（工具与中间件）

**7 个工具**（签名与描述需一致）

| 工具 | 签名 | 行为 |
|---|---|---|
| `rag_summarize` | `(query: str) -> str` | 调用 `RagSummarizeService` |
| `get_weather` | `(city: str) -> str` | 返回固定天气描述（模拟） |
| `get_user_location` | `() -> str` | 从 `["深圳","合肥","杭州"]` 随机 |
| `get_user_id` | `() -> str` | 从 `1001`~`1010` 随机 |
| `get_current_month` | `() -> str` | 从 `2025-01`~`2025-12` 随机 |
| `fetch_external_data` | `(user_id: str, month: str) -> str` | 读 `records.csv`，未命中返回空串 |
| `fill_context_for_report` | `() -> str` | 无实际逻辑，仅作为中间件触发信号 |

`records.csv` 列：`user_id, 特征, 效率, 耗材, 对比, 时间`，首行为表头。加载后组织为 `{user_id: {month: {特征,效率,耗材,对比}}}`。

**3 个中间件**

| 中间件 | 装饰器 | 行为 |
|---|---|---|
| `monitor_tool` | `@wrap_tool_call` | 记录工具名与参数；**当工具名为 `fill_context_for_report` 时，置 `request.runtime.context["report"] = True`** |
| `log_before_model` | `@before_model` | 记录当前消息条数与最后一条类型 |
| `report_prompt_switch` | `@dynamic_prompt` | `context["report"]` 为 True 时返回 `report_prompt.txt`，否则返回 `main_prompt.txt` |

**验收**：单独 import 三个中间件不报错；工具用 `tool.name` / `tool.description` 能看到正确值。

---

### 阶段 5 · agent（Agent 主体 + 会话记忆）

**文件**：`agent/react_agent.py`

**要点**
- `create_agent(model=..., system_prompt=..., tools=[7个], middleware=[3个])`
- Checkpointer：`SqliteSaver`，库文件如 `agent_memory.db`
- 上下文压缩：`SummarizationMiddleware`，触发条件按 token 数（如 3000）或消息条数（如 20）
- 两个出口方法：
  - `execute_stream(query) -> Iterator[str]`：命令行调试用，纯文本
  - `stream_events(question, thread_id) -> Iterator[dict]`：API 用，结构化事件
- `stream_events` 内部用 `stream_mode="messages"`，同时捕获工具调用信息

**验收（命令行，不接 HTTP）**
```bash
python -m agent.react_agent
```
连续输入三轮，验证：
1. 第二轮能引用第一轮内容 → 记忆生效
2. 输入「帮我生成使用报告」→ 日志中出现 `fill_context_for_report`，且随后提示词切换为报告版
3. 知识库问题能触发 `rag_summarize` 工具

---

### 阶段 6 · api（FastAPI 接口层）

**文件**：`api/main.py`、`schemas.py`、`routes/chat.py`

**要点**
- `lifespan` 中初始化 `ReactAgent` 单例，存入 `app.state`
- CORS 允许 `localhost:8501`
- SSE 用 `StreamingResponse(media_type="text/event-stream")`

**接口契约**

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat/stream` | body `{thread_id?, question}` → SSE 流 |
| GET | `/chat/history?thread_id=x` | 返回该会话历史消息 |
| POST | `/chat/new` | 生成新 `thread_id` |
| GET | `/health` | 健康检查 |

**SSE 事件格式**
```
event: token
data: {"content": "扫"}

event: tool_call
data: {"name": "rag_summarize", "args": {"query": "..."}}

event: tool_result
data: {"name": "rag_summarize", "content": "..."}

event: done
data: {"thread_id": "xxx"}

event: error
data: {"message": "..."}
```

**验收**
```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "{\"question\":\"扫地机器人有哪些功能\"}"
```
终端能看到逐条 `event:` / `data:`。

---

### 阶段 7 · ui（Streamlit 前端）

**文件**：`ui/app.py`

**要点**
- **禁止 import `agent` / `model` / `rag`**，只允许 `httpx` + `streamlit`
- `session_state` 只存：`thread_id`、渲染用的消息列表
- 用 `httpx.stream("POST", ...)` 逐行解析 SSE（`response.iter_lines()`）
- 工具调用事件用 `st.status()` 或 `st.expander()` 展示，形成推理链路
- 首次进入页面调 `/chat/new` 或本地生成 uuid

**验收**
1. 前端启动时后端未启动 → 页面给出明确错误提示，不白屏
2. 正常对话 → 逐字流式输出
3. 触发 RAG → 界面上能看到「调用了 rag_summarize」及参数
4. 输入报告类问题 → 提示词切换，输出变为报告结构
5. 刷新页面 → 通过 `/chat/history` 恢复历史

---

## 6. 已知风险清单

| 风险 | 表现 | 应对 |
|---|---|---|
| LangChain 版本错配 | `ImportError: cannot import name 'create_agent'` | 确认 `langchain>=1.0`，卸载 0.x |
| `ChatTongyi` 导入失败 | `ModuleNotFoundError` | 改用 `init_chat_model("dashscope:qwen3-max")` |
| chromadb 装不上 | 编译失败 / 无 wheel | 换 Python 3.11 环境 |
| Chroma 集合为空 | 检索返回空列表 | 确认已执行入库脚本，检查 `chroma_db/` 是否生成 |
| RAG 命中但模型不引用 | 回答发散 | 检查 `rag_summarize.txt` 提示词是否明确要求「仅依据参考资料」 |
| 多轮对话重复回答 | 历史未生效 | 确认 `thread_id` 在 `config["configurable"]` 中正确传递 |
| SSE 前端不逐字 | 一次性全出 | 检查是否用了 `requests`；必须 `httpx.stream` + `iter_lines` |
| 报告场景不切换 | 始终用客服提示词 | 检查中间件里 `runtime.context["report"]` 的写入时机 |

---

## 7. 面试可讲的技术点

按含金量排序，每个都能展开成一段：

1. **存储与注入分离**：Checkpointer 存全量历史，`SummarizationMiddleware` 控制注入量，token 成本与对话长度解耦
2. **Agent 单例 vs 会话隔离**：Agent 全局单例（Chroma/模型客户端复用），会话状态按 `thread_id` 隔离
3. **动态提示词切换**：通过工具调用改写 `runtime.context`，中间件在下一轮模型调用前切换 System Prompt
4. **流式协议设计**：`stream_mode="messages"` 与结构化 SSE 事件，前端可还原完整推理链路
5. **前后端职责边界**：前端持 `thread_id` 不持历史，裁剪权留在后端
6. **RAG 的工程细节**：MD5 去重避免重复入库、chunk 参数与 Top-K 的取舍

---

## 8. 每阶段自检清单

进入下一阶段前逐条确认：

- [ ] 当前阶段验收命令能跑通，且输出符合预期
- [ ] 没有为了「先跑起来」而写死的临时值（如硬编码路径、假数据）
- [ ] 新增配置项已写入对应 YAML，而非散落在代码里
- [ ] 关键路径有日志输出，后续排错能定位
- [ ] 能用自己的话解释这个模块为什么这样设计
