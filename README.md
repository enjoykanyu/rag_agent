# 多源文档智能问答 Agent（RAG）

基于 RAG（检索增强生成）的多源文档智能问答系统，使用 LangGraph 图编排实现 retrieve → think → generate 工作流，支持 SSE 流式响应、思考过程展示、引用溯源与追问。

## 功能特性

- **多源文档接入**：本地文件夹（.md/.txt）+ 页面上传文件
- **LangGraph 图编排**：retrieve → think → generate 节点化工作流，条件路由支持拒绝策略
- **SSE 流式响应**：分阶段展示思考过程 → 引用来源 → 回答内容
- **思考过程展示**：可折叠的"深度思考"面板，显示检索分析和决策逻辑
- **引用溯源**：内联引用标记 `[1]` `[2]`，点击显示引用卡片；顶部"N 个引用内容"按钮展开完整引用面板
- **智能拒绝**：知识库中无相关内容时，明确回复「知识库中未找到」，不编造答案
- **追问支持**：支持多轮对话与追问澄清
- **可配置**：chunk 大小、top-k、embedding 方式、数据源路径、系统提示词等均可配置
- **Web UI**：参考 cherry-studio / LightRAG / MiMo 设计，简洁直观的流式交互界面

## 项目结构

```
rag_agent/
├── README.md
├── requirements.txt
├── run.py                         # 启动入口
├── config/
│   └── example.yaml               # 示例配置
├── src/
│   ├── config.py                  # 配置系统
│   ├── loaders/
│   │   └── document_loader.py     # 文档加载器（本地 + 上传）
│   ├── pipeline/
│   │   ├── chunker.py             # 文档切分
│   │   ├── embedding.py           # 向量化（TF-IDF Mock / OpenAI）
│   │   ├── vector_store.py        # 向量存储与检索
│   │   └── rag_pipeline.py        # RAG 流水线编排
│   └── agent/
│       ├── graph.py               # LangGraph 图定义（retrieve→think→generate）
│       ├── streaming.py           # SSE 流式 Agent
│       └── rag_agent.py           # （旧版，已弃用）
├── web/
│   ├── app.py                     # FastAPI 应用（SSE 流式 API）
│   ├── static/
│   └── templates/
│       └── index.html             # Web UI（流式交互）
└── docs/
    ├── design.md                  # 设计文档（架构、LangGraph、UI 设计）
    └── sample_docs/               # 5 篇示例文档
        ├── python_basics.md
        ├── python_list_dict.md
        ├── python_functions.md
        ├── python_exception_file.md
        └── python_oop.md
```

## 安装步骤

```bash
# 1. 进入项目目录
cd rag_agent

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

## 配置说明

编辑 `config/example.yaml`：

```yaml
sources:
  local:
    path: "./docs/sample_docs"         # 本地文档路径
    patterns: ["**/*.md", "**/*.txt"]  # 文件匹配模式
  upload:
    enabled: true                       # 启用文件上传
    allowed_extensions: [".md", ".txt"]

rag:
  chunk_size: 512                       # 文本块大小（字符数）
  chunk_overlap: 64                     # 文本块重叠
  top_k: 5                              # 检索返回的文本块数量
  embedding: "mock"                     # 向量化方式: mock / openai

agent:
  refuse_when_no_context: true          # 无相关内容时拒绝回答
  system_prompt: "..."                  # 系统提示词
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    api_key: ""                         # LLM API Key
    base_url: ""                        # 自定义 API 地址
    temperature: 0.3
    max_tokens: 1024

ui:
  show_citations: true                  # 显示引用来源
  title: "多源文档智能问答 Agent"
```

### 环境变量覆盖

| 环境变量 | 说明 |
|---------|------|
| `RAG_LLM_API_KEY` | LLM API Key |
| `RAG_LLM_BASE_URL` | LLM API 地址 |
| `RAG_LLM_MODEL` | LLM 模型名称 |
| `RAG_CHUNK_SIZE` | 文本块大小 |
| `RAG_TOP_K` | 检索 top-k |
| `RAG_EMBEDDING` | 向量化方式 |
| `RAG_CONFIG` | 配置文件路径 |
| `PORT` | 服务端口（默认 8000） |

## 启动命令

```bash
# 使用默认配置启动
python run.py

# 指定配置文件
RAG_CONFIG=config/example.yaml python run.py

# 使用 OpenAI LLM
RAG_LLM_API_KEY=sk-xxx python run.py

# 指定端口
PORT=3000 python run.py
```

启动后访问 http://localhost:8000

## AI 使用说明

### Mock 模式（默认）

默认使用 TF-IDF Mock embedding 和 Mock LLM，无需任何 API Key 即可运行：
- **Embedding**：基于 TF-IDF 的词频向量，共享关键词的文本有更高相似度
- **LLM**：基于检索到的上下文拼接生成带引用标记的回答

### OpenAI 模式

设置 `RAG_LLM_API_KEY` 环境变量后，系统自动切换到 OpenAI 模式：
- **Embedding**：使用 `text-embedding-3-small`
- **LLM**：使用 `gpt-3.5-turbo`（可在配置中修改）
- **LangGraph**：通过 `langchain-openai` 的 `ChatOpenAI` 在 generate 节点中调用

## SSE 流式 API

### 事件类型

POST `/api/ask` 返回 SSE 流，事件类型如下：

| 事件 | 数据 | 说明 |
|------|------|------|
| `references` | `{count, items}` | 检索到的引用来源列表 |
| `thinking` | `{content, ms}` | 思考过程内容（可折叠） |
| `thinking_end` | `{ms}` | 思考完成，含用时 |
| `answer` | `{content, has_context, follow_up_hint}` | 生成的回答 |
| `done` | `{}` | 流结束 |
| `error` | `{message}` | 错误信息 |

### 示例

```bash
curl -N -X POST http://localhost:8000/api/ask \
  -F "question=Python 有哪些基本数据类型？"
```

响应：
```
event: references
data: {"count": 5, "items": [...]}

event: thinking
data: {"content": "1. 用户问题: ...\n2. 检索到 5 个相关文本块...", "ms": 0}

event: thinking_end
data: {"ms": 0}

event: answer
data: {"content": "根据知识库...[1][2]", "has_context": true, "follow_up_hint": "..."}

event: done
data: {}
```

## 标准测试问题与期望引用来源

内置 5 篇 Python 知识库文档，以下是 3 个标准问题及期望命中来源：

| # | 问题 | 期望命中的引用来源 |
|---|------|-------------------|
| 1 | Python 有哪些基本数据类型？ | `python_basics.md` - 数字类型、字符串、布尔类型章节 |
| 2 | 如何使用装饰器？请举例说明 | `python_functions.md` - 装饰器章节 |
| 3 | Python 的列表推导式怎么写？ | `python_list_dict.md` - 列表推导式章节 |

### 知识库外问题验证

提问「法国的首都是哪里？」等与知识库无关的问题，系统应：
1. 显示思考过程："检索结果最高相似度 0.1355，低于阈值 0.15"
2. 拒绝回答："知识库中未找到相关信息，无法回答该问题。"

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web UI 页面 |
| POST | `/api/ask` | 提问（SSE 流式响应） |
| POST | `/api/upload` | 上传文档（file） |
| POST | `/api/reindex` | 重新索引本地文档 |
| POST | `/api/reset` | 重置对话历史 |
| GET | `/api/status` | 系统状态 |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **Agent 编排**：LangGraph（retrieve → think → generate 节点图）
- **LLM 调用**：LangChain OpenAI（支持真实模型 / Mock 回退）
- **向量化**：TF-IDF Mock / OpenAI Embedding
- **向量存储**：NumPy 内存存储 + 余弦相似度检索
- **流式传输**：SSE（Server-Sent Events）
- **前端**：原生 HTML/CSS/JS，零构建工具

## 参考设计

- **cherry-studio**：思考过程独立 Block 设计、引用来源状态流转
- **LightRAG**：流式 SSE 响应、思考时间展示、查询接口设计
- **MiMo**："已深度思考（用时 X 秒）"可折叠设计、内联引用标记、引用卡片悬浮展示
