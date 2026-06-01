# 多源文档智能问答 Agent（RAG）

基于 RAG（检索增强生成）的多源文档智能问答系统，使用 LangGraph 图编排实现 rewrite → retrieve → think → generate 工作流，支持 SSE 流式响应、思考过程展示、引用溯源与追问。

## 功能特性

- **多源文档接入**：本地文件夹（.md/.txt）+ 页面上传文件
- **LangGraph 图编排**：rewrite → retrieve → think → generate 节点化工作流，条件路由支持拒绝策略
- **混合检索**：Dense 向量检索 + BM25 关键词检索融合，提升召回质量
- **SSE 流式响应**：分阶段展示思考过程 → 引用来源 → 回答内容
- **思考过程展示**：可折叠的"深度思考"面板，显示检索分析和决策逻辑
- **引用溯源**：内联引用标记 `[1]` `[2]`，点击显示引用卡片；顶部"N 个引用内容"按钮展开完整引用面板
- **智能拒绝**：知识库中无相关内容时，明确回复「知识库中未找到」，不编造答案
- **多轮对话**：支持多轮对话与追问澄清，自动进行查询改写（Query Rewrite）
- **会话管理**：支持多会话创建、切换、重命名、删除与持久化
- **可配置**：chunk 大小、top-k、embedding 方式、混合检索权重、数据源路径、系统提示词等均可配置
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
│   ├── config.py                  # 配置系统（Pydantic 模型 + 环境变量覆盖）
│   ├── session_manager.py         # 会话管理（多会话 CRUD、历史压缩）
│   ├── loaders/
│   │   └── document_loader.py     # 文档加载器（本地 + 上传）
│   └── pipeline/
│       ├── chunker.py             # 文档切分
│       ├── embedding.py           # 向量化（Ollama / OpenAI / Mock TF-IDF）
│       ├── bm25_index.py          # BM25 关键词索引（jieba 分词）
│       ├── vector_store.py        # 向量存储与检索（Milvus / 内存）
│       └── rag_pipeline.py        # RAG 流水线编排（混合检索融合）
│   └── agent/
│       ├── graph.py               # LangGraph 图定义（rewrite→retrieve→think→generate）
│       ├── streaming.py           # SSE 流式 Agent
│       └── rag_agent.py           # （旧版，已弃用）
├── web/
│   ├── app.py                     # FastAPI 应用（SSE 流式 API + 会话接口）
│   ├── static/
│   └── templates/
│       └── index.html             # Web UI（流式交互）
├── storage/
│   └── sessions/                  # 会话持久化存储目录
└── docs/
    ├── design.md                  # 设计文档（架构、LangGraph、UI 设计）
    └── sample_docs/               # 5 篇示例文档
        ├── 三国演义.md
        ├── 水浒传.md
        ├── 计算机网络.md
        ├── 暗影蜘蛛侠.md
        └── 大模型技术概览.md
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

### 可选依赖

- **Ollama 本地 Embedding**（默认）：需本地安装 [Ollama](https://ollama.com) 并拉取模型 `ollama pull bge-m3`
- **Milvus 向量数据库**：配置 `milvus_uri` 后自动启用，否则回退到内存存储

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
  embedding: "ollama"                   # 向量化方式: ollama / openai / mock
  ollama_embedding_model: "bge-m3"      # Ollama embedding 模型
  ollama_base_url: "http://localhost:11434"  # Ollama 服务地址
  milvus_uri: ""                        # Milvus 地址（空字符串则使用内存存储）
  milvus_collection: "rag_chunks"       # Milvus 集合名
  dense_weight: 0.6                     # 向量检索权重（混合检索）
  keyword_weight: 0.4                   # BM25 关键词检索权重（混合检索）

agent:
  refuse_when_no_context: true          # 无相关内容时拒绝回答
  system_prompt: "..."                  # 系统提示词
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    api_key: ""                         # LLM API Key
    base_url: ""                        # 自定义 API 地址
    temperature: 0.3
    max_tokens: 2048

ui:
  show_citations: true                  # 显示引用来源
  title: "多源文档智能问答 Agent"

session:
  storage_dir: "./storage/sessions"     # 会话存储目录
  compress_trigger_messages: 50         # 触发历史压缩的消息数阈值
  compress_keep_messages: 20            # 压缩后保留的最近消息数
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
| `RAG_DOTENV` | .env 文件路径 |
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

### Ollama 模式（默认）

默认使用 Ollama 本地 Embedding（bge-m3），无需 API Key 即可运行：
- **Embedding**：通过 Ollama 调用 `bge-m3` 模型生成向量，失败时自动降级为 TF-IDF Mock
- **LLM**：若未配置 API Key，使用基于检索上下文的 Mock 回答拼接

### OpenAI 模式

设置 `RAG_LLM_API_KEY` 环境变量后，系统启用真实 LLM：
- **Embedding**：可选 `text-embedding-3-small`（将 `embedding` 设为 `openai`）
- **LLM**：使用配置的 OpenAI 兼容模型（如 `gpt-3.5-turbo`、`mimo-v2.5-pro`）
- **LangGraph**：在 generate 节点中通过 `openai` 客户端调用

### Mock 模式

将 `embedding` 设为 `mock`：
- **Embedding**：基于 TF-IDF 的词频向量，零外部依赖
- **LLM**：基于检索到的上下文拼接生成带引用标记的回答

## SSE 流式 API

### 事件类型

POST `/api/ask` 返回 SSE 流，事件类型如下：

| 事件 | 数据 | 说明 |
|------|------|------|
| `references` | `{count, items}` | 检索到的引用来源列表 |
| `thinking` | `{content, ms}` | 思考过程内容（可折叠） |
| `thinking_end` | `{ms}` | 思考完成，含用时 |
| `answer_start` | `{has_context}` | 真实 LLM 流式回答开始 |
| `answer_token` | `{token}` | 真实 LLM 流式回答片段 |
| `answer_end` | `{follow_up_hint}` | 真实 LLM 流式回答结束 |
| `answer` | `{content, has_context, follow_up_hint}` | Mock 模式完整回答 |
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

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web UI 页面 |
| POST | `/api/ask` | 提问（SSE 流式响应） |
| POST | `/api/upload` | 上传文档（file） |
| POST | `/api/reindex` | 重新索引本地文档 |
| POST | `/api/reset` | 重置指定会话对话历史 |
| GET | `/api/status` | 系统状态 |
| POST | `/api/sessions` | 创建新会话 |
| GET | `/api/sessions` | 列出所有会话 |
| GET | `/api/sessions/{session_id}` | 获取会话信息 |
| DELETE | `/api/sessions/{session_id}` | 删除会话 |
| PUT | `/api/sessions/{session_id}/title` | 重命名会话 |

## 标准测试问题与期望引用来源

内置 5 篇示例文档（三国演义、水浒传、计算机网络、暗影蜘蛛侠、大模型技术概览），以下是 3 个标准测试用例：

### Case 1：多轮对话上下文保持

**第 1 轮**
- **问题**：三国演义的主要人物有哪些？
- **预期行为**：召回 `三国演义.md` 相关文本块，回答包含刘备、关羽、张飞、诸葛亮等主要人物
- **期望引用**：`三国演义.md` - 主要人物章节

**第 2 轮（追问）**
- **问题**：作者是谁呢？
- **预期行为**：基于多轮对话上下文，通过 Query Rewrite 将问题改写为"三国演义的作者是谁"，召回 `三国演义.md` 相关文本块
- **预期回答**：罗贯中（只回答三国演义的作者，不应混入水浒传等其他文档内容）
- **期望引用**：`三国演义.md` - 作者信息章节

> 此用例验证多轮对话的上下文保持与查询改写能力。

### Case 2：新上传文件知识库问答

- **问题**：minicalw作者是谁呢
- **预期行为**：召回 `新的知识库.md` 相关文本块
- **期望引用**：`新的知识库.md` 

### Case 3：知识库外问题拒绝

- **问题**：巴甫洛夫的狗
- **预期行为**：检索结果最高相似度低于阈值（0.2），系统明确拒绝回答
- **预期回答**："知识库中未找到相关信息，无法回答该问题。"
- **验证点**：系统不编造答案，不 hallucinate

### 其他知识库外问题验证

提问「法国的首都是哪里？」等与知识库无关的问题，系统应：
1. 显示思考过程："检索结果最高相似度 0.1355，低于阈值 0.2"
2. 拒绝回答："知识库中未找到相关信息，无法回答该问题。"

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **Agent 编排**：LangGraph（rewrite → retrieve → think → generate 节点图）
- **LLM 调用**：OpenAI 兼容 API（支持真实模型 / Mock 回退）
- **向量化**：Ollama（bge-m3）/ OpenAI Embedding / TF-IDF Mock
- **关键词检索**：BM25Okapi + jieba 中文分词
- **向量存储**：Milvus（可选）/ NumPy 内存存储 + 余弦相似度
- **混合检索**：Dense 向量检索 + BM25 关键词检索加权融合
- **流式传输**：SSE（Server-Sent Events）
- **会话管理**：JSON 文件持久化 + 历史压缩
- **前端**：原生 HTML/CSS/JS，零构建工具
