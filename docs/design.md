# 设计文档

## 架构说明

系统采用分层架构，从下到上分为六层：

```
┌─────────────────────────────────┐
│          Web UI (前端)           │  展示层 — SSE 流式渲染
├─────────────────────────────────┤
│       FastAPI (API 路由)         │  接口层 — SSE 流式接口 + 会话管理
├─────────────────────────────────┤
│   StreamingRAGAgent (流式 Agent) │  Agent 层 — SSE 事件编排
├─────────────────────────────────┤
│   LangGraph (rewrite→retrieve→  │  图编排层 — 节点化工作流
│              think→generate)     │
├─────────────────────────────────┤
│   RAGPipeline (切分→向量化→      │  流水线层 — 混合检索融合
│              索引→检索)          │
├─────────────────────────────────┤
│  DocumentLoader (文档接入)       │  数据层
└─────────────────────────────────┘
```

### 各层职责

1. **数据层（DocumentLoader）**：负责从不同来源加载原始文档，统一输出 `Document` 对象
2. **流水线层（RAGPipeline）**：编排文档切分、向量化、BM25 索引构建、向量存储与混合检索的完整 RAG 流程
3. **图编排层（LangGraph）**：使用 LangGraph 定义 `rewrite → retrieve → think → generate` 节点图，实现可观测、可扩展的 Agent 工作流
4. **Agent 层（StreamingRAGAgent）**：将 LangGraph 的节点产出转换为 SSE 流式事件，支持思考过程、引用来源、回答内容的分阶段展示
5. **接口层（FastAPI）**：提供 SSE 流式 API、会话管理 API 和 Web UI，处理 HTTP 请求
6. **展示层（Web UI）**：用户交互界面，实时展示思考过程、引用来源、内联引用标记

## LangGraph 图编排设计

### 节点定义

```
START ──→ rewrite ──→ retrieve ──→ think ──→ [conditional] ──→ generate ──→ END
                                              │
                                              └── should_answer=false ──→ END
```

| 节点 | 职责 | 产出 |
|------|------|------|
| **rewrite** | 基于对话历史改写用户问题为独立完整的问题 | `rewritten_question` |
| **retrieve** | 从向量库和 BM25 索引混合检索相关文本块 | `retrieved_chunks` |
| **think** | 判断是否有足够上下文，生成思考过程 | `thinking`, `thinking_ms`, `should_answer`, `context_text`, `citations` |
| **generate** | 调用 LLM 生成带引用的回答 | `answer`, `follow_up_hint` |

### 条件路由

`think` 节点后根据 `should_answer` 字段路由：
- `should_answer=true` → 进入 `generate` 节点
- `should_answer=false` → 直接结束（返回拒绝回答）

### 查询改写（Query Rewrite）

当存在多轮对话历史时，`rewrite` 节点会将依赖上下文的简短问题改写为完整独立的问题：

**示例**：
- 第 1 轮："三国演义的主要人物有哪些？"
- 第 2 轮（用户）："作者是谁呢？"
- 改写后："三国演义的作者是谁？"

这样可以确保后续 `retrieve` 节点不依赖对话历史也能准确召回相关文本块。

### 流式事件映射

| LangGraph 节点完成 | SSE 事件 | 前端展示 |
|-------------------|---------|---------|
| retrieve | `references` | "📎 N 个引用内容" 按钮 |
| think (进行中) | `thinking` | 旋转图标 + "深度思考中..." |
| think (完成) | `thinking_end` | ✓ 图标 + "已深度思考（用时 X 秒）" |
| generate (真实 LLM) | `answer_start` → `answer_token` → `answer_end` | 逐字流式输出 |
| generate (Mock) | `answer` | 完整回答内容 + 内联引用标记 |
| 结束 | `done` | 完成状态 |

## Agent 设计

### 核心流程（流式）

```
用户提问 → StreamingRAGAgent.ask_stream()
  ├── 1. LangGraph 执行 rewrite 节点
  │     └── 基于对话历史改写查询
  ├── 2. LangGraph 执行 retrieve 节点
  │     └── 发送 SSE: references {count, items}
  ├── 3. LangGraph 执行 think 节点
  │     ├── 发送 SSE: thinking {content, ms}
  │     └── 发送 SSE: thinking_end {ms}
  ├── 4. 条件判断
  │     ├── 无相关内容 → 发送 SSE: answer {拒绝内容} → done
  │     └── 有相关内容 → 继续
  ├── 5. LangGraph 执行 generate 节点
  │     ├── 真实 LLM: answer_start → answer_token → answer_end
  │     └── Mock 模式: answer {content, has_context, follow_up_hint}
  └── 6. 发送 SSE: done
```

### 思考过程设计

参考 cherry-studio 和 MiMo 的设计，思考过程包含：
- **思考中**：旋转加载图标 + "深度思考中..."
- **思考完成**：✓ 图标 + "已深度思考（用时 X 秒）"
- **思考内容**：可折叠展开，展示检索结果分析和决策逻辑，包括：
  - 用户问题分析
  - Dense+BM25 混合检索召回的文本块数量
  - 最相关的来源及相似度分数
  - 是否具备足够上下文的结论

### 引用溯源设计

参考 MiMo 图片和 LightRAG 的设计：
- **引用按钮**：回答顶部显示 "📎 N 个引用内容" 按钮，点击展开引用面板
- **内联引用标记**：回答文本中使用 `[1]`、`[2]` 等上标数字标记，点击显示引用卡片悬浮提示
- **引用卡片**：包含文档名、章节、内容摘要、相似度分数

### 拒绝策略

当检索结果的最高相似度分数低于阈值（默认 0.2）时：
1. `think` 节点判定为「知识库中无相关内容」
2. 直接发送拒绝回答的 SSE 事件
3. 不调用 `generate` 节点，避免编造答案

## 混合检索设计

系统采用 Dense + BM25 混合检索策略，兼顾语义相关性和关键词匹配：

```
用户查询
    ├──→ Embedding 模型 → 向量检索（VectorStore）→ Dense Results
    │                                                    │
    └──→ jieba 分词 → BM25 检索（BM25Index）→ Keyword Results
                                                         │
                                              加权融合（Weighted Fusion）
                                                         │
                                              最终排序 Top-K 结果
```

### 融合公式

```
score_final = dense_weight × (score_dense / max_dense) + keyword_weight × (score_keyword / max_keyword)
```

默认权重：`dense_weight=0.6`，`keyword_weight=0.4`

### 各检索方式特点

| 检索方式 | 优势 | 适用场景 |
|---------|------|---------|
| Dense（向量） | 语义理解，同义词、近义词召回 | 概念性、描述性问题 |
| BM25（关键词） | 精确匹配，关键词命中率高 | 专有名词、术语查询 |
| 混合融合 | 两者互补，提升整体召回质量 | 通用场景 |

## 会话管理设计

### 会话模型

```
Session
├── id: UUID
├── title: 会话标题
├── created_at: 创建时间
├── updated_at: 更新时间
├── compressed_context: 压缩后的历史摘要
└── messages: 消息列表 [{role, content}, ...]
```

### 多会话支持

- **创建会话**：`POST /api/sessions` 生成新会话 ID
- **切换会话**：前端通过 `session_id` 参数隔离不同对话上下文
- **会话列表**：`GET /api/sessions` 返回所有会话概览（按更新时间倒序）
- **重命名**：`PUT /api/sessions/{id}/title`
- **删除**：`DELETE /api/sessions/{id}`

### 历史压缩

当消息数超过 `compress_trigger_messages`（默认 50）时：
1. 提取前 `compress_trigger_messages - compress_keep_messages` 条消息
2. 通过 LLM 生成对话摘要
3. 将摘要存入 `compressed_context`
4. 保留最近 `compress_keep_messages`（默认 20）条完整消息

加载会话时，`compressed_context` 以 assistant 角色注入为系统提示，确保长对话的上下文不丢失。

## 前端设计

### 参考来源

- **cherry-studio**：思考过程的独立 Block 设计、引用来源的状态流转（PROCESSING → SUCCESS）
- **LightRAG**：流式 SSE 响应处理、思考时间的计算和展示
- **MiMo（用户图片）**："已深度思考（用时 X 秒）"的可折叠设计、"N 个引用内容"按钮、内联引用标记 `[1]`、引用卡片悬浮展示

### UI 布局

```
┌─────────────────────────────────────┐
│  [R] RAG Agent    [上传][重置][索引] │  Header
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │  用户消息气泡（右对齐蓝色）    │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌──┐  [深度思考中... ▼]          │   │
│  │A │  📎 5 个引用内容 ▼           │   │  Assistant
│  └──┘  根据知识库...[1][2][3]      │   │  Message
│        如需更多细节，请继续追问。    │   │
│                                     │
├─────────────────────────────────────┤
│  [输入问题...               ] [➤]  │  Input
│  支持多轮追问 · 知识库外问题将被拒绝  │
├─────────────────────────────────────┤
│  ● 服务正常  📄 5 文档  🧩 17 块   │  Status
└─────────────────────────────────────┘
```

## 关键取舍

### 1. LangGraph 图编排

**选择**：使用 LangGraph 定义 rewrite → retrieve → think → generate 节点图

**理由**：
- 节点化的工作流使每个阶段可独立观测和调试
- 条件路由支持复杂的决策逻辑（如拒绝回答）
- 天然支持流式输出（每个节点完成可发送事件）
- 查询改写节点使多轮对话的上下文保持更可靠
- 未来可轻松扩展新节点（如 rerank、multi-query 等）

**代价**：引入额外依赖，学习成本略高于直接函数调用

### 2. SSE 流式响应

**选择**：Server-Sent Events 而非 WebSocket

**理由**：
- 单向流式（服务端→客户端）完全满足需求
- 基于 HTTP，兼容性好，无需额外协议握手
- FastAPI 原生支持 StreamingResponse

**代价**：不支持客户端向服务端发送消息（追问通过新请求实现）

### 3. 混合检索：Dense + BM25

**选择**：Dense 向量检索 + BM25 关键词检索加权融合

**理由**：
- Dense 检索擅长语义匹配，BM25 擅长关键词精确匹配
- 两者互补可显著提升召回率和准确率
- jieba + BM25Okapi 对中文支持良好
- 权重可配置，便于针对不同场景调优

**代价**：检索耗时约为单一检索方式的 1.5-2 倍，需维护两套索引

### 4. Embedding：Ollama 优先

**选择**：默认 Ollama 本地 Embedding（bge-m3），降级到 TF-IDF Mock

**理由**：
- Ollama 本地运行无需网络依赖，保护数据隐私
- bge-m3 是开源中文 Embedding 模型，效果优于 TF-IDF
- 失败时自动降级到 Mock，保证系统可用性
- 生产环境可切换到 OpenAI Embedding

**代价**：需要本地安装 Ollama 并拉取模型，首次配置有门槛

### 5. 向量存储：Milvus / 内存双模式

**选择**：优先 Milvus（配置后），否则回退到 NumPy 内存存储

**理由**：
- 项目定位为演示/练习，数据量有限，内存存储足够
- Milvus 支持后可无缝切换，适合生产环境扩展
- 避免强制引入外部数据库依赖

**代价**：重启后需重新索引（内存模式），不适合大规模数据

### 6. 会话管理：JSON 文件持久化

**选择**：JSON 文件存储会话，而非数据库

**理由**：
- 零额外依赖，部署简单
- 会话数据量小，文件 IO 性能足够
- 便于调试和手动查看会话内容

**代价**：高并发场景下文件锁可能成为瓶颈，不支持分布式部署
