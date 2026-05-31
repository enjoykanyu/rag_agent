# 设计文档

## 架构说明

系统采用分层架构，从下到上分为五层：

```
┌─────────────────────────────────┐
│          Web UI (前端)           │  展示层 — SSE 流式渲染
├─────────────────────────────────┤
│       FastAPI (API 路由)         │  接口层 — SSE 流式接口
├─────────────────────────────────┤
│   StreamingRAGAgent (流式 Agent) │  Agent 层 — SSE 事件编排
├─────────────────────────────────┤
│   LangGraph (retrieve→think→gen)│  图编排层 — 节点化工作流
├─────────────────────────────────┤
│  RAGPipeline (切分→向量化→检索)  │  流水线层
├─────────────────────────────────┤
│  DocumentLoader (文档接入)       │  数据层
└─────────────────────────────────┘
```

### 各层职责

1. **数据层（DocumentLoader）**：负责从不同来源加载原始文档，统一输出 `Document` 对象
2. **流水线层（RAGPipeline）**：编排文档切分、向量化、向量存储与检索的完整 RAG 流程
3. **图编排层（LangGraph）**：使用 LangGraph 定义 `retrieve → think → generate` 节点图，实现可观测、可扩展的 Agent 工作流
4. **Agent 层（StreamingRAGAgent）**：将 LangGraph 的节点产出转换为 SSE 流式事件，支持思考过程、引用来源、回答内容的分阶段展示
5. **接口层（FastAPI）**：提供 SSE 流式 API 和 Web UI，处理 HTTP 请求
6. **展示层（Web UI）**：用户交互界面，实时展示思考过程、引用来源、内联引用标记

## LangGraph 图编排设计

### 节点定义

```
START ──→ retrieve ──→ think ──→ [conditional] ──→ generate ──→ END
                          │
                          └── should_answer=false ──→ END
```

| 节点 | 职责 | 产出 |
|------|------|------|
| **retrieve** | 从向量库检索相关文本块 | `retrieved_chunks`, `context_text`, `citations` |
| **think** | 判断是否有足够上下文，生成思考过程 | `thinking`, `thinking_ms`, `should_answer` |
| **generate** | 调用 LLM 生成带引用的回答 | `answer`, `follow_up_hint` |

### 条件路由

`think` 节点后根据 `should_answer` 字段路由：
- `should_answer=true` → 进入 `generate` 节点
- `should_answer=false` → 直接结束（返回拒绝回答）

### 流式事件映射

| LangGraph 节点完成 | SSE 事件 | 前端展示 |
|-------------------|---------|---------|
| retrieve | `references` | "📎 N 个引用内容" 按钮 |
| think (进行中) | `thinking` | 旋转图标 + "深度思考中..." |
| think (完成) | `thinking_end` | ✓ 图标 + "已深度思考（用时 X 秒）" |
| generate | `answer` | 回答内容 + 内联引用标记 |
| 结束 | `done` | 完成状态 |

## Agent 设计

### 核心流程（流式）

```
用户提问 → StreamingRAGAgent.ask_stream()
  ├── 1. LangGraph 执行 retrieve 节点
  │     └── 发送 SSE: references {count, items}
  ├── 2. LangGraph 执行 think 节点
  │     ├── 发送 SSE: thinking {content, ms}
  │     └── 发送 SSE: thinking_end {ms}
  ├── 3. 条件判断
  │     ├── 无相关内容 → 发送 SSE: answer {拒绝内容} → done
  │     └── 有相关内容 → 继续
  ├── 4. LangGraph 执行 generate 节点
  │     └── 发送 SSE: answer {content, has_context, follow_up_hint}
  └── 5. 发送 SSE: done
```

### 思考过程设计

参考 cherry-studio 和 MiMo 的设计，思考过程包含：
- **思考中**：旋转加载图标 + "深度思考中..."
- **思考完成**：✓ 图标 + "已深度思考（用时 X 秒）"
- **思考内容**：可折叠展开，展示检索结果分析和决策逻辑

### 引用溯源设计

参考 MiMo 图片和 LightRAG 的设计：
- **引用按钮**：回答顶部显示 "📎 N 个引用内容" 按钮，点击展开引用面板
- **内联引用标记**：回答文本中使用 `[1]`、`[2]` 等上标数字标记，点击显示引用卡片悬浮提示
- **引用卡片**：包含文档名、内容摘要、相似度分数

### 拒绝策略

当检索结果的最高相似度分数低于阈值（默认 0.15）时：
1. `think` 节点判定为「知识库中无相关内容」
2. 直接发送拒绝回答的 SSE 事件
3. 不调用 `generate` 节点，避免编造答案

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

**选择**：使用 LangGraph 定义 retrieve → think → generate 节点图

**理由**：
- 节点化的工作流使每个阶段可独立观测和调试
- 条件路由支持复杂的决策逻辑（如拒绝回答）
- 天然支持流式输出（每个节点完成可发送事件）
- 未来可轻松扩展新节点（如 rerank、multi-query 等）

**代价**：引入额外依赖，学习成本略高于直接函数调用

### 2. SSE 流式响应

**选择**：Server-Sent Events 而非 WebSocket

**理由**：
- 单向流式（服务端→客户端）完全满足需求
- 基于 HTTP，兼容性好，无需额外协议握手
- FastAPI 原生支持 StreamingResponse

**代价**：不支持客户端向服务端发送消息（追问通过新请求实现）

### 3. 向量存储：内存 vs 持久化

**选择**：纯内存向量存储（NumPy ndarray）

**理由**：
- 项目定位为演示/练习，数据量有限
- 避免引入 FAISS/ChromaDB 等外部依赖
- 启动时从源文件重新索引，保证数据一致性

**代价**：重启后需重新索引，不适合大规模数据

### 4. Embedding：TF-IDF Mock vs 真实模型

**选择**：默认 TF-IDF Mock，可选 OpenAI

**理由**：
- Mock 模式零配置即可运行，降低体验门槛
- TF-IDF 基于词频，共享关键词的文本有更高相似度
- 生产环境可切换到 OpenAI embedding

**代价**：Mock 向量的语义检索能力不如真实 embedding 模型
