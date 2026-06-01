from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import AppConfig
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    index: int
    source: str
    document_name: str = ""
    section: str = ""
    snippet: str
    score: float


class GraphState(BaseModel):
    question: str = ""
    rewritten_question: str = ""
    session_id: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # retrieve
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)

    # think
    context_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    thinking: str = ""
    thinking_ms: int = 0
    should_answer: bool = True

    # generate
    answer: str = ""
    has_context: bool = True
    follow_up_hint: str = ""

    class Config:
        arbitrary_types_allowed = True


def create_rag_graph(config: AppConfig, pipeline: RAGPipeline) -> StateGraph:

    def rewrite_node(state: GraphState) -> dict[str, Any]:
        if not state.conversation_history:
            return {"rewritten_question": state.question}

        history_lines: list[str] = []
        for msg in state.conversation_history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

        rewrite_prompt = (
            f"以下是一段对话历史:\n{history_text}\n\n"
            f"用户最新问题: {state.question}\n\n"
            f"请将用户最新问题改写为一个独立、完整、自包含的问题，使其不需要对话历史也能被理解。"
            f"只输出改写后的问题，不要解释，不要加引号。"
            f"如果问题本身已经完整，直接原样输出。"
        )

        rewritten = _call_llm_sync(
            config,
            "你是一个查询改写助手，擅长将依赖上下文的简短问题改写为完整独立的问题。",
            rewrite_prompt,
            [],
        ).strip()

        if not rewritten:
            rewritten = state.question

        logger.info("Query rewrite: '%s' -> '%s'", state.question, rewritten)
        return {"rewritten_question": rewritten}

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        query = state.rewritten_question or state.question
        results = pipeline.retrieve(query, top_k=config.rag.top_k)
        return {"retrieved_chunks": results}

    def think_node(state: GraphState) -> dict[str, Any]:
        t0 = time.time()
        chunks = state.retrieved_chunks

        if not chunks or (chunks and chunks[0].score < 0.2):
            if config.agent.refuse_when_no_context:
                thinking = (
                    f"分析用户问题: 「{state.question}」\n\n"
                    f"检索结果: 共检索到 {len(chunks)} 个文本块，"
                    f"最高相关度 {chunks[0].score if chunks else 0:.4f}，低于阈值。\n\n"
                    f"结论: 知识库中未找到与该问题相关的内容，无法基于知识库回答。"
                )
                return {
                    "thinking": thinking,
                    "thinking_ms": int((time.time() - t0) * 1000),
                    "should_answer": False,
                    "has_context": False,
                    "answer": "知识库中未找到相关信息，无法回答该问题。",
                    "follow_up_hint": "您可以尝试换一种方式提问，或上传相关文档到知识库。",
                }

        citations: list[Citation] = []
        context_parts: list[str] = []
        for i, r in enumerate(chunks, 1):
            source = r.chunk.source
            doc_name = r.chunk.metadata.get("document_name", source)
            section = r.chunk.metadata.get("section", "")
            snippet = r.chunk.content[:400]
            section_label = f" > {section}" if section else ""
            context_parts.append(f"[{i}] 来源: {doc_name}{section_label}\n{snippet}")
            citations.append(Citation(
                index=i,
                source=source,
                document_name=doc_name,
                section=section,
                snippet=r.chunk.content[:200],
                score=round(r.score, 4),
            ))

        top_scores = [
            f"[{c.index}]{c.document_name}" + (f" > {c.section}" if c.section else "") + f"({c.score:.3f})"
            for c in citations[:3]
        ]
        thinking = (
            f"分析用户问题: 「{state.question}」\n\n"
            f"检索与融合: 通过 Dense+BM25 混合检索召回 {len(chunks)} 个文本块。\n\n"
            f"最相关来源:\n"
        )
        for ts in top_scores:
            thinking += f"  - {ts}\n"
        thinking += f"\n结论: 知识库中有足够的相关信息，可以基于检索内容生成带引用的回答。"

        return {
            "thinking": thinking,
            "thinking_ms": int((time.time() - t0) * 1000),
            "should_answer": True,
            "has_context": True,
            "context_text": "\n\n".join(context_parts),
            "citations": citations,
        }

    def generate_node(state: GraphState) -> dict[str, Any]:
        if not state.should_answer:
            return {}

        system = config.agent.system_prompt
        query = state.rewritten_question or state.question

        prompt = (
            f"以下是检索到的上下文内容:\n\n{state.context_text}\n\n"
            f"请严格根据以上上下文回答问题。要求:\n"
            f"1. 使用 Markdown 格式组织回答，包括标题(##/###)、列表、代码块、表格等\n"
            f"2. 回答要准确、完整，结构清晰，直接回答问题\n"
            f"3. 只回答与问题直接相关的内容，不要回答问题未涉及的其他信息\n"
            f"4. 在相关事实后标注引用来源编号，格式如 [1] [2]，引用标记放在句末\n"
            f"5. 代码示例使用 ```language 代码块格式\n"
            f"6. 如果上下文中没有足够信息，请明确说明\n\n"
            f"问题: {query}"
        )

        answer = _call_llm_sync(config, system, prompt, state.conversation_history)

        follow_up = ""
        if len(state.conversation_history) <= 2:
            follow_up = "如果您想进一步了解，可以继续追问。"

        return {"answer": answer, "follow_up_hint": follow_up}

    def route_after_think(state: GraphState) -> Literal["generate", END]:
        if state.should_answer:
            return "generate"
        return END

    builder = StateGraph(GraphState)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("think", think_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "think")
    builder.add_conditional_edges("think", route_after_think)
    builder.add_edge("generate", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _call_llm_sync(config: AppConfig, system: str, prompt: str, history: list[dict[str, str]]) -> str:
    llm_cfg = config.agent.llm
    if llm_cfg.api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url or None)
            messages: list[dict[str, str]] = [{"role": "system", "content": system}]
            for msg in history:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=llm_cfg.model,
                messages=messages,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return _fallback_answer(prompt)
    return _fallback_answer(prompt)


async def stream_llm_answer(
    config: AppConfig, system: str, prompt: str, history: list[dict[str, str]]
) -> AsyncGenerator[str, None]:
    llm_cfg = config.agent.llm
    if llm_cfg.api_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url or None)
            messages: list[dict[str, str]] = [{"role": "system", "content": system}]
            for msg in history:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            stream = await client.chat.completions.create(
                model=llm_cfg.model,
                messages=messages,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("LLM stream failed: %s", e)
            for token in _fallback_answer(prompt):
                yield token
    else:
        for token in _fallback_answer(prompt):
            yield token


def _fallback_answer(prompt: str) -> str:
    context_marker = "以下是检索到的上下文内容:"
    context_section = prompt.split(context_marker)
    if len(context_section) < 2:
        return "知识库中未找到相关信息。"
    context = context_section[1].split("请严格根据以上上下文回答问题")[0]
    question_part = prompt.split("问题:")[-1].strip() if "问题:" in prompt else ""

    sources: list[str] = []
    content_lines: list[str] = []
    for line in context.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "] 来源:" in line:
            sources.append(line)
        elif not line.startswith("["):
            content_lines.append(line)

    if content_lines:
        summary = " ".join(content_lines[:6])
        refs = " ".join([f"[{i}]" for i in range(1, min(len(sources) + 1, 4))])
        return f"根据知识库中的信息，关于「{question_part}」：\n\n{summary}\n\n以上内容参考了相关文档资料 {refs}。"
    return "知识库中未找到相关信息。"
