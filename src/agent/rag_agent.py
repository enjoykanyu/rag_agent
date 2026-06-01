from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import AppConfig
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    source: str
    chunk_index: int
    snippet: str
    score: float
    document_name: str = ""
    section: str = ""


@dataclass
class AgentResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    has_context: bool = True
    follow_up_hint: str = ""


class RAGAgent:
    def __init__(self, config: AppConfig, pipeline: RAGPipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self._conversation_history: list[dict[str, str]] = []

    def ask(self, question: str) -> AgentResponse:
        query = self._rewrite_query(question)
        results = self.pipeline.retrieve(query)

        if not results or results[0].score < 0.15:
            if self.config.agent.refuse_when_no_context:
                self._conversation_history.append({"role": "user", "content": question})
                no_context_msg = "知识库中未找到相关信息，无法回答该问题。"
                self._conversation_history.append({"role": "assistant", "content": no_context_msg})
                return AgentResponse(
                    answer=no_context_msg,
                    has_context=False,
                    follow_up_hint="您可以尝试换一种方式提问，或上传相关文档到知识库。",
                )

        context_text = self._build_context(results)
        citations = self._build_citations(results)

        prompt = self._build_prompt(query, context_text)

        history_snapshot = list(self._conversation_history)
        self._conversation_history.append({"role": "user", "content": question})

        answer = self._call_llm(prompt, history_snapshot)

        self._conversation_history.append({"role": "assistant", "content": answer})

        follow_up = ""
        if len(self._conversation_history) <= 2:
            follow_up = "如果您想进一步了解，可以继续追问。"

        return AgentResponse(
            answer=answer,
            citations=citations,
            has_context=True,
            follow_up_hint=follow_up,
        )

    def _rewrite_query(self, question: str) -> str:
        if not self._conversation_history:
            return question

        history_lines: list[str] = []
        for msg in self._conversation_history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

        rewrite_prompt = (
            f"以下是一段对话历史:\n{history_text}\n\n"
            f"用户最新问题: {question}\n\n"
            f"请将用户最新问题改写为一个独立、完整、自包含的问题，使其不需要对话历史也能被理解。"
            f"只输出改写后的问题，不要解释，不要加引号。"
            f"如果问题本身已经完整，直接原样输出。"
        )

        llm_cfg = self.config.agent.llm
        if llm_cfg.provider == "openai" and llm_cfg.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url or None)
                response = client.chat.completions.create(
                    model=llm_cfg.model,
                    messages=[
                        {"role": "system", "content": "你是一个查询改写助手，擅长将依赖上下文的简短问题改写为完整独立的问题。"},
                        {"role": "user", "content": rewrite_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=128,
                )
                rewritten = (response.choices[0].message.content or "").strip()
                if rewritten:
                    logger.info("Query rewrite: '%s' -> '%s'", question, rewritten)
                    return rewritten
            except Exception as e:
                logger.error("Query rewrite LLM call failed: %s", e)

        return question

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.chunk.source
            doc_name = r.chunk.metadata.get("document_name", source)
            section = r.chunk.metadata.get("section", "")
            snippet = r.chunk.content[:300]
            section_label = f" > {section}" if section else ""
            parts.append(f"[来源{i}] 文档: {doc_name}{section_label}\n{snippet}")
        return "\n\n".join(parts)

    def _build_citations(self, results: list[RetrievalResult]) -> list[Citation]:
        citations: list[Citation] = []
        for r in results:
            citations.append(
                Citation(
                    source=r.chunk.source,
                    document_name=r.chunk.metadata.get("document_name", r.chunk.source),
                    section=r.chunk.metadata.get("section", ""),
                    chunk_index=r.chunk.metadata.get("chunk_index", 0),
                    snippet=r.chunk.content[:200],
                    score=round(r.score, 4),
                )
            )
        return citations

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"以下是检索到的上下文内容:\n\n{context}\n\n"
            f"请根据以上上下文回答问题。要求:\n"
            f"1. 只回答与问题直接相关的内容，不要回答问题未涉及的其他信息\n"
            f"2. 如果上下文中没有相关信息，请明确说明\n"
            f"3. 回答时请标注引用来源编号\n\n"
            f"问题: {query}"
        )

    def _call_llm(self, prompt: str, history: list[dict[str, str]]) -> str:
        llm_cfg = self.config.agent.llm

        if llm_cfg.provider == "openai" and llm_cfg.api_key:
            return self._call_openai(prompt, history)
        else:
            return self._mock_answer(prompt)

    def _call_openai(self, prompt: str, history: list[dict[str, str]]) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.agent.llm.api_key,
                base_url=self.config.agent.llm.base_url or None,
            )
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.config.agent.system_prompt},
            ]
            for msg in history:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.config.agent.llm.model,
                messages=messages,
                temperature=self.config.agent.llm.temperature,
                max_tokens=self.config.agent.llm.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return self._mock_answer(prompt)

    def _mock_answer(self, prompt: str) -> str:
        if not prompt.strip():
            return "知识库中未找到相关信息。"

        context_marker = "以下是检索到的上下文内容:"
        context_section = prompt.split(context_marker)
        if len(context_section) < 2:
            return "知识库中未找到相关信息。"

        context = context_section[1].split("请根据以上上下文回答问题")[0]
        question_part = prompt.split("问题:")[-1].strip() if "问题:" in prompt else ""

        sources: list[str] = []
        content_lines: list[str] = []
        for line in context.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[来源"):
                sources.append(line)
            else:
                content_lines.append(line)

        if content_lines:
            summary_parts = content_lines[:5]
            summary = " ".join(summary_parts)
            source_refs = ", ".join(sources[:3])
            return (
                f"根据知识库中的信息，关于「{question_part}」：\n\n"
                f"{summary}\n\n"
                f"参考来源: {source_refs}\n\n"
                f"如需更多细节，请继续追问。"
            )

        return "知识库中未找到相关信息。"

    def reset_conversation(self) -> None:
        self._conversation_history = []
