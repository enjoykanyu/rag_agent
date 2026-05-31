from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import Citation, GraphState, stream_llm_answer
from src.config import AppConfig
from src.session_manager import SessionManager

logger = logging.getLogger(__name__)


class StreamingRAGAgent:
    def __init__(self, graph: CompiledStateGraph, config: AppConfig, session_manager: SessionManager) -> None:
        self.graph = graph
        self.config = config
        self.session_manager = session_manager
        self._conversation_histories: dict[str, list[dict[str, str]]] = {}

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        if session_id not in self._conversation_histories:
            self._conversation_histories[session_id] = self.session_manager.load_session(session_id)
        return self._conversation_histories[session_id]

    def _append_history(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._conversation_histories:
            self._conversation_histories[session_id] = []
        self._conversation_histories[session_id].append({"role": role, "content": content})
        self.session_manager.save_message(session_id, role, content)

    async def ask_stream(self, question: str, session_id: str = "default") -> AsyncGenerator[str, None]:
        history = self._get_history(session_id)

        state = GraphState(
            question=question,
            session_id=session_id,
            conversation_history=list(history),
        )

        self._append_history(session_id, "user", question)

        try:
            final_state = await self.graph.ainvoke(state.model_dump())

            citations = []
            for c in final_state.get("citations", []):
                if isinstance(c, dict):
                    citations.append(Citation(**c))
                elif isinstance(c, Citation):
                    citations.append(c)

            thinking = final_state.get("thinking", "")
            thinking_ms = final_state.get("thinking_ms", 0)
            should_answer = final_state.get("should_answer", True)
            has_context = final_state.get("has_context", True)
            answer = final_state.get("answer", "")
            follow_up = final_state.get("follow_up_hint", "")

            if citations:
                yield _sse("references", {
                    "count": len(citations),
                    "items": [
                        {
                            "index": c.index,
                            "source": c.source,
                            "document_name": c.document_name,
                            "section": c.section,
                            "snippet": c.snippet,
                            "score": c.score,
                        }
                        for c in citations
                    ],
                })

            if thinking:
                yield _sse("thinking", {"content": thinking, "ms": thinking_ms})
                yield _sse("thinking_end", {"ms": thinking_ms})

            if not should_answer:
                yield _sse("answer", {
                    "content": answer,
                    "has_context": has_context,
                    "follow_up_hint": follow_up,
                })
                yield _sse("done", {})
                self._append_history(session_id, "assistant", answer)
                return

            if self.config.agent.llm.api_key and answer:
                yield _sse("answer_start", {"has_context": has_context})

                chunk_size = 2
                for i in range(0, len(answer), chunk_size):
                    token = answer[i:i + chunk_size]
                    yield _sse("answer_token", {"token": token})

                yield _sse("answer_end", {"follow_up_hint": follow_up})
            elif answer:
                yield _sse("answer", {
                    "content": answer,
                    "has_context": has_context,
                    "follow_up_hint": follow_up,
                })

            yield _sse("done", {})
            self._append_history(session_id, "assistant", answer)

        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            yield _sse("error", {"message": str(e)})

    def reset_conversation(self, session_id: str = "default") -> None:
        if session_id in self._conversation_histories:
            del self._conversation_histories[session_id]


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
