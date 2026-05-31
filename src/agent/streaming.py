from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import Citation, GraphState, stream_llm_answer
from src.config import AppConfig

logger = logging.getLogger(__name__)


class StreamingRAGAgent:
    def __init__(self, graph: CompiledStateGraph, config: AppConfig) -> None:
        self.graph = graph
        self.config = config
        self._conversation_history: list[dict[str, str]] = []

    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        state = GraphState(
            question=question,
            conversation_history=list(self._conversation_history),
        )

        self._conversation_history.append({"role": "user", "content": question})

        try:
            # Phase 1: Run LangGraph (retrieve → rerank → think → generate)
            # We run it synchronously to get all non-streaming results first
            final_state = await self.graph.ainvoke(state.model_dump())

            # Reconstruct state from final_state (dict)
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

            # Phase 2: Stream events in order

            # 2a. Send references
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

            # 2b. Send thinking
            if thinking:
                yield _sse("thinking", {"content": thinking, "ms": thinking_ms})
                yield _sse("thinking_end", {"ms": thinking_ms})

            # 2c. Send answer (stream if possible, otherwise send complete)
            if not should_answer:
                yield _sse("answer", {
                    "content": answer,
                    "has_context": has_context,
                    "follow_up_hint": follow_up,
                })
                yield _sse("done", {})
                self._conversation_history.append({"role": "assistant", "content": answer})
                return

            # Try streaming the answer from LLM
            if self.config.agent.llm.api_key and answer:
                # We already have the answer from LangGraph, but let's stream it token by token
                # for a better UX experience (simulate streaming from the pre-computed answer)
                yield _sse("answer_start", {"has_context": has_context})

                # Stream the answer in chunks for typewriter effect
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
            self._conversation_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            yield _sse("error", {"message": str(e)})

    def reset_conversation(self) -> None:
        self._conversation_history = []


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
