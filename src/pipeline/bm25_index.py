from __future__ import annotations

import logging
import re
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from src.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

_CJK_PUNCTUATION = set(
    "，。！？；：、""''（）【】《》〈〉…—～·「」『』"
    ",.!?;:()[]<>{}\"'`~@#$%^&*+=|/\\\n\r\t"
)


def _is_noise_token(t: str) -> bool:
    if not t or t.isspace():
        return True
    if all(c in _CJK_PUNCTUATION for c in t):
        return True
    if len(t) == 1:
        if re.match(r"^[\u3400-\u4DBF\u4E00-\u9FFF]$", t):
            return False
        if not t.isalnum():
            return True
    return False


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    jieba_tokens = jieba.lcut(text)
    for t in jieba_tokens:
        t = t.strip()
        if not _is_noise_token(t):
            tokens.append(t.lower())
    return tokens


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._tokenized_docs: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = []
        self._tokenized_docs = []
        for chunk in chunks:
            tokens = tokenize(chunk.content)
            if tokens:
                self._chunk_ids.append(chunk.chunk_id)
                self._tokenized_docs.append(tokens)
        if self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)
        else:
            self._bm25 = None
        logger.info("BM25 index built: %d chunks indexed", len(self._chunk_ids))

    def search(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in scored:
            if score <= 0:
                continue
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "keyword_score": float(score),
            })
        return results

    @property
    def size(self) -> int:
        return len(self._chunk_ids)
