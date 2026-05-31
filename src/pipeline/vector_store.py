from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.chunker import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._chunks.extend(chunks)
        new_vectors = np.array(embeddings, dtype=np.float32)

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        scores = self._vectors @ query_vec / (norms * query_norm + 1e-8)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            results.append(
                RetrievalResult(chunk=self._chunks[idx], score=float(scores[idx]))
            )
        return results

    def clear(self) -> None:
        self._chunks = []
        self._vectors = None

    @property
    def size(self) -> int:
        return len(self._chunks)
