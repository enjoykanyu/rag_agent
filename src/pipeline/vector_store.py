from __future__ import annotations

import logging
from dataclasses import dataclass

from src.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dim: int = 1024, milvus_uri: str = "", collection_name: str = "rag_chunks") -> None:
        self._chunks: dict[str, Chunk] = {}
        self._dim = dim
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name
        self._milvus_client = None
        self._use_milvus = bool(milvus_uri)

        if self._use_milvus:
            self._init_milvus()

    def _init_milvus(self) -> None:
        try:
            from pymilvus import MilvusClient

            self._milvus_client = MilvusClient(uri=self._milvus_uri)
            if self._milvus_client.has_collection(self._collection_name):
                self._milvus_client.drop_collection(self._collection_name)

            self._milvus_client.create_collection(
                collection_name=self._collection_name,
                dimension=self._dim,
                metric_type="COSINE",
                auto_id=True,
            )
            logger.info("Milvus collection '%s' created (dim=%d)", self._collection_name, self._dim)
        except Exception as e:
            logger.warning("Milvus init failed, falling back to in-memory: %s", e)
            self._use_milvus = False
            self._milvus_client = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

        if self._use_milvus and self._milvus_client is not None:
            self._add_milvus(chunks, embeddings)
        else:
            self._add_memory(embeddings)

    def _add_milvus(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        try:
            data = []
            for chunk, emb in zip(chunks, embeddings):
                data.append({
                    "vector": emb,
                    "chunk_id": chunk.chunk_id,
                })
            self._milvus_client.insert(collection_name=self._collection_name, data=data)
            logger.info("Inserted %d vectors into Milvus", len(data))
        except Exception as e:
            logger.error("Milvus insert failed: %s", e)
            self._use_milvus = False
            self._milvus_client = None

    def _add_memory(self, embeddings: list[list[float]]) -> None:
        import numpy as np

        new_vectors = np.array(embeddings, dtype=np.float32)
        if not hasattr(self, "_vectors") or self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        if self._use_milvus and self._milvus_client is not None:
            return self._search_milvus(query_embedding, top_k)
        return self._search_memory(query_embedding, top_k)

    def _search_milvus(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        try:
            results = self._milvus_client.search(
                collection_name=self._collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["chunk_id"],
            )
            retrieval_results: list[RetrievalResult] = []
            if results and len(results) > 0:
                for hit in results[0]:
                    chunk_id = hit["entity"]["chunk_id"]
                    score = hit["distance"]
                    chunk = self._chunks.get(chunk_id)
                    if chunk:
                        retrieval_results.append(RetrievalResult(chunk=chunk, score=float(score)))
            return retrieval_results
        except Exception as e:
            logger.error("Milvus search failed: %s", e)
            return self._search_memory(query_embedding, top_k)

    def _search_memory(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        import numpy as np

        if not hasattr(self, "_vectors") or self._vectors is None or len(self._chunks) == 0:
            return []

        chunk_list = list(self._chunks.values())
        query_vec = np.array(query_embedding, dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        scores = self._vectors @ query_vec / (norms * query_norm + 1e-8)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if idx < len(chunk_list):
                results.append(
                    RetrievalResult(chunk=chunk_list[idx], score=float(scores[idx]))
                )
        return results

    def clear(self) -> None:
        self._chunks = {}
        if hasattr(self, "_vectors"):
            self._vectors = None
        if self._use_milvus and self._milvus_client is not None:
            try:
                if self._milvus_client.has_collection(self._collection_name):
                    self._milvus_client.drop_collection(self._collection_name)
                self._milvus_client.create_collection(
                    collection_name=self._collection_name,
                    dimension=self._dim,
                    metric_type="COSINE",
                    auto_id=True,
                )
            except Exception as e:
                logger.error("Milvus clear failed: %s", e)

    @property
    def size(self) -> int:
        return len(self._chunks)
