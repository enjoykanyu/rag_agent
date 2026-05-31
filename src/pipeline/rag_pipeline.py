from __future__ import annotations

import logging
from typing import Any

from src.config import AppConfig
from src.loaders.document_loader import Document, load_local_docs, load_uploaded_file
from src.pipeline.bm25_index import BM25Index
from src.pipeline.chunker import Chunk, split_documents
from src.pipeline.embedding import create_embedding_provider
from src.pipeline.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        embedding_dim = 1024
        if config.rag.embedding == "mock":
            embedding_dim = 256

        self.store = VectorStore(
            dim=embedding_dim,
            milvus_uri=config.rag.milvus_uri,
            collection_name=config.rag.milvus_collection,
        )
        self.bm25 = BM25Index()
        self.embedding_provider = create_embedding_provider(
            config.rag.embedding,
            api_key=config.agent.llm.api_key,
            base_url=config.agent.llm.base_url,
            ollama_embedding_model=config.rag.ollama_embedding_model,
            ollama_base_url=config.rag.ollama_base_url,
        )
        self._documents: list[Document] = []
        self._all_chunks: list[Chunk] = []

    def index_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        self._documents.extend(documents)
        chunks = split_documents(
            documents,
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
        )

        if not chunks:
            return 0

        self._all_chunks.extend(chunks)

        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.store.add(chunks, embeddings)
        self.bm25.build(self._all_chunks)

        logger.info(
            "Indexed %d documents, %d chunks (total store: %d, bm25: %d)",
            len(documents),
            len(chunks),
            self.store.size,
            self.bm25.size,
        )
        return len(chunks)

    def index_local(self) -> int:
        local_cfg = self.config.sources.local
        docs = load_local_docs(local_cfg.path, local_cfg.patterns)
        logger.info("Found %d local documents in %s", len(docs), local_cfg.path)
        return self.index_documents(docs)

    def index_upload(self, filename: str, content: bytes) -> int:
        doc = load_uploaded_file(filename, content)
        if doc is None:
            return 0
        return self.index_documents([doc])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.config.rag.top_k
        query_emb = self.embedding_provider.embed([query])[0]
        dense_results = self.store.search(query_emb, top_k=k * 2)
        keyword_results = self.bm25.search(query, top_k=k * 2)
        return self._hybrid_fusion(dense_results, keyword_results, top_k=k)

    def _hybrid_fusion(
        self,
        dense_results: list[RetrievalResult],
        keyword_results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        dense_weight = self.config.rag.dense_weight
        keyword_weight = self.config.rag.keyword_weight

        dense_max = max((r.score for r in dense_results), default=1.0) or 1.0
        keyword_max = max((r["keyword_score"] for r in keyword_results), default=1.0) or 1.0

        scores: dict[str, float] = {}
        meta: dict[str, RetrievalResult] = {}

        for r in dense_results:
            cid = r.chunk.chunk_id
            normalized = r.score / dense_max
            scores[cid] = scores.get(cid, 0.0) + dense_weight * normalized
            meta[cid] = r

        chunk_map = {c.chunk_id: c for c in self._all_chunks}
        for item in keyword_results:
            cid = item["chunk_id"]
            normalized = item["keyword_score"] / keyword_max
            scores[cid] = scores.get(cid, 0.0) + keyword_weight * normalized
            if cid not in meta:
                chunk = chunk_map.get(cid)
                if chunk:
                    meta[cid] = RetrievalResult(chunk=chunk, score=0.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievalResult] = []
        for cid, fused_score in ranked:
            r = meta.get(cid)
            if r:
                results.append(RetrievalResult(chunk=r.chunk, score=fused_score))
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return self.store.size
