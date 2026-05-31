from __future__ import annotations

import logging
from typing import Any

from src.config import AppConfig
from src.loaders.document_loader import Document, load_local_docs, load_uploaded_file
from src.pipeline.chunker import Chunk, split_documents
from src.pipeline.embedding import create_embedding_provider
from src.pipeline.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = VectorStore()
        self.embedding_provider = create_embedding_provider(
            config.rag.embedding,
            api_key=config.agent.llm.api_key,
            base_url=config.agent.llm.base_url,
            ollama_embedding_model=config.rag.ollama_embedding_model,
            ollama_base_url=config.rag.ollama_base_url,
        )
        self._documents: list[Document] = []

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

        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.store.add(chunks, embeddings)
        logger.info(
            "Indexed %d documents, %d chunks (total store: %d)",
            len(documents),
            len(chunks),
            self.store.size,
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
        return self.store.search(query_emb, top_k=k)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return self.store.size
