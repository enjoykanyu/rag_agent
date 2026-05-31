from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from src.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingProvider:
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            try:
                resp = httpx.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    results.append(embeddings[0])
                else:
                    logger.warning("Ollama returned empty embedding, using fallback")
                    results.append(self._fallback_embedding(text))
            except Exception as e:
                logger.error("Ollama embedding failed: %s", e)
                results.append(self._fallback_embedding(text))
        return results

    def _fallback_embedding(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        vec = rng.randn(1024).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class MockEmbeddingProvider:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-z]+|[0-9]+', text)
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]}{tokens[i+1]}")
        return tokens + bigrams

    def _build_vocab(self, texts: list[str]) -> None:
        doc_freq: dict[str, int] = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            for t in set(tokens):
                doc_freq[t] += 1
        sorted_tokens = sorted(doc_freq.keys(), key=lambda x: (-doc_freq[x], x))
        self._vocab = {t: i for i, t in enumerate(sorted_tokens[:self.dim])}
        n_docs = len(texts)
        self._idf = {}
        for t, idx in self._vocab.items():
            self._idf[t] = np.log((n_docs + 1) / (doc_freq.get(t, 0) + 1)) + 1
        self._fitted = True

    def _text_to_tfidf(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = np.zeros(self.dim, dtype=np.float32)
        for t, count in tf.items():
            if t in self._vocab:
                idx = self._vocab[t]
                vec[idx] = (count / total) * self._idf.get(t, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self._build_vocab(texts)
        return [self._text_to_tfidf(t).tolist() for t in texts]


class OpenAIEmbeddingProvider:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "", base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


def create_embedding_provider(provider_name: str, **kwargs) -> EmbeddingProvider:
    if provider_name == "ollama":
        return OllamaEmbeddingProvider(
            model=kwargs.get("ollama_embedding_model", "bge-m3"),
            base_url=kwargs.get("ollama_base_url", "http://localhost:11434"),
        )
    elif provider_name == "mock":
        return MockEmbeddingProvider(dim=kwargs.get("dim", 256))
    elif provider_name == "openai":
        return OpenAIEmbeddingProvider(
            model=kwargs.get("model", "text-embedding-3-small"),
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", ""),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
