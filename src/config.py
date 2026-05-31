import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LocalSourceConfig(BaseModel):
    path: str = "./docs/sample_docs"
    patterns: list[str] = Field(default_factory=lambda: ["**/*.md", "**/*.txt"])


class UploadSourceConfig(BaseModel):
    enabled: bool = True
    allowed_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt"])


class SourcesConfig(BaseModel):
    local: LocalSourceConfig = Field(default_factory=LocalSourceConfig)
    upload: UploadSourceConfig = Field(default_factory=UploadSourceConfig)


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding: str = "ollama"
    ollama_embedding_model: str = "bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    milvus_uri: str = ""
    milvus_collection: str = "rag_chunks"
    dense_weight: float = 0.6
    keyword_weight: float = 0.4


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "mimo-v2.5-pro"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048


class AgentConfig(BaseModel):
    refuse_when_no_context: bool = True
    system_prompt: str = (
        "你是一个基于知识库的智能问答助手。请严格根据检索到的上下文内容回答问题，"
        "不要编造信息。如果知识库中没有相关内容，请明确回复「知识库中未找到相关信息」。"
        "每条回答必须附带引用来源，格式如 [1] [2]。"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)


class UIConfig(BaseModel):
    show_citations: bool = True
    title: str = "多源文档智能问答 Agent"


class SessionConfig(BaseModel):
    storage_dir: str = "./storage/sessions"
    compress_trigger_messages: int = 50
    compress_keep_messages: int = 20


class AppConfig(BaseModel):
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)


def _load_dotenv(dotenv_path: str | None = None) -> None:
    if dotenv_path is None:
        dotenv_path = os.environ.get("RAG_DOTENV", ".env")

    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(config_path: str | None = None) -> AppConfig:
    _load_dotenv()

    if config_path is None:
        config_path = os.environ.get("RAG_CONFIG", "config/example.yaml")

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw = {}

    env_overrides(raw)

    return AppConfig(**raw)


def env_overrides(raw: dict[str, Any]) -> None:
    env_map = {
        "RAG_LLM_API_KEY": ("agent", "llm", "api_key"),
        "RAG_LLM_BASE_URL": ("agent", "llm", "base_url"),
        "RAG_LLM_MODEL": ("agent", "llm", "model"),
        "RAG_CHUNK_SIZE": ("rag", "chunk_size"),
        "RAG_TOP_K": ("rag", "top_k"),
        "RAG_EMBEDDING": ("rag", "embedding"),
    }
    for env_key, path in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            node = raw
            for key in path[:-1]:
                node = node.setdefault(key, {})
            leaf_key = path[-1]
            if leaf_key in ("chunk_size", "top_k"):
                val = int(val)
            node[leaf_key] = val
