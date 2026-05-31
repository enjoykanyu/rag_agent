from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator


@dataclass
class Document:
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


def load_local_docs(base_path: str, patterns: list[str]) -> list[Document]:
    base = Path(base_path)
    if not base.exists():
        return []

    docs: list[Document] = []
    seen: set[str] = set()
    for pattern in patterns:
        for fp in base.glob(pattern):
            resolved = fp.resolve()
            if resolved in seen or not fp.is_file():
                continue
            seen.add(resolved)
            try:
                text = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            if text.strip():
                docs.append(
                    Document(
                        content=text,
                        source=str(fp.relative_to(base)),
                        metadata={"filename": fp.name, "loader": "local"},
                    )
                )
    return docs


def load_uploaded_file(filename: str, content: bytes) -> Document | None:
    try:
        text = content.decode("utf-8")
    except Exception:
        return None
    if not text.strip():
        return None
    return Document(
        content=text,
        source=f"upload://{filename}",
        metadata={"filename": filename, "loader": "upload"},
    )
