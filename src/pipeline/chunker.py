from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from src.loaders.document_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


def _parse_sections(text: str) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    for line in text.split("\n"):
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            result.append((title, line, level))
    return result


def _build_section_path(sections: list[tuple[str, str, int]], up_to_line_idx: int, lines: list[str]) -> str:
    active: list[tuple[str, int]] = []
    for line_idx in range(min(up_to_line_idx, len(lines))):
        line = lines[line_idx]
        for title, _, level in sections:
            if line.strip() == title or line.strip().startswith("#"):
                m = re.match(r'^(#{1,6})\s+(.+)$', line)
                if m and m.group(2).strip() == title:
                    active = [(t, l) for t, l in active if l < level]
                    active.append((title, level))
                    break
    if not active:
        return ""
    return " > ".join(t for t, _ in active)


def split_documents(
    documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = _split_single(doc, chunk_size, chunk_overlap)
        chunks.extend(doc_chunks)
    return chunks


def _split_single(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    lines = doc.content.split("\n")
    sections = _parse_sections(doc.content)

    header_map: dict[int, str] = {}
    current_path: list[tuple[str, int]] = []
    for line_idx, line in enumerate(lines):
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            current_path = [(t, l) for t, l in current_path if l < level]
            current_path.append((title, level))
            header_map[line_idx] = " > ".join(t for t, _ in current_path)

    paragraphs = _split_paragraphs(doc.content)
    chunks: list[Chunk] = []
    current_text = ""
    current_meta: list[dict] = []
    current_section = ""
    current_line_idx = 0

    for para_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue

        first_line = para.split("\n")[0]
        para_line_idx = _find_line_index(lines, first_line, current_line_idx)
        if para_line_idx >= 0:
            for li in range(current_line_idx, para_line_idx + 1):
                if li in header_map:
                    current_section = header_map[li]
            current_line_idx = para_line_idx

        if len(current_text) + len(para) + 1 <= chunk_size:
            current_text = f"{current_text}\n{para}".strip() if current_text else para
            current_meta.append({"paragraph_index": para_idx, "source": doc.source, "section": current_section})
        else:
            if current_text:
                chunks.append(
                    _make_chunk(current_text, doc, current_meta, len(chunks), current_section)
                )
            if len(para) > chunk_size:
                sub_chunks = _split_by_chars(para, chunk_size, chunk_overlap)
                for sc in sub_chunks:
                    chunks.append(
                        _make_chunk(
                            sc,
                            doc,
                            [{"paragraph_index": para_idx, "source": doc.source, "section": current_section}],
                            len(chunks),
                            current_section,
                        )
                    )
                current_text = ""
                current_meta = []
            else:
                current_text = para
                current_meta = [{"paragraph_index": para_idx, "source": doc.source, "section": current_section}]

    if current_text:
        chunks.append(_make_chunk(current_text, doc, current_meta, len(chunks), current_section))

    return chunks


def _find_line_index(lines: list[str], target: str, start: int) -> int:
    stripped = target.strip()
    for i in range(start, len(lines)):
        if lines[i].strip() == stripped:
            return i
    return -1


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    result: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - chunk_overlap
    return result


def _make_chunk(
    text: str, doc: Document, meta_parts: list[dict], index: int, section: str = ""
) -> Chunk:
    doc_name = doc.metadata.get("filename", doc.source)
    if doc_name.endswith(".md") or doc_name.endswith(".txt"):
        doc_name = doc_name.rsplit(".", 1)[0]

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        content=text,
        source=doc.source,
        metadata={
            **doc.metadata,
            "chunk_index": index,
            "paragraphs": meta_parts,
            "document_name": doc_name,
            "section": section,
        },
    )
