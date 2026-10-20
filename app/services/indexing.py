"""Turn a knowledge document into embedded chunks in the database.

document -> chunks (app/services/chunking.py) -> embeddings -> knowledge_chunks rows

Like the other services, these functions never commit: the caller does.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.chunking import DEFAULT_MAX_CHARS, Chunk, chunk_document


@dataclass(frozen=True)
class IndexResult:
    document_id: uuid.UUID
    title: str
    status: Literal["indexed", "unchanged", "skipped_inactive"]
    chunk_count: int
    embedding_model: str


def embedding_text(title: str, content: str) -> str:
    """What is embedded for a chunk: the document title plus the chunk, so a chunk such as
    "Symptoms" still carries the topic it belongs to. Must be the same at every re-index."""
    return f"{title}\n{content}"


def _is_current(existing: list[KnowledgeChunk], chunks: list[Chunk], version: int, model: str) -> bool:
    """True if the stored chunks are exactly what indexing would produce now."""
    return len(existing) == len(chunks) and all(
        old.document_version == version
        and old.embedding_model == model
        and old.chunk_index == new.chunk_index
        and old.content == new.content
        for old, new in zip(existing, chunks)
    )


def index_document(db: Session, document: KnowledgeDocument, embedder: EmbeddingClient) -> IndexResult:
    """Make the document's stored chunks match its current text.

    - Inactive documents are not indexed (and are never retrieved anyway).
    - If the chunks are already up to date, nothing happens: no embedding call, no writes,
      so running it twice never creates duplicates.
    - Otherwise the old chunks are replaced by new ones in the same transaction, so old and
      new chunks are never both present.
    - If embedding fails, EmbeddingError is raised before anything is changed.
    """
    model = embedder.model
    if not document.is_active:
        return IndexResult(document.id, document.title, "skipped_inactive", 0, model)

    chunks = chunk_document(
        document_id=document.id,
        title=document.title,
        category=document.category,
        version=document.version,
        content=document.content,
        max_chars=DEFAULT_MAX_CHARS,
    )
    existing = list(
        db.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document.id)
            .order_by(KnowledgeChunk.chunk_index)
        )
    )
    if _is_current(existing, chunks, document.version, model):
        return IndexResult(document.id, document.title, "unchanged", len(existing), model)

    vectors = embedder.embed_many([embedding_text(document.title, c.content) for c in chunks])
    if len(vectors) != len(chunks) or any(len(v) != embedder.dimension for v in vectors):
        raise EmbeddingError("invalid_response", f"expected {len(chunks)} vectors of {embedder.dimension} numbers")

    db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
    db.add_all(
        KnowledgeChunk(
            document_id=document.id,
            chunk_index=chunk.chunk_index,
            section=chunk.section,
            content=chunk.content,
            document_version=document.version,
            embedding_model=model,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors)
    )
    db.flush()
    return IndexResult(document.id, document.title, "indexed", len(chunks), model)
