import uuid
from datetime import datetime

from sqlalchemy import ARRAY, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.models.ticket import utcnow


class KnowledgeChunk(Base):
    """One retrievable piece of a knowledge document, with its embedding vector.

    The document's title and category are not copied here: retrieval joins to the document,
    so they can never go out of date. `document_version` and `embedding_model` record what
    the chunk was built from, so retrieval can ignore chunks of an older version and refuse
    to mix vectors from different embedding models.

    The vector is a plain PostgreSQL float array and similarity is computed in Python
    (exact cosine similarity). That is fast for a knowledge base of hundreds of chunks;
    pgvector becomes worth it for many thousands.
    """

    __tablename__ = "knowledge_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("knowledge_documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text, default="")  # heading of the section this chunk came from
    content: Mapped[str] = mapped_column(Text)
    document_version: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
