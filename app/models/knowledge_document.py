import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.models.ticket import utcnow


class KnowledgeDocument(Base):
    """A document in the internal IT knowledge base (trusted, company-maintained content).

    One row per document. `version` starts at 1 and goes up by 1 whenever the document's
    content changes (see app/services/knowledge.py), so a citation like "VPN Guide v2"
    always points at one exact text. Documents are deactivated, never deleted:
    only active documents are meant to be used for retrieval.
    """

    __tablename__ = "knowledge_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), unique=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    content: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    # The column and API field are called "metadata", but that name is reserved by
    # SQLAlchemy's declarative base, so the Python attribute is doc_metadata.
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
