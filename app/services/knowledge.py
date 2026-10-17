"""Knowledge base operations. Like the rest of the app, these functions never commit:
the caller (route or script) commits, so a request succeeds or fails as one unit."""

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import KnowledgeDocument
from app.schemas.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentUpdate


class DuplicateDocumentTitle(Exception):
    def __init__(self, title: str):
        super().__init__(f"A knowledge document titled {title!r} already exists")
        self.title = title


def _new_document(data: KnowledgeDocumentCreate) -> KnowledgeDocument:
    return KnowledgeDocument(
        title=data.title,
        category=data.category.value,
        content=data.content,
        version=data.version,
        is_active=data.is_active,
        doc_metadata=data.metadata,
    )


def _flush_or_duplicate(db: Session, title: str) -> None:
    """Send pending changes to the database. A unique-title violation becomes DuplicateDocumentTitle."""
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        # Only report "duplicate" if a document with that title really exists.
        if db.scalar(select(KnowledgeDocument.id).where(KnowledgeDocument.title == title)):
            raise DuplicateDocumentTitle(title) from None
        raise


def create_document(db: Session, data: KnowledgeDocumentCreate) -> KnowledgeDocument:
    document = _new_document(data)
    db.add(document)
    _flush_or_duplicate(db, data.title)
    return document


def get_document(db: Session, document_id: uuid.UUID, *, lock: bool = False) -> KnowledgeDocument | None:
    """lock=True holds the row until the transaction ends, so updates cannot overlap."""
    return db.get(KnowledgeDocument, document_id, with_for_update=True if lock else None)


def list_documents(
    db: Session,
    *,
    category: str | None = None,
    is_active: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[KnowledgeDocument]:
    stmt = select(KnowledgeDocument).order_by(KnowledgeDocument.title, KnowledgeDocument.id)
    if category is not None:
        stmt = stmt.where(KnowledgeDocument.category == category)
    if is_active is not None:
        stmt = stmt.where(KnowledgeDocument.is_active == is_active)
    return list(db.scalars(stmt.limit(limit).offset(offset)))


def update_document(db: Session, document: KnowledgeDocument, data: KnowledgeDocumentUpdate) -> KnowledgeDocument:
    """Apply the provided changes.

    Version rule: if the title, category, content or metadata actually changes, the version
    goes up by 1. Changing only is_active (activate / deactivate) does not, and neither does
    sending values that are already stored.
    """
    changes = data.model_dump(mode="json", exclude_unset=True)
    content_changed = False

    for field_name in ("title", "category", "content"):
        if field_name in changes and changes[field_name] != getattr(document, field_name):
            setattr(document, field_name, changes[field_name])
            content_changed = True
    if "metadata" in changes and changes["metadata"] != document.doc_metadata:
        document.doc_metadata = changes["metadata"]
        content_changed = True
    if "is_active" in changes:
        document.is_active = changes["is_active"]

    if content_changed:
        document.version += 1
    _flush_or_duplicate(db, document.title)
    return document


@dataclass
class SeedResult:
    inserted: list[str] = field(default_factory=list)  # titles added
    skipped: list[str] = field(default_factory=list)  # titles that already existed


def seed_documents(db: Session, documents: Iterable[KnowledgeDocumentCreate]) -> SeedResult:
    """Insert the documents that are not in the database yet (matched by title).

    Safe to run repeatedly. It never updates or overwrites an existing document, so edits
    made after seeding are kept.
    """
    existing = set(db.scalars(select(KnowledgeDocument.title)))
    result = SeedResult()
    for data in documents:
        if data.title in existing:
            result.skipped.append(data.title)
            continue
        db.add(_new_document(data))
        existing.add(data.title)
        result.inserted.append(data.title)
    db.flush()
    return result
