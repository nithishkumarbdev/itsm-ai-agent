import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_embedding_client, get_retrieval_service
from app.db.database import get_db
from app.embeddings.base import EmbeddingClient
from app.models import KnowledgeDocument
from app.schemas.classification import Category
from app.schemas.retrieval import IndexResultRead, KnowledgeSearchRequest, KnowledgeSearchResponse
from app.schemas.knowledge import (
    KnowledgeDocumentCreate,
    KnowledgeDocumentRead,
    KnowledgeDocumentSummary,
    KnowledgeDocumentUpdate,
)
from app.services import knowledge as service
from app.services.indexing import index_document
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

NOT_FOUND = {404: {"description": "Document not found"}}
DUPLICATE_TITLE = {409: {"description": "A document with this title already exists"}}


def get_document_or_404(db: Session, document_id: uuid.UUID, *, lock: bool = False) -> KnowledgeDocument:
    document = service.get_document(db, document_id, lock=lock)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge document not found")
    return document


@router.post("", response_model=KnowledgeDocumentRead, status_code=status.HTTP_201_CREATED, responses=DUPLICATE_TITLE)
def create_knowledge_document(payload: KnowledgeDocumentCreate, db: Session = Depends(get_db)):
    """Add a document to the knowledge base. Titles are unique."""
    document = service.create_document(db, payload)
    db.commit()
    db.refresh(document)
    return document


@router.get("", response_model=list[KnowledgeDocumentSummary])
def list_knowledge_documents(
    category: Category | None = None,
    is_active: bool | None = Query(None, description="true = active only, false = inactive only, omitted = all"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List documents (without their content), ordered by title."""
    return service.list_documents(db, category=category.value if category else None, is_active=is_active, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=KnowledgeDocumentRead, responses=NOT_FOUND)
def get_knowledge_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """One document with its full content. Inactive documents can still be read here."""
    return get_document_or_404(db, document_id)


@router.patch("/{document_id}", response_model=KnowledgeDocumentRead, responses={**NOT_FOUND, **DUPLICATE_TITLE})
def update_knowledge_document(document_id: uuid.UUID, payload: KnowledgeDocumentUpdate, db: Session = Depends(get_db)):
    """Change a document, or activate / deactivate it with {"is_active": false}.

    The version goes up by 1 when the title, category, content or metadata changes.
    Documents are never deleted: deactivating keeps them stored but out of retrieval.
    """
    document = get_document_or_404(db, document_id, lock=True)
    service.update_document(db, document, payload)
    db.commit()
    db.refresh(document)
    return document


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    payload: KnowledgeSearchRequest,
    db: Session = Depends(get_db),
    retriever: RetrievalService = Depends(get_retrieval_service),
):
    """Find the knowledge chunks most relevant to a text, with similarity scores and sources.

    Only active documents that are indexed at their current version are searched. If nothing
    is relevant enough the result list is empty; it never returns weak matches to fill the list.
    This finds evidence only: it does not write an answer.
    """
    return retriever.retrieve(
        db,
        payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        category=payload.category.value if payload.category else None,
    )


@router.post("/{document_id}/index", response_model=IndexResultRead, responses=NOT_FOUND)
def index_knowledge_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    embedder: EmbeddingClient = Depends(get_embedding_client),
):
    """Chunk and embed one document so it can be retrieved.

    Safe to repeat: if the stored chunks already match the document, nothing is changed.
    If the document changed, its old chunks are replaced. Inactive documents are skipped.
    """
    document = get_document_or_404(db, document_id, lock=True)
    result = index_document(db, document, embedder)
    db.commit()
    return result
