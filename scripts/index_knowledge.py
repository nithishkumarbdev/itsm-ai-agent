"""Chunk and embed the active knowledge documents so they can be retrieved.

Uses the REAL embedding provider from .env (EMBEDDING_MODEL, EMBEDDING_API_KEY, ...), so it
calls the provider's API. Safe to repeat: documents that are already up to date are skipped
without any API call. It is not run automatically when the app starts.

Run from the repository root, after `python -m scripts.seed_knowledge`:

    python -m scripts.index_knowledge
"""

import sys

from sqlalchemy import select

from app.api.deps import get_embedding_client
from app.db.database import Base, SessionLocal, engine
from app.embeddings.base import EmbeddingError
from app.models import KnowledgeDocument
from app.services.indexing import index_document


def main() -> int:
    embedder = get_embedding_client()
    print(f"Embedding model: {embedder.model} ({embedder.dimension} dimensions)\n")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        documents = list(db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.title)))
        if not documents:
            print("No knowledge documents found. Run: python -m scripts.seed_knowledge")
            return 1
        for document in documents:
            try:
                result = index_document(db, document, embedder)
            except EmbeddingError as exc:
                db.rollback()
                print(f"FAILED  {document.title}: {exc}")
                return 1
            db.commit()  # one document at a time, so earlier work is kept if a later one fails
            print(f"{result.status:17s} {result.chunk_count:2d} chunks  {document.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
