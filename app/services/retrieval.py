"""Find the knowledge chunks most relevant to a piece of text.

query text -> embedding -> cosine similarity against stored chunk vectors -> top-k

This only *finds evidence*. It never writes an answer. If nothing is relevant enough the
result is an empty list; it never returns "the closest thing" just to have something.

Similarity is cosine similarity between the query vector and each chunk vector, computed
exactly in Python over all eligible chunks. Eligible means: the document is active and the
chunk was built from the document's current version.
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingClient, embed_one
from app.models import KnowledgeChunk, KnowledgeDocument

logger = logging.getLogger(__name__)

# Ranking bonus for a chunk whose document category equals the ticket's category. It only
# affects ORDER: the relevance threshold uses the raw similarity, and the raw similarity is
# what is reported. So the bonus can prefer a same-category chunk but can never make an
# irrelevant chunk pass the threshold, and other categories are still found.
CATEGORY_BOOST = 0.10

MAX_QUERY_CHARS = 2000  # keeps the embedding request small; the problem is stated up front anyway


class KnowledgeIndexError(Exception):
    """The stored index cannot be searched with the configured embedding model."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One piece of evidence, with everything needed to trace it back to its source."""

    document_id: uuid.UUID
    document_title: str
    category: str
    version: int
    chunk_id: uuid.UUID
    chunk_index: int
    section: str
    content: str
    similarity_score: float  # raw cosine similarity, never adjusted
    category_match: bool

    def reference(self) -> dict:
        """The concise form stored on a ticket: ids and scores, not the text."""
        return {
            "document_id": str(self.document_id),
            "document_title": self.document_title,
            "category": self.category,
            "version": self.version,
            "chunk_id": str(self.chunk_id),
            "chunk_index": self.chunk_index,
            "similarity_score": self.similarity_score,
        }


@dataclass(frozen=True)
class RetrievalOutcome:
    results: list[RetrievedChunk]  # empty means "no sufficiently relevant knowledge"
    candidates_considered: int  # 0 means the index is empty: nothing has been indexed yet
    top_k: int
    min_score: float


def build_query(title: str, description: str) -> str:
    """The text that is searched for: what the ticket says, and nothing else."""
    return f"{title}\n{description}"[:MAX_QUERY_CHARS]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class RetrievalService:
    def __init__(self, embedder: EmbeddingClient, *, top_k: int, min_score: float):
        self._embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    @property
    def embedding_model(self) -> str:
        return self._embedder.model

    def embed_query(self, text: str) -> list[float]:
        """Turn the query into a vector. This is the external call; it touches no database."""
        started = time.perf_counter()
        vector = embed_one(self._embedder, text)
        logger.info(
            "query embedded model=%s latency_ms=%d", self._embedder.model, round((time.perf_counter() - started) * 1000)
        )
        return vector

    def retrieve(
        self,
        db: Session,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        category: str | None = None,
    ) -> RetrievalOutcome:
        """embed_query + search."""
        return self.search(db, self.embed_query(query), top_k=top_k, min_score=min_score, category=category)

    def search(
        self,
        db: Session,
        query_vector: list[float],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        category: str | None = None,
    ) -> RetrievalOutcome:
        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started = time.perf_counter()
        rows = db.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .where(KnowledgeDocument.is_active.is_(True))
            .where(KnowledgeChunk.document_version == KnowledgeDocument.version)  # never a stale version
            .execution_options(populate_existing=True)  # current database values, not cached objects
        ).all()

        # Vectors from different embedding models (or sizes) cannot be compared. Say so
        # loudly instead of returning meaningless scores.
        for chunk, _ in rows:
            if chunk.embedding_model != self._embedder.model or len(chunk.embedding) != len(query_vector):
                raise KnowledgeIndexError(
                    "The knowledge index was built with a different embedding model; re-run indexing."
                )

        scored = []
        for chunk, document in rows:
            similarity = cosine_similarity(query_vector, chunk.embedding)
            if similarity < min_score:
                continue  # not relevant enough: never returned, however few results that leaves
            match = category is not None and document.category == category
            scored.append((similarity + (CATEGORY_BOOST if match else 0.0), similarity, match, chunk, document))
        # Best first. Ties are broken by title and position so the order is always the same.
        scored.sort(key=lambda s: (-s[0], s[4].title, s[3].chunk_index))

        results = [
            RetrievedChunk(
                document_id=document.id,
                document_title=document.title,
                category=document.category,
                version=document.version,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                section=chunk.section,
                content=chunk.content,
                similarity_score=round(similarity, 4),
                category_match=match,
            )
            for _, similarity, match, chunk, document in scored[:top_k]
        ]
        logger.info(
            "retrieval completed candidates=%d results=%d top_score=%s latency_ms=%d",
            len(rows),
            len(results),
            max((r.similarity_score for r in results), default=None),
            round((time.perf_counter() - started) * 1000),
        )
        if not rows:
            logger.warning("retrieval found no indexed chunks: has `python -m scripts.index_knowledge` been run?")
        return RetrievalOutcome(results, len(rows), top_k, min_score)
