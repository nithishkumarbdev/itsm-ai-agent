from collections.abc import Sequence
from typing import Literal, Protocol


class EmbeddingError(Exception):
    """The provider could not produce usable embeddings (timeout, outage, bad key, bad response)."""

    def __init__(
        self,
        kind: Literal["timeout", "provider_error", "invalid_response"],
        detail: str | None = None,
    ):
        super().__init__(f"Embedding {kind}" + (f": {detail}" if detail else ""))
        self.kind = kind
        self.detail = detail


class EmbeddingClient(Protocol):
    """All the app needs from an embedding model: turn texts into vectors."""

    model: str  # identifies the vector space; stored with every indexed chunk
    dimension: int  # every vector has exactly this many numbers

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in the same order. Raises EmbeddingError on any failure.

        Implementations must return vectors of exactly `dimension` numbers, or raise.
        """
        ...


def embed_one(client: EmbeddingClient, text: str) -> list[float]:
    return client.embed_many([text])[0]
