from collections.abc import Sequence
from typing import Any

import httpx

from app.embeddings.base import EmbeddingError

BATCH_SIZE = 64  # texts per request; a document has only a handful of chunks


class OpenAICompatibleEmbeddings:
    """EmbeddingClient for any service with an OpenAI-style `POST {base_url}/embeddings`.

    Request:  {"model": ..., "input": [text, ...]}
    Response: {"data": [{"index": 0, "embedding": [...]}, ...]}
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None,
        dimension: int,
        base_url: str,
        timeout: float,
        max_retries: int = 1,
        http_client: Any = None,  # lets tests pass an httpx client with a fake transport
    ):
        self._api_key = api_key
        self._model = model
        self.dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = http_client

    @property
    def model(self) -> str:
        return self._model or "unconfigured"

    def _http(self) -> Any:
        # Built lazily so the app starts fine without embedding settings.
        if self._client is None:
            if not self._api_key or not self._model:
                raise EmbeddingError("provider_error", "EMBEDDING_API_KEY and EMBEDDING_MODEL are not configured")
            self._client = httpx.Client(
                timeout=self._timeout,
                transport=httpx.HTTPTransport(retries=self._max_retries),  # retries connection errors only
            )
        return self._client

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + BATCH_SIZE]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        client = self._http()
        try:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": batch},
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingError("timeout") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError("provider_error", type(exc).__name__) from exc
        if response.status_code >= 400:
            # Only the status code: the response body is never copied into errors or logs.
            raise EmbeddingError("provider_error", f"HTTP {response.status_code}")

        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("invalid_response", "unexpected response shape") from exc
        if len(vectors) != len(batch):
            raise EmbeddingError("invalid_response", "wrong number of embeddings returned")
        for vector in vectors:
            if not isinstance(vector, list) or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector
            ):
                raise EmbeddingError("invalid_response", "embedding is not a list of numbers")
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    "invalid_response",
                    f"expected {self.dimension} dimensions, got {len(vector)} (check EMBEDDING_DIMENSION)",
                )
        return [[float(x) for x in vector] for vector in vectors]
