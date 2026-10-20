"""Embedding clients: the mock used by the tests, cosine similarity, and the real HTTP adapter
(driven by a fake transport, so no network and no API key)."""

import json
import math

import httpx
import pytest

from app.embeddings.base import EmbeddingError, embed_one
from app.embeddings.openai_compatible import BATCH_SIZE, OpenAICompatibleEmbeddings
from app.services.retrieval import MAX_QUERY_CHARS, build_query, cosine_similarity

# ---------- the mock embedder ----------


def test_same_text_always_gives_the_same_vector(embedder):
    text = "VPN keeps disconnecting after I connect"

    assert embed_one(embedder, text) == embed_one(embedder, text)
    assert embed_one(embedder, text) == embed_one(type(embedder)(), text)  # a fresh instance too


def test_vectors_have_the_configured_dimension_and_unit_length(embedder):
    vector = embed_one(embedder, "printer queue is stuck")

    assert len(vector) == embedder.dimension == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_dimension_is_configurable_and_part_of_the_model_name(embedder):
    small = type(embedder)(dimension=64)

    assert small.model == "mock-bow-64"
    assert len(embed_one(small, "vpn problem")) == 64
    assert small.model != embedder.model  # different sizes are different vector spaces


def test_different_texts_give_different_vectors(embedder):
    assert embed_one(embedder, "vpn disconnects") != embed_one(embedder, "printer jammed")


def test_texts_that_share_words_are_more_similar_than_unrelated_texts(embedder):
    vpn_a = embed_one(embedder, "the VPN keeps disconnecting every few minutes")
    vpn_b = embed_one(embedder, "VPN disconnects when I work from home")
    printer = embed_one(embedder, "the printer has a paper jam")

    assert cosine_similarity(vpn_a, vpn_b) > cosine_similarity(vpn_a, printer)


def test_batch_gives_the_same_vectors_as_single_calls_in_order(embedder):
    texts = ["vpn problem", "printer problem", "mfa prompt missing"]

    assert embedder.embed_many(texts) == [embed_one(embedder, t) for t in texts]


def test_text_with_no_meaningful_words_gives_a_zero_vector_that_matches_nothing(embedder):
    empty = embed_one(embedder, "I need the help of a")  # only filler words

    assert all(x == 0.0 for x in empty)
    assert cosine_similarity(empty, embed_one(embedder, "vpn problem")) == 0.0


def test_the_mock_can_be_made_to_fail(embedder):
    embedder.fails_with(EmbeddingError("timeout"))

    with pytest.raises(EmbeddingError) as excinfo:
        embed_one(embedder, "anything")

    assert excinfo.value.kind == "timeout"


# ---------- cosine similarity ----------


def test_cosine_of_identical_orthogonal_and_opposite_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_ignores_vector_length():
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_cosine_with_a_zero_vector_is_zero_not_an_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_never_exceeds_one_because_of_rounding():
    vector = [0.1, 0.2, 0.3, 0.7, 0.11]
    assert cosine_similarity(vector, vector) <= 1.0


def test_cosine_of_vectors_with_different_sizes_is_an_error_not_a_silent_number():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------- what is searched for ----------


def test_query_is_the_title_and_description_only():
    assert build_query("VPN down", "I cannot connect.") == "VPN down\nI cannot connect."


def test_a_very_long_query_is_cut_to_a_safe_length():
    assert len(build_query("t", "x" * 10_000)) == MAX_QUERY_CHARS


# ---------- the real provider adapter (fake HTTP transport) ----------

BASE_URL = "https://embeddings.example.test/v1/"


def make(handler, **overrides):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = {"api_key": "test-key", "model": "test-model", "dimension": 3, "base_url": BASE_URL, "timeout": 5}
    settings.update(overrides)
    return OpenAICompatibleEmbeddings(http_client=client, **settings)


def echo_handler(request):
    """One 3-number vector per input, returned in REVERSE order to prove sorting by index."""
    inputs = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(i), 0.0, 1.0]} for i in range(len(inputs))]
    return httpx.Response(200, json={"data": list(reversed(data))})


def test_request_is_built_as_the_provider_expects_and_results_are_in_input_order():
    seen = []

    def handler(request):
        seen.append(request)
        return echo_handler(request)

    vectors = make(handler).embed_many(["first", "second"])

    assert vectors == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    request = seen[0]
    assert str(request.url) == "https://embeddings.example.test/v1/embeddings"  # trailing slash handled
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {"model": "test-model", "input": ["first", "second"]}


def test_many_texts_are_sent_in_batches():
    sizes = []

    def handler(request):
        sizes.append(len(json.loads(request.content)["input"]))
        return echo_handler(request)

    vectors = make(handler).embed_many([f"text {i}" for i in range(BATCH_SIZE * 2 + 2)])

    assert sizes == [BATCH_SIZE, BATCH_SIZE, 2]
    assert len(vectors) == BATCH_SIZE * 2 + 2


def test_no_texts_means_no_request():
    calls = []

    assert make(lambda request: calls.append(request)).embed_many([]) == []
    assert calls == []


def test_a_vector_of_the_wrong_size_is_rejected_with_a_helpful_message():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    with pytest.raises(EmbeddingError) as excinfo:
        make(handler).embed_many(["x"])

    assert excinfo.value.kind == "invalid_response"
    assert "expected 3 dimensions, got 2" in excinfo.value.detail
    assert "EMBEDDING_DIMENSION" in excinfo.value.detail


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},  # wrong number of embeddings
        {"data": [{"index": 0, "embedding": ["a", "b", "c"]}]},  # not numbers
        {"data": [{"index": 0, "embedding": [True, False, True]}]},  # booleans are not numbers
        {"data": [{"index": 0}]},  # no embedding
        {"unexpected": "shape"},
        "not an object",
    ],
    ids=["wrong-count", "not-numbers", "booleans", "no-embedding", "wrong-shape", "not-an-object"],
)
def test_malformed_responses_are_rejected(body):
    with pytest.raises(EmbeddingError) as excinfo:
        make(lambda request: httpx.Response(200, json=body)).embed_many(["x"])

    assert excinfo.value.kind == "invalid_response"


def test_a_response_that_is_not_json_is_rejected():
    with pytest.raises(EmbeddingError) as excinfo:
        make(lambda request: httpx.Response(200, text="<html>oops</html>")).embed_many(["x"])

    assert excinfo.value.kind == "invalid_response"


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_http_errors_report_only_the_status_never_the_body_or_the_key(status):
    def handler(request):
        return httpx.Response(status, text="secret body mentioning test-key")

    with pytest.raises(EmbeddingError) as excinfo:
        make(handler).embed_many(["x"])

    assert excinfo.value.kind == "provider_error"
    assert excinfo.value.detail == f"HTTP {status}"
    assert "secret body" not in str(excinfo.value) and "test-key" not in str(excinfo.value)


def test_a_timeout_is_reported_as_a_timeout():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(EmbeddingError) as excinfo:
        make(handler).embed_many(["x"])

    assert excinfo.value.kind == "timeout"


def test_a_connection_failure_is_a_provider_error_with_only_the_class_name():
    def handler(request):
        raise httpx.ConnectError("connection refused to embeddings.example.test", request=request)

    with pytest.raises(EmbeddingError) as excinfo:
        make(handler).embed_many(["x"])

    assert excinfo.value.kind == "provider_error"
    assert excinfo.value.detail == "ConnectError"


@pytest.mark.parametrize(("api_key", "model"), [(None, "m"), ("k", None), (None, None), ("", "")])
def test_missing_configuration_fails_safely_without_any_network_call(api_key, model):
    client = OpenAICompatibleEmbeddings(api_key=api_key, model=model, dimension=3, base_url=BASE_URL, timeout=5)

    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_many(["x"])

    assert excinfo.value.kind == "provider_error"
    assert "not configured" in excinfo.value.detail


def test_the_model_and_dimension_are_exposed_and_the_key_is_not():
    client = make(echo_handler)

    assert client.model == "test-model"
    assert client.dimension == 3
    assert "test-key" not in client.model
    assert OpenAICompatibleEmbeddings(api_key="k", model=None, dimension=3, base_url=BASE_URL, timeout=5).model == "unconfigured"
