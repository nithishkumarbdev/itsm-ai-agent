"""Chunking on its own: pure functions, no database, no LLM."""

import dataclasses

import pytest

from app.services.chunking import MIN_MAX_CHARS, Chunk, chunk_document

DOC = {"document_id": "doc-1", "title": "VPN Troubleshooting Guide", "category": "vpn", "version": 2}

SAMPLE = """## Symptoms
- The VPN keeps disconnecting.

## Troubleshooting Steps
1. Reconnect.
2. Restart the client.

## When to Escalate
Escalate to the Network team."""


def chunks(content, **kwargs):
    return chunk_document(content=content, **DOC, **kwargs)


# ---------- sections become chunks ----------


def test_each_section_becomes_one_chunk_in_reading_order():
    result = chunks(SAMPLE)

    assert [c.section for c in result] == ["Symptoms", "Troubleshooting Steps", "When to Escalate"]
    assert [c.chunk_index for c in result] == [0, 1, 2]


def test_chunk_content_is_the_heading_followed_by_the_section_text():
    result = chunks(SAMPLE)

    assert result[0].content == "Symptoms\n- The VPN keeps disconnecting."
    assert result[1].content == "Troubleshooting Steps\n1. Reconnect.\n2. Restart the client."


def test_every_chunk_keeps_the_document_metadata():
    for chunk in chunks(SAMPLE):
        assert chunk.document_id == "doc-1"
        assert chunk.document_title == "VPN Troubleshooting Guide"
        assert chunk.category == "vpn"
        assert chunk.version == 2


def test_output_is_deterministic():
    assert chunks(SAMPLE) == chunks(SAMPLE)


def test_windows_line_endings_give_the_same_chunks():
    assert chunks(SAMPLE.replace("\n", "\r\n")) == chunks(SAMPLE)


def test_chunks_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunks(SAMPLE)[0].content = "changed"


# ---------- short and odd documents ----------


def test_short_document_without_headings_is_one_chunk():
    result = chunks("Just a short note about the printer.")

    assert len(result) == 1
    assert result[0] == Chunk("doc-1", "VPN Troubleshooting Guide", "vpn", 2, 0, "", "Just a short note about the printer.")


def test_text_before_the_first_heading_is_kept_as_its_own_chunk():
    result = chunks("Intro text.\n\n## Symptoms\nBody text.")

    assert [(c.section, c.content) for c in result] == [("", "Intro text."), ("Symptoms", "Symptoms\nBody text.")]


@pytest.mark.parametrize("content", ["", "   ", "\n\n  \n"], ids=["empty", "spaces", "blank-lines"])
def test_document_without_text_has_no_chunks(content):
    assert chunks(content) == []


def test_heading_without_text_is_dropped():
    result = chunks("## Empty Section\n\n## Real Section\nSome text.")

    assert [c.section for c in result] == ["Real Section"]
    assert result[0].chunk_index == 0


# ---------- long documents ----------


def test_long_section_is_split_without_losing_or_reordering_lines():
    lines = [f"{i}. Step number {i} of the procedure" for i in range(1, 80)]
    result = chunks("## Troubleshooting Steps\n" + "\n".join(lines), max_chars=200)

    prefix = "Troubleshooting Steps\n"
    assert len(result) > 1
    assert all(c.content.startswith(prefix) for c in result)  # heading repeated on every chunk
    assert all(len(c.content) - len(prefix) <= 200 for c in result)
    assert [c.chunk_index for c in result] == list(range(len(result)))
    rebuilt = "\n".join(c.content[len(prefix):] for c in result)
    assert rebuilt.split("\n") == lines


def test_a_very_long_line_is_split_at_word_boundaries():
    words = [f"word{i}" for i in range(300)]
    result = chunks(" ".join(words), max_chars=100)

    bodies = [c.content for c in result]
    assert len(result) > 1
    assert all(len(b) <= 100 for b in bodies)
    assert " ".join(bodies) == " ".join(words)


def test_a_single_word_longer_than_the_limit_is_cut_exactly():
    result = chunks("x" * 250, max_chars=100)

    assert [len(c.content) for c in result] == [100, 100, 50]


def test_text_exactly_at_the_limit_stays_in_one_chunk():
    assert len(chunks("a" * 100, max_chars=100)) == 1
    assert len(chunks("a" * 101, max_chars=100)) == 2


def test_no_chunk_is_empty_or_whitespace():
    content = "## A\n" + "\n\n\n".join(f"paragraph {i}" for i in range(50)) + "\n\n## B\n   \n"

    for chunk in chunks(content, max_chars=80):
        assert chunk.content.strip() == chunk.content
        assert chunk.content


def test_max_chars_below_the_minimum_is_rejected():
    with pytest.raises(ValueError):
        chunks(SAMPLE, max_chars=MIN_MAX_CHARS - 1)
