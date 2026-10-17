"""Split a knowledge document into chunks, ready for retrieval in a later phase.

Pure functions: no database, no LLM, no embeddings. Same input, same output.

Documents are written in sections ("## Symptoms", "## Troubleshooting Steps", ...), so
each section becomes one chunk, which keeps every chunk about one topic. A section longer
than `max_chars` is split at line boundaries (and at word boundaries if a single line is
too long). The section heading is repeated at the top of each of its chunks.
"""

import uuid
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1000
MIN_MAX_CHARS = 50


@dataclass(frozen=True)
class Chunk:
    document_id: uuid.UUID | str
    document_title: str
    category: str
    version: int
    chunk_index: int  # position within the document, starting at 0
    section: str  # heading of the section this chunk came from ("" before the first heading)
    content: str  # the heading (if any) followed by the text


def chunk_document(
    *,
    document_id: uuid.UUID | str,
    title: str,
    category: str,
    version: int,
    content: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Return the document's chunks in reading order.

    `max_chars` limits the body text of a chunk; the section heading is added on top of that.
    """
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}")

    chunks: list[Chunk] = []
    for heading, body in _sections(content):
        prefix = f"{heading}\n" if heading else ""
        for piece in _split(body, max_chars):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    document_title=title,
                    category=category,
                    version=version,
                    chunk_index=len(chunks),
                    section=heading,
                    content=prefix + piece,
                )
            )
    return chunks


def _sections(content: str) -> list[tuple[str, str]]:
    """Split on level-2 markdown headings into (heading, body). Empty sections are dropped."""
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    for line in content.replace("\r\n", "\n").split("\n"):
        if line.startswith("## "):
            sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines).strip()))
    return [(h, b) for h, b in sections if b]


def _split(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters, keeping the order and all the text."""
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            pieces.append(current.strip())
        current = ""

    for line in text.split("\n"):
        while len(line) > limit:  # a single very long line: cut at the last space that fits
            flush()
            cut = line.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            if line[:cut].strip():
                pieces.append(line[:cut].strip())
            line = line[cut:].lstrip()
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            flush()
            current = line
    flush()
    return pieces
