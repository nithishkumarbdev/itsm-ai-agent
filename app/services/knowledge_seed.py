"""Read the seed knowledge documents from knowledge/documents/*.md.

Each file starts with a small header, then the document text:

    ---
    title: VPN Troubleshooting Guide
    category: vpn
    version: 1
    document_type: troubleshooting
    ---

    ## Symptoms
    ...

`title` and `category` are required. `version` defaults to 1. Any other header line
(like document_type) is stored in the document's metadata.
"""

from pathlib import Path

from app.schemas.knowledge import KnowledgeDocumentCreate

# knowledge/ sits at the repository root, next to app/
SEED_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "documents"

DEFAULT_METADATA = {
    "department": "IT Support",
    "source": "internal_kb",
    "audience": "employees",
}


def parse_seed_file(text: str) -> dict:
    """Split a seed file into its fields. Raises ValueError if the file is malformed."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines[0].strip() != "---":
        raise ValueError("file must start with a '---' header")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("header is not closed with '---'") from None

    header: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"bad header line: {line!r}")
        header[key.strip()] = value.strip()

    for required in ("title", "category"):
        if not header.get(required):
            raise ValueError(f"header is missing '{required}'")

    title = header.pop("title")
    category = header.pop("category")
    version = int(header.pop("version", "1"))  # ValueError if it is not a number
    return {
        "title": title,
        "category": category,
        "version": version,
        "content": "\n".join(lines[end + 1 :]).strip(),
        "metadata": {**DEFAULT_METADATA, **header},
    }


def load_seed_documents(directory: Path = SEED_DIR) -> list[KnowledgeDocumentCreate]:
    """Load and validate every seed document, in file-name order (so the result is deterministic)."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ValueError(f"no seed documents found in {directory}")
    documents = []
    for path in paths:
        try:
            documents.append(KnowledgeDocumentCreate(**parse_seed_file(path.read_text(encoding="utf-8"))))
        except ValueError as exc:  # includes pydantic's ValidationError
            raise ValueError(f"{path.name}: {exc}") from exc
    return documents
