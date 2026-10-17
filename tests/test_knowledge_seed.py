"""The seed documents (knowledge/documents/*.md): parsing, content quality, and seeding."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.main import app
from app.models import KnowledgeDocument
from app.schemas.classification import Category
from app.services.chunking import DEFAULT_MAX_CHARS, chunk_document
from app.services.knowledge import seed_documents
from app.services.knowledge_seed import DEFAULT_METADATA, load_seed_documents, parse_seed_file

# Hand-written: the documents the knowledge base must start with.
EXPECTED_DOCUMENTS = {
    "VPN Troubleshooting Guide": "vpn",
    "Password Reset Guide": "account",
    "Wi-Fi Troubleshooting Guide": "network",
    "Email Sync Troubleshooting Guide": "email",
    "Microsoft Teams Troubleshooting Guide": "software",
    "Printer Troubleshooting Guide": "hardware",
    "Laptop Performance Troubleshooting Guide": "hardware",
    "MFA Troubleshooting Guide": "account",
    "Account Lockout Guide": "account",
    "Software Installation Guide": "software",
    "Suspected Account Compromise Guide": "security",
}
SECTIONS = ["Symptoms", "Common Causes", "Troubleshooting Steps", "Restricted Actions", "When to Escalate"]

VALID_FILE = """---
title: Example Guide
category: vpn
version: 3
document_type: procedure
---

## Symptoms
- Something is wrong.
"""


def count_documents(db):
    return db.scalar(select(func.count()).select_from(KnowledgeDocument))


# ---------- parsing a seed file ----------


def test_parse_reads_header_and_body():
    parsed = parse_seed_file(VALID_FILE)

    assert parsed["title"] == "Example Guide"
    assert parsed["category"] == "vpn"
    assert parsed["version"] == 3
    assert parsed["content"] == "## Symptoms\n- Something is wrong."


def test_other_header_lines_become_metadata_on_top_of_the_defaults():
    parsed = parse_seed_file(VALID_FILE)

    assert parsed["metadata"] == {**DEFAULT_METADATA, "document_type": "procedure"}


def test_version_defaults_to_1():
    assert parse_seed_file(VALID_FILE.replace("version: 3\n", ""))["version"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "title: X\ncategory: vpn\n\nBody",  # no --- header
        "---\ntitle: X\ncategory: vpn\n\nBody",  # header never closed
        "---\ncategory: vpn\n---\nBody",  # no title
        "---\ntitle: X\n---\nBody",  # no category
        "---\ntitle: X\ncategory: vpn\nnot a header line\n---\nBody",
        "---\ntitle: X\ncategory: vpn\nversion: two\n---\nBody",
    ],
    ids=["no-header", "unclosed-header", "no-title", "no-category", "bad-line", "bad-version"],
)
def test_malformed_seed_file_is_rejected(text):
    with pytest.raises(ValueError):
        parse_seed_file(text)


def test_loading_reports_which_file_is_invalid(tmp_path):
    (tmp_path / "bad.md").write_text(VALID_FILE.replace("category: vpn", "category: banana"))

    with pytest.raises(ValueError, match="bad.md"):
        load_seed_documents(tmp_path)


def test_loading_an_empty_directory_is_an_error_not_a_silent_no_op(tmp_path):
    with pytest.raises(ValueError, match="no seed documents"):
        load_seed_documents(tmp_path)


# ---------- the seed documents themselves ----------


def test_expected_documents_exist_with_the_expected_categories():
    documents = load_seed_documents()

    assert {d.title: d.category.value for d in documents} == EXPECTED_DOCUMENTS
    assert len({d.title for d in documents}) == len(documents)  # titles are unique


def test_all_seed_categories_are_valid_classification_categories():
    valid = {c.value for c in Category}
    assert all(d.category.value in valid for d in load_seed_documents())


def test_loading_is_deterministic():
    assert load_seed_documents() == load_seed_documents()


def test_seed_documents_start_at_version_1_active_with_metadata():
    for d in load_seed_documents():
        assert d.version == 1
        assert d.is_active is True
        assert d.metadata["source"] == "internal_kb"
        assert d.metadata["department"] == "IT Support"
        assert d.metadata["audience"] == "employees"
        assert d.metadata["document_type"] in {"troubleshooting", "procedure", "incident"}


@pytest.mark.parametrize("document", load_seed_documents(), ids=lambda d: d.title)
def test_seed_document_is_structured_and_meaningful(document):
    headings = re.findall(r"^## (.+)$", document.content, flags=re.MULTILINE)

    assert headings == SECTIONS
    assert len(document.content) > 800  # real guidance, not a stub
    assert re.search(r"^\d+\. ", document.content, flags=re.MULTILINE)  # numbered steps


@pytest.mark.parametrize("document", load_seed_documents(), ids=lambda d: d.title)
def test_seed_document_contains_no_urls_emails_or_secrets_and_no_commands(document):
    text = document.content

    assert not re.search(r"https?://|www\.|\S+@\S+", text)  # no URLs or email addresses
    assert not re.search(r"password\s*[:=]", text, flags=re.IGNORECASE)  # no credentials
    for command in ["sudo", "rm -", "chmod", "powershell", "net user", "regedit"]:
        assert command not in text.lower()


def test_every_document_states_what_must_not_be_done():
    for d in load_seed_documents():
        section = d.content.split("## Restricted Actions")[1].split("## When to Escalate")[0]
        assert section.count("- ") >= 2, d.title


def test_security_guide_always_escalates_and_is_never_resolved_at_first_level():
    guide = next(d for d in load_seed_documents() if d.category.value == "security")

    assert "Always escalate to the Security team" in guide.content
    assert "must not close or resolve" in guide.content


@pytest.mark.parametrize("document", load_seed_documents(), ids=lambda d: d.title)
def test_every_seed_document_chunks_into_its_five_sections(document):
    chunks = chunk_document(
        document_id="x", title=document.title, category=document.category.value,
        version=document.version, content=document.content,
    )

    # All five sections are covered, in order. (A long section may span several chunks.)
    assert list(dict.fromkeys(c.section for c in chunks)) == SECTIONS
    assert all(c.content and len(c.content) <= DEFAULT_MAX_CHARS + len(c.section) + 1 for c in chunks)


# ---------- seeding the database ----------


def test_seeding_inserts_every_document(db):
    result = seed_documents(db, load_seed_documents())
    db.commit()

    assert len(result.inserted) == len(EXPECTED_DOCUMENTS)
    assert result.skipped == []
    assert count_documents(db) == len(EXPECTED_DOCUMENTS)


def test_seeding_twice_creates_no_duplicates(db):
    seed_documents(db, load_seed_documents())
    db.commit()

    second = seed_documents(db, load_seed_documents())
    db.commit()

    assert second.inserted == []
    assert len(second.skipped) == len(EXPECTED_DOCUMENTS)
    assert count_documents(db) == len(EXPECTED_DOCUMENTS)


def test_seeded_rows_are_stored_correctly(db):
    seed_documents(db, load_seed_documents())
    db.commit()

    stored = db.scalars(select(KnowledgeDocument)).all()
    assert {d.title: d.category for d in stored} == EXPECTED_DOCUMENTS
    assert all(d.version == 1 and d.is_active for d in stored)
    assert all(d.doc_metadata["source"] == "internal_kb" for d in stored)


def test_reseeding_never_overwrites_edited_documents(client, db):
    seed_documents(db, load_seed_documents())
    db.commit()
    guide = client.get("/knowledge", params={"category": "vpn"}).json()[0]
    edited = client.patch(f"/knowledge/{guide['id']}", json={"content": "## Symptoms\n- Edited by the IT team."}).json()

    seed_documents(db, load_seed_documents())
    db.commit()

    after = client.get(f"/knowledge/{guide['id']}").json()
    assert after["content"] == "## Symptoms\n- Edited by the IT team."
    assert after["version"] == edited["version"] == 2


def test_duplicate_titles_inside_one_batch_are_inserted_once(db):
    document = load_seed_documents()[0]

    result = seed_documents(db, [document, document])
    db.commit()

    assert (len(result.inserted), len(result.skipped)) == (1, 1)
    assert count_documents(db) == 1


def test_seeded_documents_are_available_through_the_api(client, db):
    seed_documents(db, load_seed_documents())
    db.commit()

    documents = client.get("/knowledge").json()

    assert [d["title"] for d in documents] == sorted(EXPECTED_DOCUMENTS)
    assert all("content" not in d for d in documents)


def test_starting_the_app_does_not_seed_the_database(db):
    with TestClient(app) as started_client:  # `with` runs the startup hook
        assert started_client.get("/knowledge").json() == []
    assert count_documents(db) == 0
