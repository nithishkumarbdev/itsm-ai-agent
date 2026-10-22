# ITSM AI Agent

AI automation for internal IT support (ITSM): ticket triage, RAG-grounded resolution,
deterministic rules and guardrails, human escalation, and a full audit trail.

**Status: Phase 7 (grounded resolution generation).** Tickets are stored in PostgreSQL, move
through a controlled lifecycle, every important change is audited, and `POST /tickets/{id}/process`
classifies the ticket with an LLM, routes it to a team with deterministic backend rules,
retrieves the most relevant internal knowledge, and generates a suggested resolution grounded
in that evidence.
**A resolution is a suggestion for a person to review, never an executed action: the ticket is
never marked resolved by this endpoint, and no action is ever performed.** Auto-resolution
policy and guardrails start in Phase 8.
This README will be expanded as phases are completed.

## Setup

Requires Python 3.11+ and a running PostgreSQL server.

Create two databases: one to run the app, and a separate one for the tests.
The test database name must end in `_test` because the tests drop and recreate its tables.

    createdb itsm
    createdb itsm_test

Install dependencies and configure:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt
    cp .env.example .env      # then edit DATABASE_URL

## Run

    uvicorn app.main:app --reload

The `tickets` and `audit_events` tables are created on startup if missing. API docs: http://localhost:8000/docs

    curl -X POST http://localhost:8000/tickets \
      -H "Content-Type: application/json" \
      -d '{"source":"web","requester":"user@example.com","title":"VPN not connecting","description":"I cannot connect to the company VPN from home."}'

## API

| Endpoint | What it does |
|---|---|
| `GET /health` | App and database health |
| `POST /tickets` | Create a ticket (also records `ticket_created`) |
| `GET /tickets` | List tickets, newest first (`limit`, `offset`) |
| `GET /tickets/{id}` | One ticket, 404 if unknown |
| `POST /tickets/{id}/status` | Change status, only if the lifecycle allows it (409 otherwise) |
| `POST /tickets/{id}/process` | Start processing: `processing`, LLM classification, rule-based routing, retrieval, resolution suggestion |
| `GET /tickets/{id}/events` | The ticket's audit history, oldest first |

## Ticket lifecycle

The allowed moves are defined in one table in `app/lifecycle.py`:

    new            -> processing
    processing     -> awaiting_human, resolved, escalated
    awaiting_human -> processing, resolved, escalated
    escalated      -> processing, closed
    resolved       -> closed
    closed         -> (nothing, terminal)

Every successful move writes a `status_changed` audit event in the same transaction as the
status update. Audit events are append-only: there are no endpoints to edit or delete them.

## Classification

`/process` moves the ticket to `processing`, then asks the LLM for a structured classification:
`category`, `priority`, `urgency`, `impact`, `confidence`, `reason`. The reply is validated
against a strict schema (`app/schemas/classification.py`); only a valid result is saved.
The LLM only proposes; the backend decides what is stored.

If the provider fails or its output is invalid, nothing is invented: a `classification_failed`
event is recorded and the ticket moves to `awaiting_human` (call `/process` again to retry).
The response is still 200 with the ticket; check `status` and the ticket's events.

`confidence` is the model's own stated confidence. It is a signal, not a measured probability.

Configure the LLM in `.env` (`LLM_MODEL`, `LLM_API_KEY`, see `.env.example`). To check your
real configuration (calls the real API, not part of the tests):

    python -m scripts.check_llm

## Routing

The LLM understands the ticket; the backend decides where it goes. The LLM never chooses or
sets a team (if it returns one, it is ignored). After the classification is validated,
`app/services/routing.py` maps the category to a team with a fixed table:

| Category | Team |
|---|---|
| security | Security |
| vpn, network | Network |
| hardware | Hardware / IT Support |
| access, account | Access Management |
| email, software, other | IT Support |

Routing is a pure function of the category: same input, same team, no LLM, no database.
Nothing else (priority, confidence, ticket text) can change it, so security tickets always go to
Security. A missing or invalid category gets the fallback team (IT Support), never a guess.
If classification fails, no team is assigned at all.

Every routing decision is stored in a `routing_decided` audit event with the rule and reason
(`GET /tickets/{id}/events`), so "why did it go to Network?" is always answerable. Classification,
team and both audit events are saved in one transaction.

Not in Phase 4 on purpose: no priority overrides, and no confidence threshold or review flag.
Confidence is stored and audited so later policy can use it.

## Knowledge base

The knowledge base is the company's trusted internal IT documentation (troubleshooting guides
and procedures). Later phases will retrieve from it so the model answers from company knowledge
instead of inventing procedures. Right now it is only stored and managed; ticket processing
does not use it.

A document has: `title` (unique), `category` (the same values as ticket classification),
`content`, `version`, `is_active`, `metadata` (free-form JSON) and timestamps.

| Endpoint | What it does |
|---|---|
| `POST /knowledge` | Add a document (409 if the title exists) |
| `GET /knowledge` | List documents without content; filter with `category`, `is_active`; `limit`, `offset` |
| `GET /knowledge/{id}` | One document with its content (inactive ones too) |
| `PATCH /knowledge/{id}` | Change fields, or activate/deactivate with `{"is_active": false}` |
| `POST /knowledge/{id}/index` | Chunk and embed one document (safe to repeat) |
| `POST /knowledge/search` | Find the most relevant chunks for a text, with scores and sources |

Rules:
- **Version.** It starts at 1 and goes up by 1 when the title, category, content or metadata
  changes. Activating/deactivating alone does not change it, and clients cannot set it on update.
- **No deletion.** Documents are deactivated, never deleted. Only active documents are meant for
  retrieval.
- **Not audited yet.** The audit trail is per ticket (`audit_events.ticket_id` is required), so
  knowledge changes are not recorded there. `version` and `updated_at` are the only trace for now.

Seed documents live in `knowledge/documents/*.md`: 11 synthetic guides (VPN, password reset,
Wi-Fi, email sync, Teams, printers, laptop performance, MFA, account lockout, software
installation, suspected account compromise). Each has the sections Symptoms, Common Causes,
Troubleshooting Steps, Restricted Actions and When to Escalate. "Restricted Actions" lists what
must not be done (for example resetting passwords without identity verification), because the
knowledge base is information, not permission to act. They contain no URLs, credentials or commands.

Load them with (safe to repeat: existing titles are skipped and never overwritten; it does not
run when the app starts):

    python -m scripts.seed_knowledge

`app/services/chunking.py` splits a document into one chunk per `##` section (long sections are
split further), keeping the document id, title, category and version on every chunk. It is pure
Python with no embeddings; Phase 6 will use it and can tune the chunk size once an embedding model
is chosen.

## Retrieval (RAG, retrieval half)

    ticket -> query (title + description) -> embedding -> cosine similarity -> top-k evidence

1. **Chunking** (`app/services/chunking.py`): one chunk per `##` section of a document. No overlap:
   each section is a complete topic, so overlap would only produce near-duplicate hits.
2. **Embedding** (`app/embeddings/`): text to vectors through a small interface. The real provider is
   any OpenAI-style `/embeddings` endpoint over `httpx` (Anthropic has no embeddings API). The title is
   embedded together with each chunk so a chunk like "Symptoms" still carries its topic.
3. **Storage**: `knowledge_chunks` holds the text and the vector as a PostgreSQL float array, plus the
   document version and embedding model it was built from.
4. **Search** (`app/services/retrieval.py`): exact cosine similarity against every eligible chunk.
   Eligible = the document is active **and** the chunk was built from the document's current version.
   Only chunks with similarity >= `RETRIEVAL_MIN_SCORE` are returned, best `RETRIEVAL_TOP_K` first.
   **If nothing is relevant enough the result is an empty list**; the closest weak match is never
   returned to fill the space. Each result carries document id, title, category, version, chunk id,
   chunk index, section, text and similarity score.
5. **Category**: the ticket's category adds a small ranking bonus to chunks of that category. It only
   affects order: the threshold and the reported score use the raw similarity, and other categories can
   still be found.

Why not pgvector? It needs an OS package and a superuser `CREATE EXTENSION`, which I could not check for
your environment, and exact search over a few hundred chunks is instant. Search is one function, so
moving to pgvector later (worth it at many thousands of chunks) is a contained change.

**Staleness and models.** Editing a document raises its version, and its old chunks are ignored until you
re-index (`POST /knowledge/{id}/index`); old and new chunks are never returned together. Deactivating hides
a document from search immediately and reactivating brings it back without re-indexing. Vectors from
different embedding models cannot be compared, so if the index was built with another model, search fails
with a clear error (503) and ticket processing records `retrieval_failed` instead of returning nonsense.

**In ticket processing** the evidence is stored on the ticket as references only (`retrieved_documents`:
ids, titles, versions and scores, not the text) and audited as `knowledge_retrieved`. `[]` means "searched,
nothing relevant"; `null` means "not searched". If retrieval fails, the valid classification and routing
are kept, a `retrieval_failed` event is written, and the ticket goes to `awaiting_human`. Retrieval never
changes the routing. Note that the ticket text is sent to the embedding provider, as it is to the LLM.

### Real embeddings

Set `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` and `EMBEDDING_DIMENSION` in `.env` (see `.env.example`), then:

    python -m scripts.seed_knowledge
    python -m scripts.index_knowledge

Indexing is explicit and never runs at startup, so the app starts without an embedding provider. Then try
`POST /knowledge/search` at `/docs`. `RETRIEVAL_MIN_SCORE=0.30` is only a starting point for
`text-embedding-3-small`; it has not been validated on real data.

**Tests never need a key.** They use a deterministic bag-of-words mock embedder (`tests/conftest.py`)
with its own threshold (0.20). It matches on shared words, not meaning, so it proves the retrieval
mechanics (ranking, threshold, staleness, wiring), not the quality of real embeddings.

## Resolution generation

    ticket + classification + retrieved evidence -> LLM -> structured output -> validation -> ticket

After retrieval finds relevant knowledge, `app/services/resolution.py` asks the LLM for a
resolution suggestion grounded in that evidence, reusing the same configured LLM client as
classification (one provider, two separate prompts). **If retrieval found nothing relevant,
resolution generation is not attempted at all** — no evidence means no invented answer, and
that is not treated as a failure.

The prompt sends the ticket, the backend's validated classification, and the retrieved
chunks as three separate, escaped, labelled blocks (`<ticket>`, `<classification>`,
`<evidence>`). The ticket is untrusted data; so, in the same way, is any imperative-sounding
text inside a retrieved document — neither can redefine the model's task. The model must
always write in suggestion language ("try", "check"), never claim the issue is fixed or that
an action was performed, and never recommend disabling security controls.

**Source references are constrained, then re-checked.** The JSON schema sent to the model
limits `chunk_id` to an enum of exactly the chunk ids that were retrieved for this ticket, so
a compliant model cannot even express a fabricated citation. The backend re-checks this
anyway (defense in depth, in case a provider ignores the schema): any citation to a chunk that
was not actually retrieved makes the whole resolution invalid, exactly like a bad category
would fail classification.

**What is stored.** `suggested_resolution` and `resolution_confidence` (the model's own stated
confidence — a signal, not a validated probability, and not acted on yet) go straight onto the
ticket. `resolution_sources` holds only the chunks the model actually cited (ids and titles, a
subset of `retrieved_documents`, never the chunk text) for traceability. The `resolution_generated`
audit event records the source count, confidence and the model's one-sentence `reason`. The
model may legitimately cite nothing and say the evidence was insufficient — that is a
successful, valid result (low confidence, empty `resolution_sources`), not a failure.

**The ticket's status never changes because of a resolution.** It stays in `processing`,
whether or not a resolution was generated: `suggested_resolution` is a draft for a person,
not proof the issue is fixed, and no action is ever executed. If generation fails (bad
provider output, a fabricated citation, a timeout), the previously valid classification,
routing and retrieved evidence are kept, a `resolution_failed` event is written, and the
ticket moves to `awaiting_human` for retry.

Configuration is unchanged: resolution reuses `LLM_MODEL` / `LLM_API_KEY` from `.env`. Tests
use the same mock LLM as classification, extended with its own resolution schema/response, so
no key is needed. The mock's default resolution cites the first piece of retrieved evidence.

## Upgrading an existing database

Phase 3 added one column, `tickets.classification_reason`. Phases 4 and 5 add none: Phase 5 only adds a new table, `knowledge_documents`, which `create_all` creates by itself. `create_all` does not alter existing
tables, so on a database created in an earlier phase run this once (or drop and recreate the
dev database):

    psql itsm -c "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS classification_reason TEXT;"

This is the point where Alembic migrations start to make sense; we will add them before deploying.

## Tests

The tests use a real PostgreSQL database and a mock LLM (no API key needed, no network calls).
Export the test database URL (pytest does not read `.env`):

    export TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/itsm_test
    pytest
