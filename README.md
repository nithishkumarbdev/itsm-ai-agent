# ITSM AI Agent

AI automation for internal IT support (ITSM): ticket triage, RAG-grounded resolution,
deterministic rules and guardrails, human escalation, and a full audit trail.

**Status: Phase 5 (knowledge base foundation).** Tickets are stored in PostgreSQL, move through a
controlled lifecycle, every important change is audited, and `POST /tickets/{id}/process`
classifies the ticket with an LLM and routes it to a team with deterministic backend rules.
A separate internal IT knowledge base is stored and managed through `/knowledge`. It is not
connected to ticket processing yet: retrieval (RAG) and resolution start in Phase 6.
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
| `POST /tickets/{id}/process` | Start processing: `processing`, then LLM classification, then rule-based team routing |
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
