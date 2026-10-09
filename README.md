# ITSM AI Agent

AI automation for internal IT support (ITSM): ticket triage, RAG-grounded resolution,
deterministic rules and guardrails, human escalation, and a full audit trail.

**Status: Phase 2 (ticket lifecycle + audit trail).** Tickets are stored in PostgreSQL, move
through a controlled lifecycle, and every important change is recorded.
There is no AI processing yet. This README will be expanded as phases are completed.

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
| `POST /tickets/{id}/process` | Start processing: moves the ticket to `processing` (no AI yet) |
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

## Tests

The tests use a real PostgreSQL database. Export its URL (pytest does not read `.env`):

    export TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/itsm_test
    pytest
