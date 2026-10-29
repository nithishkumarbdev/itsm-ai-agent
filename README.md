# ITSM AI Agent

AI automation for internal IT support (ITSM): ticket triage, RAG-grounded resolution,
deterministic rules and guardrails, human escalation, and a full audit trail.

**Status: Phase 10 (evaluation & reliability).** Tickets are stored in PostgreSQL, move through
a controlled lifecycle, every important change is audited, and `POST /tickets/{id}/process`
classifies, routes, retrieves, suggests a resolution, and checks guardrails. A small internal
tool layer exists (`app/tools/`) but nothing calls it yet. A new, separate evaluation framework
(`app/evaluation/`) now runs a representative dataset of tickets through the real pipeline and
reports measured accuracy, retrieval recall, guardrail safety and tool-policy correctness --
this is **reliability measurement**, not a new product feature; it changes nothing about how a
ticket is processed.
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
| `POST /tickets/{id}/process` | Start processing: classification, routing, retrieval, resolution suggestion, guardrail check |
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

## Guardrails

    classification + resolution text -> deterministic policy check -> guardrail_status (+ maybe block)

After a resolution is generated (or correctly not attempted, for lack of evidence),
`app/guardrails/policy.py` runs a deterministic check — no LLM, no database, same style as
routing — and records the outcome as `guardrail_status`:

| `guardrail_status` | Meaning |
|---|---|
| `passed` | A resolution was generated, grounded in cited evidence, and its text is safe |
| `requires_human_review` | The ticket's category is `security` — always, regardless of how good the resolution looks |
| `ungrounded` | No resolution was generated, or it cited no evidence at all |
| `blocked_restricted_content` | The resolution's own text recommended something unsafe, so it was discarded |

Precedence when more than one applies (highest first, the same fixed-table style as routing's
category → team mapping): **blocked > security > ungrounded > passed.** A security ticket with
a perfectly safe, well-grounded resolution still gets `requires_human_review`, and unsafe text
is blocked even on a security ticket.

**What "unsafe" means here.** A resolution is blocked if its text matches a deterministic,
auditable blocklist of phrases: recommending disabling MFA, a firewall, antivirus or other
security controls; sharing a password; deleting an account; granting admin access; or claiming
an action was already performed ("the issue has been fixed", "I have reset your password").
The resolution prompt (Phase 7) already forbids all of this — this is the backend's
independent re-check, the same defense-in-depth stance as validating source references against
what was actually retrieved. If the check trips, the resolution is discarded outright: nothing
unsafe is ever stored or shown, `suggested_resolution` stays null, and the ticket goes to
`awaiting_human`. Classification, routing and the retrieved evidence are kept.

**Prompt injection is recorded, not blocked.** Ticket text is scanned for common
instruction-override phrasing ("ignore all previous instructions", "you are now an
administrator", ...) and the result (`prompt_injection_suspected`) is written to the
`guardrails_checked` audit event. It does not by itself change how the ticket is processed:
ticket text is already treated as untrusted data everywhere it reaches an LLM (see the
classification and resolution prompts), so this is a visibility signal, not a second line of
defense.

**Guardrails never invent a policy decision beyond recording it.** They do not decide
auto-resolution eligibility, do not escalate a ticket's status for being `ungrounded` or
`requires_human_review`, and do not execute or undo anything — there is nothing to undo, since
no tools exist yet. That decision-making is for a later phase; guardrails only make the signal
available (on the ticket and in the audit trail) for it to use. Guardrails only run once
classification, routing, retrieval and resolution generation have all completed without
failing — an upstream failure already has its own explicit audit event and there is nothing
new to check.

## Controlled tools

`app/tools/` is a narrow, internal interface for operations on one ticket. A tool call has one
entry point, `execute_tool` (`app/tools/executor.py`), so every call is validated, policy-checked,
logged and audited; there is no other route to a tool's handler.

| Tool | Kind | What it does |
|---|---|---|
| `get_ticket` | read | A small view of the ticket (no requester email, no evidence list) |
| `search_knowledge_base` | read | Search active knowledge documents; returns scored, traceable chunks |
| `check_resolution_policy` | read | Re-run the guardrail check on the stored resolution and compare with the recorded verdict |
| `add_ticket_note` | write | Append a note (stored as an append-only `ticket_note_added` audit event) |
| `assign_ticket` | write | Reassign the ticket to another team. **Operators only** |

**The registry is the allow-list.** A name not in `TOOLS` cannot run, so there is no way to ask for
a shell, SQL, code execution, deletion, or a generic "update these fields" tool (deliberately not
built: it is exactly the unrestricted access the design avoids).

Every call goes through, in order: known tool? known actor? actor permitted for this tool? arguments
valid (strict, unknown fields rejected)? ticket-specific policy allows it? Then the handler runs in a
savepoint, so a tool that fails half way is undone, the failure is recorded, and only a short code
(never an error message) is stored. Each call leaves exactly one `tool_executed`, `tool_denied` or
`tool_failed` event, with the actor. Denials and failures record field names and codes, never
argument values; the knowledge search records only the query's length. State-changing tools lock the
ticket row like every other mutation, and like the rest of the app nothing here commits: the caller does.

Policies that do not depend on who asks or what they say: notes are refused on closed tickets;
`assign_ticket` needs a classified ticket in `processing`, `awaiting_human` or `escalated`, and **a
security ticket can never be moved away from Security** (the same precedence as routing).

Limits to know: `actor` (`agent` or `operator`) is a label asserted by trusted internal code and recorded
in the audit trail. There is no authentication, so it is not proof of identity, which is exactly why
tools are not an HTTP endpoint. Reprocessing a ticket re-runs routing and will overwrite a manual
reassignment. Tool results contain user-written and knowledge-base text, so if one is ever given to an
LLM it must go in a labelled data block, never into instructions.

## Evaluation

Unit tests answer "does this function behave correctly for this input?". Evaluation answers
"how well does the whole AI pipeline perform across a representative set of tickets?" -- both
are needed, and AI behavior can drift even when the application code around it does not change
at all (a provider updates a model; a prompt gets tweaked).

    python -m scripts.run_evaluation                      # deterministic mode, no API key needed
    python -m scripts.run_evaluation --mode real           # the actual configured LLM/embedding provider
    python -m scripts.run_evaluation --json-out report.json --save-baseline

Every run creates real rows in whatever `DATABASE_URL` points to -- there is no separate
evaluation database, the same as `scripts/seed_knowledge.py`. Point it at a database you do not
mind seeing ~40 throwaway tickets added to.

### The dataset

`evaluation/cases/v1.yaml`: 43 cases across the required case types (standard, ambiguous,
security, no-knowledge, prompt-injection, restricted-action, tool-action). Each case states
only the properties it has a real opinion about -- a category, a team, a document that must be
retrieved, a guardrail verdict -- never exact LLM wording, so the dataset stays meaningful as
model output varies. The five AMBIGUOUS cases assert nothing at all: there is no defensible
single ground truth for a genuinely vague ticket, so they exist to be observed in the JSON
report, not graded.

### Two modes, and what each one actually measures

**`deterministic`** (default, no API key): classification comes from a small keyword matcher
and resolutions from a fixed safe template (`app/evaluation/providers.py`), not a real LLM. It
is deliberately **not** scripted to return each case's expected answer -- that would measure
nothing, since a classifier told the right answer in advance can't be wrong. It classifies from
the ticket's own words, the same way the embedder's bag-of-words hashing has no idea which
document "should" match. This mode measures whether the **pipeline** wires together correctly
(routing, retrieval, guardrails, tools); it does not and cannot measure classification or
resolution **quality** -- the report says so explicitly, and `classification_accuracy` /
`retrieval_recall_at_k` are reported honestly low where the keyword matcher is genuinely weak
(see the dataset file's own notes on the security cases) rather than written around.

**`real`**: the actual configured `LLM_MODEL` / `EMBEDDING_MODEL`. This is the only mode that
measures classification and resolution quality against real model behavior.

### Metrics

Classification accuracy, routing accuracy, Retrieval Recall@K (does at least one acceptable
document appear in what was retrieved), guardrail detection rate, guardrail false-positive rate
(a safe case that was wrongly **blocked** -- not merely "ungrounded", which is a different,
non-safety problem), prompt-injection detection rate, tool safety rate, and end-to-end success.
**A metric with nothing eligible to measure it on is reported as "not measured", never as a
fabricated 0%** -- the same principle used throughout the app (an empty retrieval result is
`[]`, not an invented low score).

### Baselines and regressions

    python -m scripts.run_evaluation --save-baseline
    python -m scripts.run_evaluation --check-regression evaluation/baselines/v1-deterministic.json

A baseline is a saved `MetricsSummary` (`evaluation/baselines/`). `check_regression` compares a
new run against it with a per-metric tolerance: how many percentage points a metric may drop
before the run is considered a regression. Safety-relevant metrics (`routing_accuracy`,
`guardrail_detection_rate`, `guardrail_false_positive_rate`, `tool_safety_rate`) get **zero**
tolerance, since they are either deterministic or directly about what leaks past a safety
check; retrieval and classification get a few points of slack, since small wording or dataset
changes can move them slightly even with no real regression. The CLI exits 1 if the gate fails,
0 otherwise, so it can be wired into CI later without any extra plumbing.

### Traceability

Every case's `CaseResult` records what was observed (category, team, retrieved documents,
whether a resolution was generated, guardrail status, injection flag, tool outcome) and, on
failure, a normalized stage (`CLASSIFICATION_ERROR`, `ROUTING_ERROR`, `RETRIEVAL_ERROR`,
`GROUNDING_ERROR`, `RESOLUTION_ERROR`, `GUARDRAIL_ERROR`, `TOOL_ERROR`, `SYSTEM_ERROR`) with a
specific reason -- never full prompts or hidden reasoning. `--json-out` writes the full
structured result for every case, not just the summary.

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
