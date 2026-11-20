# RAG-Based Personalized Interview Preparation System

An agentic AI platform that prepares students and job seekers for technical interviews. It reads your resume and
your own study material, plans an interview, asks adaptive questions, grades your answers with a structured rubric,
remembers your weak topics across sessions, and lets you stay in control through human-in-the-loop checkpoints.

## 1. Problem statement
Interview prep is generic: question lists ignore your resume, your gaps and your level, and nobody tells you *why* an
answer scored poorly. This project personalises every step — questions come from your resume, job description and
study notes (RAG), difficulty adapts to your scores, and long-term memory steers future sessions to what you
repeatedly get wrong.

## 2. Features
| Area | What you get |
|---|---|
| Accounts | Register/login, Argon2 password hashing, JWT, per-user data isolation |
| Resume | PDF upload → skills, projects, experience, education (LLM + heuristic fallback) |
| Study material | PDF/TXT/DOCX ingestion → chunking → BGE embeddings → ChromaDB (or FAISS); optional BGE reranker; cited answers |
| Interview modes | Resume-based · Skill-based · Job-description-based · Mock (easy→hard, follow-up, behavioral) · Practice weak topics |
| Agent | LangGraph workflow with 10 core nodes + 4 HITL gates, conditional routing, Mongo checkpoints, `thread_id` per interview |
| HITL | Correct extracted skills · approve/edit/regenerate the plan · accept/reject/re-evaluate feedback · skip · pause · resume · end |
| Evaluation | Structured JSON (score, missing points, strengths, improvements, ideal answer, follow-up), configurable weighted rubric |
| Personalization | Adaptive difficulty (configurable thresholds), weak-topic prioritisation, memory-aware prompts |
| Memory | Short-term = LangGraph state + checkpoints. Long-term = MongoDB `user_memory` (CRUD, relevance filter, summaries, compaction) |
| MCP | 4 FastMCP servers / 13 tools, called by the graph through an MCP client (with in-process fallback) |
| Streaming | SSE (`text/event-stream`) for questions, feedback, summaries and the document chatbot; keep-alives; disconnect handling |
| Observability | LangSmith tracing (optional) + evaluation-dataset script |
| UI | 12-page Streamlit app with live token streaming |

## 3. Tech stack
Python 3.11+ · FastAPI · Pydantic v2 · Uvicorn · Streamlit · LangChain · LangGraph · LangSmith · FastMCP +
langchain-mcp-adapters · MongoDB (PyMongo async client; sync client for the LangGraph saver) · ChromaDB (FAISS
optional) · `BAAI/bge-small-en-v1.5` · `BAAI/bge-reranker-base` · `meta-llama/Llama-3.1-8B-Instruct` through Hugging
Face Inference Providers · pytest.

## 4. Architecture
```
                         USER
                           │
                  Streamlit Frontend  (httpx + SSE client, JWT in session)
                           │
                    FastAPI Backend  (rate limit · logging · CORS)
        ┌──────────────────┼──────────────────┐
   Auth Service      Resume Service      Document Service
        └──────────────────┼──────────────────┘
                           ▼
                 InterviewService  ── ownership + thread_id validation + HITL command queue
                           ▼
                  LangGraph Agent  ◄──── MongoDBSaver checkpoints (thread_id)
        ┌──────────────────┼──────────────────┐
   RAG Service        Memory Service      MCP Client ──► FastMCP servers
 (retrieve+rerank)    (long-term, Mongo)                   ├─ resume_analysis
        │                    │                             ├─ interview_questions
   ChromaDB/FAISS         MongoDB                          ├─ performance_analytics
        ▲                                                  └─ job_description
   BGE embeddings
                           ▼
            Meta Llama 3.1 8B (HF Inference Providers)
                           ▼
      Evaluation → Feedback → MongoDB persistence      (LangSmith traces everything)
```

### RAG workflow
`upload → loader (PyPDF/Docx2txt/Text) → clean → RecursiveCharacterTextSplitter → metadata (document_id, user_id,
document_name, skill_category, page, chunk_id, uploaded_at) → BGE embeddings → Chroma`. Retrieval always filters on
`user_id` (and re-checks it on every hit), applies a cosine threshold, optionally reranks, assembles numbered context
`[1] (source: file, p.3)`, and wraps it as untrusted data in the prompt. Four strategies exist: `technical`, `resume`,
`jd`, `evaluation`. If nothing relevant is found, the chatbot says so and answers from general knowledge, clearly
labelled (set `RAG_ALLOW_GENERAL_FALLBACK=false` to refuse instead).

### LangGraph workflow
```
START → analyze_resume → extract_skills ─┬─(HITL) review_skills ─┐
                                         └───────────────────────┴→ plan_interview ─(HITL) approve_plan ─┐
        ┌────────────── regenerate plan ◄─────────────────────────────────────────────────────────────────┘
        ▼
  retrieve_knowledge → generate_question → interviewer (HITL: waits for answer/skip/end)
        ▲                                         │ answer                       │ skip
        │                                  evaluate_answer ◄── re-evaluate      record_skip ─┐
        │                                         ▼                │                         │
        │                                     feedback → (HITL) review_feedback              │
        │                                                           │ accept/reject          │
        └─── adjust_difficulty ◄── update_performance ◄─────────────┘                        │
        └──────────────────────────────── more questions? ◄──────────────────────────────────┘
                                   no / end requested → summary → END
```
State is a `TypedDict` (`graph/state.py`); nodes are methods of `InterviewNodes` (`graph/nodes.py`) so they can be
instantiated with fakes; routing lives in `graph/edges.py`. Interrupt nodes contain no side effects before
`interrupt()` because LangGraph re-runs a node from its start when a thread resumes.

### Memory architecture
* **Short-term** – LangGraph state (current interview id/question/difficulty/topic, previous questions, answers,
  scores, messages, progress) persisted by the Mongo checkpointer, keyed by
  `thread_id = user_<user_id>_interview_<interview_id>`. The API derives the thread id from the interview document
  (never from client input) and validates it embeds the authenticated user.
* **Long-term** – `user_memory` collection (`topic_stat`, `mistake`, `skill`, `preferred_role`,
  `preferred_difficulty`, `learning_pref`, `progress`, `progress_summary`). Repeated low scores on a topic make it a
  *weak topic*, which gets double weight in plans and drives "Practice weak topics". Old `progress` entries are
  compacted into one summary.

### MCP architecture
| Server | Tools |
|---|---|
| resume_analysis | `extract_resume_skills`, `analyze_resume_projects`, `summarize_resume` |
| interview_questions | `generate_interview_questions`, `retrieve_interview_questions`, `search_technical_knowledge` |
| performance_analytics | `get_user_performance`, `get_weak_topics`, `get_interview_history`, `generate_progress_report` |
| job_description | `analyze_job_description`, `extract_required_skills`, `compare_resume_with_job_description` |

Graph nodes call them deterministically (the LLM never chooses `user_id`): resume nodes → resume server; planning →
analytics + JD servers; retrieval/evaluation → `search_technical_knowledge`; the analytics API → analytics + question
servers. Each tool has a validated schema (ids are 32-hex, enums for difficulty/strategy, length limits), checks
resource ownership, and converts errors to clean `ToolError`s. HTTP endpoints require
`Authorization: Bearer $MCP_INTERNAL_TOKEN`.
Modes: `MCP_MODE=embedded` (default; servers are mounted in FastAPI at `/mcp-servers/<name>/mcp`),
`external` (`python scripts/run_mcp_servers.py`), `disabled`. With `MCP_FALLBACK_LOCAL=true` an unreachable server
falls back to the identical in-process implementation (`app/mcp/logic.py`).

## 5. Installation
Prerequisites: Python 3.11+, Docker (for MongoDB) or a MongoDB Atlas URI, a Hugging Face account.

```bash
unzip rag-interview-preparation.zip && cd rag-interview-preparation
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only machines (saves GBs)
pip install -r backend/requirements.txt -r frontend/requirements.txt
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into JWT_SECRET_KEY (and set MCP_INTERNAL_TOKEN)
docker compose up -d mongo                                 # or set MONGODB_URI to Atlas
```

### Hugging Face / Llama setup
1. Open <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept Meta's license (approval can take time).
2. Create a **fine-grained** token (Settings → Access Tokens) with **"Make calls to Inference Providers"**; put it in `HF_TOKEN`.
3. Keep `HF_PROVIDER=auto` (Hugging Face picks an available provider) or pin one. Provider support for a model can
   change — run `python scripts/check_env.py`, which makes a real test call.
4. To use another Meta Llama instruct model, change only `LLM_MODEL_ID` (e.g. `meta-llama/Llama-3.2-3B-Instruct`).
   No code references the model name. Only Meta models are configured; no Qwen/Chinese-developed models are used.

Verify: `python scripts/check_env.py` (add `--skip-llm` to avoid the test call).

## 6. Environment variables
Every setting is documented in [`.env.example`](.env.example) and read in one place, `backend/app/config.py`.
Key ones: `MONGODB_URI`, `HF_TOKEN`, `LLM_MODEL_ID`, `HF_PROVIDER`, `JWT_SECRET_KEY`, `VECTOR_STORE`
(`chroma|faiss`), `RERANKER_ENABLED`, `CHECKPOINTER` (`mongo|memory`), `MCP_MODE`, `LANGSMITH_*`,
`DIFFICULTY_UP_THRESHOLD`/`DIFFICULTY_DOWN_THRESHOLD`, `EVAL_RUBRIC_JSON`.

## 7. Run
```bash
# backend (from repo root) – http://localhost:8000/docs
cd backend && uvicorn app.main:app --reload --port 8000
# frontend (new terminal, repo root) – http://localhost:8501
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```
First start downloads the embedding model (and reranker if enabled). Set `RERANKER_ENABLED=false` on small machines.

### MCP servers as separate processes (optional)
```bash
# .env: MCP_MODE=external  CHROMA_MODE=http   (Chroma's PersistentClient is not multi-process safe)
docker compose --profile external-mcp up -d chroma
python scripts/run_mcp_servers.py        # ports 8101-8104
```
Default `embedded` mode needs nothing extra.

### Docker (whole stack)
`cp .env.example .env` (set `HF_TOKEN`, secrets) then `docker compose up --build` → UI on :8501, API on :8000.

## 8. API overview (`/docs` has the full schema)
`POST /auth/register|login`, `GET /auth/me` · `POST /resume/upload`, `GET /resume`, `DELETE /resume/{id}` ·
`POST /documents/upload`, `POST /documents/job-description`, `GET /documents`, `DELETE /documents/{id}` ·
`POST /interviews/start`, `POST /interviews/{id}/answer|next|pause|resume|end`, `GET /interviews/{id}`,
`GET /interviews/{id}/stream` · `POST /chat/ask|stream` · `GET /memory`, `DELETE /memory/{id}`, `GET|PUT /preferences` ·
`GET /analytics/dashboard|history|weak-topics|performance|recommendations|report|practice-questions` · `GET /health`.

**Interview protocol.** Every action is a `POST` that validates the HITL step (409 if it isn't valid now). With
`"stream": true` it returns `{queued: true, stream_url}`; the client then opens `GET /interviews/{id}/stream`, which
claims the command exactly once (a second open returns 409), runs the graph and emits SSE events:
`run_start`, `node_start`, `token {node,text}`, `status`, `node_done`, `state`, `interrupt {kind,…}` or
`complete`, `error`, always finishing with `done`. With `"stream": false` the call blocks and returns the new state.
`waiting_for.kind` tells the client what the graph needs: `review_skills`, `approve_plan`, `await_answer`,
`review_feedback`. A failed step leaves status `needs_retry` → `POST …/next {"decision":"retry"}`.

## 9. Tests
```bash
pip install -r backend/requirements.txt && pytest -q          # 52 tests, no network/GPU/paid APIs
```
Uses a scripted fake chat model, an in-memory async Mongo double, deterministic hash embeddings, **real** Chroma and
FAISS, the **real** LangGraph graph (in-memory checkpointer), **real** FastMCP servers (in-memory protocol and over
HTTP with bearer auth) and Streamlit's `AppTest` against a live backend. Covered: auth, resume parsing, ingestion,
retrieval + cross-user isolation, RAG answers, graph workflow + HITL, thread persistence across a simulated restart,
long-term memory, adaptive difficulty, streaming/SSE, MCP execution + validation + authorization, evaluation,
prompt-injection handling, upload validation.

## 10. LangSmith
1. Create a key at <https://smith.langchain.com> → Settings → API keys.
2. `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=…`, `LANGSMITH_PROJECT=rag-interview-prep`. Restart the API.
3. Every graph run carries `user_id`/`interview_id` metadata; retrieval and evaluation are separate traced runs, along
   with LLM calls, token usage, latency and errors.
4. Evaluation: `python scripts/langsmith_eval.py --user-id <id>` creates five datasets (retrieval relevance, answer
   correctness, groundedness, question relevance, personalization quality) and runs experiments with heuristic
   evaluators. With tracing off, the app behaves identically.

## 11. Deployment
* **MongoDB Atlas** – create a free cluster, add a DB user, allow your server's IP, set `MONGODB_URI=mongodb+srv://…`.
  Indexes and checkpoint collections are created automatically on first start.
* **Backend** – any container host (Render, Railway, Fly.io, a VM): build `backend/Dockerfile` from the repo root
  (`docker build -f backend/Dockerfile .`), set env vars, mount a **persistent volume** at `/srv/data` for ChromaDB
  and uploads, set `APP_ENV=production` (refuses default JWT secret), `CORS_ORIGINS` to your frontend URL, and
  `BACKEND_URL` to the service's own reachable URL (embedded MCP calls it). Run a single worker unless you move rate
  limiting to a shared store.
* **Frontend** – Streamlit Community Cloud or any container: point at `frontend/app.py`, set `BACKEND_URL`.
* **Hugging Face** – nothing to host; the backend calls Inference Providers with `HF_TOKEN`. Re-run
  `scripts/check_env.py` after changing model/provider. Provider billing/limits apply.
* **ChromaDB persistence** – persistent volume (above) or a Chroma server (`CHROMA_MODE=http`).

## 12. Security notes
Argon2id hashing · JWT with expiry · every query scoped by `user_id` · thread ids validated against the user ·
upload allow-list + size cap + magic-byte checks + random filenames · untrusted text (resume, JD, documents, answers)
is sanitised and wrapped in `<untrusted_*>` tags with system rules to ignore embedded instructions · rate limiting
(stricter on auth) · no secrets in the frontend · `.env` git-ignored · MCP HTTP endpoints bearer-protected.

## 13. Screenshots
Add your own after first run: `docs/screenshots/dashboard.png`, `live_interview.png`, `feedback.png`, `analytics.png`.

## 14. Known limitations
* Llama 3.1 8B can emit malformed JSON; the service validates with Pydantic, retries once with a repair prompt, and
  evaluation falls back to a clearly flagged heuristic score. Scores from an 8B model are a *guide*, not ground truth.
* `langgraph-checkpoint-mongodb` currently ships only the **synchronous** `MongoDBSaver`; it is used through
  LangGraph's async wrappers (thread-pool), with a separate sync `MongoClient`.
* Pause is a server-side flag (the graph already sits at an interrupt); ending an interview that failed mid-step
  (`needs_retry`) closes it without a generated summary.
* Rate limiting is in-process (single instance). Scanned/image-only PDFs are rejected (no OCR).
* The reranker and embedding models download on first use and need RAM (≈1–2 GB with reranker).

## 15. Future improvements
OCR for scanned resumes · voice interviews · LLM-as-judge LangSmith evaluators · Redis-backed rate limiting and
checkpoint TTLs · Atlas Vector Search as an additional store · multi-user study groups · CI pipeline.
