# 🛡️ Sentinel-RAG: Self-Correcting News Agent

A production-ready agentic RAG system designed to monitor, analyze, and synthesize geopolitical news, with a focus on the Israel-Iran conflict, using a self-correcting feedback loop.

## Implementation Status

The first executable scaffold is now in place: a Python package, a LangGraph retrieval/correction loop, a Tavily search adapter, local ChromaDB persistence, a Telegram bot worker, a scheduled digest flow, and a FastAPI health endpoint. External credentials are still loaded from environment variables, so the project can be run locally with placeholders before real secrets are added.

## Local Development

1. Create a virtual environment with `python -m venv .venv`.
2. Activate it with `source .venv/bin/activate`.
3. Install dependencies with `python -m pip install -e ".[dev]"`.
4. Copy `.env.example` to `.env` and fill in any available keys.
5. Run the API with `python -m rag_news.interfaces.cli api`.
6. Run the Telegram worker with `python -m rag_news.interfaces.cli worker`.
7. Generate one digest with `python -m rag_news.interfaces.cli digest`.
8. Run `python -m pytest` to cross-check the current implementation.

Docker files are kept in the repo, but they are optional for now.

## 🚀 Key Features

- **Self-Correction Loop**  
  Built with LangGraph, the system evaluates the quality of retrieved documents. If the context is irrelevant or hallucinated, it automatically rewrites the search query and reruns retrieval.

- **Daily Intelligence Digest**  
  Automatically scrapes and summarizes the last 24 hours of news using Tavily AI, delivering a structured brief to a Telegram channel every morning.

- **On-Demand Q&A**  
  A separate Telegram bot interface that answers complex queries by combining internal RAG context with real-time web search fallback.

- **Anti-Hallucination Guardrails**  
  Uses a grader node to score document relevance before generation, improving factual accuracy.

## 🛠️ Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLMs:** GPT-4o / GPT-4o-mini
- **Search Engine:** Tavily AI
- **Vector Database:** ChromaDB
- **Backend & Automation:** Python, FastAPI, APScheduler
- **Interface:** Telegram Bot API

## Project Layout

- `src/rag_news/config` contains environment loading and typed settings.
- `src/rag_news/domain` holds the shared news and grading data models.
- `src/rag_news/core` contains the LangGraph flow, LLM logic, digest formatting, and service wiring.
- `src/rag_news/adapters` isolates external systems such as ChromaDB and Tavily.
- `src/rag_news/interfaces` contains the API, CLI, and Telegram bot entrypoints.
- `src/rag_news/jobs` contains the scheduled digest job wiring.
- `tests` contains repository and graph tests.

## 🔧 Configuration

All configuration is managed via environment variables. At least one LLM provider (Google, Groq, or Mistral) must be configured. See `.env.example` for a complete template.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| **EMBEDDING_BACKEND** | `semantic` | `semantic` (recommended) or `hash`. Controls whether retrieval uses learned semantic meaning or legacy hash-based matching. Phase 3.x default is semantic. |
| **EMBEDDING_MODEL** | `all-MiniLM-L6-v2` | Sentence-transformers model for semantic embeddings. Larger models (e.g., `all-mpnet-base-v2`) improve quality but reduce speed. |
| **HTTP_API_KEY** | (optional) | Bearer token for HTTP API authentication. If set, all requests must include `Authorization: Bearer {token}`. |
| **MAX_QUESTION_LENGTH** | `1000` | Maximum character length for user questions. Prevents abuse and limits LLM cost. |
| **MAX_REQUESTS_PER_MINUTE** | `20` | Rate limit per client IP (HTTP API) or Telegram chat_id (bot). Requests over limit receive HTTP 429. |
| **NEWS_RETENTION_DAYS** | `30` | Documents older than this are automatically purged from the vector database. |
| **NEWS_RETENTION_ENABLED** | `true` | Enables/disables scheduled retention purge job. |
| **PURGE_HOUR** | `2` | Hour (0-23) for scheduled purge execution. |
| **PURGE_MINUTE** | `0` | Minute (0-59) for scheduled purge execution. |
| **PURGE_BATCH_SIZE** | `500` | Batch size for legacy metadata purge scanning. |
| **API_TIMEOUT_SECONDS** | `5.0` | Timeout for external API calls (Tavily, LLM providers). Tune based on network latency. |
| **API_MAX_RETRIES** | `3` | Number of retry attempts for transient failures (network timeouts, 5xx errors, rate limits). |
| **LLM_API_*** | inherits `API_*` | Optional LLM-specific timeout/retry overrides. |
| **TAVILY_API_*** | inherits `API_*` | Optional Tavily-specific timeout/retry overrides. |
| **SCHEDULER_DIGEST_MAX_RETRIES** | `3` | Outer retry count for scheduled digest delivery. |
| **SCHEDULER_DIGEST_BACKOFF_SECONDS** | `2.0` | Base delay between scheduled digest retries. |

### LLM Providers (Fallback Chain)

The system attempts providers in order of availability: **Google** → **Groq** → **Mistral** → **Heuristics** (fallback).

Each provider is optional; configure at least one:

```bash
# Recommended: Configure your preferred provider + one backup
GOOGLE_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

Configure provider-specific models:
- `GOOGLE_MODEL`: Default `gemini-2.5-pro`
- `GROQ_MODEL`: Default `llama-3.1-8b-instant`
- `MISTRAL_GRADER_MODEL` / `MISTRAL_REWRITER_MODEL`: Default `mistral-large-latest`

### Telegram Integration (Optional)

```bash
TELEGRAM_BOT_TOKEN=12345:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=-987654321
```

### Scheduling & Digest

```bash
NEWS_TIMEZONE=Asia/Hong_Kong              # IANA timezone format
DIGEST_HOUR=9                             # Hour to send daily digest (0–23)
DIGEST_MINUTE=0                           # Minute of the hour (0–59)
NEWS_DAILY_QUERY=...                      # Custom query for digest generation
```

See `.env.example` for all available settings.

## 🏗️ Architecture

The system follows a CRAG (Corrective Retrieval-Augmented Generation) pattern with observable LLM provider orchestration:

### Retrieval Flow

1. **Trigger**  
   Scheduled task (daily digest) or user query (API/Telegram).

2. **Retrieval**  
   Query local ChromaDB using semantic embeddings. If fewer than `LOCAL_TOP_K` relevant documents found, expand to web search via Tavily.

3. **Grading**  
   LLM evaluates document relevance to the query. Relevant documents (score > `MIN_RELEVANCE_SCORE`) are kept.

4. **Decision**  
   - **If 1+ relevant documents:** Generate answer directly (Phase 3.x change: reduced from 2+)
   - **If no relevant documents and retries remain:** Transform query and repeat web search
   - **If retries exhausted:** Generate answer with available context or fallback heuristics

5. **Generation**  
   LLM synthesizes final answer with source citations from selected documents.

### LLM Orchestration & Fallback

The system implements a **provider-agnostic LLM abstraction** with transparent fallback:

```
User Question
    ↓
Try Google (if GOOGLE_API_KEY set)
    ├─ Success → Return answer (logged: llm.provider.success)
    └─ Failure → Try next provider (logged: llm.provider.failure)
         ↓
    Try Groq (if GROQ_API_KEY set)
         ├─ Success → Return answer (logged: llm.provider.success)
         └─ Failure → Try next provider
              ↓
         Try Mistral (if MISTRAL_API_KEY set)
              ├─ Success → Return answer (logged: llm.provider.success)
              └─ Failure → Use heuristic fallback (logged: llm.provider.exhausted)
                    ↓
              Return hardcoded grading/rewrite logic
```

**Fallback Behavior:**
- If all providers fail or are unconfigured, system uses heuristic fallbacks (keyword-based grading, query duplication)
- Fallback is observable: structured logs emit events showing which provider was attempted and why it failed
- Times out after `API_TIMEOUT_SECONDS` per provider

### Embedding Strategy (Phase 3.x)

The system uses **semantic embeddings** (sentence-transformers) for meaning-aware retrieval:

- **Semantic Backend:** Converts text to 384-dimensional vectors using local model (all-MiniLM-L6-v2 default)
- **Similarity Metric:** Cosine distance to find contextually similar documents
- **Collection Namespacing:** Collections named by backend + model (e.g., `sentinel_news_semantic_all-MiniLM-L6-v2`)
- **Gradual Indexing:** New documents re-indexed on digest scheduler runs

### Data Retention & Cleanup

- Documents older than `NEWS_RETENTION_DAYS` are automatically purged
- Purge happens during scheduler execution (nightly)
- ChromaDB indexes are updated incrementally (no full reindex needed)

### Rate Limiting & Security

- **API Rate Limit:** `MAX_REQUESTS_PER_MINUTE` per client IP
- **Question Length Limit:** `MAX_QUESTION_LENGTH` characters enforced at interface
- **Retry Policy:** Transient failures (5xx, timeouts) retry with exponential backoff; auth failures (401/403) fail fast

## 📊 Operations & Telemetry

RAG-news emits structured telemetry logs to help diagnose issues and monitor provider behavior.

### Telemetry Events

The system logs the following event types at `INFO` level (use `LOG_LEVEL=DEBUG` for granular inspection):

#### LLM Provider Events

```json
{
  "timestamp": "2026-04-14T12:34:56.789Z",
  "level": "INFO",
  "event": "llm.provider.success",
  "provider": "google",
  "operation": "grade_document",
  "latency_ms": 324,
  "request_id": "q_abc123"
}
```

| Event | Meaning | Action |
|-------|---------|--------|
| `llm.provider.attempt` | Provider being tried | None (informational) |
| `llm.provider.success` | Provider succeeded | Monitor latency; confirm correct provider used |
| `llm.provider.failure` | Provider failed (will retry) | Check error reason; verify API key is valid |
| `llm.provider.exhausted` | All providers failed, using heuristics | **Alert:** System degraded, check provider status |

#### Embedding Events

```json
{
  "timestamp": "2026-04-14T12:34:56.789Z",
  "level": "INFO",
  "event": "embedding.model_loaded",
  "model": "all-MiniLM-L6-v2",
  "size_mb": 22,
  "duration_seconds": 2.1
}
```

| Event | Meaning |
|-------|---------|
| `embedding.model_loaded` | Embedding model downloaded/loaded on startup |
| `embedding.documents_indexed` | Batch of documents converted to embeddings |
| `embedding.retrieval` | Query converted to embedding and searched |

### Common Issues & Solutions

#### Issue: `llm.provider.exhausted` in logs

**Diagnosis:**
- All LLM providers have failed or are unconfigured
- System falls back to keyword-based heuristics (lower quality)

**Solutions:**
1. Check API keys in `.env`: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`
2. Verify network connectivity: `curl -I https://api.openai.com` (or relevant provider)
3. Check for rate limits: Provider may be throttling; reduce QPS or wait
4. Review provider status pages for outages
5. Temporary workaround: Configure multiple providers for redundancy

#### Issue: Slow response times or embedding timeouts

**Diagnosis:**
- Embedding model download stalled or very slow
- API calls hitting `API_TIMEOUT_SECONDS` consistently

**Solutions:**
1. **First run:** Embedding model downloads ~22 MB; this takes 30s–2m on slow networks
2. Increase timeout: Set `API_TIMEOUT_SECONDS=10.0` and redeploy
3. Run model download manually in Docker: `docker exec rag-news python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
4. Check disk space: Embeddings require ~500 MB free

#### Issue: Questions answered with wrong or irrelevant context

**Diagnosis:**
- Semantic embeddings retrieving low-quality documents
- `MIN_RELEVANCE_SCORE` threshold too permissive

**Solutions:**
1. Increase `MIN_RELEVANCE_SCORE` from 0.45 → 0.55–0.60
2. Verify embedding model is being used: Check logs for `embedding.retrieval` events with correct model name
3. Try alternative embedding model: Set `EMBEDDING_MODEL=bge-small-en-v1.5` (better quality, slower)
4. Check document quality: Ensure indexed documents are relevant to domain

### Monitoring Dashboard Setup

Key metrics to track:

| Metric | Query | Alert Threshold |
|--------|-------|-----------------|
| Provider Success Rate | `count(llm.provider.success) / count(llm.provider.attempt)` | < 95% |
| Fallback Usage | `count(llm.provider.exhausted)` per hour | > 5 |
| API Timeouts | `count(llm.provider.failure AND reason="timeout")` | > 10 per hour |
| Answer Quality Score | Average `grade.score` for generated answers | < 0.6 (investigate) |
| Q&A Latency (p95) | Response time percentile | > 10 seconds |
| Embedding Model Load Time | First deployment latency | > 60 seconds (expected) |
| Collection Size | Document count in ChromaDB | Growing; peaks before `NEWS_RETENTION_DAYS` purge |

### Logs for Troubleshooting

#### Enable verbose logging
```bash
export LOG_LEVEL=DEBUG
docker-compose restart worker api
```

#### Follow logs in real-time
```bash
docker-compose logs -f worker api | grep -E "llm.provider|embedding|error"
```

#### Export logs for analysis
```bash
docker-compose logs worker api > logs.txt
# Or for JSON logs (if configured):
docker-compose logs worker api --timestamps | grep "llm.provider"
```

### Recommended Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| All LLM Providers Down | `llm.provider.exhausted` > 10/hour | Critical |
| High Provider Latency | `llm.provider.success.latency_ms` > 5000 | Warning |
| Rate Limit Exceeded | `http.429.count` > 100/hour | High |
| Embedding Model Not Loaded | `embedding.model_loaded` missing after restart | High |
| Database Space Low | `df -h data/chroma > 90%` | Warning |

---

**For Phase 3.x Migration Details, see [MIGRATION.md](MIGRATION.md)**