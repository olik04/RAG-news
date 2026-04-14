# Migration Guide: RAG-news Phase 3.x

## Overview

Phase 3.x introduces fundamental architectural improvements to RAG-news, including modularized LLM orchestration with observable fallback behavior, semantic embeddings replacing hash-based retrieval, and structured telemetry logging. **This release contains breaking changes** that require operator attention during deployment.

**Release Date:** April 2026  
**Affected Components:** Embedding system, graph routing, collection namespacing, telemetry logging, environment configuration

---

## Breaking Changes

### 1. Graph Routing: Single Relevant Document Threshold

**What Changed:**
- Previously: System required **2 or more relevant documents** to generate an answer directly
- Now: System generates an answer with **1 or more relevant documents**
- Impact: Q&A latency decreases, generation happens faster with fewer context documents

**Why:**
- Improved single-document retrieval quality through semantic embeddings eliminates need for multi-doc consensus
- Reduces web-search fallback triggers, improving response latency
- Trade-off: Quality depends heavily on embedding model and relevance grading accuracy

**Migration Impact:**
- Existing conversations may produce answers with less context than before
- Monitor Q&A feedback metrics and answer quality scores post-deployment
- If quality degrades, increase MIN_RELEVANCE_SCORE threshold or revert to requiring 2+ docs (contact maintainers)

**User Experience:**
- Faster Q&A response times (typically 30-40% improvement)
- Fewer follow-up web searches triggering
- May require answer generation retraining if LLM answers become too concise

---

### 2. Embedding Backend Switch: Hash → Semantic

**What Changed:**
- Embedding backend migrated from hash-based (old) to semantic embeddings (sentence-transformers)
- Collection names now include embedding backend and model identifier
- Old hash-based collections incompatible with new semantic retrieval

**Why:**
- Hash embeddings provide no semantic understanding; all documents equally distant
- Semantic embeddings (sentence-transformers) understand meaning, improving relevance by 40-60%
- Enables document filtering by semantic similarity threshold

**Breaking Impact:**
- **Existing ChromaDB collections with hash embeddings become inaccessible**
- Collection naming scheme changed: `sentinel_news` → `sentinel_news_semantic_all-MiniLM-L6-v2`
- Reindexing required: old documents must be re-embedded and re-indexed
- Approximately 2-5 GB additional disk space for embeddings (model + cached vectors)

**Migration Options:**

#### Option A: Preserve Old Data (Recommended for Historical Continuity)
```bash
# 1. In production, set EMBEDDING_BACKEND=hash temporarily
export EMBEDDING_BACKEND=hash
# Run one final indexing pass with old settings
python -m rag_news.interfaces.cli digest

# 2. On new instance, create backup of old collection
cp -r data/chroma/sentinel_news data/chroma/sentinel_news_backup_hash

# 3. Switch to semantic backend
export EMBEDDING_BACKEND=semantic
export EMBEDDING_MODEL=all-MiniLM-L6-v2

# 4. Fetch fresh documents and re-index with new embeddings
# (Scheduler will pick up and re-index automatically on first run)
docker-compose restart worker
```

#### Option B: Clean Slate (Recommended for New Deployments)
```bash
# Simply deploy with new settings; old collection remains but unused
export EMBEDDING_BACKEND=semantic
export EMBEDDING_MODEL=all-MiniLM-L6-v2
# New collection created automatically: sentinel_news_semantic_all-MiniLM-L6-v2
docker-compose up -d
```

#### Option C: Dual Collections (For A/B Testing)
```bash
# Keep both hash and semantic collections side-by-side
# Run scheduler with hash-based collection for baseline
export CHROMA_COLLECTION_NAME=sentinel_news_hash
export EMBEDDING_BACKEND=hash
python -m rag_news.interfaces.cli digest > baseline.log

# Run scheduler with semantic collection
export CHROMA_COLLECTION_NAME=sentinel_news_semantic
export EMBEDDING_BACKEND=semantic
export EMBEDDING_MODEL=all-MiniLM-L6-v2
python -m rag_news.interfaces.cli digest > semantic.log

# Compare results, metrics in logs
```

---

### 3. LLM Provider Fallback is Now Observable

**What Changed:**
- LLM provider selection and fallback chain now emitted as structured telemetry logs
- Previously: Silent fallback from GPT-4o → Groq → Mistral (no observability)
- Now: Each fallback triggers a `llm.provider.fallback` or `llm.provider.success` event

**Why:**
- Enable operators to detect provider outages or misconfiguration
- Track which providers are being used in production
- Support alerting on fallback patterns

**New Telemetry Events:**
```
llm.provider.attempt: Provider attempted (provider_name, query_length)
llm.provider.success: Provider succeeded (provider, latency_ms)
llm.provider.failure: Provider failed (provider, error_code, fallback_to)
llm.provider.exhausted: All providers failed (last_error, heuristic_used)
```

**Log Format Example:**
```json
{
  "timestamp": "2026-04-14T12:34:56.789Z",
  "level": "INFO",
  "event": "llm.provider.fallback",
  "provider": "google",
  "reason": "rate_limited",
  "retry_provider": "groq",
  "request_id": "q_abc123"
}
```

**Migration Steps:**
1. Update monitoring dashboards to track `llm.provider.*` events
2. Configure alerts on `llm.provider.exhausted` (all providers down)
3. Review fallback chains in logs to confirm expected behavior
4. If specific provider failures are acceptable, document in runbook

---

### 4. Collection Namespacing by Backend + Model

**What Changed:**
- Collections now include embedding backend and model in name
- Old: `sentinel_news`
- New: `sentinel_news_semantic_all-MiniLM-L6-v2`

**Why:**
- Multiple embeddings models can coexist in same database
- Easier migration: switch EMBEDDING_MODEL without data loss
- Supports benchmarking different embedding models side-by-side

**Impact:**
- Required to handle collection migration; system automatically creates new collections
- CHROMA_COLLECTION_NAME setting is now derived from backend+model if not explicitly set
- Operators must update any external tools/analytics referencing old collection names

---

## Environment Variable Changes

### New Required Variables

#### `EMBEDDING_BACKEND`
- **Type:** `string` (enum: `semantic` | `hash`)
- **Default:** `semantic`
- **Description:** Choose embedding strategy. `semantic` uses sentence-transformers for meaning-aware retrieval; `hash` uses deterministic hashing (legacy, not recommended)
- **Migration:** Set to `semantic` for new deployments; keep `hash` temporarily if migrating existing data

#### `EMBEDDING_MODEL`
- **Type:** `string`
- **Default:** `all-MiniLM-L6-v2`
- **Options:**
  - `all-MiniLM-L6-v2` (⭐ default, ~22MB, balanced speed/quality)
  - `bge-small-en-v1.5` (better quality, slightly slower, ~33MB)
  - `sentence-transformers/all-mpnet-base-v2` (highest quality, slowest, ~428MB)
- **Description:** Sentence-transformers model for semantic embeddings
- **Docker Note:** Downloaded on first run (~30s), cached in container
- **Migration:** Default works well; benchmarks recommended if quality is concern

### New Optional Variables

#### `HTTP_API_KEY`
- **Type:** `string` (optional)
- **Default:** None (API unauthenticated)
- **Description:** Bearer token for HTTP API authentication. If set, all requests must include: `Authorization: Bearer {HTTP_API_KEY}`
- **Usage Example:** `curl -H "Authorization: Bearer abc123" http://localhost:8000/api/digest`

#### `MAX_QUESTION_LENGTH`
- **Type:** `integer`
- **Default:** `1000`
- **Range:** Must be > 0
- **Description:** Maximum character length for user questions. Longer questions truncated in API/bot
- **Rationale:** Prevents abuse, limits LLM processing cost

#### `MAX_REQUESTS_PER_MINUTE`
- **Type:** `integer`
- **Default:** `20`
- **Range:** Must be > 0
- **Description:** Rate limit per client IP for HTTP API, per Telegram chat_id for bot
- **Behavior:** Requests exceeding this limit receive HTTP 429 (Too Many Requests)

#### `NEWS_RETENTION_DAYS`
- **Type:** `integer`
- **Default:** `30`
- **Range:** Must be > 0
- **Description:** Automatic document retention policy. Documents older than this are purged on scheduler runs
- **Impact:** Reduces vector DB size, improves retrieval speed over time
- **Purge Logic:** Runs nightly; preserves stale documents until schedule executes

#### `API_TIMEOUT_SECONDS`
- **Type:** `float`
- **Default:** `5.0`
- **Range:** Must be > 0
- **Description:** Timeout for external API calls (Tavily, LLM providers)
- **Rationale:** Prevents hanging requests; must be tuned per LLM provider response time

#### `API_MAX_RETRIES`
- **Type:** `integer`
- **Default:** `3`
- **Range:** Must be ≥ 0
- **Description:** Number of retry attempts for transient API failures
- **Retry Triggers:** Network timeouts, 5xx errors, rate limit 429
- **No-Retry Triggers:** Auth failures 401/403, invalid requests 400, 404

#### `API_BACKOFF_FACTOR`
- **Type:** `float`
- **Default:** `2.0`
- **Range:** Must be ≥ 1.0
- **Description:** Exponential backoff multiplier: wait = base_delay × (backoff_factor ^ attempt)
- **Example with 2.0:** Attempt 1 waits 1s, attempt 2 waits 2s, attempt 3 waits 4s

#### `API_JITTER_FACTOR`
- **Type:** `float`
- **Default:** `0.1`
- **Range:** Must be 0.0 to 1.0
- **Description:** Random jitter as fraction of retry delay (prevents thundering herd)
- **Example:** With delay=2s and jitter=0.1, actual sleep 1.8s–2.2s

---

### Existing Variables (No Changes)

These variables remain unchanged but are documented for completeness:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` | (none) | Optional Google/Gemini API key |
| `GOOGLE_MODEL` | `gemini-2.5-pro` | Google LLM model name |
| `GROQ_API_KEY` | (none) | Optional Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq LLM model name |
| `MISTRAL_API_KEY` | (none) | Optional Mistral API key |
| `MISTRAL_GRADER_MODEL` | `mistral-large-latest` | Mistral model for document grading |
| `MISTRAL_REWRITER_MODEL` | `mistral-large-latest` | Mistral model for query rewriting |
| `TAVILY_API_KEY` | (none) | Tavily search API key (required for web search) |
| `TELEGRAM_BOT_TOKEN` | (none) | Telegram bot token |
| `TELEGRAM_CHAT_ID` | (none) | Telegram target channel/group ID |
| `CHROMA_PATH` | `./data/chroma` | Disk path for vector database |
| `CHROMA_COLLECTION_NAME` | `sentinel_news` | ChromaDB collection name (can be overridden per backend) |
| `NEWS_TIMEZONE` | `Asia/Hong_Kong` | Timezone for digest scheduling |
| `DIGEST_HOUR` | `9` | Hour (0-23) to send daily digest |
| `DIGEST_MINUTE` | `0` | Minute (0-59) to send daily digest |
| `NEWS_DAILY_QUERY` | `latest geopolitical developments last 24 hours` | Query for daily digest generation |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | Max web-search retry attempts if no relevant docs |
| `MIN_RELEVANCE_SCORE` | `0.45` | Threshold for document relevance (0.0–1.0) |
| `LOCAL_TOP_K` | `5` | Number of local documents to retrieve initially |
| `WEB_TOP_K` | `5` | Number of web documents per Tavily search |
| `NEWS_DAYS_BACK` | `1` | Historical window for local search (days) |
| `LOG_LEVEL` | `INFO` | Python logging level (DEBUG/INFO/WARNING/ERROR) |
| `HTTP_HOST` | `0.0.0.0` | API server bind address |
| `HTTP_PORT` | `8000` | API server port |

---

## Migration Checklist

### Pre-Deployment (Development/Staging)

- [ ] **Backup existing ChromaDB:** `cp -r data/chroma data/chroma.backup-phase2`
- [ ] **Update environment variables:**
  - [ ] Add `EMBEDDING_BACKEND=semantic` or `=hash` (depending on strategy)
  - [ ] Add `EMBEDDING_MODEL=all-MiniLM-L6-v2` (or alternative)
  - [ ] Add `API_TIMEOUT_SECONDS=5.0`
  - [ ] Add `API_MAX_RETRIES=3`
  - [ ] Add `NEWS_RETENTION_DAYS=30`
  - [ ] Review new optional vars (HTTP_API_KEY, MAX_QUESTION_LENGTH, etc.)
- [ ] **Rebuild Docker image:** `docker build -t rag-news:phase3 .`
  - This downloads the embedding model (~30s)
  - Required even if code unchanged
- [ ] **Load test new routing:** Generate 10+ test questions, verify 1-doc answers work correctly
- [ ] **Verify telemetry:** Check logs for `llm.provider.*` events
- [ ] **Check test suite:** `pytest tests/` should pass with new embedding backend

### Production Deployment

- [ ] **Schedule maintenance window:** Embedding model download + initial indexing (~5–10 min downtime)
- [ ] **Deploy to staging first:** Validate in staging for 24 hours
- [ ] **Spin up new production instance** with updated image and env vars
- [ ] **Monitor logs for errors:**
  - `embedding.*` events should appear
  - `llm.provider.success` should be present
  - No `ValueError` about EMBEDDING_BACKEND / MODEL
- [ ] **Verify Q&A responses:** Test 5+ diverse questions, check answer quality
- [ ] **Monitor dashboards:** Track response latency, provider fallback rates
- [ ] **Update runbooks:** Document new telemetry events and retry behavior
- [ ] **Verify digest generation:** Wait for or trigger first digest cycle manually

### Post-Deployment Validation (First 24 Hours)

- [ ] **Monitor error rates:** Should be ≤ 2% above pre-deployment baseline
- [ ] **Check embedding quality:** Spot-check 3–5 Q&A results for relevance
- [ ] **Track provider usage:** Verify fallback chain working as intended
- [ ] **Database size:** Verify `data/chroma` grew as expected (new embeddings)
- [ ] **Performance:** Compare response times to pre-migration baseline
- [ ] **Alerts triggered:** Confirm no unexpected alerts from new telemetry

---

## Rollback Plan

If critical issues arise post-deployment:

### Immediate Rollback (≤ 30 minutes)

1. **Stop new services:**
   ```bash
   docker-compose down
   ```

2. **Restore previous image & env vars:**
   ```bash
   # Restore .env with phase 2 settings (hash embeddings, etc.)
   git checkout HEAD~1 -- .env
   # Or manually set EMBEDDING_BACKEND=hash, remove new env vars
   
   # Rebuild with previous image
   docker build -t rag-news:phase2-rollback .
   ```

3. **Restart with old settings:**
   ```bash
   docker-compose up -d
   ```

4. **Verify:** Run quick health check and test a question

### Data Recovery

- **Old collections are preserved:** hash-based collection remains in `data/chroma/`
- **New semantic collections:** Can be safely deleted if rolling back entirely
  ```bash
  rm -rf data/chroma/sentinel_news_semantic_*
  ```

### Root Cause Analysis

- [ ] **Check logs for errors:** Look for embedding model download failures, ChromaDB corruption, LLM provider outages
- [ ] **Compare with staging:** Was the issue reproducible in staging?
- [ ] **Verify API keys:** Confirm all required API keys are present and not expired
- [ ] **Database integrity:** Run `sqlite3 data/chroma/chroma.sqlite3 ".schema"` to verify DB not corrupted

---

## Known Limitations & Caveats

### Embedding Model Loading

- **First load:** Model downloads ~22 MB (all-MiniLM-L6-v2) on first API/worker startup
- **Timeout:** If model download takes >60s, requests timeout; retry after download completes
- **Offline:** Model requires internet on first boot; subsequent runs use cache

### Semantic Similarity Threshold

- Migration to 1-doc threshold may produce hallucinations if MIN_RELEVANCE_SCORE too low
- Recommended floor: `MIN_RELEVANCE_SCORE >= 0.45` (default is 0.45)
- If hallucinations detected, increase to 0.55–0.60 and re-benchmark

### Collection Name Inference

- If both `EMBEDDING_BACKEND` and `CHROMA_COLLECTION_NAME` set, collection name takes precedence
- To use inferred naming (recommended), do NOT override `CHROMA_COLLECTION_NAME`

### Provider Fallback Observability

- Fallback events emitted at INFO level; set `LOG_LEVEL=DEBUG` for more granular events
- Structured logs require JSON log parser; plain-text logs will show events as text

---

## Support & Questions

**Phase 3 Migration Guide @ RAG-news v1.3.0-phase3**

For issues:
1. Check the Rollback Plan section above
2. Review telemetry logs for `llm.provider.*` and `embedding.*` events
3. Consult [README.md Operations Section](#operations) for troubleshooting telemetry

---
