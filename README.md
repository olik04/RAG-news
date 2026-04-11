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

## 🏗️ Architecture

The system follows a CRAG (Corrective Retrieval-Augmented Generation) pattern:

1. **Trigger**  
   Scheduled task or user query.

2. **Retrieval**  
   Fetch latest news via Tavily.

3. **Grading**  
   LLM evaluates document relevance.

4. **Action**  
   If documents are insufficient, the agent initiates query transformation and performs a broader web search.

5. **Generation**  
   Final synthesis of information into a concise report.