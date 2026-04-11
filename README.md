# 🛡️ Sentinel-RAG: Self-Correcting News Agent

A production-ready agentic RAG system designed to monitor, analyze, and synthesize geopolitical news, with a focus on the Israel-Iran conflict, using a self-correcting feedback loop.

## 🚀 Key Features

- **Self-Correction Loop**  
  Built with LangGraph, the system evaluates the quality of retrieved documents. If the context is irrelevant or hallucinated, it automatically rewrites the search query and reruns retrieval.

- **Daily Intelligence Digest**  
  Automatically scrapes and summarizes the last 24 hours of news using Tavily AI, delivering a structured brief to a Telegram channel every morning.

- **On-Demand Q&A**  
  A Telegram bot interface that answers complex queries by combining internal RAG context with real-time web search fallback.

- **Anti-Hallucination Guardrails**  
  Uses a grader node to score document relevance before generation, improving factual accuracy.

## 🛠️ Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLMs:** GPT-4o / GPT-4o-mini
- **Search Engine:** Tavily AI
- **Vector Database:** ChromaDB
- **Backend & Automation:** Python, FastAPI, APScheduler
- **Interface:** Telegram Bot API

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