from __future__ import annotations

from pathlib import Path

from rag_news.adapters.chroma_repository import ChromaNewsRepository
from rag_news.config.settings import Settings
from rag_news.domain.models import NewsDocument


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        google_api_key=None,
        google_model="gemini-2.5-pro",
        groq_api_key=None,
        groq_model="llama-3.1-8b-instant",
        mistral_api_key=None,
        mistral_grader_model="mistral-large-latest",
        mistral_rewriter_model="mistral-large-latest",
        tavily_api_key=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        chroma_path=tmp_path / "chroma",
        chroma_collection_name="test_news",
        timezone_name="Asia/Hong_Kong",
        digest_hour=9,
        digest_minute=0,
        news_daily_query="latest geopolitical developments last 24 hours",
        max_retrieval_attempts=2,
        min_relevance_score=0.45,
        local_top_k=5,
        web_top_k=5,
        news_days_back=1,
        log_level="INFO",
        http_host="0.0.0.0",
        http_port=8000,
    )


def test_repository_round_trip(tmp_path: Path) -> None:
    repository = ChromaNewsRepository(_settings(tmp_path))
    document = NewsDocument(
        title="Example update",
        content="The situation changed after new reports emerged.",
        url="https://example.com/article",
        source="example",
        published_at="2026-04-11T00:00:00Z",
        query="example update",
    )

    assert repository.upsert_documents([document]) == 1
    results = repository.search("situation changed after reports")
    assert results
    assert results[0].title == "Example update"
