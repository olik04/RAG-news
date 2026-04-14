from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
        news_retention_days=30,
        news_retention_enabled=True,
        purge_hour=2,
        purge_minute=0,
        purge_batch_size=500,
        log_level="INFO",
        http_host="0.0.0.0",
        http_port=8000,
        http_api_key=None,
        max_question_length=1000,
        max_requests_per_minute=20,
        embedding_backend="hash",
        embedding_model="all-MiniLM-L6-v2",
        api_timeout_seconds=5.0,
        api_max_retries=3,
        api_backoff_factor=2.0,
        api_jitter_factor=0.1,
        llm_api_timeout_seconds=None,
        llm_api_max_retries=None,
        llm_api_backoff_factor=None,
        llm_api_jitter_factor=None,
        tavily_api_timeout_seconds=None,
        tavily_api_max_retries=None,
        tavily_api_backoff_factor=None,
        tavily_api_jitter_factor=None,
        scheduler_digest_max_retries=3,
        scheduler_digest_backoff_seconds=2.0,
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


@pytest.mark.asyncio
async def test_purge_stale_documents(tmp_path: Path) -> None:
    """Test that purge_stale_documents removes old documents and keeps recent ones."""
    settings = _settings(tmp_path)
    repository = ChromaNewsRepository(settings)

    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=settings.news_retention_days + 5)).isoformat()
    recent_date = (now - timedelta(days=settings.news_retention_days - 5)).isoformat()

    # Create documents with different ages
    old_doc = NewsDocument(
        title="Old article",
        content="This is old news.",
        url="https://example.com/old",
        source="archive",
        published_at=old_date,
        query="test",
    )
    recent_doc = NewsDocument(
        title="Recent article",
        content="This is recent news.",
        url="https://example.com/recent",
        source="archive",
        published_at=recent_date,
        query="test",
    )

    # Upsert both documents
    assert repository.upsert_documents([old_doc, recent_doc]) == 2
    assert repository.count() == 2

    # Purge stale documents
    result = await repository.purge_stale_documents()

    # Verify the old document was deleted
    assert result["deleted_count"] == 1
    assert result["remaining_count"] == 1
    assert repository.count() == 1

    # Verify that the recent document still exists
    remaining = repository.search("recent", top_k=10)
    assert len(remaining) == 1
    assert remaining[0].title == "Recent article"


@pytest.mark.asyncio
async def test_purge_stale_documents_with_source_filter(tmp_path: Path) -> None:
    """Test that purge_stale_documents respects source filter."""
    settings = _settings(tmp_path)
    repository = ChromaNewsRepository(settings)

    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=settings.news_retention_days + 5)).isoformat()

    # Create old documents from different sources
    old_doc_archive = NewsDocument(
        title="Old archive article",
        content="This is old archive news.",
        url="https://example.com/old-archive",
        source="archive",
        published_at=old_date,
        query="test",
    )
    old_doc_news = NewsDocument(
        title="Old news article",
        content="This is old news outlet news.",
        url="https://example.com/old-news",
        source="news_outlet",
        published_at=old_date,
        query="test",
    )

    # Upsert both documents
    assert repository.upsert_documents([old_doc_archive, old_doc_news]) == 2
    assert repository.count() == 2

    # Purge only archive source
    result = await repository.purge_stale_documents(source="archive")

    # Verify only the archive document was deleted
    assert result["deleted_count"] == 1
    assert result["remaining_count"] == 1

    # Verify the news_outlet document still exists
    remaining = repository.search("news outlet", top_k=10)
    assert len(remaining) == 1
    assert remaining[0].source == "news_outlet"


@pytest.mark.asyncio
async def test_purge_stale_documents_empty_collection(tmp_path: Path) -> None:
    """Test purge on empty collection."""
    repository = ChromaNewsRepository(_settings(tmp_path))

    # Purge from empty collection
    result = await repository.purge_stale_documents()

    assert result["deleted_count"] == 0
    assert result["remaining_count"] == 0
