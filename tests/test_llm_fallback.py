from __future__ import annotations

from pathlib import Path

import pytest

from rag_news.config.settings import Settings
from rag_news.core.llm import NewsLLM
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
        http_api_key=None,
        max_question_length=1000,
        max_requests_per_minute=20,
        embedding_backend="hash",
        embedding_model="all-MiniLM-L6-v2",
    )


@pytest.mark.asyncio
async def test_news_llm_falls_back_when_provider_clients_unavailable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = NewsLLM(settings)

    document = NewsDocument(
        title="Border escalation reported",
        content="Officials confirmed escalation and response plans.",
        url="https://example.com/escalation",
        source="test",
        published_at="2026-04-12T00:00:00Z",
        query="escalation",
    )

    grade = await llm.grade_document("What happened at the border?", document)
    rewrite = await llm.rewrite_query(
        "What happened at the border?",
        "border updates",
        [document],
        1,
    )
    chat_answer = await llm.generate_chat_answer(
        "What happened at the border?",
        "border updates",
        [document],
    )
    analysis_answer = await llm.generate_analysis_answer(
        "What happened at the border?",
        "border updates",
        [document],
    )

    assert grade.reason
    assert isinstance(grade.relevant, bool)
    assert isinstance(grade.score, float)
    assert rewrite
    assert "recent" in rewrite
    assert "Question: What happened at the border?" in chat_answer.answer
    assert chat_answer.sources == ["https://example.com/escalation"]
    assert "Question: What happened at the border?" in analysis_answer.answer
    assert analysis_answer.sources == ["https://example.com/escalation"]
