from __future__ import annotations

from pathlib import Path

import pytest

from rag_news.adapters.chroma_repository import ChromaNewsRepository
from rag_news.adapters.tavily_search import TavilyNewsSearch
from rag_news.config.settings import Settings
from rag_news.core.graph import NewsSentinelGraph
from rag_news.core.llm import AnswerText
from rag_news.domain.models import GradeResult, NewsDocument


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


class FakeSearch:
    def __init__(self, document: NewsDocument) -> None:
        self.document = document

    def search(self, query: str, *, days: int | None = None, top_k: int | None = None):
        return [self.document]


class FakeLLM:
    def __init__(self, relevant: bool) -> None:
        self.relevant = relevant

    async def grade_document(
        self, question: str, document: NewsDocument
    ) -> GradeResult:
        return GradeResult(
            relevant=self.relevant, score=0.9 if self.relevant else 0.1, reason="test"
        )

    async def rewrite_query(
        self, question: str, previous_query: str, documents, attempt: int
    ) -> str:
        return "more specific news query"

    async def generate_answer(self, question: str, query: str, documents) -> AnswerText:
        return AnswerText(
            answer=f"answer for {question}",
            sources=[document.url for document in documents if document.url],
        )


@pytest.mark.asyncio
async def test_graph_generates_answer_from_local_documents(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = ChromaNewsRepository(settings)
    document = NewsDocument(
        title="Escalation reported",
        content="New reports describe escalation in the region.",
        url="https://example.com/escalation",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="escalation",
    )
    repository.upsert_documents([document])

    graph = NewsSentinelGraph(
        settings, repository, TavilyNewsSearch(settings), FakeLLM(relevant=True)
    )
    result = await graph.answer_question("What is happening with the escalation?")

    assert result.answer == "answer for What is happening with the escalation?"
    assert result.documents


@pytest.mark.asyncio
async def test_graph_retries_with_web_mode(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repository = ChromaNewsRepository(settings)
    document = NewsDocument(
        title="Unrelated business headline",
        content="The company announced quarterly earnings.",
        url="https://example.com/business",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="business",
    )

    graph = NewsSentinelGraph(
        settings, repository, FakeSearch(document), FakeLLM(relevant=False)
    )
    result = await graph.answer_question("What is happening with the conflict?")

    assert result.attempts >= 1
