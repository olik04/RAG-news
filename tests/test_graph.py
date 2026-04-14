from __future__ import annotations

from pathlib import Path

import pytest

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


class FakeSearch:
    def __init__(self, responses: list[list[NewsDocument]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def search(
        self, query: str, *, days: int | None = None, top_k: int | None = None
    ):
        self.calls.append({"query": query, "days": days, "top_k": top_k})
        if self.responses:
            return self.responses.pop(0)
        return []


class FakeRepository:
    def __init__(self, responses: list[list[NewsDocument]]) -> None:
        self.responses = responses
        self.search_calls: list[dict[str, object]] = []
        self.upserted_documents: list[NewsDocument] = []

    def search(
        self, query: str, *, top_k: int | None = None, days_back: int | None = None
    ) -> list[NewsDocument]:
        self.search_calls.append(
            {"query": query, "top_k": top_k, "days_back": days_back}
        )
        if self.responses:
            return self.responses.pop(0)
        return []

    def upsert_documents(self, documents: list[NewsDocument]) -> int:
        self.upserted_documents.extend(documents)
        return len(documents)


class FakeLLM:
    def __init__(self, relevant_sequence: list[bool] | None = None) -> None:
        self.relevant_sequence = relevant_sequence or [True]
        self.rewrite_calls = 0
        self.chat_calls = 0
        self.analysis_calls = 0

    async def grade_document(
        self, question: str, document: NewsDocument
    ) -> GradeResult:
        relevant = self.relevant_sequence.pop(0) if self.relevant_sequence else False
        return GradeResult(
            relevant=relevant,
            score=0.9 if relevant else 0.1,
            reason="test",
        )

    async def rewrite_query(
        self, question: str, previous_query: str, documents, attempt: int
    ) -> str:
        self.rewrite_calls += 1
        return "more specific news query"

    async def generate_chat_answer(
        self, question: str, query: str, documents
    ) -> AnswerText:
        self.chat_calls += 1
        return AnswerText(
            answer=f"answer for {question}",
            sources=[document.url for document in documents if document.url],
        )

    async def generate_analysis_answer(
        self, question: str, query: str, documents
    ) -> AnswerText:
        self.analysis_calls += 1
        return AnswerText(
            answer=f"analysis for {question}",
            sources=[document.url for document in documents if document.url],
        )


@pytest.mark.asyncio
async def test_graph_generates_answer_from_local_documents(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = NewsDocument(
        title="Escalation reported",
        content="New reports describe escalation in the region.",
        url="https://example.com/escalation",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="escalation",
    )
    repository = FakeRepository([[document]])
    llm = FakeLLM(relevant_sequence=[True])
    search = FakeSearch([])

    graph = NewsSentinelGraph(settings, repository, search, llm)
    result = await graph.answer_question("What is happening with the escalation?")

    assert result.answer == "answer for What is happening with the escalation?"
    assert result.documents
    assert result.attempts == 0
    assert llm.chat_calls == 1
    assert llm.analysis_calls == 0
    assert llm.rewrite_calls == 0
    assert not search.calls


@pytest.mark.asyncio
async def test_graph_retries_with_web_mode(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    local_document = NewsDocument(
        title="Unrelated business headline",
        content="The company announced quarterly earnings.",
        url="https://example.com/business",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="business",
    )
    web_document = NewsDocument(
        title="Conflict escalates after border incident",
        content="Officials reported new developments overnight.",
        url="https://example.com/conflict",
        source="web",
        published_at="2026-04-12T00:00:00Z",
        query="conflict",
    )
    repository = FakeRepository([[local_document]])
    search = FakeSearch([[web_document]])
    llm = FakeLLM(relevant_sequence=[False, True])

    graph = NewsSentinelGraph(settings, repository, search, llm)
    result = await graph.answer_question("What is happening with the conflict?")

    assert result.answer == "answer for What is happening with the conflict?"
    assert result.attempts == 1
    assert llm.rewrite_calls == 1
    assert len(search.calls) == 1
    assert result.sources == ["https://example.com/conflict"]
    assert repository.upserted_documents == [web_document]


@pytest.mark.asyncio
async def test_graph_stops_after_max_attempts_when_no_relevant_docs(
    tmp_path: Path,
) -> None:
    base_settings = _settings(tmp_path)
    # Create new settings with custom max_retrieval_attempts (frozen dataclass requires construction)
    settings = Settings(
        google_api_key=base_settings.google_api_key,
        google_model=base_settings.google_model,
        groq_api_key=base_settings.groq_api_key,
        groq_model=base_settings.groq_model,
        mistral_api_key=base_settings.mistral_api_key,
        mistral_grader_model=base_settings.mistral_grader_model,
        mistral_rewriter_model=base_settings.mistral_rewriter_model,
        tavily_api_key=base_settings.tavily_api_key,
        telegram_bot_token=base_settings.telegram_bot_token,
        telegram_chat_id=base_settings.telegram_chat_id,
        chroma_path=base_settings.chroma_path,
        chroma_collection_name=base_settings.chroma_collection_name,
        timezone_name=base_settings.timezone_name,
        digest_hour=base_settings.digest_hour,
        digest_minute=base_settings.digest_minute,
        news_daily_query=base_settings.news_daily_query,
        max_retrieval_attempts=2,
        min_relevance_score=base_settings.min_relevance_score,
        local_top_k=base_settings.local_top_k,
        web_top_k=base_settings.web_top_k,
        news_days_back=base_settings.news_days_back,
        news_retention_days=base_settings.news_retention_days,
        news_retention_enabled=base_settings.news_retention_enabled,
        purge_hour=base_settings.purge_hour,
        purge_minute=base_settings.purge_minute,
        purge_batch_size=base_settings.purge_batch_size,
        log_level=base_settings.log_level,
        http_host=base_settings.http_host,
        http_port=base_settings.http_port,
        http_api_key=base_settings.http_api_key,
        max_question_length=base_settings.max_question_length,
        max_requests_per_minute=base_settings.max_requests_per_minute,
        embedding_backend=base_settings.embedding_backend,
        embedding_model=base_settings.embedding_model,
        api_timeout_seconds=base_settings.api_timeout_seconds,
        api_max_retries=base_settings.api_max_retries,
        api_backoff_factor=base_settings.api_backoff_factor,
        api_jitter_factor=base_settings.api_jitter_factor,
        llm_api_timeout_seconds=base_settings.llm_api_timeout_seconds,
        llm_api_max_retries=base_settings.llm_api_max_retries,
        llm_api_backoff_factor=base_settings.llm_api_backoff_factor,
        llm_api_jitter_factor=base_settings.llm_api_jitter_factor,
        tavily_api_timeout_seconds=base_settings.tavily_api_timeout_seconds,
        tavily_api_max_retries=base_settings.tavily_api_max_retries,
        tavily_api_backoff_factor=base_settings.tavily_api_backoff_factor,
        tavily_api_jitter_factor=base_settings.tavily_api_jitter_factor,
        scheduler_digest_max_retries=base_settings.scheduler_digest_max_retries,
        scheduler_digest_backoff_seconds=base_settings.scheduler_digest_backoff_seconds,
    )

    local_document = NewsDocument(
        title="Local unrelated",
        content="No overlap with question.",
        url="https://example.com/local-unrelated",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="local",
    )
    web_document_1 = NewsDocument(
        title="Web unrelated one",
        content="No overlap here either.",
        url="https://example.com/web-1",
        source="web",
        published_at="2026-04-12T00:00:00Z",
        query="web1",
    )
    web_document_2 = NewsDocument(
        title="Web unrelated two",
        content="Still no overlap.",
        url="https://example.com/web-2",
        source="web",
        published_at="2026-04-13T00:00:00Z",
        query="web2",
    )

    repository = FakeRepository([[local_document]])
    search = FakeSearch([[web_document_1], [web_document_2]])
    llm = FakeLLM(relevant_sequence=[False, False, False])

    graph = NewsSentinelGraph(settings, repository, search, llm)
    result = await graph.answer_question("Question with no relevant coverage")

    assert result.attempts == settings.max_retrieval_attempts
    assert llm.rewrite_calls == settings.max_retrieval_attempts
    assert len(search.calls) == settings.max_retrieval_attempts
    assert llm.chat_calls == 1
    assert llm.analysis_calls == 0


@pytest.mark.asyncio
async def test_graph_selects_analysis_branch_for_digest_mode(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = NewsDocument(
        title="Daily summary item",
        content="Relevant development for daily digest.",
        url="https://example.com/digest",
        source="local",
        published_at="2026-04-11T00:00:00Z",
        query="digest",
    )
    repository = FakeRepository([[document]])
    llm = FakeLLM(relevant_sequence=[True])
    search = FakeSearch([])

    graph = NewsSentinelGraph(settings, repository, search, llm)
    result = await graph.build_digest("daily digest custom query")

    assert result.answer == "analysis for daily digest custom query"
    assert llm.analysis_calls == 1
    assert llm.chat_calls == 0
