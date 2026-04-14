from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from time import perf_counter

from rag_news.config.settings import Settings
from rag_news.core.llm_components.heuristics import (
    HeuristicAnswerGenerator,
    HeuristicDocumentGrader,
    HeuristicQueryRewriter,
)
from rag_news.core.llm_components.provider_clients import (
    GoogleJsonProviderClient,
    OpenAIJsonProviderClient,
)
from rag_news.core.llm_components.telemetry import log_llm_event
from rag_news.core.llm_components.transforms import doc_payload, normalize_answer_text
from rag_news.core.llm_components.types import AnswerText
from rag_news.domain.models import GradeResult, NewsDocument


logger = logging.getLogger(__name__)


class MistralDocumentGrader:
    def __init__(
        self,
        settings: Settings,
        client: OpenAIJsonProviderClient,
        fallback: HeuristicDocumentGrader,
    ) -> None:
        self.settings = settings
        self.client = client
        self.fallback = fallback

    async def grade(self, question: str, document: NewsDocument) -> GradeResult:
        prompt = (
            "Assess whether a news document is relevant to the user's question. "
            "Return JSON with keys relevant (boolean), score (0 to 1), reason (string)."
        )
        payload = {
            "question": question,
            "title": document.title,
            "content": document.display_text[:4000],
        }
        start = perf_counter()
        log_llm_event(
            logger,
            operation="grade_document",
            provider="mistral",
            model=self.settings.mistral_grader_model,
            outcome="attempt",
            duration_ms=0,
            doc_count=1,
        )
        response = await self.client.chat_json(
            self.settings.mistral_grader_model,
            prompt,
            json.dumps(payload),
            operation="grade_document",
            doc_count=1,
        )
        provider_duration_ms = int((perf_counter() - start) * 1000)
        if response:
            log_llm_event(
                logger,
                operation="grade_document",
                provider="mistral",
                model=self.settings.mistral_grader_model,
                outcome="success",
                duration_ms=provider_duration_ms,
                doc_count=1,
            )
            return GradeResult(
                relevant=bool(response.get("relevant", False)),
                score=float(response.get("score", 0.0) or 0.0),
                reason=str(response.get("reason", "")),
            )
        log_llm_event(
            logger,
            operation="grade_document",
            provider="mistral",
            model=self.settings.mistral_grader_model,
            outcome="fallback",
            duration_ms=provider_duration_ms,
            reason="empty_or_invalid_response",
            doc_count=1,
        )
        fallback_start = perf_counter()
        result = await self.fallback.grade(question, document)
        log_llm_event(
            logger,
            operation="grade_document",
            provider="heuristic",
            model="heuristic_document_grader",
            outcome="success",
            duration_ms=int((perf_counter() - fallback_start) * 1000),
            reason="empty_or_invalid_response",
            doc_count=1,
        )
        return result


class MistralQueryRewriter:
    def __init__(
        self,
        settings: Settings,
        client: OpenAIJsonProviderClient,
        fallback: HeuristicQueryRewriter,
    ) -> None:
        self.settings = settings
        self.client = client
        self.fallback = fallback

    async def rewrite(
        self,
        question: str,
        previous_query: str,
        documents: Sequence[NewsDocument],
        attempt: int,
    ) -> str:
        prompt = (
            "Rewrite a news search query to improve retrieval. Return JSON with key query. "
            "Keep it concise and bias toward recent, verifiable reporting."
        )
        payload = {
            "question": question,
            "previous_query": previous_query,
            "attempt": attempt,
            "documents": [document.title for document in documents[:3]],
        }
        start = perf_counter()
        log_llm_event(
            logger,
            operation="rewrite_query",
            provider="mistral",
            model=self.settings.mistral_rewriter_model,
            outcome="attempt",
            duration_ms=0,
            doc_count=len(documents),
        )
        response = await self.client.chat_json(
            self.settings.mistral_rewriter_model,
            prompt,
            json.dumps(payload),
            operation="rewrite_query",
            doc_count=len(documents),
        )
        provider_duration_ms = int((perf_counter() - start) * 1000)
        if response and response.get("query"):
            log_llm_event(
                logger,
                operation="rewrite_query",
                provider="mistral",
                model=self.settings.mistral_rewriter_model,
                outcome="success",
                duration_ms=provider_duration_ms,
                doc_count=len(documents),
            )
            return str(response["query"]).strip()
        fallback_reason = (
            "missing_query_field"
            if response and not response.get("query")
            else "empty_or_invalid_response"
        )
        log_llm_event(
            logger,
            operation="rewrite_query",
            provider="mistral",
            model=self.settings.mistral_rewriter_model,
            outcome="fallback",
            duration_ms=provider_duration_ms,
            reason=fallback_reason,
            doc_count=len(documents),
        )
        fallback_start = perf_counter()
        result = await self.fallback.rewrite(question, previous_query, documents, attempt)
        log_llm_event(
            logger,
            operation="rewrite_query",
            provider="heuristic",
            model="heuristic_query_rewriter",
            outcome="success",
            duration_ms=int((perf_counter() - fallback_start) * 1000),
            reason=fallback_reason,
            doc_count=len(documents),
        )
        return result


class GroqChatAnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        client: OpenAIJsonProviderClient,
        fallback: HeuristicAnswerGenerator,
    ) -> None:
        self.settings = settings
        self.client = client
        self.fallback = fallback

    async def generate(
        self,
        question: str,
        query: str,
        documents: Sequence[NewsDocument],
    ) -> AnswerText:
        prompt = (
            "Write a concise Telegram-ready news answer with a short headline, key points, and a cautionary note if evidence is thin. "
            "Return JSON with keys answer and sources."
        )
        payload = {
            "question": question,
            "query": query,
            "documents": [doc_payload(document) for document in documents[:8]],
        }
        start = perf_counter()
        log_llm_event(
            logger,
            operation="generate_chat_answer",
            provider="groq",
            model=self.settings.groq_model,
            outcome="attempt",
            duration_ms=0,
            doc_count=len(documents),
        )
        response = await self.client.chat_json(
            self.settings.groq_model,
            prompt,
            json.dumps(payload),
            operation="generate_chat_answer",
            doc_count=len(documents),
        )
        provider_duration_ms = int((perf_counter() - start) * 1000)
        if response and response.get("answer"):
            log_llm_event(
                logger,
                operation="generate_chat_answer",
                provider="groq",
                model=self.settings.groq_model,
                outcome="success",
                duration_ms=provider_duration_ms,
                doc_count=len(documents),
            )
            sources = response.get("sources") or [
                document.url for document in documents if document.url
            ]
            return AnswerText(
                answer=normalize_answer_text(response.get("answer")),
                sources=[str(item) for item in sources],
            )
        fallback_reason = (
            "missing_answer_field"
            if response and not response.get("answer")
            else "empty_or_invalid_response"
        )
        log_llm_event(
            logger,
            operation="generate_chat_answer",
            provider="groq",
            model=self.settings.groq_model,
            outcome="fallback",
            duration_ms=provider_duration_ms,
            reason=fallback_reason,
            doc_count=len(documents),
        )
        fallback_start = perf_counter()
        result = await self.fallback.generate(question, query, documents)
        log_llm_event(
            logger,
            operation="generate_chat_answer",
            provider="heuristic",
            model="heuristic_answer_generator",
            outcome="success",
            duration_ms=int((perf_counter() - fallback_start) * 1000),
            reason=fallback_reason,
            doc_count=len(documents),
        )
        return result


class GoogleAnalysisAnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        client: GoogleJsonProviderClient,
        fallback: HeuristicAnswerGenerator,
    ) -> None:
        self.settings = settings
        self.client = client
        self.fallback = fallback

    async def generate(
        self,
        question: str,
        query: str,
        documents: Sequence[NewsDocument],
    ) -> AnswerText:
        prompt = (
            "Write a longer news analysis with a concise summary, key points, and a cautionary note if evidence is thin. "
            "Return JSON with keys answer and sources."
        )
        payload = {
            "question": question,
            "query": query,
            "documents": [doc_payload(document) for document in documents[:12]],
        }
        start = perf_counter()
        log_llm_event(
            logger,
            operation="generate_analysis_answer",
            provider="google",
            model=self.settings.google_model,
            outcome="attempt",
            duration_ms=0,
            doc_count=len(documents),
        )
        response = self.client.chat_json(
            self.settings.google_model,
            prompt,
            json.dumps(payload),
            operation="generate_analysis_answer",
            doc_count=len(documents),
        )
        provider_duration_ms = int((perf_counter() - start) * 1000)
        if response and response.get("answer"):
            log_llm_event(
                logger,
                operation="generate_analysis_answer",
                provider="google",
                model=self.settings.google_model,
                outcome="success",
                duration_ms=provider_duration_ms,
                doc_count=len(documents),
            )
            sources = response.get("sources") or [
                document.url for document in documents if document.url
            ]
            return AnswerText(
                answer=normalize_answer_text(response.get("answer")),
                sources=[str(item) for item in sources],
            )
        fallback_reason = (
            "missing_answer_field"
            if response and not response.get("answer")
            else "empty_or_invalid_response"
        )
        log_llm_event(
            logger,
            operation="generate_analysis_answer",
            provider="google",
            model=self.settings.google_model,
            outcome="fallback",
            duration_ms=provider_duration_ms,
            reason=fallback_reason,
            doc_count=len(documents),
        )
        fallback_start = perf_counter()
        result = await self.fallback.generate(question, query, documents)
        log_llm_event(
            logger,
            operation="generate_analysis_answer",
            provider="heuristic",
            model="heuristic_answer_generator",
            outcome="success",
            duration_ms=int((perf_counter() - fallback_start) * 1000),
            reason=fallback_reason,
            doc_count=len(documents),
        )
        return result
