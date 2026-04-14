from __future__ import annotations

from collections.abc import Sequence

from google import genai
from openai import AsyncOpenAI

from rag_news.config.settings import Settings
from rag_news.core.llm_components import (
    AnswerText,
    GoogleAnalysisAnswerGenerator,
    GoogleJsonProviderClient,
    GroqChatAnswerGenerator,
    HeuristicAnswerGenerator,
    HeuristicDocumentGrader,
    HeuristicQueryRewriter,
    MistralDocumentGrader,
    MistralQueryRewriter,
    OpenAIJsonProviderClient,
)
from rag_news.core.resilience import ResilienceConfig
from rag_news.domain.models import GradeResult, NewsDocument


class NewsLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.google_client = (
            genai.Client(api_key=settings.google_api_key)
            if settings.google_api_key
            else None
        )
        self.groq_client = (
            AsyncOpenAI(
                api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1"
            )
            if settings.groq_api_key
            else None
        )
        self.mistral_client = (
            AsyncOpenAI(
                api_key=settings.mistral_api_key, base_url="https://api.mistral.ai/v1"
            )
            if settings.mistral_api_key
            else None
        )

        self.heuristic_grader = HeuristicDocumentGrader(settings)
        self.heuristic_rewriter = HeuristicQueryRewriter()
        self.heuristic_answer_generator = HeuristicAnswerGenerator()

        # Create resilience config for LLM provider calls
        resilience_config = ResilienceConfig(
            base_timeout_sec=settings.llm_resilience_timeout_seconds,
            max_retries=settings.llm_resilience_max_retries,
            backoff_factor=settings.llm_resilience_backoff_factor,
            jitter_factor=settings.llm_resilience_jitter_factor,
        )

        mistral_provider = OpenAIJsonProviderClient(
            self.mistral_client, resilience_config=resilience_config
        )
        groq_provider = OpenAIJsonProviderClient(
            self.groq_client, resilience_config=resilience_config
        )
        google_provider = GoogleJsonProviderClient(
            self.google_client, resilience_config=resilience_config
        )

        self.document_grader = MistralDocumentGrader(
            settings,
            mistral_provider,
            self.heuristic_grader,
        )
        self.query_rewriter = MistralQueryRewriter(
            settings,
            mistral_provider,
            self.heuristic_rewriter,
        )
        self.chat_generator = GroqChatAnswerGenerator(
            settings,
            groq_provider,
            self.heuristic_answer_generator,
        )
        self.analysis_generator = GoogleAnalysisAnswerGenerator(
            settings,
            google_provider,
            self.heuristic_answer_generator,
        )

    async def grade_document(
        self, question: str, document: NewsDocument
    ) -> GradeResult:
        return await self.document_grader.grade(question, document)

    async def rewrite_query(
        self,
        question: str,
        previous_query: str,
        documents: Sequence[NewsDocument],
        attempt: int,
    ) -> str:
        return await self.query_rewriter.rewrite(
            question,
            previous_query,
            documents,
            attempt,
        )

    async def generate_chat_answer(
        self, question: str, query: str, documents: Sequence[NewsDocument]
    ) -> AnswerText:
        return await self.chat_generator.generate(question, query, documents)

    async def generate_analysis_answer(
        self, question: str, query: str, documents: Sequence[NewsDocument]
    ) -> AnswerText:
        return await self.analysis_generator.generate(question, query, documents)
