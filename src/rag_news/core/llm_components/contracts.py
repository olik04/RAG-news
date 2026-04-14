from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rag_news.core.llm_components.types import AnswerText
from rag_news.domain.models import GradeResult, NewsDocument


class JsonProviderClient(Protocol):
    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]: ...


class DocumentGrader(Protocol):
    async def grade(self, question: str, document: NewsDocument) -> GradeResult: ...


class QueryRewriter(Protocol):
    async def rewrite(
        self,
        question: str,
        previous_query: str,
        documents: Sequence[NewsDocument],
        attempt: int,
    ) -> str: ...


class AnswerGenerator(Protocol):
    async def generate(
        self,
        question: str,
        query: str,
        documents: Sequence[NewsDocument],
    ) -> AnswerText: ...
