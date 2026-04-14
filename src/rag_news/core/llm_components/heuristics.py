from __future__ import annotations

from collections.abc import Sequence

from rag_news.config.settings import Settings
from rag_news.core.llm_components.transforms import important_terms, tokens
from rag_news.core.llm_components.types import AnswerText
from rag_news.domain.models import GradeResult, NewsDocument


class HeuristicDocumentGrader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def grade(self, question: str, document: NewsDocument) -> GradeResult:
        question_tokens = tokens(question)
        document_tokens = tokens(f"{document.title} {document.display_text}")
        if not question_tokens or not document_tokens:
            return GradeResult(
                relevant=False, score=0.0, reason="Insufficient lexical overlap."
            )

        overlap = len(question_tokens & document_tokens)
        score = min(1.0, overlap / max(3, len(question_tokens)))
        relevant = score >= self.settings.min_relevance_score or overlap >= 2
        reason = (
            "Lexical overlap indicates topical relevance."
            if relevant
            else "Document appears off-topic."
        )
        return GradeResult(relevant=relevant, score=score, reason=reason)


class HeuristicQueryRewriter:
    async def rewrite(
        self,
        question: str,
        previous_query: str,
        documents: Sequence[NewsDocument],
        attempt: int,
    ) -> str:
        question_terms = important_terms(question)
        previous_terms = important_terms(previous_query)
        combined_terms = list(dict.fromkeys(question_terms + previous_terms))
        suffix = "recent verified reporting last 24 hours"
        if attempt > 1:
            suffix = (
                "breaking developments official statements verified reporting last 24 hours"
            )
        if combined_terms:
            return " ".join(combined_terms[:8] + [suffix])
        return f"{question.strip()} {suffix}".strip()


class HeuristicAnswerGenerator:
    async def generate(
        self,
        question: str,
        query: str,
        documents: Sequence[NewsDocument],
    ) -> AnswerText:
        if not documents:
            return AnswerText(
                answer=(
                    f"I could not verify enough recent reporting for: {question}. "
                    f"The search query was: {query}. Try narrowing the geography, actor, or time window."
                ),
                sources=[],
            )

        key_points = []
        sources = []
        for document in documents[:5]:
            key_points.append(
                f"- {document.title}: {document.display_text[:220].strip()}"
            )
            if document.url:
                sources.append(document.url)

        answer = [
            f"Question: {question}",
            f"Query: {query}",
            "Key points:",
            *key_points,
        ]
        return AnswerText(answer="\n".join(answer), sources=sources)
