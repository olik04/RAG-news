from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from google import genai
from openai import AsyncOpenAI

from rag_news.config.settings import Settings
from rag_news.domain.models import GradeResult, NewsDocument


logger = logging.getLogger(__name__)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "latest",
    "news",
    "of",
    "on",
    "or",
    "tell",
    "the",
    "to",
    "what",
    "with",
    "within",
    "today",
    "yesterday",
    "update",
    "updates",
}


@dataclass(slots=True)
class AnswerText:
    answer: str
    sources: list[str]


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

    async def grade_document(
        self, question: str, document: NewsDocument
    ) -> GradeResult:
        if self.mistral_client is not None:
            prompt = (
                "Assess whether a news document is relevant to the user's question. "
                "Return JSON with keys relevant (boolean), score (0 to 1), reason (string)."
            )
            payload = {
                "question": question,
                "title": document.title,
                "content": document.display_text[:4000],
            }
            response = await self._chat_json_openai(
                self.mistral_client,
                self.settings.mistral_grader_model,
                prompt,
                json.dumps(payload),
            )
            if response:
                return GradeResult(
                    relevant=bool(response.get("relevant", False)),
                    score=float(response.get("score", 0.0) or 0.0),
                    reason=str(response.get("reason", "")),
                )

        return self._heuristic_grade(question, document)

    async def rewrite_query(
        self,
        question: str,
        previous_query: str,
        documents: Sequence[NewsDocument],
        attempt: int,
    ) -> str:
        if self.mistral_client is not None:
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
            response = await self._chat_json_openai(
                self.mistral_client,
                self.settings.mistral_rewriter_model,
                prompt,
                json.dumps(payload),
            )
            if response and response.get("query"):
                return str(response["query"]).strip()

        return self._heuristic_rewrite(question, previous_query, attempt)

    async def generate_chat_answer(
        self, question: str, query: str, documents: Sequence[NewsDocument]
    ) -> AnswerText:
        if self.groq_client is not None:
            prompt = (
                "Write a concise Telegram-ready news answer with a short headline, key points, and a cautionary note if evidence is thin. "
                "Return JSON with keys answer and sources."
            )
            payload = {
                "question": question,
                "query": query,
                "documents": [
                    self._doc_payload(document) for document in documents[:8]
                ],
            }
            response = await self._chat_json_openai(
                self.groq_client,
                self.settings.groq_model,
                prompt,
                json.dumps(payload),
            )
            if response and response.get("answer"):
                sources = response.get("sources") or [
                    document.url for document in documents if document.url
                ]
                return AnswerText(
                    answer=self._normalize_answer_text(response.get("answer")),
                    sources=[str(item) for item in sources],
                )

        return self._heuristic_answer(question, query, documents)

    async def generate_analysis_answer(
        self, question: str, query: str, documents: Sequence[NewsDocument]
    ) -> AnswerText:
        if self.google_client is not None:
            prompt = (
                "Write a longer news analysis with a concise summary, key points, and a cautionary note if evidence is thin. "
                "Return JSON with keys answer and sources."
            )
            payload = {
                "question": question,
                "query": query,
                "documents": [
                    self._doc_payload(document) for document in documents[:12]
                ],
            }
            response = self._chat_json_google(
                model=self.settings.google_model,
                system_prompt=prompt,
                user_prompt=json.dumps(payload),
            )
            if response and response.get("answer"):
                sources = response.get("sources") or [
                    document.url for document in documents if document.url
                ]
                return AnswerText(
                    answer=self._normalize_answer_text(response.get("answer")),
                    sources=[str(item) for item in sources],
                )

        return self._heuristic_answer(question, query, documents)

    async def _chat_json_openai(
        self,
        client: AsyncOpenAI | None,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        if client is None:
            return {}

        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("OpenAI request failed: %s", type(exc).__name__)
            return {}

        content = response.choices[0].message.content if response.choices else None
        if not content:
            return {}

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned non-JSON content")
            return {}

    def _chat_json_google(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        if self.google_client is None:
            return {}

        try:
            response = self.google_client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    system_instruction=system_prompt,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Google request failed: %s", type(exc).__name__)
            return {}

        text = getattr(response, "text", None) or ""
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Google returned non-JSON content")
            return {}

    def _heuristic_grade(self, question: str, document: NewsDocument) -> GradeResult:
        question_tokens = self._tokens(question)
        document_tokens = self._tokens(f"{document.title} {document.display_text}")
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

    def _heuristic_rewrite(
        self, question: str, previous_query: str, attempt: int
    ) -> str:
        question_terms = self._important_terms(question)
        previous_terms = self._important_terms(previous_query)
        combined_terms = list(dict.fromkeys(question_terms + previous_terms))
        suffix = "recent verified reporting last 24 hours"
        if attempt > 1:
            suffix = "breaking developments official statements verified reporting last 24 hours"
        if combined_terms:
            return " ".join(combined_terms[:8] + [suffix])
        return f"{question.strip()} {suffix}".strip()

    def _heuristic_answer(
        self, question: str, query: str, documents: Sequence[NewsDocument]
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

    def _tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {token for token in tokens if token not in STOPWORDS and len(token) > 2}

    def _important_terms(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS and len(token) > 2
        ]

    @staticmethod
    def _doc_payload(document: NewsDocument) -> dict[str, Any]:
        return {
            "title": document.title,
            "content": document.display_text[:1000],
            "url": document.url,
            "source": document.source,
            "published_at": document.published_at,
        }

    @staticmethod
    def _normalize_answer_text(answer: Any) -> str:
        if isinstance(answer, dict):
            allowed = {"summary", "headline", "key_points", "cautionary_note"}
            filtered = {key: answer[key] for key in allowed if key in answer}
            if filtered:
                return json.dumps(filtered, ensure_ascii=False)
            return json.dumps(answer, ensure_ascii=False)
        if isinstance(answer, list):
            return json.dumps(answer, ensure_ascii=False)
        return str(answer or "").strip()
