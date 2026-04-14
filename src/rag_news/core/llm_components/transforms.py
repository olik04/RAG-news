from __future__ import annotations

import json
import re
from typing import Any

from rag_news.domain.models import NewsDocument


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


def tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in raw if token not in STOPWORDS and len(token) > 2}


def important_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def doc_payload(document: NewsDocument) -> dict[str, Any]:
    return {
        "title": document.title,
        "content": document.display_text[:1000],
        "url": document.url,
        "source": document.source,
        "published_at": document.published_at,
    }


def normalize_answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        allowed = {"summary", "headline", "key_points", "cautionary_note"}
        filtered = {key: answer[key] for key in allowed if key in answer}
        if filtered:
            return json.dumps(filtered, ensure_ascii=False)
        return json.dumps(answer, ensure_ascii=False)
    if isinstance(answer, list):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer or "").strip()
