from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def _stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "::".join(part.strip() for part in parts if part)))


@dataclass(slots=True)
class NewsDocument:
    title: str
    content: str
    url: str = ""
    source: str = ""
    published_at: str = ""
    query: str = ""
    summary: str = ""
    score: float = 0.0
    id: str = field(default_factory=str)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _stable_id(self.title, self.url, self.content[:128])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "query": self.query,
            "summary": self.summary,
            "score": self.score,
        }

    @property
    def display_text(self) -> str:
        if self.summary:
            return self.summary
        return self.content


@dataclass(slots=True)
class GradeResult:
    relevant: bool
    score: float
    reason: str


@dataclass(slots=True)
class AnswerBundle:
    question: str
    answer: str
    documents: list[NewsDocument]
    query: str
    attempts: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class DigestBundle:
    title: str
    body: str
    documents: list[NewsDocument]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
