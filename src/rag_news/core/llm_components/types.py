from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnswerText:
    answer: str
    sources: list[str]
