from __future__ import annotations

from rag_news.core.digest import format_digest
from rag_news.core.graph import GraphResult


def test_format_digest_renders_structured_answer() -> None:
    answer = (
        '{"summary": "Daily summary text.", '
        '"key_points": ["Point one", "Point two"], '
        '"cautionary_note": "Use caution."}'
    )
    result = GraphResult(
        question="q",
        answer=answer,
        query="query",
        documents=[],
        attempts=1,
        sources=["https://example.com/a"],
    )

    output = format_digest(result)
    assert "Daily summary text." in output
    assert "📌 Key Points:" in output
    assert "• Point one" in output
    assert "⚠️ Cautionary Note:" in output
    assert "Use caution." in output


def test_format_digest_keeps_plain_text_answer() -> None:
    result = GraphResult(
        question="q",
        answer="Plain text answer",
        query="query",
        documents=[],
        attempts=1,
        sources=[],
    )

    output = format_digest(result)
    assert "Plain text answer" in output
