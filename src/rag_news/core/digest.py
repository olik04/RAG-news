from __future__ import annotations

import ast
import html
import json
from typing import Any

from rag_news.core.graph import GraphResult


def format_digest(result: GraphResult, *, rich_text: bool = False) -> str:
    body = _format_digest_body(result.answer.strip(), rich_text=rich_text)
    if rich_text:
        lines = [
            "<b>Sentinel-RAG Daily Digest</b>",
            f"<i>Query:</i> {html.escape(result.query)}",
            "",
            body,
        ]
        if result.sources:
            lines.extend(["", "<b>Sources:</b>"])
            lines.extend(f"• {html.escape(source)}" for source in result.sources[:5])
        return "\n".join(lines).strip()

    lines = [
        "Sentinel-RAG Daily Digest",
        f"Query: {result.query}",
        "",
        body,
    ]
    if result.sources:
        lines.extend(["", "Sources:"])
        lines.extend(f"- {source}" for source in result.sources[:5])
    return "\n".join(lines).strip()


def format_answer(result: GraphResult, *, rich_text: bool = False) -> str:
    body = _format_digest_body(result.answer.strip(), rich_text=rich_text)
    lines = [body]
    if result.sources:
        if rich_text:
            lines.extend(["", "<b>Sources:</b>"])
            lines.extend(f"• {html.escape(source)}" for source in result.sources[:5])
        else:
            lines.extend(["", "Sources:"])
            lines.extend(f"- {source}" for source in result.sources[:5])
    return "\n".join(lines).strip()


def _format_digest_body(answer_text: str, *, rich_text: bool = False) -> str:
    parsed = _parse_structured_answer(answer_text)
    if not parsed:
        return html.escape(answer_text) if rich_text else answer_text

    # Support both 'summary' and 'headline' keys
    summary = str(parsed.get("summary") or parsed.get("headline", "")).strip()
    key_points = parsed.get("key_points")
    cautionary_note = str(parsed.get("cautionary_note", "")).strip()

    lines: list[str] = []
    if summary:
        if rich_text:
            lines.extend([f"<b>📰 {html.escape(summary)}</b>", ""])
        else:
            lines.extend(["📰 " + summary, ""])

    if isinstance(key_points, list) and key_points:
        if rich_text:
            lines.append("<b>📌 Key Points:</b>")
            lines.extend(
                f"  • {html.escape(str(point).strip())}"
                for point in key_points
                if str(point).strip()
            )
        else:
            lines.append("📌 Key Points:")
            lines.extend(
                f"  • {str(point).strip()}"
                for point in key_points
                if str(point).strip()
            )
        lines.append("")

    if cautionary_note:
        if rich_text:
            lines.extend(
                [
                    "<b>⚠️ Cautionary Note:</b>",
                    f"  <i>{html.escape(cautionary_note)}</i>",
                ]
            )
        else:
            lines.extend(["⚠️ Cautionary Note:", f"  {cautionary_note}"])

    formatted = "\n".join(lines).strip()
    if formatted:
        return formatted
    return html.escape(answer_text) if rich_text else answer_text


def _parse_structured_answer(answer_text: str) -> dict[str, Any] | None:
    if not answer_text.startswith("{"):
        return None

    try:
        parsed = json.loads(answer_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(answer_text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
