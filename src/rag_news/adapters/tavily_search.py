from __future__ import annotations

from logging import getLogger
from typing import Any

from tavily import TavilyClient
from tavily.errors import BadRequestError

from rag_news.config.settings import Settings
from rag_news.domain.models import NewsDocument


logger = getLogger(__name__)


class TavilyNewsSearch:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = (
            TavilyClient(api_key=settings.tavily_api_key)
            if settings.tavily_api_key
            else None
        )

    def search(
        self, query: str, *, days: int | None = None, top_k: int | None = None
    ) -> list[NewsDocument]:
        if self.client is None:
            logger.info(
                "Tavily is not configured; returning no external results for %s", query
            )
            return []

        request_days = days or self.settings.news_days_back
        try:
            response = self.client.search(
                query=query,
                topic="news",
                days=request_days,
                max_results=top_k or self.settings.web_top_k,
            )
        except BadRequestError as exc:
            # Tavily may reject combinations of days and implicit date windows from upstream.
            if "When days is set" in str(exc):
                logger.warning(
                    "Tavily rejected days-based query; retrying without days for query: %s",
                    query,
                )
                response = self.client.search(
                    query=query,
                    topic="news",
                    max_results=top_k or self.settings.web_top_k,
                )
            else:
                logger.warning("Tavily search failed: %s", exc)
                return []
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Tavily search failed: %s", exc)
            return []

        results = response.get("results", []) if isinstance(response, dict) else []
        documents: list[NewsDocument] = []
        for result in results:
            document = self._to_document(query, result)
            if document is not None:
                documents.append(document)
        return documents

    def _to_document(self, query: str, result: dict[str, Any]) -> NewsDocument | None:
        title = str(result.get("title", "Untitled"))
        content = str(result.get("content") or result.get("raw_content") or "").strip()
        if not content:
            return None
        summary = str(result.get("content", "")).strip()
        return NewsDocument(
            title=title,
            content=content,
            url=str(result.get("url", "")),
            source=str(result.get("source", "tavily")),
            published_at=str(
                result.get("published_date") or result.get("published_at") or ""
            ),
            query=query,
            summary=summary,
            score=float(result.get("score", 0.0) or 0.0),
            id=str(result.get("id") or ""),
        )
