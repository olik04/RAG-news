from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient
from tavily.errors import BadRequestError

from rag_news.config.settings import Settings
from rag_news.core.resilience import ResilienceConfig, with_timeout_and_retry
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

    async def search(
        self, query: str, *, days: int | None = None, top_k: int | None = None
    ) -> list[NewsDocument]:
        if self.client is None:
            logger.info(
                "Tavily is not configured; returning no external results for %s", query
            )
            return []

        request_days = days or self.settings.news_days_back
        config = ResilienceConfig(
            base_timeout_sec=self.settings.tavily_resilience_timeout_seconds,
            max_retries=self.settings.tavily_resilience_max_retries,
            backoff_factor=self.settings.tavily_resilience_backoff_factor,
            jitter_factor=self.settings.tavily_resilience_jitter_factor,
        )

        response: dict[str, Any] = {}
        try:
            response = await with_timeout_and_retry(
                "tavily_search",
                config,
                lambda: asyncio.to_thread(
                    self.client.search,
                    query=query,
                    topic="news",
                    days=request_days,
                    max_results=top_k or self.settings.web_top_k,
                ),
            )
        except Exception as exc:
            # Handle validation/user errors: Tavily may reject certain query combinations
            if isinstance(exc, BadRequestError) and "When days is set" in str(exc):
                logger.warning(
                    "Tavily rejected days-based query; retrying without days for query: %s",
                    query,
                )
                try:
                    response = await with_timeout_and_retry(
                        "tavily_search_no_days",
                        config,
                        lambda: asyncio.to_thread(
                            self.client.search,
                            query=query,
                            topic="news",
                            max_results=top_k or self.settings.web_top_k,
                        ),
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "Tavily search without days failed: %s", type(retry_exc).__name__
                    )
                    return []
            else:
                logger.warning("Tavily search failed: %s", type(exc).__name__)
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
        url = str(result.get("url", ""))
        if url and not self._is_valid_url(url):
            return None
        summary = str(result.get("content", "")).strip()
        return NewsDocument(
            title=title,
            content=content,
            url=url,
            source=str(result.get("source", "tavily")),
            published_at=str(
                result.get("published_date") or result.get("published_at") or ""
            ),
            query=query,
            summary=summary,
            score=float(result.get("score", 0.0) or 0.0),
            id=str(result.get("id") or ""),
        )

    @staticmethod
    def _is_valid_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
