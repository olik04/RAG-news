from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from logging import getLogger
import re
from typing import Any

import chromadb

from rag_news.config.settings import Settings
from rag_news.core.embeddings import HashingEmbeddingFunction, SemanticEmbeddingFunction
from rag_news.core.exceptions import RepositoryError
from rag_news.domain.models import NewsDocument


logger = getLogger(__name__)


class ChromaNewsRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        if settings.embedding_backend == "semantic":
            try:
                self.embedding_function = SemanticEmbeddingFunction(
                    settings.embedding_model
                )
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                logger.warning(
                    "Semantic embedding init failed; falling back to hash embeddings: %s",
                    type(exc).__name__,
                )
                self.embedding_function = HashingEmbeddingFunction()
        else:
            self.embedding_function = HashingEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collection_name = self._resolve_collection_name(settings)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_documents(self, documents: Iterable[NewsDocument]) -> int:
        items = list(documents)
        if not items:
            return 0

        texts = [self._compose_document_text(item) for item in items]
        try:
            self.collection.upsert(
                ids=[item.id for item in items],
                documents=texts,
                metadatas=[item.to_metadata() for item in items],
                embeddings=[self.embedding_function.embed_text(text) for text in texts],
            )
        except Exception as exc:  # pragma: no cover - persistence backend dependent
            raise RepositoryError(f"Failed to upsert {len(items)} documents") from exc
        return len(items)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source: str | None = None,
        days_back: int | None = None,
    ) -> list[NewsDocument]:
        limit = top_k or self.settings.local_top_k
        expanded_limit = max(limit, limit * 3)
        indexed_count = max(1, self.count())
        n_results = min(expanded_limit, indexed_count)

        try:
            response = self.collection.query(
                query_embeddings=[self.embedding_function.embed_text(query)],
                n_results=n_results,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RepositoryError("Local retrieval failed") from exc

        results = self._parse_query_response(response, query)
        filtered = self._apply_filters(results, source=source, days_back=days_back)
        return filtered[:limit]

    def _parse_query_response(
        self, response: dict[str, Any], query: str
    ) -> list[NewsDocument]:
        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]
        ids = response.get("ids") or [[]]

        if not documents or not documents[0]:
            return []

        results: list[NewsDocument] = []
        for index, document_text in enumerate(documents[0]):
            metadata = (metadatas[0][index] if metadatas and metadatas[0] else {}) or {}
            distance = (
                distances[0][index]
                if distances and distances[0] and index < len(distances[0])
                else 0.0
            )
            result_id = ids[0][index] if ids and ids[0] and index < len(ids[0]) else ""
            results.append(
                NewsDocument(
                    title=str(metadata.get("title", "Untitled")),
                    content=document_text,
                    url=str(metadata.get("url", "")),
                    source=str(metadata.get("source", "local")),
                    published_at=str(metadata.get("published_at", "")),
                    query=str(metadata.get("query", query)),
                    summary=str(metadata.get("summary", "")),
                    score=max(0.0, 1.0 - float(distance)),
                    id=str(result_id),
                )
            )
        return results

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to count collection documents: %s", type(exc).__name__
            )
            return 0

    @staticmethod
    def _compose_document_text(document: NewsDocument) -> str:
        parts = [
            document.title.strip(),
            document.summary.strip(),
            document.content.strip(),
        ]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _resolve_collection_name(settings: Settings) -> str:
        suffix = f"{settings.embedding_backend}_{settings.embedding_model}".lower()
        sanitized = re.sub(r"[^a-z0-9_-]+", "_", suffix).strip("_")
        return f"{settings.chroma_collection_name}_{sanitized}"

    @staticmethod
    def _apply_filters(
        documents: list[NewsDocument],
        *,
        source: str | None,
        days_back: int | None,
    ) -> list[NewsDocument]:
        if not source and (not days_back or days_back <= 0):
            return documents

        cutoff: datetime | None = None
        if days_back and days_back > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        filtered: list[NewsDocument] = []
        for document in documents:
            if source and document.source != source:
                continue
            if cutoff and document.published_at:
                published = ChromaNewsRepository._parse_datetime(document.published_at)
                if published is not None and published < cutoff:
                    continue
            filtered.append(document)
        return filtered

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
