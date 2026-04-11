from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger

import chromadb

from rag_news.config.settings import Settings
from rag_news.core.embeddings import HashingEmbeddingFunction
from rag_news.domain.models import NewsDocument


logger = getLogger(__name__)


class ChromaNewsRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.embedding_function = HashingEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection_name)

    def upsert_documents(self, documents: Iterable[NewsDocument]) -> int:
        items = list(documents)
        if not items:
            return 0

        texts = [self._compose_document_text(item) for item in items]
        self.collection.upsert(
            ids=[item.id for item in items],
            documents=texts,
            metadatas=[item.to_metadata() for item in items],
            embeddings=[self.embedding_function.embed_text(text) for text in texts],
        )
        return len(items)

    def search(self, query: str, top_k: int | None = None) -> list[NewsDocument]:
        limit = top_k or self.settings.local_top_k
        try:
            response = self.collection.query(query_embeddings=[self.embedding_function.embed_text(query)], n_results=limit)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Local retrieval failed: %s", exc)
            return []

        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]
        ids = response.get("ids") or [[]]

        results: list[NewsDocument] = []
        for index, document_text in enumerate(documents[0]):
            metadata = (metadatas[0][index] if metadatas and metadatas[0] else {}) or {}
            distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else 0.0
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
        except Exception:  # pragma: no cover - defensive fallback
            return 0

    @staticmethod
    def _compose_document_text(document: NewsDocument) -> str:
        parts = [document.title.strip(), document.summary.strip(), document.content.strip()]
        return "\n\n".join(part for part in parts if part)
