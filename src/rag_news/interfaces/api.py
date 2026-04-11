from __future__ import annotations

from fastapi import FastAPI

from rag_news.core.service import get_service


def create_app() -> FastAPI:
    app = FastAPI(title="Sentinel-RAG", version="0.1.0")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "Sentinel-RAG", "status": "ok"}

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        service = get_service()
        return {
            "status": "ok",
            "documents_indexed": service.repository.count(),
            "has_tavily": service.settings.has_tavily,
            "has_telegram": service.settings.has_telegram,
        }

    return app


app = create_app()
