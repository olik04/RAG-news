from __future__ import annotations

from fastapi import FastAPI
from fastapi import Header, HTTPException, status

from rag_news.core.service import get_service


def _authorize_request(x_api_key: str | None) -> None:
    service = get_service()
    if not service.settings.has_http_api_key:
        return
    if x_api_key != service.settings.http_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Sentinel-RAG", version="0.1.0")

    @app.get("/")
    async def root(x_api_key: str | None = Header(default=None)) -> dict[str, str]:
        _authorize_request(x_api_key)
        return {"service": "Sentinel-RAG", "status": "ok"}

    @app.get("/healthz")
    async def healthz(
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, int | str]:
        _authorize_request(x_api_key)
        service = get_service()
        return {
            "status": "ok",
            "documents_indexed": service.repository.count(),
        }

    return app


app = create_app()
