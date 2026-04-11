from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rag_news.adapters.chroma_repository import ChromaNewsRepository
from rag_news.adapters.tavily_search import TavilyNewsSearch
from rag_news.config.settings import Settings, load_settings
from rag_news.core.graph import GraphResult, NewsSentinelGraph
from rag_news.core.llm import NewsLLM


@dataclass(slots=True)
class ServiceBundle:
    settings: Settings
    repository: ChromaNewsRepository
    search: TavilyNewsSearch
    llm: NewsLLM
    graph: NewsSentinelGraph


def build_service(settings: Settings | None = None) -> ServiceBundle:
    resolved_settings = settings or load_settings()
    repository = ChromaNewsRepository(resolved_settings)
    search = TavilyNewsSearch(resolved_settings)
    llm = NewsLLM(resolved_settings)
    graph = NewsSentinelGraph(resolved_settings, repository, search, llm)
    return ServiceBundle(
        settings=resolved_settings,
        repository=repository,
        search=search,
        llm=llm,
        graph=graph,
    )


@lru_cache(maxsize=1)
def get_service() -> ServiceBundle:
    return build_service()


async def answer_question(question: str, settings: Settings | None = None) -> GraphResult:
    bundle = build_service(settings) if settings is not None else get_service()
    return await bundle.graph.answer_question(question)


async def build_digest(query: str | None = None, settings: Settings | None = None) -> GraphResult:
    bundle = build_service(settings) if settings is not None else get_service()
    resolved_query = query or bundle.settings.news_daily_query
    return await bundle.graph.build_digest(resolved_query)
