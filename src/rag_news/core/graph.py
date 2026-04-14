from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag_news.adapters.chroma_repository import ChromaNewsRepository
from rag_news.adapters.tavily_search import TavilyNewsSearch
from rag_news.config.settings import Settings
from rag_news.core.exceptions import RepositoryError
from rag_news.core.llm import NewsLLM
from rag_news.domain.models import GradeResult, NewsDocument, SearchMode


logger = getLogger(__name__)


class GraphState(TypedDict, total=False):
    question: str
    query: str
    search_mode: SearchMode
    attempts: int
    documents: list[NewsDocument]
    graded_documents: list[NewsDocument]
    grades: list[GradeResult]
    answer: str
    sources: list[str]
    last_reason: str


@dataclass(slots=True)
class GraphResult:
    question: str
    answer: str
    query: str
    documents: list[NewsDocument]
    attempts: int
    sources: list[str]


class NewsSentinelGraph:
    def __init__(
        self,
        settings: Settings,
        repository: ChromaNewsRepository,
        search: TavilyNewsSearch,
        llm: NewsLLM,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.search = search
        self.llm = llm
        self.graph = self._build_graph()

    async def answer_question(self, question: str) -> GraphResult:
        final_state = await self.graph.ainvoke(
            {
                "question": question,
                "query": question,
                "search_mode": SearchMode.LOCAL,
                "attempts": 0,
                "documents": [],
                "graded_documents": [],
                "grades": [],
            }
        )
        documents = (
            final_state.get("graded_documents") or final_state.get("documents") or []
        )
        return GraphResult(
            question=question,
            answer=final_state.get("answer", ""),
            query=final_state.get("query", question),
            documents=documents,
            attempts=int(final_state.get("attempts", 0)),
            sources=final_state.get("sources", []),
        )

    async def build_digest(self, query: str) -> GraphResult:
        final_state = await self.graph.ainvoke(
            {
                "question": query,
                "query": query,
                "search_mode": SearchMode.ANALYSIS,
                "attempts": 0,
                "documents": [],
                "graded_documents": [],
                "grades": [],
            }
        )
        documents = (
            final_state.get("graded_documents") or final_state.get("documents") or []
        )
        return GraphResult(
            question=query,
            answer=final_state.get("answer", ""),
            query=final_state.get("query", query),
            documents=documents,
            attempts=int(final_state.get("attempts", 0)),
            sources=final_state.get("sources", []),
        )

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("transform_query", self._transform_query)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self._route_after_grading)
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    async def _retrieve(self, state: GraphState) -> GraphState:
        query = state.get("query") or state["question"]
        mode = state.get("search_mode", SearchMode.LOCAL)

        if mode == SearchMode.WEB:
            try:
                documents = await self.search.search(
                    query,
                    days=self.settings.news_days_back,
                    top_k=self.settings.web_top_k,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Web retrieval failed: %s", type(exc).__name__)
                documents = []
            if documents:
                try:
                    self.repository.upsert_documents(documents)
                except RepositoryError as exc:  # pragma: no cover - storage fallback
                    logger.warning(
                        "Failed to persist web documents: %s", type(exc).__name__
                    )
            return {"documents": documents}

        try:
            recency_days = (
                self.settings.news_days_back if mode == SearchMode.ANALYSIS else None
            )
            local_documents = self.repository.search(
                query,
                top_k=self.settings.local_top_k,
                days_back=recency_days,
            )
        except RepositoryError as exc:
            logger.warning("Local retrieval failed: %s", type(exc).__name__)
            local_documents = []
        return {"documents": local_documents}

    async def _grade_documents(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        graded_documents: list[NewsDocument] = []
        grades: list[GradeResult] = []
        last_reason = ""

        for document in documents:
            grade = await self.llm.grade_document(question, document)
            grades.append(grade)
            last_reason = grade.reason
            if grade.relevant:
                graded_documents.append(document)

        return {
            "graded_documents": graded_documents,
            "grades": grades,
            "last_reason": last_reason,
        }

    async def _transform_query(self, state: GraphState) -> GraphState:
        attempts = int(state.get("attempts", 0)) + 1
        documents = state.get("documents", [])
        query = await self.llm.rewrite_query(
            state["question"],
            state.get("query", state["question"]),
            documents,
            attempts,
        )
        return {
            "query": query,
            "search_mode": SearchMode.WEB,
            "attempts": attempts,
            "documents": documents,
            "graded_documents": state.get("graded_documents", []),
            "grades": state.get("grades", []),
        }

    async def _generate_answer(self, state: GraphState) -> GraphState:
        question = state["question"]
        query = state.get("query", question)
        documents = state.get("graded_documents") or state.get("documents") or []
        if (
            state.get("question") == self.settings.news_daily_query
            or state.get("search_mode") == SearchMode.ANALYSIS
        ):
            answer = await self.llm.generate_analysis_answer(question, query, documents)
        else:
            answer = await self.llm.generate_chat_answer(question, query, documents)
        return {
            "answer": answer.answer,
            "sources": answer.sources,
            "documents": documents,
            "graded_documents": documents,
        }

    def _route_after_grading(self, state: GraphState) -> str:
        graded_documents = state.get("graded_documents") or []
        attempts = int(state.get("attempts", 0))
        if graded_documents:
            return "generate_answer"
        if attempts >= self.settings.max_retrieval_attempts:
            return "generate_answer"
        return "transform_query"
