from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    google_api_key: str | None
    google_model: str
    groq_api_key: str | None
    groq_model: str
    mistral_api_key: str | None
    mistral_grader_model: str
    mistral_rewriter_model: str
    tavily_api_key: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    chroma_path: Path
    chroma_collection_name: str
    timezone_name: str
    digest_hour: int
    digest_minute: int
    news_daily_query: str
    max_retrieval_attempts: int
    min_relevance_score: float
    local_top_k: int
    web_top_k: int
    news_days_back: int
    log_level: str
    http_host: str
    http_port: int
    http_api_key: str | None
    max_question_length: int
    max_requests_per_minute: int
    embedding_backend: str
    embedding_model: str

    def __post_init__(self) -> None:
        if not 0 <= self.digest_hour <= 23:
            raise ValueError("DIGEST_HOUR must be between 0 and 23")
        if not 0 <= self.digest_minute <= 59:
            raise ValueError("DIGEST_MINUTE must be between 0 and 59")
        if self.max_retrieval_attempts < 0:
            raise ValueError("MAX_RETRIEVAL_ATTEMPTS must be non-negative")
        if self.local_top_k <= 0 or self.web_top_k <= 0:
            raise ValueError("LOCAL_TOP_K and WEB_TOP_K must be positive")
        if self.max_question_length <= 0:
            raise ValueError("MAX_QUESTION_LENGTH must be positive")
        if self.max_requests_per_minute <= 0:
            raise ValueError("MAX_REQUESTS_PER_MINUTE must be positive")
        if self.embedding_backend not in {"semantic", "hash"}:
            raise ValueError("EMBEDDING_BACKEND must be either 'semantic' or 'hash'")

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_mistral(self) -> bool:
        return bool(self.mistral_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def has_http_api_key(self) -> bool:
        return bool(self.http_api_key)


def load_settings() -> Settings:
    # Load environment values from .env for local runs.
    load_dotenv(override=False)
    return Settings(
        google_api_key=_get_env("GOOGLE_API_KEY"),
        google_model=_get_env("GOOGLE_MODEL", "gemini-2.5-pro") or "gemini-2.5-pro",
        groq_api_key=_get_env("GROQ_API_KEY"),
        groq_model=_get_env("GROQ_MODEL", "llama-3.1-8b-instant")
        or "llama-3.1-8b-instant",
        mistral_api_key=_get_env("MISTRAL_API_KEY"),
        mistral_grader_model=_get_env("MISTRAL_GRADER_MODEL", "mistral-large-latest")
        or "mistral-large-latest",
        mistral_rewriter_model=_get_env(
            "MISTRAL_REWRITER_MODEL", "mistral-large-latest"
        )
        or "mistral-large-latest",
        tavily_api_key=_get_env("TAVILY_API_KEY"),
        telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get_env("TELEGRAM_CHAT_ID"),
        chroma_path=Path(_get_env("CHROMA_PATH", "./data/chroma") or "./data/chroma"),
        chroma_collection_name=_get_env("CHROMA_COLLECTION_NAME", "sentinel_news")
        or "sentinel_news",
        timezone_name=_get_env("NEWS_TIMEZONE", "Asia/Hong_Kong") or "Asia/Hong_Kong",
        digest_hour=int(_get_env("DIGEST_HOUR", "9") or 9),
        digest_minute=int(_get_env("DIGEST_MINUTE", "0") or 0),
        news_daily_query=_get_env(
            "NEWS_DAILY_QUERY", "latest geopolitical developments last 24 hours"
        )
        or "latest geopolitical developments last 24 hours",
        max_retrieval_attempts=int(_get_env("MAX_RETRIEVAL_ATTEMPTS", "2") or 2),
        min_relevance_score=float(_get_env("MIN_RELEVANCE_SCORE", "0.45") or 0.45),
        local_top_k=int(_get_env("LOCAL_TOP_K", "5") or 5),
        web_top_k=int(_get_env("WEB_TOP_K", "5") or 5),
        news_days_back=int(_get_env("NEWS_DAYS_BACK", "1") or 1),
        log_level=_get_env("LOG_LEVEL", "INFO") or "INFO",
        http_host=_get_env("HTTP_HOST", "0.0.0.0") or "0.0.0.0",
        http_port=int(_get_env("HTTP_PORT", "8000") or 8000),
        http_api_key=_get_env("HTTP_API_KEY"),
        max_question_length=int(_get_env("MAX_QUESTION_LENGTH", "1000") or 1000),
        max_requests_per_minute=int(_get_env("MAX_REQUESTS_PER_MINUTE", "20") or 20),
        embedding_backend=_get_env("EMBEDDING_BACKEND", "semantic") or "semantic",
        embedding_model=_get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        or "all-MiniLM-L6-v2",
    )
