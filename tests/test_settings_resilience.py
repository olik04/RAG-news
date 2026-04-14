from __future__ import annotations

from pathlib import Path

from rag_news.config.settings import Settings


def _base_settings(tmp_path: Path) -> dict[str, object]:
    return {
        "google_api_key": None,
        "google_model": "gemini-2.5-pro",
        "groq_api_key": None,
        "groq_model": "llama-3.1-8b-instant",
        "mistral_api_key": None,
        "mistral_grader_model": "mistral-large-latest",
        "mistral_rewriter_model": "mistral-large-latest",
        "tavily_api_key": None,
        "telegram_bot_token": None,
        "telegram_chat_id": None,
        "chroma_path": tmp_path / "chroma",
        "chroma_collection_name": "test_news",
        "timezone_name": "Asia/Hong_Kong",
        "digest_hour": 9,
        "digest_minute": 0,
        "news_daily_query": "latest geopolitical developments last 24 hours",
        "max_retrieval_attempts": 2,
        "min_relevance_score": 0.45,
        "local_top_k": 5,
        "web_top_k": 5,
        "news_days_back": 1,
        "news_retention_days": 30,
        "news_retention_enabled": True,
        "purge_hour": 2,
        "purge_minute": 0,
        "purge_batch_size": 500,
        "log_level": "INFO",
        "http_host": "0.0.0.0",
        "http_port": 8000,
        "http_api_key": None,
        "max_question_length": 1000,
        "max_requests_per_minute": 20,
        "embedding_backend": "hash",
        "embedding_model": "all-MiniLM-L6-v2",
        "api_timeout_seconds": 5.0,
        "api_max_retries": 3,
        "api_backoff_factor": 2.0,
        "api_jitter_factor": 0.1,
        "llm_api_timeout_seconds": None,
        "llm_api_max_retries": None,
        "llm_api_backoff_factor": None,
        "llm_api_jitter_factor": None,
        "tavily_api_timeout_seconds": None,
        "tavily_api_max_retries": None,
        "tavily_api_backoff_factor": None,
        "tavily_api_jitter_factor": None,
        "scheduler_digest_max_retries": 3,
        "scheduler_digest_backoff_seconds": 2.0,
    }


def test_resilience_overrides_fall_back_to_api_defaults(tmp_path: Path) -> None:
    settings = Settings(**_base_settings(tmp_path))

    assert settings.llm_resilience_timeout_seconds == settings.api_timeout_seconds
    assert settings.llm_resilience_max_retries == settings.api_max_retries
    assert settings.llm_resilience_backoff_factor == settings.api_backoff_factor
    assert settings.llm_resilience_jitter_factor == settings.api_jitter_factor

    assert settings.tavily_resilience_timeout_seconds == settings.api_timeout_seconds
    assert settings.tavily_resilience_max_retries == settings.api_max_retries
    assert settings.tavily_resilience_backoff_factor == settings.api_backoff_factor
    assert settings.tavily_resilience_jitter_factor == settings.api_jitter_factor


def test_resilience_overrides_can_be_customized(tmp_path: Path) -> None:
    values = _base_settings(tmp_path)
    values.update(
        {
            "llm_api_timeout_seconds": 8.0,
            "llm_api_max_retries": 1,
            "llm_api_backoff_factor": 1.5,
            "llm_api_jitter_factor": 0.2,
            "tavily_api_timeout_seconds": 3.5,
            "tavily_api_max_retries": 4,
            "tavily_api_backoff_factor": 2.5,
            "tavily_api_jitter_factor": 0.05,
        }
    )

    settings = Settings(**values)

    assert settings.llm_resilience_timeout_seconds == 8.0
    assert settings.llm_resilience_max_retries == 1
    assert settings.llm_resilience_backoff_factor == 1.5
    assert settings.llm_resilience_jitter_factor == 0.2

    assert settings.tavily_resilience_timeout_seconds == 3.5
    assert settings.tavily_resilience_max_retries == 4
    assert settings.tavily_resilience_backoff_factor == 2.5
    assert settings.tavily_resilience_jitter_factor == 0.05
