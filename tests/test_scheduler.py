from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_news.config.settings import Settings
from rag_news.jobs.scheduler import DigestScheduler


class FakeScheduler:
    def __init__(self) -> None:
        self.running = False
        self.jobs: list[dict[str, object]] = []

    def add_job(self, func, trigger: str, **kwargs) -> None:  # noqa: ANN001
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})

    def start(self) -> None:
        self.running = True


class FakeApp:
    bot = object()


def _settings(tmp_path: Path, *, retention_enabled: bool = True) -> Settings:
    return Settings(
        google_api_key=None,
        google_model="gemini-2.5-pro",
        groq_api_key=None,
        groq_model="llama-3.1-8b-instant",
        mistral_api_key=None,
        mistral_grader_model="mistral-large-latest",
        mistral_rewriter_model="mistral-large-latest",
        tavily_api_key=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        chroma_path=tmp_path / "chroma",
        chroma_collection_name="test_news",
        timezone_name="Asia/Hong_Kong",
        digest_hour=9,
        digest_minute=15,
        news_daily_query="latest geopolitical developments last 24 hours",
        max_retrieval_attempts=2,
        min_relevance_score=0.45,
        local_top_k=5,
        web_top_k=5,
        news_days_back=1,
        news_retention_days=30,
        news_retention_enabled=retention_enabled,
        purge_hour=3,
        purge_minute=30,
        purge_batch_size=250,
        log_level="INFO",
        http_host="0.0.0.0",
        http_port=8000,
        http_api_key=None,
        max_question_length=1000,
        max_requests_per_minute=20,
        embedding_backend="hash",
        embedding_model="all-MiniLM-L6-v2",
        api_timeout_seconds=5.0,
        api_max_retries=3,
        api_backoff_factor=2.0,
        api_jitter_factor=0.1,
        llm_api_timeout_seconds=None,
        llm_api_max_retries=None,
        llm_api_backoff_factor=None,
        llm_api_jitter_factor=None,
        tavily_api_timeout_seconds=None,
        tavily_api_max_retries=None,
        tavily_api_backoff_factor=None,
        tavily_api_jitter_factor=None,
        scheduler_digest_max_retries=3,
        scheduler_digest_backoff_seconds=2.0,
    )


def test_scheduler_registers_purge_job_when_enabled(tmp_path: Path) -> None:
    service = SimpleNamespace(settings=_settings(tmp_path, retention_enabled=True))
    scheduler = DigestScheduler(service=service, application=FakeApp())
    fake_scheduler = FakeScheduler()
    scheduler.scheduler = fake_scheduler  # type: ignore[assignment]

    scheduler.start()

    purge_jobs = [job for job in fake_scheduler.jobs if job.get("id") == "daily_purge"]
    assert len(purge_jobs) == 1
    assert purge_jobs[0]["hour"] == 3
    assert purge_jobs[0]["minute"] == 30


def test_scheduler_skips_purge_job_when_disabled(tmp_path: Path) -> None:
    service = SimpleNamespace(settings=_settings(tmp_path, retention_enabled=False))
    scheduler = DigestScheduler(service=service, application=FakeApp())
    fake_scheduler = FakeScheduler()
    scheduler.scheduler = fake_scheduler  # type: ignore[assignment]

    scheduler.start()

    purge_jobs = [job for job in fake_scheduler.jobs if job.get("id") == "daily_purge"]
    assert not purge_jobs

    digest_jobs = [job for job in fake_scheduler.jobs if job.get("id") == "daily_digest"]
    assert len(digest_jobs) == 1
    assert digest_jobs[0]["hour"] == 9
    assert digest_jobs[0]["minute"] == 15
