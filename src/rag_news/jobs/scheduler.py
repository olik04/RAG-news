from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from logging import getLogger

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.constants import ParseMode

from rag_news.core.digest import format_digest
from rag_news.core.service import ServiceBundle


logger = getLogger(__name__)


@dataclass(slots=True)
class DigestScheduler:
    service: ServiceBundle
    application: object
    scheduler: AsyncIOScheduler = field(init=False)

    def __post_init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=self.service.settings.timezone)

    def start(self) -> None:
        if self.scheduler.running:
            logger.info("Digest scheduler already running; skipping duplicate start")
            return

        self.scheduler.add_job(
            self._send_digest,
            trigger="cron",
            hour=self.service.settings.digest_hour,
            minute=self.service.settings.digest_minute,
            id="daily_digest",
            replace_existing=True,
        )
        if self.service.settings.news_retention_enabled:
            self.scheduler.add_job(
                self._purge_stale_documents,
                trigger="cron",
                hour=self.service.settings.purge_hour,
                minute=self.service.settings.purge_minute,
                id="daily_purge",
                replace_existing=True,
            )
        self.scheduler.start()

    async def _send_digest(self) -> None:
        if not self.service.settings.has_telegram:
            logger.warning("Telegram is not configured; skipping scheduled digest")
            return

        attempts = self.service.settings.scheduler_digest_max_retries
        for attempt in range(1, attempts + 1):
            try:
                result = await self.service.graph.build_digest(
                    self.service.settings.news_daily_query
                )
                message = format_digest(result, rich_text=True)
                await self.application.bot.send_message(
                    chat_id=self.service.settings.telegram_chat_id,
                    text=message,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                return
            except Exception as exc:  # pragma: no cover - provider/network failures
                logger.warning(
                    "Scheduled digest attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    type(exc).__name__,
                )
                if attempt < attempts:
                    await asyncio.sleep(
                        self.service.settings.scheduler_digest_backoff_seconds * attempt
                    )

        logger.error("Scheduled digest failed after %s attempts", attempts)

    async def _purge_stale_documents(self) -> None:
        """Purge documents older than the retention period."""
        try:
            result = await self.service.repository.purge_stale_documents()
            logger.info(
                "Scheduled document purge completed: deleted=%d, remaining=%d",
                result["deleted_count"],
                result["remaining_count"],
            )
        except Exception as exc:  # pragma: no cover - persistence failures
            logger.error("Scheduled document purge failed: %s", type(exc).__name__)
