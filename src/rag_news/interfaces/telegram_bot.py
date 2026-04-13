from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag_news.core.digest import format_answer
from rag_news.core.rate_limiter import SlidingWindowRateLimiter
from rag_news.core.service import ServiceBundle
from rag_news.jobs.scheduler import DigestScheduler


logger = getLogger(__name__)


@dataclass(slots=True)
class TelegramNewsBot:
    service: ServiceBundle
    application: Application = field(init=False)
    scheduler: DigestScheduler = field(init=False)
    rate_limiter: SlidingWindowRateLimiter = field(init=False)

    def __post_init__(self) -> None:
        if not self.service.settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required to start the bot")
        self.application = (
            Application.builder()
            .token(self.service.settings.telegram_bot_token)
            .post_init(self._post_init)
            .build()
        )
        self.scheduler = DigestScheduler(self.service, self.application)
        self.rate_limiter = SlidingWindowRateLimiter(
            max_requests=self.service.settings.max_requests_per_minute,
            window_seconds=60,
        )
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("help", self._help))
        self.application.add_handler(CommandHandler("ask", self._ask))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._ask_from_text)
        )
        self.application.add_error_handler(self._on_error)

    def run(self) -> None:
        self.application.run_polling()

    async def _post_init(self, application: Application) -> None:
        logger.info("Starting digest scheduler")
        self.scheduler.start()

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Sentinel-RAG is ready. Send /ask <question> or just type a query about recent news."
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Use /ask to search recent news. The bot first checks local ChromaDB and then falls back to Tavily."
        )

    async def _ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        question = " ".join(context.args).strip()
        if not question:
            await update.effective_message.reply_text("Usage: /ask your question here")
            return
        await self._respond(update, question)

    async def _ask_from_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        text = (update.effective_message.text or "").strip()
        if not text:
            return
        await self._respond(update, text)

    async def _respond(self, update: Update, question: str) -> None:
        message = update.effective_message
        if message is None:
            return

        if len(question) > self.service.settings.max_question_length:
            await message.reply_text(
                f"Question is too long. Maximum length is {self.service.settings.max_question_length} characters."
            )
            return

        user_id = update.effective_user.id if update.effective_user else "anonymous"
        if not self.rate_limiter.allow(user_id):
            await message.reply_text(
                "Rate limit exceeded. Please try again in a minute."
            )
            return

        try:
            result = await self.service.graph.answer_question(question)
        except Exception as exc:  # pragma: no cover - network/provider failures
            logger.error("Failed to answer Telegram question: %s", type(exc).__name__)
            await message.reply_text(
                "I hit a temporary error while processing that request. Please try again in a moment."
            )
            return

        await message.reply_text(
            format_answer(result, rich_text=True),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    async def _on_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error(
            "Unhandled Telegram update error: %s", type(context.error).__name__
        )
        if isinstance(update, Update) and update.effective_message is not None:
            await update.effective_message.reply_text(
                "I hit a temporary error while processing that request. Please try again in a moment."
            )
