from __future__ import annotations

import argparse
import asyncio

from rag_news.core.digest import format_digest
from rag_news.core.service import build_digest, get_service
from rag_news.config.logging_config import configure_logging
from rag_news.config.settings import load_settings
from rag_news.interfaces.telegram_bot import TelegramNewsBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-news")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("api", help="Run the FastAPI app")
    subparsers.add_parser("worker", help="Run the Telegram bot and digest scheduler")
    subparsers.add_parser(
        "digest", help="Generate a single digest and print it to stdout"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()
    configure_logging(settings.log_level)

    if args.command == "api":
        from uvicorn import run

        run(
            "rag_news.interfaces.api:app",
            host=settings.http_host,
            port=settings.http_port,
            reload=False,
        )
        return

    if args.command == "worker":
        service = get_service()
        TelegramNewsBot(service).run()
        return

    if args.command == "digest":
        result = asyncio.run(build_digest())
        print(format_digest(result))
        return


if __name__ == "__main__":
    main()
