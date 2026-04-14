from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from typing import Any

from google import genai
from openai import AsyncOpenAI
from rag_news.core.llm_components.telemetry import log_llm_event
from rag_news.core.resilience import ResilienceConfig, with_timeout_and_retry


logger = logging.getLogger(__name__)


class OpenAIJsonProviderClient:
    def __init__(self, client: AsyncOpenAI | None, resilience_config: ResilienceConfig | None = None) -> None:
        self.client = client
        self.resilience_config = resilience_config or ResilienceConfig()

    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str = "provider_chat_json",
        request_id: str | None = None,
        doc_count: int | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            log_llm_event(
                logger,
                operation=operation,
                provider="openai",
                model=model,
                outcome="fallback",
                duration_ms=0,
                reason="client_unavailable",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        start = perf_counter()
        log_llm_event(
            logger,
            operation=operation,
            provider="openai",
            model=model,
            outcome="attempt",
            duration_ms=0,
            request_id=request_id,
            doc_count=doc_count,
        )
        try:
            response = await with_timeout_and_retry(
                f"{operation}_openai",
                self.resilience_config,
                lambda: self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                ),
            )
        except Exception:  # pragma: no cover
            log_llm_event(
                logger,
                operation=operation,
                provider="openai",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="request_failed",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        content = response.choices[0].message.content if response.choices else None
        if not content:
            log_llm_event(
                logger,
                operation=operation,
                provider="openai",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="empty_response",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            log_llm_event(
                logger,
                operation=operation,
                provider="openai",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="invalid_json",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}
        log_llm_event(
            logger,
            operation=operation,
            provider="openai",
            model=model,
            outcome="success",
            duration_ms=int((perf_counter() - start) * 1000),
            request_id=request_id,
            doc_count=doc_count,
        )
        return parsed


class GoogleJsonProviderClient:
    def __init__(self, client: genai.Client | None, resilience_config: ResilienceConfig | None = None) -> None:
        self.client = client
        self.resilience_config = resilience_config or ResilienceConfig()

    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str = "provider_chat_json",
        request_id: str | None = None,
        doc_count: int | None = None,
    ) -> dict[str, Any]:
        if self.client is None:
            log_llm_event(
                logger,
                operation=operation,
                provider="google",
                model=model,
                outcome="fallback",
                duration_ms=0,
                reason="client_unavailable",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        start = perf_counter()
        log_llm_event(
            logger,
            operation=operation,
            provider="google",
            model=model,
            outcome="attempt",
            duration_ms=0,
            request_id=request_id,
            doc_count=doc_count,
        )
        try:
            response = await with_timeout_and_retry(
                f"{operation}_google",
                self.resilience_config,
                lambda: asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        system_instruction=system_prompt,
                    ),
                ),
            )
        except Exception:  # pragma: no cover
            log_llm_event(
                logger,
                operation=operation,
                provider="google",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="request_failed",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        text = getattr(response, "text", None) or ""
        if not text:
            log_llm_event(
                logger,
                operation=operation,
                provider="google",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="empty_response",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log_llm_event(
                logger,
                operation=operation,
                provider="google",
                model=model,
                outcome="failure",
                duration_ms=int((perf_counter() - start) * 1000),
                reason="invalid_json",
                request_id=request_id,
                doc_count=doc_count,
            )
            return {}
        log_llm_event(
            logger,
            operation=operation,
            provider="google",
            model=model,
            outcome="success",
            duration_ms=int((perf_counter() - start) * 1000),
            request_id=request_id,
            doc_count=doc_count,
        )
        return parsed
