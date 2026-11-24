import asyncio
import logging
import os
import re
import time

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .agent import app

logger = logging.getLogger("discourse_anal_plugins")


class PIIRedactorPlugin(BasePlugin):
    """Automatically redacts PII (emails, phone numbers) from LLM request contents to ensure safety."""

    def __init__(self):
        super().__init__(name="pii_redactor")

    async def before_model_callback(self, *, callback_context, llm_request):
        if not llm_request.contents:
            return None

        email_regex = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        phone_regex = re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )

        modified_contents = []
        for content in llm_request.contents:
            if not content.parts:
                modified_contents.append(content)
                continue

            modified_parts = []
            for part in content.parts:
                if part.text:
                    redacted_text = email_regex.sub("[REDACTED_EMAIL]", part.text)
                    redacted_text = phone_regex.sub("[REDACTED_PHONE]", redacted_text)
                    modified_parts.append(types.Part.from_text(text=redacted_text))
                else:
                    modified_parts.append(part)

            modified_contents.append(
                types.Content(role=content.role, parts=modified_parts)
            )

        llm_request.contents = modified_contents
        return None


class ContentGuardrailPlugin(BasePlugin):
    """Checks for extreme toxicity, hate speech, or prompt injections in the input."""

    def __init__(self):
        super().__init__(name="content_guardrail")

    async def before_model_callback(self, *, callback_context, llm_request):
        if not llm_request.contents:
            return None

        # We check for basic toxicity and system prompt overrides
        toxic_patterns = [
            re.compile(
                r"\b(kill|murder|hang|assassinate|lynch)\b.*\b(all|them|him|her)\b",
                re.IGNORECASE,
            ),
            re.compile(r"ignore previous instructions", re.IGNORECASE),
            re.compile(r"you must now act as", re.IGNORECASE),
        ]

        for content in llm_request.contents:
            if not content.parts:
                continue
            for part in content.parts:
                if part.text:
                    for pattern in toxic_patterns:
                        if pattern.search(part.text):
                            logger.warning(f"Guardrail triggered: {pattern.pattern}")
                            # Block and return a mock response
                            blocked_content = types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text="Safety Alert: This request has been blocked because it violates the system's content safety policy or contains prompt injection attempts."
                                    )
                                ],
                            )
                            return LlmResponse(
                                model_version="guardrail",
                                content=blocked_content,
                                finish_reason="SAFETY",
                            )
        return None


class RateLimiterPlugin(BasePlugin):
    """Proactively throttles requests to Gemini to prevent hitting API rate limits using a Token Bucket."""

    def __init__(self, requests_per_minute: float = 15.0):
        super().__init__(name="rate_limiter")
        self.capacity = requests_per_minute
        self.fill_rate = requests_per_minute / 60.0  # tokens per second
        self.tokens = requests_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def before_model_callback(self, *, callback_context, llm_request):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)

            if self.tokens < 1.0:
                # Calculate time to wait until we have at least 1 token
                wait_time = (1.0 - self.tokens) / self.fill_rate
                logger.info(
                    f"RateLimiterPlugin: Throttling model request. Sleeping for {wait_time:.2f} seconds..."
                )
                await asyncio.sleep(wait_time)

                # Update tokens after sleeping
                now_after_sleep = time.monotonic()
                elapsed_sleep = now_after_sleep - self.last_update
                self.last_update = now_after_sleep
                self.tokens = min(
                    self.capacity, self.tokens + elapsed_sleep * self.fill_rate
                )

            # Consume a token
            self.tokens -= 1.0
            logger.info(
                f"RateLimiterPlugin: Request allowed. Tokens remaining: {self.tokens:.2f}/{self.capacity}"
            )
            return None


# Load RPM limit from environment (default is 15 RPM for free tier)
gemini_rpm_limit = float(os.getenv("GEMINI_RPM_LIMIT", "15"))

# Register plugins on the app
app.plugins.append(PIIRedactorPlugin())
app.plugins.append(ContentGuardrailPlugin())
app.plugins.append(RateLimiterPlugin(requests_per_minute=gemini_rpm_limit))

__all__ = ["app"]
