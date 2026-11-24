from unittest.mock import MagicMock

import pytest
from google.genai import types

from app.__init__ import ContentGuardrailPlugin, PIIRedactorPlugin, RateLimiterPlugin


@pytest.mark.asyncio
async def test_pii_redactor_plugin():
    plugin = PIIRedactorPlugin()

    # Construct a mock LLM request with PII
    llm_request = MagicMock()
    content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Hi, my email is test@example.com and phone is 123-456-7890."
            )
        ],
    )
    llm_request.contents = [content]

    await plugin.before_model_callback(callback_context=None, llm_request=llm_request)

    # Verify PII is redacted
    redacted_text = llm_request.contents[0].parts[0].text
    assert redacted_text is not None
    assert "[REDACTED_EMAIL]" in redacted_text
    assert "[REDACTED_PHONE]" in redacted_text
    assert "test@example.com" not in redacted_text
    assert "123-456-7890" not in redacted_text


@pytest.mark.asyncio
async def test_content_guardrail_plugin_safe():
    plugin = ContentGuardrailPlugin()

    # Safe request
    llm_request = MagicMock()
    content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="Let's analyze Pareto's sociological framework.")
        ],
    )
    llm_request.contents = [content]

    res = await plugin.before_model_callback(
        callback_context=None, llm_request=llm_request
    )
    # Safe request should return None (allowing it to proceed)
    assert res is None


@pytest.mark.asyncio
async def test_content_guardrail_plugin_toxic():
    plugin = ContentGuardrailPlugin()

    # Toxic request
    llm_request = MagicMock()
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text="We should kill all them immediately!")],
    )
    llm_request.contents = [content]

    res = await plugin.before_model_callback(
        callback_context=None, llm_request=llm_request
    )

    # Toxic request should return a blocked LlmResponse mock response
    assert res is not None
    assert "Safety Alert" in res.content.parts[0].text


@pytest.mark.asyncio
async def test_rate_limiter_plugin():
    from unittest.mock import patch

    # Create a plugin with 2.0 RPM (capacity = 2.0, fill_rate = 2/60 = 1/30 tokens per second)
    plugin = RateLimiterPlugin(requests_per_minute=2.0)
    assert plugin.capacity == 2.0
    assert plugin.fill_rate == pytest.approx(1 / 30)

    llm_request = MagicMock()

    # 1st request should pass immediately and consume 1 token (2.0 -> 1.0)
    res1 = await plugin.before_model_callback(
        callback_context=None, llm_request=llm_request
    )
    assert res1 is None
    assert plugin.tokens == pytest.approx(1.0, abs=1e-3)

    # 2nd request should pass immediately and consume 1 token (1.0 -> 0.0)
    res2 = await plugin.before_model_callback(
        callback_context=None, llm_request=llm_request
    )
    assert res2 is None
    assert plugin.tokens == pytest.approx(0.0, abs=1e-3)

    # 3rd request would need 1 token, but tokens = 0.0.
    # It must sleep for (1.0 - 0.0) / (2.0/60.0) = 30 seconds.
    # We patch asyncio.sleep to verify the delay instantly without sleeping.
    sleep_called_with = []

    async def mock_sleep(delay):
        sleep_called_with.append(delay)
        # Manually advance the limiter state time by mock delay to satisfy calculation
        plugin.last_update -= delay

    with patch("asyncio.sleep", side_effect=mock_sleep):
        res3 = await plugin.before_model_callback(
            callback_context=None, llm_request=llm_request
        )
        assert res3 is None
        assert len(sleep_called_with) == 1
        # Throttler sleep delay must be approximately 30 seconds
        assert sleep_called_with[0] == pytest.approx(30.0, abs=1e-2)
