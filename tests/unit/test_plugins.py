from unittest.mock import MagicMock

import pytest
from google.genai import types

from app.__init__ import ContentGuardrailPlugin, PIIRedactorPlugin


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
