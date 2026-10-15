"""
Regression test for NVIDIA Nemotron reasoning content separation.
Ensures that intermediate reasoning traces/planning text are never exposed as user-facing answers.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.llm.provider import OpenAICompatibleProvider
from app.llm.base import LLMProviderError


@pytest.mark.asyncio
async def test_nemotron_discards_reasoning_content_when_final_content_present():
    """Verify provider returns only 'content', never leaking 'reasoning_content'."""
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    mock_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "### Comparison\nElectric vehicles have lower operational emissions.",
                        "reasoning_content": "We need to produce a response to compare electric and petrol vehicles... We have evidence from T1...",
                    },
                }
            ],
            "usage": {"total_tokens": 100},
        },
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await provider.generate("Compare electric and petrol vehicles.")

        # Ensure enable_thinking=False was passed in payload for Nemotron
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("chat_template_kwargs") == {"enable_thinking": False}

        # Ensure returned output is strictly final content without reasoning text
        assert "We need to produce a response" not in result
        assert "We have evidence from T1" not in result
        assert result == "### Comparison\nElectric vehicles have lower operational emissions."


@pytest.mark.asyncio
async def test_nemotron_rejects_fallback_to_reasoning_content_on_null_content():
    """Verify provider raises error rather than substituting reasoning_content when content is null."""
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    mock_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "We need to produce a response... [planning notes truncated]",
                    },
                }
            ],
            "usage": {"total_tokens": 100},
        },
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(LLMProviderError, match="Provider returned null or empty content"):
            await provider.generate("Compare electric and petrol vehicles.")
