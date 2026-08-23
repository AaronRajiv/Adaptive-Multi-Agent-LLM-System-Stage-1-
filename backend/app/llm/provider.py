"""
LLM Provider Implementations for Stage 1.

Includes:
1. GeminiProvider: Live Gemini API integration using standard HTTP requests.
2. MockLLMProvider: Deterministic, schema-compliant mock provider for testing and offline development.
3. Factory function get_llm_provider(): Instantiates the configured provider based on system settings.
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.base import (
    LLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
)
from app.models.schemas import PlannerOutput, EvaluationResult

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM Provider for unit testing and offline development.
    Returns deterministic, valid data without calling external APIs.
    """

    def __init__(self, custom_responses: Optional[Dict[str, Any]] = None):
        self.custom_responses = custom_responses or {}

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Check custom override first
        if "generate" in self.custom_responses:
            return str(self.custom_responses["generate"])

        # Context-based mock text generation
        if "Analyst Agent" in (system_prompt or ""):
            return (
                "### Synthesized Analytical Assessment\n\n"
                "Based on the collected research, here is a structured synthesis:\n"
                "1. **Cost & Financial Factors**: Initial investment varies significantly, with running costs offsetting upfront premiums over time.\n"
                "2. **Operational Practicality**: User suitability depends on daily infrastructure access and maintenance overhead.\n"
                "3. **Long-Term Implications**: Environmental and policy benefits present compelling incentives.\n\n"
                "**Conclusion**: The optimal choice is contingent on specific user constraints and localized infrastructure."
            )
        elif "Synthesizer Agent" in (system_prompt or ""):
            return (
                "### Final Comprehensive Evaluation & Recommendation\n\n"
                "After decomposing the task, executing targeted research subtasks, synthesizing findings, and passing evaluation quality checks, the comprehensive result is as follows:\n\n"
                "- **Key Finding 1**: Upfront acquisition expenses vs operational efficiency require balanced budgeting.\n"
                "- **Key Finding 2**: Practical everyday usability is strongly correlated with available supporting infrastructure.\n"
                "- **Key Finding 3**: Environmental and sustainability considerations favor modern alternatives.\n\n"
                "**Final Guidance**: Tailor the decision to your specific financial horizon and daily operational requirements."
            )
        elif "Research Agent" in (system_prompt or ""):
            return (
                "Parametric Knowledge Findings:\n"
                "- Factual overview indicates significant divergence across key comparison criteria.\n"
                "- Maintenance requirements and operating costs scale proportionally with usage intensity.\n"
                "- Environmental lifecycle analysis highlights distinct trade-offs between manufacturing vs operational phases."
            )
        return "Mock response generated for prompt: " + prompt[:60] + "..."

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        if "structured" in self.custom_responses:
            return schema.model_validate(self.custom_responses["structured"])

        if schema == PlannerOutput:
            mock_data = {
                "tasks": [
                    {
                        "id": "T1",
                        "description": "Analyze acquisition and initial setup costs",
                        "type": "research",
                    },
                    {
                        "id": "T2",
                        "description": "Examine operational, maintenance, and running costs",
                        "type": "research",
                    },
                    {
                        "id": "T3",
                        "description": "Evaluate environmental and sustainability impact",
                        "type": "research",
                    },
                    {
                        "id": "T4",
                        "description": "Synthesize overall feasibility and user suitability",
                        "type": "analysis",
                    },
                ]
            }
            return schema.model_validate(mock_data)

        if schema == EvaluationResult:
            mock_data = {
                "score": 88,
                "status": "PASS",
                "feedback": "The analytical synthesis is comprehensive, well-structured, covers all core dimensions, and directly addresses the user query with clarity.",
            }
            return schema.model_validate(mock_data)

        raise LLMStructuredOutputError(f"MockLLMProvider has no default fixture for schema: {schema.__name__}")


class GeminiProvider(LLMProvider):
    """
    Google Gemini API Provider implementation.
    Calls Google Generative Language REST API directly using httpx.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "Gemini API key is missing. Please set LLM_API_KEY in backend/.env"
            )
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def _clean_json_markdown(self, text: str) -> str:
        """Extract and clean raw JSON from markdown code fences if present."""
        text = text.strip()
        # Look for ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        return text

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System Instructions]: {system_prompt}\n\nPlease acknowledge."}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will strictly follow these instructions."}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
            }
        }

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                if response.status_code == 503 or response.status_code == 429:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                if response.status_code != 200:
                    error_body = response.text
                    try:
                        err_json = response.json()
                        error_body = err_json.get("error", {}).get("message", error_body)
                    except Exception:
                        pass
                    raise LLMProviderError(
                        f"Gemini API error (HTTP {response.status_code}): {error_body}"
                    )

                res_data = response.json()
                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise LLMProviderError("Gemini API returned no candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise LLMProviderError("Gemini API returned an empty response body.")

                # Filter and join all text parts (supporting reasoning/thinking parts)
                text_parts = [p.get("text", "") for p in parts if "text" in p and p.get("text")]
                if not text_parts:
                    raise LLMProviderError("Gemini API candidate contained no text content.")

                return "".join(text_parts).strip()

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMProviderError(f"Network error connecting to Gemini API: {str(e)}")

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        structured_instruction = (
            f"You MUST output ONLY valid JSON matching the following JSON schema.\n"
            f"Do not include any explanation or extra text outside the JSON block.\n\n"
            f"JSON Schema:\n{schema_json}\n\n"
            f"Task Prompt:\n{prompt}"
        )

        full_system_prompt = system_prompt or "You are a precise, structured data generation agent."
        raw_text = await self.generate(structured_instruction, system_prompt=full_system_prompt)

        cleaned_json_text = self._clean_json_markdown(raw_text)

        try:
            parsed_data = json.loads(cleaned_json_text)
            return schema.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as err:
            raise LLMStructuredOutputError(
                f"Failed to parse structured model response as {schema.__name__}: {str(err)}\n"
                f"Raw output received:\n{raw_text}"
            )


def get_llm_provider(config: Settings) -> LLMProvider:
    """
    Factory function to obtain the configured LLMProvider instance.

    Raises:
        LLMConfigurationError: If provider is misconfigured or API key is absent.
    """
    provider_name = config.llm_provider.lower().strip()

    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "gemini":
        if not config.is_api_key_configured:
            raise LLMConfigurationError(
                "Gemini API key is not configured. Please set LLM_API_KEY in backend/.env "
                "or set LLM_PROVIDER=mock for offline development/testing."
            )
        return GeminiProvider(api_key=config.llm_api_key, model=config.llm_model)

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{config.llm_provider}'. Supported in Stage 1: 'gemini', 'mock'."
    )
