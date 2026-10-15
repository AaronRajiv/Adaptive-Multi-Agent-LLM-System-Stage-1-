"""
LLM Provider Implementations for Stage 1.

Includes:
1. GeminiProvider: Live Gemini API integration using standard HTTP requests.
2. MockLLMProvider: Deterministic, schema-compliant mock provider for testing and offline development.
3. Factory function get_llm_provider(): Instantiates the configured provider based on system settings.
"""

import asyncio
import json
import random
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
from app.logging_config import get_logger
from app.models.schemas import PlannerOutput, EvaluationResult

logger = get_logger("app.llm.provider")
T = TypeVar("T", bound=BaseModel)


def pydantic_to_gemini_schema(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Pydantic V2 model JSON schema ($defs, anyOf, etc.) into
    a clean, inlined OpenAPI 3.0 schema compatible with Gemini REST API responseSchema.
    """
    defs = schema_dict.get("$defs", {})

    def resolve(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"].split("/")[-1]
            if ref_path in defs:
                return resolve(dict(defs[ref_path]))
        if "anyOf" in node:
            # Pick non-null variant
            non_null = [s for s in node["anyOf"] if s.get("type") != "null"]
            if non_null:
                return resolve(non_null[0])
            return resolve(node["anyOf"][0])

        res: Dict[str, Any] = {}
        for k, v in node.items():
            if k in ("$defs", "$ref", "title", "default", "minItems", "additionalProperties"):
                continue
            if k == "type":
                res["type"] = str(v).upper()
            elif k == "properties" and isinstance(v, dict):
                res["properties"] = {pk: resolve(pv) for pk, pv in v.items()}
            elif k == "items":
                res["items"] = resolve(v)
            elif k in ("required", "description", "enum"):
                res[k] = v
        return res

    return resolve(dict(schema_dict))


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

        sys_str = (system_prompt or "").lower()
        if "analyst" in sys_str:
            return (
                "### Synthesized Analytical Assessment\n\n"
                "Based on the collected research and evidence ledger, here is a structured synthesis:\n"
                "1. **Cost & Financial Factors**: Initial investment varies significantly, with running costs offsetting upfront premiums over time [FIA 2025, art 3.4.1, p.24].\n"
                "2. **Operational Practicality**: User suitability depends on daily infrastructure access and maintenance overhead [FIA 2025, art 10.2, p.112].\n"
                "3. **Cross-System Trade-offs**: Aerodynamic load limits directly interact with suspension stiffness [FIA 2025, art 3.10, p.45] and [FIA 2025, art 10.1, p.110].\n\n"
                "**Conclusion**: The optimal choice is contingent on specific user constraints and localized infrastructure."
            )
        elif "synthesizer" in sys_str:
            return (
                "### Explicit Regulatory Requirements\n"
                "- Article 3.4.1 prescribes the overall dimensional envelope boundaries [FIA 2025, art 3.4.1, p.24].\n"
                "- Article 4.1 establishes the minimum mass specification of 798 kg [FIA 2025, art 4.1, p.38].\n\n"
                "### Engineering Interpretation / Inference\n"
                "- Aerodynamic balance shifts dynamically under braking, requiring balanced mechanical suspension compensation [INFERENCE].\n"
                "- Packaging constraints in the sidepod area influence cooling efficiency and airflow toward the diffuser [INFERENCE].\n\n"
                "### Limitations and Unknowns\n"
                "- Specific aerodynamic elasticity deflection measurements outside standard test load conditions are not detailed in the available text."
            )
        elif "research" in sys_str:
            return (
                "Grounding Research Findings:\n"
                "- [EXPLICIT] Dimensional requirements mandate strict envelope boundaries [FIA 2025, art 3.4.1, p.24].\n"
                "- [INFERENCE] Reducing front wing chord shifts aerodynamic balance rearward [FIA 2025, art 3.10, p.45].\n"
                "- [NOT_IN_SOURCE] Standard track setup procedures involve adjusting roll bar stiffness accordingly."
            )
        return "Mock response generated for prompt: " + prompt[:60] + "..."

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        if "structured" in self.custom_responses:
            return schema.model_validate(self.custom_responses["structured"])

        if schema == PlannerOutput:
            has_rag = any(
                term in prompt.lower()
                for term in [
                    "document", "knowledge base", "retrieval", "pdf", "rules", "regulations",
                    "article", "fia", "report", "solar", "sources", "chunks", "attached",
                ]
            )
            mock_data = {
                "tasks": [
                    {
                        "id": "T1",
                        "description": "Analyze acquisition and initial setup costs",
                        "type": "research",
                        **({"metadata": {
                            "domain": "financial",
                            "retrieval_queries": ["initial acquisition costs capital expense", "setup fees"],
                            "retrieval_query": "initial acquisition costs capital expense",
                        }} if has_rag else {}),
                    },
                    {
                        "id": "T2",
                        "description": "Examine operational, maintenance, and running costs",
                        "type": "research",
                        **({"metadata": {
                            "domain": "operations",
                            "retrieval_queries": ["operational maintenance schedules", "running costs"],
                            "retrieval_query": "operational maintenance schedules",
                        }} if has_rag else {}),
                    },
                    {
                        "id": "T3",
                        "description": "Evaluate environmental and sustainability impact",
                        "type": "research",
                        **({"metadata": {
                            "domain": "environmental",
                            "retrieval_queries": ["environmental lifecycle emissions", "carbon footprint"],
                            "retrieval_query": "environmental lifecycle emissions",
                        }} if has_rag else {}),
                    },
                    {
                        "id": "T4",
                        "description": "Cross-system interaction: Evaluate financial vs operational trade-offs",
                        "type": "analysis",
                        "dependencies": ["T1", "T2"],
                    },
                    {
                        "id": "T5",
                        "description": "Synthesize overall feasibility and user suitability",
                        "type": "analysis",
                        "dependencies": ["T3", "T4"],
                    },
                ]
            }
            return schema.model_validate(mock_data)

        if schema == EvaluationResult:
            mock_data = {
                "score": 88,
                "status": "PASS",
                "feedback": "The analytical synthesis is comprehensive, well-grounded in cited evidence, and directly addresses the user query with clarity.",
                "groundedness_score": 23,
                "coverage_score": 22,
                "distinction_score": 22,
                "honesty_score": 21,
                "uncovered_domains": [],
            }
            return schema.model_validate(mock_data)

        if schema.__name__ == "StrategyDecision":
            from app.orchestration.strategy import ExecutionStrategy
            cleaned = prompt.lower()
            strat = ExecutionStrategy.MULTI_AGENT
            req_rag = False
            # Check user task extracted from prompt
            task_match = re.search(r'user task:\s*"(.*?)"', cleaned, re.DOTALL)
            task_content = task_match.group(1).strip() if task_match else cleaned
            if re.search(r"\b(hello|hi|hey|greetings|howdy)\b", task_content) and len(task_content) < 80:
                strat = ExecutionStrategy.DIRECT
            elif re.search(r"\b(draft|write|function|python|code|calc|calculate|math)\b", task_content) and not re.search(r"\b(compare|lifecycle|tradeoff|versus|vs)\b", task_content):
                strat = ExecutionStrategy.SINGLE_AGENT
            if "attached documents present: true" in cleaned or "knowledge base" in cleaned:
                if any(w in task_content for w in ["document", "file", "pdf", "solar", "knowledge base"]):
                    req_rag = True
            has_cross_doc = bool(re.search(r"\b(compare|cross-reference|between|tradeoff|versus|vs)\b", task_content))
            domains_found = [w for w in ["aerodynamics", "mass", "suspension", "solar", "cost", "financial", "regulations"] if w in task_content]
            mock_data = {
                "strategy": strat.value,
                "reasoning": "Mock semantic strategy selection for testing.",
                "estimated_complexity": "HIGH" if strat == ExecutionStrategy.MULTI_AGENT else "LOW",
                "requires_rag": req_rag,
                "domains": domains_found,
                "cross_document_reasoning": has_cross_doc,
            }
            return schema.model_validate(mock_data)

        raise LLMStructuredOutputError(f"MockLLMProvider has no default fixture for schema: {schema.__name__}")


class GeminiProvider(LLMProvider):
    """
    Google Gemini API Provider implementation.
    Calls Google Generative Language REST API directly using httpx.
    Supports native systemInstruction, responseMimeType, and responseSchema.
    """

    # Global concurrency semaphore to prevent thundering herd and RPM exhaustion
    _semaphore = asyncio.Semaphore(2)

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
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        return text

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/{self.model}:generateContent"

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        max_retries = 5
        backoff = 2.5

        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    # Minimal pacing to protect free-tier rate limits
                    await asyncio.sleep(0.2)
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "x-goog-api-key": self.api_key,
                            },
                        )

                if response.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                        if delay is None:
                            match = re.search(r"retry in ([\d\.]+)s", response.text, re.IGNORECASE)
                            if match:
                                try:
                                    delay = min(float(match.group(1)), 30.0)
                                except ValueError:
                                    delay = None
                        delay = delay if delay is not None else backoff
                        # Add random jitter to prevent concurrent tasks from retrying in lockstep
                        delay += random.uniform(0.5, 2.5)
                        logger.warning(
                            "Gemini API rate-limit or high demand (HTTP %s). Backing off for %.1fs (attempt %d/%d).",
                            response.status_code,
                            delay,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(delay)
                        backoff = min(backoff * 2.0, 30.0)
                        continue

                if response.status_code != 200:
                    error_body = response.text
                    try:
                        err_json = response.json()
                        error_body = err_json.get("error", {}).get("message", error_body)
                    except Exception:
                        pass
                    logger.error("Gemini API call failed (HTTP %s): %s", response.status_code, error_body)
                    raise LLMProviderError(
                        f"Gemini API error (HTTP {response.status_code}): {error_body}"
                    )

                res_data = response.json()
                self.last_usage = res_data.get("usageMetadata", {})
                candidates = res_data.get("candidates", [])
                if not candidates:
                    logger.error("Gemini API returned no candidates.")
                    raise LLMProviderError("Gemini API returned no candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    logger.error("Gemini API returned empty response parts.")
                    raise LLMProviderError("Gemini API returned an empty response body.")

                text_parts = [p.get("text", "") for p in parts if "text" in p and p.get("text")]
                if not text_parts:
                    logger.error("Gemini API candidate contained no text content.")
                    raise LLMProviderError("Gemini API candidate contained no text content.")

                return "".join(text_parts).strip()

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    logger.warning("Gemini network error: %s. Retrying in %.1fs.", str(e), backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                logger.error("Gemini network error after %d attempts: %s", max_retries, str(e))
                raise LLMProviderError(f"Network error connecting to Gemini API: {str(e)}")

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        """
        Generate structured output using native Gemini responseMimeType='application/json'
        and responseSchema, with fallback parsing if needed.
        """
        url = f"{self.base_url}/{self.model}:generateContent"

        gemini_schema = None
        try:
            gemini_schema = pydantic_to_gemini_schema(schema.model_json_schema())
        except Exception as exc:
            logger.warning("Failed to convert Pydantic schema to Gemini responseSchema: %s", exc)

        generation_config: Dict[str, Any] = {
            "temperature": 0.1,
            "topP": 0.95,
            "responseMimeType": "application/json",
        }
        if gemini_schema:
            generation_config["responseSchema"] = gemini_schema

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        max_retries = 5
        backoff = 2.5
        raw_text = ""

        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    # Minimal pacing to protect free-tier rate limits
                    await asyncio.sleep(0.2)
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "x-goog-api-key": self.api_key,
                            },
                        )

                # If schema rejected with 400 (e.g. unsupported schema construct), retry without responseSchema
                if response.status_code == 400 and "responseSchema" in payload.get("generationConfig", {}):
                    logger.warning("Gemini rejected responseSchema (HTTP 400). Retrying without responseSchema...")
                    payload["generationConfig"].pop("responseSchema", None)
                    schema_json = json.dumps(schema.model_json_schema(), indent=2)
                    payload["contents"][0]["parts"][0]["text"] = (
                        f"You MUST output ONLY valid JSON matching this schema:\n{schema_json}\n\nTask:\n{prompt}"
                    )
                    continue

                if response.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                        if delay is None:
                            match = re.search(r"retry in ([\d\.]+)s", response.text, re.IGNORECASE)
                            if match:
                                try:
                                    delay = min(float(match.group(1)), 30.0)
                                except ValueError:
                                    delay = None
                        delay = delay if delay is not None else backoff
                        # Add random jitter to prevent concurrent tasks from retrying in lockstep
                        delay += random.uniform(0.5, 2.5)
                        logger.warning(
                            "Gemini API rate-limit or high demand (HTTP %s). Backing off for %.1fs (attempt %d/%d).",
                            response.status_code,
                            delay,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(delay)
                        backoff = min(backoff * 2.0, 30.0)
                        continue

                if response.status_code != 200:
                    err_msg = response.text
                    try:
                        err_msg = response.json().get("error", {}).get("message", err_msg)
                    except Exception:
                        pass
                    raise LLMProviderError(f"Gemini API structured call failed (HTTP {response.status_code}): {err_msg}")

                res_data = response.json()
                self.last_usage = res_data.get("usageMetadata", {})
                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise LLMProviderError("Gemini API returned no candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if "text" in p and p.get("text")]
                raw_text = "".join(text_parts).strip()
                break

            except httpx.RequestError as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMProviderError(f"Network error during structured call: {exc}")

        cleaned = self._clean_json_markdown(raw_text)
        try:
            parsed = json.loads(cleaned)
            return schema.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as err:
            raise LLMStructuredOutputError(
                f"Failed to parse structured model response as {schema.__name__}: {str(err)}\n"
                f"Raw output received:\n{raw_text}"
            )


class OpenAICompatibleProvider(LLMProvider):
    """
    OpenAI-compatible LLM Provider implementation.
    Supports NVIDIA NIM, OpenAI, vLLM, and any standard OpenAI /chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nemotron-3-super-120b-a12b",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 180.0,
        max_tokens: int = 4096,
    ):
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "API key is missing for OpenAI-compatible provider. Please set NVIDIA_API_KEY in backend/.env"
            )
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_usage: Dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(4)

    def _clean_json_markdown(self, text: str) -> str:
        """Extract and clean raw JSON from markdown code fences if present."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        return text

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        effective_max_tokens = min(self.max_tokens, 2048)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": effective_max_tokens,
            "stream": False,
        }

        # For NVIDIA Nemotron models, pass chat_template_kwargs={"enable_thinking": False}
        # to ensure direct assistant answers in choices[0].message.content rather than wasting
        # token budget on internal reasoning traces that can exhaust max_tokens or leak scratchpads.
        if "nemotron" in self.model.lower() or "nvidia" in self.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                        )

                if response.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff
                        delay += random.uniform(0.5, 1.5)
                        logger.warning(
                            "Provider API rate limit or high demand (HTTP %s). Retrying in %.1fs (attempt %d/%d).",
                            response.status_code, delay, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(delay)
                        backoff *= 2
                        continue

                if response.status_code != 200:
                    err_msg = response.text
                    try:
                        err_msg = response.json().get("error", {}).get("message", err_msg)
                    except Exception:
                        pass
                    raise LLMProviderError(f"Provider API error (HTTP {response.status_code}): {err_msg}")

                res_data = response.json()
                self.last_usage = res_data.get("usage", {})
                choices = res_data.get("choices", [])
                if not choices:
                    raise LLMProviderError("Provider returned no choices in response.")

                msg = choices[0].get("message", {})
                content = msg.get("content")
                # Never extract or substitute internal reasoning_content as user-facing content.
                if not content:
                    finish_reason = choices[0].get("finish_reason")
                    raise LLMProviderError(
                        f"Provider returned null or empty content (finish_reason: {finish_reason})."
                    )
                return str(content).strip()

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMProviderError(f"Network error connecting to provider: {e}")

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        url = f"{self.base_url}/chat/completions"
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instruction = (
            f"You MUST output ONLY valid JSON matching this schema:\n{schema_json}\n"
            f"Do not output markdown code fences, pre-text, chain-of-thought, or commentary. "
            f"Return pure JSON starting with '{{' and ending with '}}' only."
        )
        if system_prompt:
            system_instruction = f"{system_prompt}\n\n{system_instruction}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": min(self.max_tokens, 2048),
            "stream": False,
        }

        # For NVIDIA Nemotron models, disable unconstrained reasoning on structured JSON calls
        # so that completion tokens are directly dedicated to valid JSON output.
        if "nemotron" in self.model.lower() or "nvidia" in self.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        max_retries = 3
        backoff = 2.0
        raw_text = ""

        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                        )

                if response.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff + random.uniform(0.5, 1.5))
                        backoff *= 2
                        continue

                # Fallback if json_object mode not accepted
                if response.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue

                if response.status_code != 200:
                    err_msg = response.text
                    try:
                        err_msg = response.json().get("error", {}).get("message", err_msg)
                    except Exception:
                        pass
                    raise LLMProviderError(f"Provider API structured error (HTTP {response.status_code}): {err_msg}")

                res_data = response.json()
                self.last_usage = res_data.get("usage", {})
                choices = res_data.get("choices", [])
                if not choices:
                    raise LLMProviderError("Provider returned no choices in response.")

                msg = choices[0].get("message", {})
                content = msg.get("content")
                raw_text = str(content or "").strip()

                # If content is empty or lacks JSON curly braces, check if reasoning_content has an embedded JSON object
                if not raw_text or "{" not in raw_text or "}" not in raw_text:
                    reasoning = msg.get("reasoning_content")
                    if reasoning and "{" in reasoning and "}" in reasoning:
                        m = re.search(r"(\{[\s\S]*\})", reasoning)
                        if m:
                            raw_text = m.group(1).strip()
                break

            except httpx.RequestError as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise LLMProviderError(f"Network error during structured call: {exc}")

        cleaned = self._clean_json_markdown(raw_text)
        try:
            parsed = json.loads(cleaned)
            return schema.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as err:
            obj_match = re.search(r"(\{[\s\S]*\})", raw_text)
            if obj_match:
                try:
                    parsed = json.loads(obj_match.group(1))
                    return schema.model_validate(parsed)
                except Exception:
                    pass

            # Fast repair pass if initial attempt didn't produce clean JSON
            try:
                repair_payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a strict JSON converter. Convert the following text into valid JSON matching this schema:\n{schema_json}\n"
                                f"Output ONLY the valid JSON object. No explanations, no thoughts."
                            ),
                        },
                        {"role": "user", "content": raw_text or prompt[:3000]},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 4096,
                }
                if "nemotron" in self.model.lower() or "nvidia" in self.model.lower():
                    repair_payload["chat_template_kwargs"] = {"enable_thinking": False}

                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        rep_resp = await client.post(
                            url,
                            json=repair_payload,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                        )
                        if rep_resp.status_code == 200:
                            rep_msg = rep_resp.json().get("choices", [{}])[0].get("message", {})
                            rep_content = rep_msg.get("content") or ""
                            rep_cleaned = self._clean_json_markdown(rep_content)
                            m = re.search(r"(\{[\s\S]*\})", rep_cleaned)
                            if m:
                                rep_cleaned = m.group(1)
                            parsed = json.loads(rep_cleaned)
                            return schema.model_validate(parsed)
            except Exception as repair_err:
                logger.warning("Structured repair pass failed: %s", repair_err)

            raise LLMStructuredOutputError(
                f"Failed to parse structured model response as {schema.__name__}: {str(err)}\n"
                f"Raw output received:\n{raw_text[:500]}"
            )


SUPPORTED_MODELS: List[Dict[str, Any]] = [
    {
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "name": "Nemotron 3 Super 120B",
        "provider": "nvidia",
        "description": "NVIDIA ultra-large 120B reasoning model with state-of-the-art accuracy",
        "recommended": True,
        "badge": "120B Flagship",
    },
    {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "name": "Nemotron 3 Nano Omni",
        "provider": "nvidia",
        "description": "NVIDIA lightweight 30B fast multimodal reasoning model",
        "recommended": False,
        "badge": "30B Fast",
    },
    {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash Lite",
        "provider": "gemini",
        "description": "Google lightweight fast multimodal model",
        "recommended": False,
        "badge": "Google",
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "provider": "gemini",
        "description": "Google general-purpose fast reasoning model",
        "recommended": False,
        "badge": "Google",
    },
    {
        "id": "mock-model",
        "name": "Mock Simulation",
        "provider": "mock",
        "description": "Deterministic offline mock simulation for testing",
        "recommended": False,
        "badge": "Offline Test",
    },
]


def get_llm_provider_for_model(model_name: Optional[str], config: Settings) -> LLMProvider:
    """
    Obtain an LLMProvider instance dynamically configured for the specified model.
    Routes to NVIDIA NIM, OpenAI-compatible APIs, Google Gemini, or MockProvider.
    """
    if not model_name or not model_name.strip():
        model_name = config.llm_model

    provider_lower = config.llm_provider.lower().strip()
    if provider_lower not in ("gemini", "mock", "nvidia", "openai"):
        raise LLMConfigurationError(f"Unsupported LLM provider: '{config.llm_provider}'")

    m_lower = model_name.lower().strip()
    if m_lower in ("mock", "mock-model") or provider_lower == "mock":
        return MockLLMProvider()

    # NVIDIA NIM and OpenAI-compatible models
    if (
        m_lower.startswith("nvidia/")
        or m_lower.startswith("meta/")
        or m_lower.startswith("mistralai/")
        or "nemotron" in m_lower
        or config.llm_provider.lower() in ("nvidia", "openai")
    ):
        api_key = config.nvidia_api_key or config.llm_api_key
        if not api_key:
            raise LLMConfigurationError(
                "NVIDIA API key is not configured. Please set NVIDIA_API_KEY in backend/.env"
            )
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model_name,
            base_url=config.nvidia_base_url,
        )

    # Gemini models
    if m_lower.startswith("gemini") or config.llm_provider.lower() == "gemini":
        if not config.llm_api_key:
            raise LLMConfigurationError(
                "Gemini API key is not configured. Please set LLM_API_KEY in backend/.env"
            )
        return GeminiProvider(api_key=config.llm_api_key, model=model_name)

    return GeminiProvider(api_key=config.llm_api_key, model=model_name)


def get_llm_provider(config: Settings) -> LLMProvider:
    """
    Factory function to obtain the configured default LLMProvider instance.

    Raises:
        LLMConfigurationError: If provider is misconfigured or API key is absent.
    """
    return get_llm_provider_for_model(config.llm_model, config)
