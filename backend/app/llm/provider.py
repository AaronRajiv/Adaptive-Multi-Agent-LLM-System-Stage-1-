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
                                    delay = min(float(match.group(1)) + 0.5, 60.0)
                                except ValueError:
                                    delay = None
                        delay = delay if delay is not None else backoff
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
                                    delay = min(float(match.group(1)) + 0.5, 60.0)
                                except ValueError:
                                    delay = None
                        delay = delay if delay is not None else backoff
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
