"""
Instrumented LLM Provider Wrapper.

Transparently decorates any LLMProvider to measure:
- Per-call latency (ms)
- Token counts (prompt tokens, completion tokens, total)
- Estimated cost ($)
- Pipeline stage attribution
- Integration with BudgetTracker for immediate limit enforcement
- Recording of CallMetric to the shared MetricsCollector
"""

import time
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from app.llm.base import LLMProvider
from app.orchestration.resource_budget import BudgetTracker
from app.telemetry.metrics import (
    CallMetric,
    MetricsCollector,
    compute_estimated_cost,
    global_metrics_collector,
)

T = TypeVar("T", bound=BaseModel)


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (average 1.3 tokens per word or 1 token per 4 characters)."""
    if not text or not text.strip():
        return 0
    words = len(text.split())
    by_words = int(words * 1.33)
    by_chars = len(text) // 4
    return max(1, max(by_words, by_chars))


class InstrumentedLLMProvider(LLMProvider):
    """Wraps an LLMProvider to record performance metrics and enforce resource budgets."""

    def __init__(
        self,
        base_provider: LLMProvider,
        run_id: str,
        metrics_collector: Optional[MetricsCollector] = None,
        budget_tracker: Optional[BudgetTracker] = None,
        model_name: str = "gemini-3.1-flash-lite",
    ):
        self.base_provider = base_provider
        self.run_id = run_id
        self.metrics_collector = metrics_collector or global_metrics_collector
        self.budget_tracker = budget_tracker
        self.model_name = getattr(base_provider, "model", model_name) or model_name
        self._current_stage: str = "Execution"

    def set_stage(self, stage: str) -> None:
        """Update the active pipeline stage for telemetry attribution."""
        self._current_stage = stage

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Pre-call budget check
        if self.budget_tracker:
            self.budget_tracker.check_limits()

        t0 = time.perf_counter()
        result = await self.base_provider.generate(prompt=prompt, system_prompt=system_prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Extract tokens from provider metadata or fallback estimator
        usage = getattr(self.base_provider, "last_usage", None) or {}
        tokens_in = usage.get("promptTokenCount")
        tokens_out = usage.get("candidatesTokenCount")

        if tokens_in is None:
            full_prompt = (f"{system_prompt}\n" if system_prompt else "") + prompt
            tokens_in = estimate_tokens(full_prompt)
        if tokens_out is None:
            tokens_out = estimate_tokens(result)

        total_tokens = tokens_in + tokens_out
        cost = compute_estimated_cost(self.model_name, tokens_in, tokens_out)

        # Record telemetry call metric
        metric = CallMetric(
            run_id=self.run_id,
            call_type="llm",
            stage=self._current_stage,
            model=self.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
        )
        self.metrics_collector.record_call(metric)

        # Enforce budget consumption
        if self.budget_tracker:
            self.budget_tracker.record_llm_call(tokens_in=tokens_in, tokens_out=tokens_out)

        return result

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        # Pre-call budget check
        if self.budget_tracker:
            self.budget_tracker.check_limits()

        t0 = time.perf_counter()
        result = await self.base_provider.generate_structured(
            prompt=prompt, schema=schema, system_prompt=system_prompt
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Extract tokens
        usage = getattr(self.base_provider, "last_usage", None) or {}
        tokens_in = usage.get("promptTokenCount")
        tokens_out = usage.get("candidatesTokenCount")

        if tokens_in is None:
            full_prompt = (f"{system_prompt}\n" if system_prompt else "") + prompt
            tokens_in = estimate_tokens(full_prompt)
        if tokens_out is None:
            out_str = result.model_dump_json() if hasattr(result, "model_dump_json") else str(result)
            tokens_out = estimate_tokens(out_str)

        total_tokens = tokens_in + tokens_out
        cost = compute_estimated_cost(self.model_name, tokens_in, tokens_out)

        metric = CallMetric(
            run_id=self.run_id,
            call_type="llm",
            stage=self._current_stage,
            model=self.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
            metadata={"schema": getattr(schema, "__name__", str(schema))},
        )
        self.metrics_collector.record_call(metric)

        if self.budget_tracker:
            self.budget_tracker.record_llm_call(tokens_in=tokens_in, tokens_out=tokens_out)

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate any unhandled attributes to the underlying provider."""
        return getattr(self.base_provider, name)
