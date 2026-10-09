"""
Telemetry and Metrics Instrumentation for Adaptive Multi-Agent Orchestration.

Instruments every LLM and embedding call:
- tokens_in (prompt tokens)
- tokens_out (completion tokens)
- total_tokens
- latency_ms
- estimated_cost ($)
- model
- run_id
- stage
Persists per-run metrics and exposes programmatic querying.
"""

from datetime import datetime, timezone
import threading
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Standard pricing benchmarks per 1,000,000 tokens
MODEL_PRICING = {
    # Gemini 3.1 Flash Lite / 1.5 Flash
    "gemini-3.1-flash-lite": {"prompt_per_m": 0.075, "completion_per_m": 0.30},
    "gemini-1.5-flash": {"prompt_per_m": 0.075, "completion_per_m": 0.30},
    "gemini-1.5-pro": {"prompt_per_m": 1.25, "completion_per_m": 5.00},
    # Embedding model
    "gemini-embedding-001": {"prompt_per_m": 0.02, "completion_per_m": 0.0},
    # Mock / testing default
    "mock": {"prompt_per_m": 0.001, "completion_per_m": 0.002},
}


def compute_estimated_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute estimated cost in USD based on model pricing."""
    pricing = MODEL_PRICING.get(model.lower(), MODEL_PRICING["gemini-3.1-flash-lite"])
    cost_in = (tokens_in / 1_000_000.0) * pricing["prompt_per_m"]
    cost_out = (tokens_out / 1_000_000.0) * pricing["completion_per_m"]
    return round(cost_in + cost_out, 6)


class CallMetric(BaseModel):
    """Instrumentation record for a single LLM or embedding call."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(..., description="Pipeline execution run ID")
    call_type: str = Field(..., description="'llm' or 'embedding'")
    stage: str = Field(..., description="Pipeline stage (e.g. Planning, Research:T1, Synthesis)")
    model: str = Field(..., description="Target model name")
    tokens_in: int = Field(default=0, description="Prompt/input tokens")
    tokens_out: int = Field(default=0, description="Completion/output tokens")
    total_tokens: int = Field(default=0, description="Sum of input and output tokens")
    latency_ms: float = Field(default=0.0, description="Call duration in milliseconds")
    estimated_cost: float = Field(default=0.0, description="Estimated USD cost")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or diagnostics")


class RunMetrics(BaseModel):
    """Aggregated metrics summary and invocation history for an execution run."""

    run_id: str = Field(..., description="Unique run UUID")
    strategy: str = Field(default="UNKNOWN", description="Selected execution strategy")
    total_llm_calls: int = Field(default=0, description="Total number of LLM invocations")
    total_embedding_calls: int = Field(default=0, description="Total number of embedding invocations")
    total_tokens_in: int = Field(default=0, description="Aggregate prompt tokens")
    total_tokens_out: int = Field(default=0, description="Aggregate completion tokens")
    total_tokens: int = Field(default=0, description="Aggregate total tokens")
    total_latency_ms: float = Field(default=0.0, description="Sum of call latencies in ms")
    total_estimated_cost: float = Field(default=0.0, description="Sum of estimated USD costs")
    wall_time_ms: float = Field(default=0.0, description="Total execution wall clock duration in ms")
    calls: List[CallMetric] = Field(default_factory=list, description="Ordered list of individual call metrics")


class MetricsCollector:
    """Thread-safe in-memory metrics registry for multi-agent runs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[str, RunMetrics] = {}

    def get_or_create(self, run_id: str, strategy: str = "UNKNOWN") -> RunMetrics:
        with self._lock:
            if run_id not in self._runs:
                self._runs[run_id] = RunMetrics(run_id=run_id, strategy=strategy)
            return self._runs[run_id]

    def record_call(self, metric: CallMetric) -> None:
        with self._lock:
            if metric.run_id not in self._runs:
                self._runs[metric.run_id] = RunMetrics(run_id=metric.run_id)

            run = self._runs[metric.run_id]
            run.calls.append(metric)

            if metric.call_type == "llm":
                run.total_llm_calls += 1
            elif metric.call_type == "embedding":
                run.total_embedding_calls += 1

            run.total_tokens_in += metric.tokens_in
            run.total_tokens_out += metric.tokens_out
            run.total_tokens += metric.total_tokens
            run.total_latency_ms = round(run.total_latency_ms + metric.latency_ms, 2)
            run.total_estimated_cost = round(run.total_estimated_cost + metric.estimated_cost, 6)

    def finalize_run(self, run_id: str, wall_time_ms: float, strategy: Optional[str] = None) -> RunMetrics:
        with self._lock:
            if run_id not in self._runs:
                self._runs[run_id] = RunMetrics(run_id=run_id, strategy=strategy or "UNKNOWN")
            run = self._runs[run_id]
            run.wall_time_ms = round(wall_time_ms, 2)
            if strategy:
                run.strategy = strategy
            return run

    def get_metrics(self, run_id: str) -> Optional[RunMetrics]:
        with self._lock:
            return self._runs.get(run_id)

    def list_all(self) -> List[RunMetrics]:
        with self._lock:
            return list(self._runs.values())

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


# Global shared metrics registry
global_metrics_collector = MetricsCollector()
