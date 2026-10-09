"""
Resource Budgets and Enforcement for Adaptive Multi-Agent Orchestration.

Defines resource limits (max LLM calls, max tokens, max wall time)
per execution strategy, and provides runtime tracking and enforcement.
"""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.config import Settings
from app.orchestration.strategy import ExecutionStrategy


class ResourceBudget(BaseModel):
    """Resource constraints allocated to a specific execution strategy."""

    strategy: ExecutionStrategy = Field(..., description="Target execution strategy")
    max_llm_calls: int = Field(..., ge=1, description="Maximum allowed LLM provider invocations")
    max_tokens: int = Field(..., ge=1, description="Maximum total tokens (prompt + completion)")
    max_wall_time_seconds: float = Field(..., ge=0.001, description="Maximum execution wall clock time in seconds")


class ResourceBudgetExceededError(RuntimeError):
    """Raised when an execution exceeds its allocated resource budget."""

    def __init__(self, message: str, limit_type: str, current_value: Any, limit_value: Any):
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class BudgetTracker:
    """Runtime tracker that strictly enforces strategy-specific resource budgets."""

    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.llm_calls: int = 0
        self.total_tokens: int = 0
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.start_time: float = time.perf_counter()

    def record_llm_call(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record an LLM call and token consumption, checking limits immediately."""
        self.llm_calls += 1
        self.tokens_in += max(0, tokens_in)
        self.tokens_out += max(0, tokens_out)
        self.total_tokens += max(0, tokens_in + tokens_out)
        self.check_limits()

    def check_limits(self) -> None:
        """Verify that current usage does not exceed the allocated budget."""
        # Check LLM calls
        if self.llm_calls > self.budget.max_llm_calls:
            raise ResourceBudgetExceededError(
                f"Resource budget exceeded for {self.budget.strategy.value}: "
                f"LLM calls ({self.llm_calls}) exceeded maximum allowance ({self.budget.max_llm_calls}).",
                limit_type="llm_calls",
                current_value=self.llm_calls,
                limit_value=self.budget.max_llm_calls,
            )

        # Check total tokens
        if self.total_tokens > self.budget.max_tokens:
            raise ResourceBudgetExceededError(
                f"Resource budget exceeded for {self.budget.strategy.value}: "
                f"Total tokens ({self.total_tokens}) exceeded maximum allowance ({self.budget.max_tokens}).",
                limit_type="tokens",
                current_value=self.total_tokens,
                limit_value=self.budget.max_tokens,
            )

        # Check elapsed wall time
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.budget.max_wall_time_seconds:
            raise ResourceBudgetExceededError(
                f"Resource budget exceeded for {self.budget.strategy.value}: "
                f"Wall time ({elapsed:.2f}s) exceeded maximum allowance ({self.budget.max_wall_time_seconds:.2f}s).",
                limit_type="wall_time",
                current_value=round(elapsed, 2),
                limit_value=self.budget.max_wall_time_seconds,
            )

    @property
    def elapsed_seconds(self) -> float:
        return round(time.perf_counter() - self.start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Diagnostic summary of budget consumption."""
        return {
            "strategy": self.budget.strategy.value,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.budget.max_llm_calls,
            "total_tokens": self.total_tokens,
            "max_tokens": self.budget.max_tokens,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "elapsed_seconds": self.elapsed_seconds,
            "max_wall_time_seconds": self.budget.max_wall_time_seconds,
        }


def get_budget_for_strategy(
    strategy: ExecutionStrategy,
    settings: Optional[Settings] = None,
) -> ResourceBudget:
    """Retrieve strategy resource budget based on system settings or sensible defaults."""
    if strategy == ExecutionStrategy.DIRECT:
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_direct_max_llm_calls", 1) if settings else 1,
            max_tokens=getattr(settings, "budget_direct_max_tokens", 3000) if settings else 3000,
            max_wall_time_seconds=getattr(settings, "budget_direct_max_wall_time_seconds", 15.0) if settings else 15.0,
        )
    elif strategy == ExecutionStrategy.SINGLE_AGENT:
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_single_agent_max_llm_calls", 3) if settings else 3,
            max_tokens=getattr(settings, "budget_single_agent_max_tokens", 12000) if settings else 12000,
            max_wall_time_seconds=getattr(settings, "budget_single_agent_max_wall_time_seconds", 45.0) if settings else 45.0,
        )
    elif strategy == ExecutionStrategy.SINGLE_AGENT_RAG:
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_single_agent_rag_max_llm_calls", 4) if settings else 4,
            max_tokens=getattr(settings, "budget_single_agent_rag_max_tokens", 18000) if settings else 18000,
            max_wall_time_seconds=getattr(settings, "budget_single_agent_rag_max_wall_time_seconds", 60.0) if settings else 60.0,
        )
    else:  # MULTI_AGENT
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_multi_agent_max_llm_calls", 20) if settings else 20,
            max_tokens=getattr(settings, "budget_multi_agent_max_tokens", 80000) if settings else 80000,
            max_wall_time_seconds=getattr(settings, "budget_multi_agent_max_wall_time_seconds", 240.0) if settings else 240.0,
        )
