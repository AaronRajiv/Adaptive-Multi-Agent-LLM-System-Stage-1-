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


class WebBudget(BaseModel):
    """Resource constraints for outbound web operations."""

    max_searches: int = Field(default=5, ge=0, description="Max web searches allowed per run")
    max_fetches: int = Field(default=5, ge=0, description="Max webpage fetches allowed per run")
    max_bytes: int = Field(default=2 * 1024 * 1024, ge=0, description="Max downloaded bytes allowed per run")
    max_wall_time_seconds: float = Field(default=60.0, ge=0.001, description="Max wall time in seconds")


class ResourceBudget(BaseModel):
    """Resource constraints allocated to a specific execution strategy."""

    strategy: ExecutionStrategy = Field(..., description="Target execution strategy")
    max_llm_calls: int = Field(..., ge=1, description="Maximum allowed LLM provider invocations")
    max_tokens: int = Field(..., ge=1, description="Maximum total tokens (prompt + completion)")
    max_wall_time_seconds: float = Field(..., ge=0.001, description="Maximum execution wall clock time in seconds")
    max_web_searches: int = Field(default=0, ge=0, description="Maximum web searches allowed per run")
    max_web_fetches: int = Field(default=0, ge=0, description="Maximum webpage fetches allowed per run")
    max_web_bytes: int = Field(default=0, ge=0, description="Maximum downloaded bytes allowed per run")
    max_web_wall_time_seconds: float = Field(default=60.0, ge=0.001, description="Maximum execution wall clock time for web operations")

    def to_web_budget(self) -> WebBudget:
        return WebBudget(
            max_searches=self.max_web_searches,
            max_fetches=self.max_web_fetches,
            max_bytes=self.max_web_bytes,
            max_wall_time_seconds=self.max_web_wall_time_seconds,
        )


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
        self.web_tracker = AtomicWebBudgetTracker(self.budget.to_web_budget())

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
            "web_budget": self.web_tracker.to_dict(),
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
            max_web_searches=0,
            max_web_fetches=0,
            max_web_bytes=0,
            max_web_wall_time_seconds=15.0,
        )
    elif strategy == ExecutionStrategy.SINGLE_AGENT:
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_single_agent_max_llm_calls", 3) if settings else 3,
            max_tokens=getattr(settings, "budget_single_agent_max_tokens", 12000) if settings else 12000,
            max_wall_time_seconds=getattr(settings, "budget_single_agent_max_wall_time_seconds", 45.0) if settings else 45.0,
            max_web_searches=getattr(settings, "budget_single_agent_max_web_searches", 3) if settings else 3,
            max_web_fetches=getattr(settings, "budget_single_agent_max_web_fetches", 3) if settings else 3,
            max_web_bytes=getattr(settings, "budget_single_agent_max_web_bytes", 2 * 1024 * 1024) if settings else 2 * 1024 * 1024,
            max_web_wall_time_seconds=getattr(settings, "budget_single_agent_max_web_wall_time_seconds", 30.0) if settings else 30.0,
        )
    elif strategy == ExecutionStrategy.SINGLE_AGENT_RAG:
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_single_agent_rag_max_llm_calls", 4) if settings else 4,
            max_tokens=getattr(settings, "budget_single_agent_rag_max_tokens", 18000) if settings else 18000,
            max_wall_time_seconds=getattr(settings, "budget_single_agent_rag_max_wall_time_seconds", 60.0) if settings else 60.0,
            max_web_searches=getattr(settings, "budget_single_agent_rag_max_web_searches", 3) if settings else 3,
            max_web_fetches=getattr(settings, "budget_single_agent_rag_max_web_fetches", 3) if settings else 3,
            max_web_bytes=getattr(settings, "budget_single_agent_rag_max_web_bytes", 2 * 1024 * 1024) if settings else 2 * 1024 * 1024,
            max_web_wall_time_seconds=getattr(settings, "budget_single_agent_rag_max_web_wall_time_seconds", 45.0) if settings else 45.0,
        )
    else:  # MULTI_AGENT
        return ResourceBudget(
            strategy=strategy,
            max_llm_calls=getattr(settings, "budget_multi_agent_max_llm_calls", 35) if settings else 35,
            max_tokens=getattr(settings, "budget_multi_agent_max_tokens", 500000) if settings else 500000,
            max_wall_time_seconds=getattr(settings, "budget_multi_agent_max_wall_time_seconds", 360.0) if settings else 360.0,
            max_web_searches=getattr(settings, "budget_multi_agent_max_web_searches", 8) if settings else 8,
            max_web_fetches=getattr(settings, "budget_multi_agent_max_web_fetches", 8) if settings else 8,
            max_web_bytes=getattr(settings, "budget_multi_agent_max_web_bytes", 10 * 1024 * 1024) if settings else 10 * 1024 * 1024,
            max_web_wall_time_seconds=getattr(settings, "budget_multi_agent_max_web_wall_time_seconds", 240.0) if settings else 240.0,
        )


class AtomicWebBudgetTracker:
    """
    Thread- and coroutine-safe atomic budget tracker for web operations.
    Enforces limits atomically under concurrent parallel tool invocations.
    """

    def __init__(self, budget: WebBudget):
        import asyncio
        self.budget = budget
        self.searches: int = 0
        self.fetches: int = 0
        self.bytes_downloaded: int = 0
        self.start_time: float = time.perf_counter()
        self._lock = asyncio.Lock()

    def _check_wall_time_locked(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.budget.max_wall_time_seconds:
            raise ResourceBudgetExceededError(
                f"Web budget exceeded: Wall time ({elapsed:.2f}s) exceeded limit ({self.budget.max_wall_time_seconds:.2f}s).",
                limit_type="web_wall_time",
                current_value=round(elapsed, 2),
                limit_value=self.budget.max_wall_time_seconds,
            )

    async def reserve_search(self) -> int:
        """Atomically check and reserve 1 web search slot."""
        async with self._lock:
            self._check_wall_time_locked()
            if self.searches + 1 > self.budget.max_searches:
                raise ResourceBudgetExceededError(
                    f"Web search budget exceeded: current {self.searches} + 1 > limit {self.budget.max_searches}.",
                    limit_type="web_searches",
                    current_value=self.searches + 1,
                    limit_value=self.budget.max_searches,
                )
            self.searches += 1
            return self.searches

    async def reserve_fetch(self) -> int:
        """Atomically check and reserve 1 page fetch slot."""
        async with self._lock:
            self._check_wall_time_locked()
            if self.fetches + 1 > self.budget.max_fetches:
                raise ResourceBudgetExceededError(
                    f"Web fetch budget exceeded: current {self.fetches} + 1 > limit {self.budget.max_fetches}.",
                    limit_type="web_fetches",
                    current_value=self.fetches + 1,
                    limit_value=self.budget.max_fetches,
                )
            self.fetches += 1
            return self.fetches

    async def record_downloaded_bytes(self, count: int) -> int:
        """Atomically check and record downloaded bytes."""
        async with self._lock:
            self._check_wall_time_locked()
            if self.bytes_downloaded + count > self.budget.max_bytes:
                raise ResourceBudgetExceededError(
                    f"Web byte budget exceeded: {self.bytes_downloaded + count} bytes > limit {self.budget.max_bytes} bytes.",
                    limit_type="web_bytes",
                    current_value=self.bytes_downloaded + count,
                    limit_value=self.budget.max_bytes,
                )
            self.bytes_downloaded += count
            return self.bytes_downloaded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "searches": self.searches,
            "max_searches": self.budget.max_searches,
            "fetches": self.fetches,
            "max_fetches": self.budget.max_fetches,
            "bytes_downloaded": self.bytes_downloaded,
            "max_bytes": self.budget.max_bytes,
            "elapsed_seconds": round(time.perf_counter() - self.start_time, 2),
            "max_wall_time_seconds": self.budget.max_wall_time_seconds,
        }
