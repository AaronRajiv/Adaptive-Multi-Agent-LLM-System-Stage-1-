"""
Execution Event Model and Per-Run Event Bus.

Provides typed execution events emitted during OrchestratedPipeline runs,
and a lightweight in-memory event bus that supports multiple async consumers
and late-subscriber buffering.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExecutionEventType(str, Enum):
    """Exhaustive set of events the pipeline may emit."""

    RUN_STARTED = "RUN_STARTED"
    STRATEGY_SELECTED = "STRATEGY_SELECTED"
    PLANNING_STARTED = "PLANNING_STARTED"
    PLANNING_COMPLETED = "PLANNING_COMPLETED"

    TASK_READY = "TASK_READY"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_BLOCKED = "TASK_BLOCKED"

    RAG_STARTED = "RAG_STARTED"
    RAG_COMPLETED = "RAG_COMPLETED"

    EVALUATION_STARTED = "EVALUATION_STARTED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"

    REPLANNING_STARTED = "REPLANNING_STARTED"
    REPLANNING_COMPLETED = "REPLANNING_COMPLETED"

    SYNTHESIS_STARTED = "SYNTHESIS_STARTED"
    SYNTHESIS_COMPLETED = "SYNTHESIS_COMPLETED"

    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


class ExecutionEvent(BaseModel):
    """A single typed event emitted during pipeline execution."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(..., description="The pipeline run this event belongs to")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    event_type: ExecutionEventType
    task_id: Optional[str] = Field(default=None, description="Graph task ID, when applicable")
    capability: Optional[str] = Field(default=None, description="Task capability/type, when applicable")
    status: Optional[str] = Field(default=None, description="Task or stage status")
    message: Optional[str] = Field(default=None, description="Human-readable summary")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra structured data")

    def to_sse(self) -> str:
        """Format this event as a Server-Sent Events data frame."""
        return f"event: {self.event_type.value}\ndata: {self.model_dump_json()}\n\n"


class RunEventBus:
    """
    Per-run async event bus with bounded history for late subscribers.

    Thread/task-safe: multiple producers and consumers may operate concurrently.
    """

    def __init__(self, run_id: str, max_history: int = 500):
        self.run_id = run_id
        self._max_history = max_history
        self._history: List[ExecutionEvent] = []
        self._subscribers: List[asyncio.Queue[ExecutionEvent | None]] = []
        self._finished = False
        self._lock = asyncio.Lock()

    async def emit(self, event: ExecutionEvent) -> None:
        """Publish an event to all current subscribers and append to history."""
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # Drop if subscriber is too slow; pipeline must not block

    async def finish(self) -> None:
        """Signal all subscribers that no more events will be emitted."""
        async with self._lock:
            self._finished = True
            for queue in self._subscribers:
                try:
                    queue.put_nowait(None)  # Sentinel
                except asyncio.QueueFull:
                    pass

    async def subscribe(self) -> asyncio.Queue[ExecutionEvent | None]:
        """
        Create a new subscriber queue.

        Returns a queue that yields ExecutionEvent instances, then None when the
        run completes. Late subscribers receive the full buffered history first.
        """
        queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            # Replay buffered history
            for event in self._history:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break
            if self._finished:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            else:
                self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[ExecutionEvent | None]) -> None:
        """Remove a subscriber queue (idempotent)."""
        async with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    @property
    def history(self) -> List[ExecutionEvent]:
        """Read-only snapshot of the event history (for testing)."""
        return list(self._history)

    @property
    def finished(self) -> bool:
        return self._finished


class EventBusRegistry:
    """
    Global registry mapping run_id → RunEventBus.

    Retains completed buses briefly so late SSE connections can still read history.
    Evicts oldest completed buses when exceeding the retention limit.
    """

    def __init__(self, max_retained: int = 100):
        self._buses: Dict[str, RunEventBus] = {}
        self._max_retained = max_retained
        self._lock = asyncio.Lock()

    async def create(self, run_id: str) -> RunEventBus:
        """Create a new event bus for a pipeline run."""
        bus = RunEventBus(run_id)
        async with self._lock:
            self._buses[run_id] = bus
            await self._evict_if_needed()
        return bus

    async def get(self, run_id: str) -> Optional[RunEventBus]:
        """Look up an event bus by run_id."""
        async with self._lock:
            return self._buses.get(run_id)

    async def get_or_create(self, run_id: str) -> RunEventBus:
        """Get an existing event bus or create one if it doesn't exist yet."""
        async with self._lock:
            if run_id not in self._buses:
                self._buses[run_id] = RunEventBus(run_id)
                await self._evict_if_needed()
            return self._buses[run_id]

    async def _evict_if_needed(self) -> None:
        """Remove oldest finished buses when over retention limit."""
        if len(self._buses) <= self._max_retained:
            return
        finished_ids = [
            rid for rid, bus in self._buses.items() if bus.finished
        ]
        for rid in finished_ids:
            if len(self._buses) <= self._max_retained:
                break
            del self._buses[rid]


# Module-level singleton
event_bus_registry = EventBusRegistry()
