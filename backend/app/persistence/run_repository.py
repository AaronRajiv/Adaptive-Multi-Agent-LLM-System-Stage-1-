"""
Persistence layer for pipeline runs and telemetry metrics.

Provides:
- RunRepository: Abstract protocol for run and telemetry persistence.
- InMemoryRunRepository: In-memory store (up to 100 recent runs) for testing and local dev.
- PostgresRunRepository: PostgreSQL JSONB persistent store for production deployments.
- get_run_repository(): Factory returning PostgresRunRepository when database_url is provided,
  falling back gracefully to InMemoryRunRepository.
"""

from abc import ABC, abstractmethod
import asyncio
from collections import OrderedDict
import json
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import RunResponse
from app.telemetry.metrics import RunMetrics

logger = get_logger("app.persistence.run_repository")


class RunRepository(ABC):
    """Abstract interface for storing and retrieving pipeline runs and telemetry metrics."""

    @abstractmethod
    async def save_run(self, run: RunResponse) -> None:
        """Persist a completed or updated run response."""
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[RunResponse]:
        """Retrieve a run response by run_id, or None if not found."""
        pass

    @abstractmethod
    async def list_runs(self, limit: int = 50) -> List[RunResponse]:
        """Retrieve recent runs in reverse chronological order."""
        pass

    @abstractmethod
    async def save_metrics(self, metrics: RunMetrics) -> None:
        """Persist telemetry metrics for a run."""
        pass

    @abstractmethod
    async def get_metrics(self, run_id: str) -> Optional[RunMetrics]:
        """Retrieve telemetry metrics for a run by run_id."""
        pass

    @abstractmethod
    async def list_metrics(self, limit: int = 50) -> List[RunMetrics]:
        """Retrieve recent telemetry metrics."""
        pass

    @abstractmethod
    async def delete_run(self, run_id: str) -> bool:
        """Delete a run and associated metrics."""
        pass


class InMemoryRunRepository(RunRepository):
    """Thread-safe in-memory repository retaining the most recent 100 runs."""

    def __init__(self, max_runs: int = 100):
        self._max_runs = max_runs
        self._runs: OrderedDict[str, RunResponse] = OrderedDict()
        self._metrics: OrderedDict[str, RunMetrics] = OrderedDict()
        self._lock = asyncio.Lock()

    async def save_run(self, run: RunResponse) -> None:
        async with self._lock:
            if run.run_id in self._runs:
                del self._runs[run.run_id]
            self._runs[run.run_id] = run
            if len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)

    async def get_run(self, run_id: str) -> Optional[RunResponse]:
        async with self._lock:
            return self._runs.get(run_id)

    async def list_runs(self, limit: int = 50) -> List[RunResponse]:
        async with self._lock:
            items = list(reversed(self._runs.values()))
            return items[:limit]

    async def save_metrics(self, metrics: RunMetrics) -> None:
        async with self._lock:
            if metrics.run_id in self._metrics:
                del self._metrics[metrics.run_id]
            self._metrics[metrics.run_id] = metrics
            if len(self._metrics) > self._max_runs:
                self._metrics.popitem(last=False)

    async def get_metrics(self, run_id: str) -> Optional[RunMetrics]:
        async with self._lock:
            return self._metrics.get(run_id)

    async def list_metrics(self, limit: int = 50) -> List[RunMetrics]:
        async with self._lock:
            items = list(reversed(self._metrics.values()))
            return items[:limit]

    async def delete_run(self, run_id: str) -> bool:
        async with self._lock:
            removed = self._runs.pop(run_id, None) is not None
            self._metrics.pop(run_id, None)
            return removed


class PostgresRunRepository(RunRepository):
    """PostgreSQL persistent repository storing runs and telemetry metrics using JSONB."""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("database_url is required for PostgresRunRepository.")
        self.database_url = database_url
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _connect(self):
        try:
            from psycopg import AsyncConnection
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL run persistence. Please install psycopg."
            ) from exc
        return await AsyncConnection.connect(self.database_url)

    async def initialize_schema(self) -> None:
        """Create runs and run_metrics tables if they do not already exist."""
        async with self._init_lock:
            if self._initialized:
                return
            conn = await self._connect()
            async with conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS runs (
                            run_id VARCHAR(128) PRIMARY KEY,
                            task TEXT NOT NULL,
                            status VARCHAR(64) NOT NULL DEFAULT 'completed',
                            execution_strategy VARCHAR(64),
                            final_answer TEXT,
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);

                        CREATE TABLE IF NOT EXISTS run_metrics (
                            run_id VARCHAR(128) PRIMARY KEY,
                            strategy VARCHAR(64),
                            total_tokens INT,
                            total_llm_calls INT,
                            total_estimated_cost NUMERIC(12, 6),
                            wall_time_ms NUMERIC(12, 2),
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_run_metrics_created_at ON run_metrics (created_at DESC);
                        """
                    )
            self._initialized = True
            logger.info("PostgresRunRepository schema initialized successfully.")

    async def save_run(self, run: RunResponse) -> None:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO runs (run_id, task, status, execution_strategy, final_answer, payload)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        task = EXCLUDED.task,
                        status = EXCLUDED.status,
                        execution_strategy = EXCLUDED.execution_strategy,
                        final_answer = EXCLUDED.final_answer,
                        payload = EXCLUDED.payload;
                    """,
                    (
                        run.run_id,
                        run.user_task,
                        "completed",
                        run.execution_strategy,
                        run.final_answer,
                        json.dumps(run.model_dump()),
                    ),
                )

    async def get_run(self, run_id: str) -> Optional[RunResponse]:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT payload FROM runs WHERE run_id = %s LIMIT 1;",
                    (run_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return RunResponse.model_validate(payload)

    async def list_runs(self, limit: int = 50) -> List[RunResponse]:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT payload FROM runs ORDER BY created_at DESC LIMIT %s;",
                    (limit,),
                )
                rows = await cur.fetchall()
                results = []
                for row in rows:
                    payload = row[0]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    results.append(RunResponse.model_validate(payload))
                return results

    async def save_metrics(self, metrics: RunMetrics) -> None:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO run_metrics (
                        run_id, strategy, total_tokens, total_llm_calls,
                        total_estimated_cost, wall_time_ms, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        strategy = EXCLUDED.strategy,
                        total_tokens = EXCLUDED.total_tokens,
                        total_llm_calls = EXCLUDED.total_llm_calls,
                        total_estimated_cost = EXCLUDED.total_estimated_cost,
                        wall_time_ms = EXCLUDED.wall_time_ms,
                        payload = EXCLUDED.payload;
                    """,
                    (
                        metrics.run_id,
                        metrics.strategy,
                        metrics.total_tokens,
                        metrics.total_llm_calls,
                        metrics.total_estimated_cost,
                        metrics.wall_time_ms,
                        json.dumps(metrics.model_dump()),
                    ),
                )

    async def get_metrics(self, run_id: str) -> Optional[RunMetrics]:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT payload FROM run_metrics WHERE run_id = %s LIMIT 1;",
                    (run_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return None
                payload = row[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return RunMetrics.model_validate(payload)

    async def list_metrics(self, limit: int = 50) -> List[RunMetrics]:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT payload FROM run_metrics ORDER BY created_at DESC LIMIT %s;",
                    (limit,),
                )
                rows = await cur.fetchall()
                results = []
                for row in rows:
                    payload = row[0]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    results.append(RunMetrics.model_validate(payload))
                return results

    async def delete_run(self, run_id: str) -> bool:
        await self.initialize_schema()
        conn = await self._connect()
        async with conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM run_metrics WHERE run_id = %s;", (run_id,))
                await cur.execute("DELETE FROM runs WHERE run_id = %s;", (run_id,))
                return cur.rowcount > 0


_global_run_repository: Optional[RunRepository] = None


def get_run_repository(settings: Optional[Settings] = None) -> RunRepository:
    """
    Factory function returning the singleton run repository.
    Instantiates PostgresRunRepository if database_url or knowledge_database_url is set,
    falling back to InMemoryRunRepository.
    """
    global _global_run_repository
    if _global_run_repository is not None:
        return _global_run_repository

    db_url = ""
    if settings:
        db_url = settings.database_url or settings.knowledge_database_url

    if db_url and db_url.strip():
        try:
            logger.info("Configuring PostgresRunRepository with db_url=%s...", db_url[:20])
            _global_run_repository = PostgresRunRepository(db_url)
            return _global_run_repository
        except Exception as exc:
            logger.warning(
                "Failed to initialize PostgresRunRepository (%s); falling back to in-memory store.",
                exc,
            )

    _global_run_repository = InMemoryRunRepository()
    return _global_run_repository


def set_run_repository(repo: RunRepository) -> None:
    """Override the run repository (useful for tests)."""
    global _global_run_repository
    _global_run_repository = repo
