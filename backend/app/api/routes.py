"""
API Endpoints for Adaptive Multi-Agent Web Application.
"""

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.events import event_bus_registry
from app.llm.base import (
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
)
from app.llm.provider import get_llm_provider
from app.models.schemas import (
    RunRequest,
    RunResponse,
    SystemStatusResponse,
)
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline
from app.pipeline.baseline_pipeline import get_recent_runs
from app.knowledge.repository import InMemoryKnowledgeRepository, KnowledgeRepository


def get_shared_knowledge_repository() -> KnowledgeRepository:
    if settings.knowledge_database_url and settings.knowledge_database_url.strip():
        from app.knowledge.repository import PostgresPgvectorKnowledgeRepository
        return PostgresPgvectorKnowledgeRepository(
            database_url=settings.knowledge_database_url.strip(),
            embedding_dimensions=settings.knowledge_embedding_dimensions,
        )
    return InMemoryKnowledgeRepository()


shared_knowledge_repository = get_shared_knowledge_repository()


router = APIRouter(prefix="/api", tags=["Multi-Agent Pipeline"])



@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """
    Check backend configuration status and LLM provider readiness.
    """
    is_ready = settings.is_api_key_configured
    if is_ready:
        if settings.llm_provider.lower() == "mock":
            msg = "System is running in Mock Mode (offline deterministic simulation for testing)."
        else:
            msg = f"System is configured with {settings.llm_provider.upper()} ({settings.llm_model}). Ready to process tasks."
    else:
        msg = (
            f"LLM Provider '{settings.llm_provider}' is missing an API key. "
            "Please set LLM_API_KEY in backend/.env to execute live tasks, "
            "or set LLM_PROVIDER=mock for offline development."
        )

    return SystemStatusResponse(
        configured=is_ready,
        provider=settings.llm_provider,
        model=settings.llm_model,
        evaluator_threshold=settings.evaluator_pass_threshold,
        message=msg,
    )


@router.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest) -> RunResponse:
    """
    Execute the Orchestrated Multi-Agent Pipeline.

    Orchestrated steps:
    Planner -> TaskGraph -> Orchestrator (Parallel + Capabilities + RAG) -> Evaluator -> Synthesizer

    An SSE event stream for this run is available at GET /api/runs/{run_id}/events
    once the run_id is returned.
    """
    task_text = request.task.strip()
    if not task_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Task prompt cannot be empty. Please provide a task or question.",
        )

    # 1. Initialize LLM Provider with clear configuration error handling
    try:
        provider = get_llm_provider(settings)
    except LLMConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "ConfigurationError",
                "message": str(e),
                "hint": "Set LLM_API_KEY in backend/.env or set LLM_PROVIDER=mock for testing.",
            },
        )

    import uuid
    run_id = request.run_id or str(uuid.uuid4())
    event_bus = await event_bus_registry.get_or_create(run_id)

    # 3. Run Orchestrated Pipeline (Stage 2-5 Graph + Orchestrator)
    pipeline = OrchestratedPipeline(
        llm_provider=provider,
        settings=settings,
        knowledge_repository=shared_knowledge_repository,
        event_bus=event_bus,
        run_id=run_id,
    )

    try:
        result = await pipeline.execute(
            task_text,
            documents=request.documents,
            history=request.history,
        )
        return result
    except LLMStructuredOutputError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_type": "StructuredOutputError",
                "message": f"Agent structured output validation failed: {str(e)}",
            },
        )
    except LLMProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_type": "ProviderError",
                "message": f"Downstream LLM provider error: {str(e)}",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalPipelineError",
                "message": f"An unexpected pipeline error occurred: {str(e)}",
            },
        )


@router.get("/runs", response_model=List[RunResponse])
async def list_recent_runs() -> List[RunResponse]:
    """
    Retrieve in-memory execution traces for recently executed runs.
    """
    return get_recent_runs()


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    """
    Server-Sent Events stream for a pipeline execution run.

    Streams typed ExecutionEvent objects as they are emitted by the pipeline.
    The connection closes after RUN_COMPLETED or RUN_FAILED.

    Late subscribers receive the full buffered event history before live events.
    """
    bus = await event_bus_registry.get_or_create(run_id)

    async def event_generator():
        queue = await bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send SSE keep-alive comment to prevent proxy/browser timeouts
                    yield ": keepalive\n\n"
                    continue

                if event is None:
                    # Run finished
                    break
                yield event.to_sse()
        finally:
            await bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
