"""
API Endpoints for Adaptive Multi-Agent Web Application.
Provides document lifecycle endpoints, SSE streams, async run handling, and pipeline orchestration.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.events import ExecutionEvent, ExecutionEventType, event_bus_registry
from app.persistence.run_repository import get_run_repository
from app.knowledge.document_service import (
    DocumentEvent,
    DocumentMetadata,
    DocumentService,
    DocumentStatus,
    DocumentUploadResponse,
)
from app.knowledge.embeddings import MockEmbeddingProvider, get_embedding_provider
from app.knowledge.repository import InMemoryKnowledgeRepository, KnowledgeRepository
from app.llm.base import (
    LLMConfigurationError,
    LLMProviderError,
    LLMStructuredOutputError,
)
from app.llm.provider import get_llm_provider
from app.logging_config import get_logger
from app.models.schemas import (
    AsyncRunResponse,
    RunRequest,
    RunResponse,
    SystemStatusResponse,
)
from app.orchestration.resource_budget import ResourceBudgetExceededError
from app.pipeline.baseline_pipeline import get_recent_runs
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline

logger = get_logger("app.api.routes")


def get_shared_knowledge_repository() -> KnowledgeRepository:
    if settings.knowledge_database_url and settings.knowledge_database_url.strip():
        from app.knowledge.repository import PostgresPgvectorKnowledgeRepository

        return PostgresPgvectorKnowledgeRepository(
            database_url=settings.knowledge_database_url.strip(),
            embedding_dimensions=settings.knowledge_embedding_dimensions,
        )
    return InMemoryKnowledgeRepository()


shared_knowledge_repository = get_shared_knowledge_repository()
_shared_document_service: Optional[DocumentService] = None


def get_shared_document_service() -> DocumentService:
    global _shared_document_service
    if _shared_document_service is None:
        try:
            embedder = get_embedding_provider(settings)
        except Exception:
            embedder = MockEmbeddingProvider()

        _shared_document_service = DocumentService(
            repository=shared_knowledge_repository,
            embedding_provider=embedder,
            chunk_size=getattr(settings, "knowledge_chunk_size", 1200),
            chunk_overlap=getattr(settings, "knowledge_chunk_overlap", 120),
            embedding_batch_size=getattr(settings, "embedding_batch_size", 32),
            embedding_rate_limit_delay_seconds=getattr(settings, "embedding_rate_limit_delay_seconds", 0.0),
        )
    return _shared_document_service


# Registry of run execution results & async tasks
RUNS_INDEX: Dict[str, Dict[str, Any]] = {}

router = APIRouter(prefix="/api", tags=["Multi-Agent Pipeline"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """Check backend configuration status and LLM provider readiness."""
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


# =====================================================================
# Document Service Endpoints (Decoupled ingestion, dedup & status)
# =====================================================================


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
) -> DocumentUploadResponse:
    """
    Upload a document for decoupled background extraction, chunking, and embedding.
    Deduplicates by SHA-256 and returns immediately with document_id.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    doc_service = get_shared_document_service()
    return await doc_service.register_and_start_ingestion(
        filename=file.filename,
        file_bytes=content_bytes,
        label=label,
    )


@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents() -> List[DocumentMetadata]:
    """List all registered documents with their status and chunk progress."""
    doc_service = get_shared_document_service()
    return doc_service.list_documents()


@router.get("/documents/{document_id}/status", response_model=DocumentMetadata)
async def get_document_status(document_id: str) -> DocumentMetadata:
    """Get the current ingestion status and chunk count of a specific document."""
    doc_service = get_shared_document_service()
    doc = doc_service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and purge all associated chunks and embeddings."""
    doc_service = get_shared_document_service()
    deleted = await doc_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return {"document_id": document_id, "deleted": True}


@router.get("/documents/{document_id}/events")
async def stream_document_events(document_id: str):
    """Server-Sent Events stream for background document ingestion progress."""
    doc_service = get_shared_document_service()
    doc = doc_service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    bus = await doc_service.get_or_create_event_bus(document_id)

    async def event_generator():
        queue = await bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if event is None:
                    break

                data = json.dumps(event.model_dump())
                yield f"event: {event.status.value}\ndata: {data}\n\n"
                if event.status in (DocumentStatus.READY, DocumentStatus.FAILED):
                    break
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


# =====================================================================
# Pipeline Execution Endpoints
# =====================================================================

ACTIVE_RUN_TASKS: Dict[str, asyncio.Task] = {}


async def _run_async_worker(
    pipeline: OrchestratedPipeline, request: RunRequest, run_id: str
) -> None:
    """Background worker for asynchronous pipeline execution with timeout, cancel handling, and persistence."""
    RUNS_INDEX[run_id] = {
        "run_id": run_id,
        "status": "running",
        "result": None,
        "error": None,
    }
    run_repo = get_run_repository(settings)
    try:
        result = await asyncio.wait_for(
            pipeline.execute(
                request.task.strip(),
                documents=request.documents,
                document_ids=request.document_ids,
                history=request.history,
            ),
            timeout=settings.run_timeout_seconds,
        )
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "completed",
            "result": result.model_dump(),
            "error": None,
        }
        await run_repo.save_run(result)
        if result.metrics:
            from app.telemetry.metrics import global_metrics_collector
            m = global_metrics_collector.get_metrics(run_id)
            if m:
                await run_repo.save_metrics(m)
    except asyncio.CancelledError:
        logger.warning("Pipeline execution cancelled for run_id=%s", run_id)
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "cancelled",
            "result": None,
            "error": "Pipeline execution cancelled by client.",
        }
    except asyncio.TimeoutError:
        logger.error(
            "Pipeline execution timed out for run_id=%s after %ss",
            run_id,
            settings.run_timeout_seconds,
        )
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": f"Pipeline execution timed out after {settings.run_timeout_seconds} seconds.",
        }
        bus = await event_bus_registry.get(run_id)
        if bus:
            await bus.emit(
                ExecutionEvent(
                    run_id=run_id,
                    event_type=ExecutionEventType.RUN_FAILED,
                    source="TIMEOUT",
                    data={"error": f"Execution timed out after {settings.run_timeout_seconds}s"},
                )
            )
    except ResourceBudgetExceededError as rbe:
        logger.error("Resource budget exceeded in async worker for run_id=%s: %s", run_id, str(rbe))
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": f"Resource budget exceeded: {str(rbe)}",
        }
    except Exception as exc:
        logger.error(
            "Async pipeline execution failed for run_id=%s: %s",
            run_id,
            exc,
            exc_info=True,
        )
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": str(exc),
        }
    finally:
        ACTIVE_RUN_TASKS.pop(run_id, None)


@router.post("/run")
async def run_pipeline(request: RunRequest):
    """
    Execute the Orchestrated Multi-Agent Pipeline.
    Supports concurrency throttling, per-run timeout, cancel registration, and persistence.
    """
    task_text = request.task.strip()
    if not task_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Task prompt cannot be empty. Please provide a task or question.",
        )

    # Concurrency limit enforcement
    if len(ACTIVE_RUN_TASKS) >= settings.max_concurrent_runs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Server is currently processing maximum concurrent runs ({settings.max_concurrent_runs}). Please try again shortly.",
        )

    # Validate referenced document_ids readiness before starting
    doc_service = get_shared_document_service()
    if request.document_ids:
        for doc_id in request.document_ids:
            meta = doc_service.get_document(doc_id)
            if not meta:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Referenced document '{doc_id}' does not exist.",
                )
            if meta.status != DocumentStatus.READY:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Referenced document '{doc_id}' is not ready (status: '{meta.status.value}'). "
                        "Ingestion must complete before running."
                    ),
                )

    # Initialize LLM Provider with clear configuration error handling
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

    run_id = request.run_id or str(uuid.uuid4())
    logger.info("Starting pipeline execution for run_id=%s, task='%s'", run_id, task_text[:100])
    event_bus = await event_bus_registry.get_or_create(run_id)

    pipeline = OrchestratedPipeline(
        llm_provider=provider,
        settings=settings,
        knowledge_repository=shared_knowledge_repository,
        document_service=doc_service,
        event_bus=event_bus,
        run_id=run_id,
    )

    # Asynchronous execution mode: returns immediately with run_id
    if request.async_run:
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "running",
            "result": None,
            "error": None,
        }
        task = asyncio.create_task(_run_async_worker(pipeline, request, run_id))
        ACTIVE_RUN_TASKS[run_id] = task
        return AsyncRunResponse(
            run_id=run_id,
            status="started",
            message="Pipeline execution started asynchronously. Stream events or poll GET /api/runs/{run_id}.",
        )

    # Synchronous execution mode (for backward compatibility and test suites)
    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_RUN_TASKS[run_id] = current_task

    try:
        result = await asyncio.wait_for(
            pipeline.execute(
                task_text,
                documents=request.documents,
                document_ids=request.document_ids,
                history=request.history,
            ),
            timeout=settings.run_timeout_seconds,
        )
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "completed",
            "result": result.model_dump(),
            "error": None,
        }
        run_repo = get_run_repository(settings)
        await run_repo.save_run(result)
        if result.metrics:
            from app.telemetry.metrics import global_metrics_collector
            m = global_metrics_collector.get_metrics(run_id)
            if m:
                await run_repo.save_metrics(m)

        logger.info(
            "Pipeline completed successfully for run_id=%s, strategy=%s",
            run_id,
            result.execution_strategy,
        )
        return result
    except asyncio.CancelledError:
        logger.warning("Synchronous pipeline execution cancelled for run_id=%s", run_id)
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "cancelled",
            "result": None,
            "error": "Pipeline execution cancelled by client.",
        }
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline execution cancelled by client.",
        )
    except asyncio.TimeoutError:
        logger.error(
            "Synchronous pipeline execution timed out for run_id=%s after %ss",
            run_id,
            settings.run_timeout_seconds,
        )
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": f"Pipeline execution timed out after {settings.run_timeout_seconds} seconds.",
        }
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_type": "TimeoutError",
                "message": f"Pipeline execution timed out after {settings.run_timeout_seconds} seconds.",
            },
        )
    except LLMStructuredOutputError as e:
        logger.error("Structured output validation failed for run_id=%s: %s", run_id, str(e))
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_type": "StructuredOutputError",
                "message": f"Agent structured output validation failed: {str(e)}",
            },
        )
    except LLMProviderError as e:
        logger.error("LLM provider error for run_id=%s: %s", run_id, str(e))
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_type": "ProviderError",
                "message": f"Downstream LLM provider error: {str(e)}",
            },
        )
    except ResourceBudgetExceededError as e:
        logger.error("Resource budget exceeded for run_id=%s: %s", run_id, str(e))
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_type": "ResourceBudgetExceededError",
                "message": f"Execution resource budget exceeded: {str(e)}",
                "limit_type": getattr(e, "limit_type", "unknown"),
                "current_value": getattr(e, "current_value", None),
                "limit_value": getattr(e, "limit_value", None),
            },
        )
    except Exception as e:
        logger.error("Internal pipeline error for run_id=%s: %s", run_id, str(e), exc_info=True)
        RUNS_INDEX[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "result": None,
            "error": str(e),
        }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalPipelineError",
                "message": f"An unexpected pipeline error occurred: {str(e)}",
            },
        )
    finally:
        ACTIVE_RUN_TASKS.pop(run_id, None)
        bus = await event_bus_registry.get(run_id)
        if bus:
            try:
                run_repo = get_run_repository(settings)
                await run_repo.save_events(run_id, bus.history)
            except Exception as e:
                logger.error("Failed to persist events for run_id=%s: %s", run_id, str(e))


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel an active pipeline execution by run_id."""
    task = ACTIVE_RUN_TASKS.get(run_id)
    if not task or task.done():
        if run_id in RUNS_INDEX:
            return {
                "run_id": run_id,
                "status": RUNS_INDEX[run_id]["status"],
                "message": f"Run '{run_id}' is already {RUNS_INDEX[run_id]['status']}.",
            }
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active run '{run_id}' not found or already terminated.",
        )

    task.cancel()
    RUNS_INDEX[run_id] = {
        "run_id": run_id,
        "status": "cancelled",
        "result": None,
        "error": "Execution cancelled by user request.",
    }
    bus = await event_bus_registry.get(run_id)
    if bus:
        await bus.emit(
            ExecutionEvent(
                run_id=run_id,
                event_type=ExecutionEventType.RUN_CANCELLED,
                source="USER",
                data={"error": "Pipeline execution cancelled by user request."},
            )
        )
        try:
            run_repo = get_run_repository(settings)
            await run_repo.save_events(run_id, bus.history)
        except Exception:
            pass
    return {
        "run_id": run_id,
        "status": "cancelled",
        "message": "Pipeline execution successfully cancelled.",
    }


@router.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    """Retrieve run execution status or completed RunResponse by run_id."""
    if run_id in RUNS_INDEX:
        return RUNS_INDEX[run_id]

    run_repo = get_run_repository(settings)
    stored = await run_repo.get_run(run_id)
    if stored:
        return {
            "run_id": run_id,
            "status": "completed",
            "result": stored.model_dump(),
            "error": None,
        }

    for r in get_recent_runs():
        if r.run_id == run_id:
            return {
                "run_id": run_id,
                "status": "completed",
                "result": r.model_dump(),
                "error": None,
            }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Run '{run_id}' not found.",
    )


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    """Retrieve telemetry metrics (tokens, latency, cost, and call breakdowns) for a run."""
    from app.telemetry.metrics import global_metrics_collector

    metrics = global_metrics_collector.get_metrics(run_id)
    if metrics:
        return metrics.model_dump()

    run_repo = get_run_repository(settings)
    stored_metrics = await run_repo.get_metrics(run_id)
    if stored_metrics:
        return stored_metrics.model_dump()

    if run_id in RUNS_INDEX:
        run_res = RUNS_INDEX[run_id].get("result")
        if run_res and isinstance(run_res, dict) and run_res.get("metrics"):
            return run_res["metrics"]

    for r in get_recent_runs():
        if r.run_id == run_id and r.metrics:
            return r.metrics

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Metrics for run '{run_id}' not found.",
    )


@router.get("/runs", response_model=List[RunResponse])
async def list_recent_runs() -> List[RunResponse]:
    """Retrieve execution traces for recently executed runs."""
    run_repo = get_run_repository(settings)
    repo_runs = await run_repo.list_runs()
    if repo_runs:
        return repo_runs
    return get_recent_runs()


@router.get("/runs/{run_id}/events")
async def get_or_stream_run_events(
    run_id: str,
    request: Request,
    after_seq: Optional[int] = Query(None, description="Return events strictly after this sequence number"),
    stream: Optional[bool] = Query(None, description="Force streaming SSE mode if true, or JSON list if false"),
):
    """
    Server-Sent Events stream or JSON replay for pipeline execution events.
    Supports ?after_seq=N for reconnects and replay without duplicated events.
    """
    accept = request.headers.get("accept", "")
    is_json = stream is False or ("application/json" in accept and "text/event-stream" not in accept)

    run_repo = get_run_repository(settings)

    if is_json:
        # Non-SSE request: return JSON replay list directly
        bus = await event_bus_registry.get(run_id)
        if bus and bus.history:
            events = await bus.get_events(after_seq=after_seq)
        else:
            events = await run_repo.get_events(run_id, after_seq=after_seq)

        if not events:
            # Check if run exists at all
            if run_id not in RUNS_INDEX and not await run_repo.get_run(run_id):
                if not any(r.run_id == run_id for r in get_recent_runs()):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Run '{run_id}' not found.",
                    )
        return [event.model_dump() for event in events]

    # SSE streaming mode
    bus = await event_bus_registry.get_or_create(run_id)
    if not bus.history:
        persisted = await run_repo.get_events(run_id)
        if persisted:
            async with bus._lock:
                bus._history.extend(persisted)

    async def event_generator():
        queue = await bus.subscribe(after_seq=after_seq)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if event is None:
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


class EvidencePassageResponse(BaseModel):
    evidence_id: str
    chunk_id: Optional[str] = None
    citation: Optional[str] = None
    source_type: str = "document"
    document_id: Optional[str] = None
    document_label: Optional[str] = None
    title: Optional[str] = None
    article_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    url: Optional[str] = None
    text: str = ""
    similarity_score: Optional[float] = None
    retrieval_timestamp: Optional[str] = None


@router.get("/runs/{run_id}/evidence/{evidence_id}", response_model=EvidencePassageResponse)
async def get_run_evidence(run_id: str, evidence_id: str):
    """
    Retrieve the stored passage for an evidence item or citation in a run,
    allowing citation clicks in the UI to open verbatim source cards.
    """
    run_repo = get_run_repository(settings)
    run_obj: Optional[Dict[str, Any]] = None

    if run_id in RUNS_INDEX and RUNS_INDEX[run_id].get("result"):
        res = RUNS_INDEX[run_id]["result"]
        run_obj = res if isinstance(res, dict) else res.model_dump()
    else:
        stored = await run_repo.get_run(run_id)
        if stored:
            run_obj = stored.model_dump()
        else:
            for r in get_recent_runs():
                if r.run_id == run_id:
                    run_obj = r.model_dump()
                    break

    if not run_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    norm_target = evidence_id.strip().strip("[]").lower()
    hash_target = norm_target[4:] if norm_target.startswith("web:") else norm_target

    # 1. Search in evidence_ledger
    ledger = run_obj.get("evidence_ledger") or {}
    items = ledger.get("items", {}) if isinstance(ledger, dict) else {}
    if isinstance(items, dict):
        item_list = list(items.values())
    elif isinstance(items, list):
        item_list = items
    else:
        item_list = []

    matched_item = None
    for item in item_list:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).lower()
        citation = str(item.get("citation", "")).strip().strip("[]").lower()
        content_hash = str(item.get("content_hash", "")).lower()
        article_path = str(item.get("article_path", "")).lower()
        url = str(item.get("url", "")).lower()

        # Exact ID or citation match
        if chunk_id == norm_target or citation == norm_target:
            matched_item = item
            break
        # Hash match
        if hash_target and (content_hash.startswith(hash_target) or hash_target in chunk_id or hash_target in citation):
            matched_item = item
            break
        # Article match (e.g. "art 3.4.1" or "3.4.1")
        if article_path and (norm_target == article_path or norm_target == f"art {article_path}" or f"art {article_path}" in norm_target):
            matched_item = item
            break
        # URL match
        if url and (norm_target in url or url in norm_target):
            matched_item = item
            break

    # 2. Search in retrieved_chunks if not matched yet
    if not matched_item:
        chunks = run_obj.get("retrieved_chunks", [])
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            c_id = str(chunk.get("id", "")).lower()
            meta = chunk.get("metadata", {}) or {}
            c_art = str(meta.get("article_path", meta.get("article", ""))).lower()
            if c_id == norm_target or (c_art and (norm_target == c_art or norm_target == f"art {c_art}" or f"art {c_art}" in norm_target)):
                matched_item = {
                    "chunk_id": chunk.get("id"),
                    "citation": meta.get("citation", f"[{meta.get('label', 'doc')}]"),
                    "source_type": "document",
                    "document_id": chunk.get("document_id"),
                    "document_label": meta.get("label") or meta.get("document_label"),
                    "title": meta.get("heading") or meta.get("document_label"),
                    "article_path": meta.get("article_path"),
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "text": chunk.get("content", ""),
                    "similarity_score": chunk.get("similarity_score"),
                }
                break

    if not matched_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence passage '{evidence_id}' not found in run '{run_id}'.",
        )

    return EvidencePassageResponse(
        evidence_id=evidence_id,
        chunk_id=matched_item.get("chunk_id"),
        citation=matched_item.get("citation"),
        source_type=matched_item.get("source_type", "document"),
        document_id=matched_item.get("document_id"),
        document_label=matched_item.get("document_label"),
        title=matched_item.get("heading") or matched_item.get("title") or matched_item.get("document_label"),
        article_path=matched_item.get("article_path"),
        page_start=matched_item.get("page_start"),
        page_end=matched_item.get("page_end"),
        url=matched_item.get("url"),
        text=matched_item.get("text", ""),
        similarity_score=matched_item.get("similarity_score"),
        retrieval_timestamp=matched_item.get("retrieval_timestamp"),
    )
