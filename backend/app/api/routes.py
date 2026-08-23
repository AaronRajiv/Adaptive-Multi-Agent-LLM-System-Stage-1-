"""
API Endpoints for Stage 1 Baseline Multi-Agent Web Application.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status

from app.config import settings
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
from app.pipeline.baseline_pipeline import (
    BaselinePipeline,
    get_recent_runs,
    record_run,
)

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
    Execute the Stage 1 Fixed Baseline Multi-Agent Pipeline.

    Sequential steps:
    Planner -> Research -> Analyst -> Evaluator -> Synthesizer -> Final Answer
    """
    task_text = request.task.strip()
    if len(task_text) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Task prompt is too short. Please provide a substantive task description.",
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

    # 2. Run Baseline Pipeline
    pipeline = BaselinePipeline(llm_provider=provider, settings=settings)

    try:
        result = await pipeline.execute(task_text)
        record_run(result)
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
