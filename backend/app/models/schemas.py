"""
Data models and schemas for Stage 1 Baseline Multi-Agent System.

Provides strict Pydantic models for agent inputs, outputs, validation,
and API request/response contracts.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.logging_config import get_logger
from app.models.task_graph import TaskGraph, TaskNode

logger = get_logger("app.models.schemas")


# Backward-compatible Stage 1 name for graph-compatible planned work.
SubTask = TaskNode


class PlannerOutput(BaseModel):
    """Structured output emitted by the Planner Agent."""

    tasks: List[TaskNode] = Field(
        ...,
        min_length=1,
        description="List of subtasks decomposing the user task"
    )

    def to_task_graph(self) -> TaskGraph:
        """Convert planner output into a validated graph without executing it."""
        return TaskGraph(tasks=self.tasks)


class ResearchResult(BaseModel):
    """Result of a Research Agent executing on a single research subtask."""

    subtask_id: str = Field(..., description="ID of the research subtask")
    subtask_description: str = Field(..., description="Description of the assigned subtask")
    findings: str = Field(..., description="Parametric knowledge findings generated for this subtask")
    citations: List[str] = Field(default_factory=list, description="Citations extracted from findings, e.g. ['[FIA 2025, art 3.4.1, p.24]']")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks referenced by findings")
    missing_evidence: Optional[str] = Field(default=None, description="Explicit statement of missing evidence or gaps")


class EvaluationResult(BaseModel):
    """Structured output emitted by the Evaluator Agent."""

    score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Evaluation quality score from 0 to 100, or None if evaluation was skipped/not applicable"
    )
    status: Literal["PASS", "FAIL", "SKIPPED", "NOT_APPLICABLE"] = Field(
        ...,
        description="PASS/FAIL for evaluated tasks, or SKIPPED/NOT_APPLICABLE when formal evaluation is bypassed"
    )
    feedback: str = Field(
        ...,
        description="Constructive critique evaluating correctness, completeness, relevance, and clarity"
    )
    groundedness_score: Optional[int] = Field(
        default=None, ge=0, le=25, description="Groundedness and source fidelity (0-25 pts)"
    )
    coverage_score: Optional[int] = Field(
        default=None, ge=0, le=25, description="Coverage of every domain in the question (0-25 pts)"
    )
    distinction_score: Optional[int] = Field(
        default=None, ge=0, le=25, description="Distinction between explicit regulatory and inference (0-25 pts)"
    )
    honesty_score: Optional[int] = Field(
        default=None, ge=0, le=25, description="Honesty about limits and unknowns without definitive claims (0-25 pts)"
    )
    uncovered_domains: List[str] = Field(
        default_factory=list, description="List of technical domains lacking sufficient coverage or evidence"
    )
    verification_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Deterministic programmatic grounding verification metrics"
    )
    score_normalized: bool = Field(
        default=False,
        description="True if any subscore exceeded 25 and had to be rescaled from 0-100 scale",
    )

    @model_validator(mode="before")
    @classmethod
    def check_and_rescale_subscores(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = False
        for field_name in ("groundedness_score", "coverage_score", "distinction_score", "honesty_score"):
            raw = data.get(field_name)
            if raw is not None:
                try:
                    val = int(raw)
                    if val > 25:
                        logger.warning(
                            "Evaluator subscore '%s' exceeds 25 (received %d). Rescaling from 0-100 to 0-25 scale and setting score_normalized=True.",
                            field_name, val,
                        )
                        data[field_name] = min(25, max(0, round(val * 25 / 100)))
                        normalized = True
                except (ValueError, TypeError):
                    pass
        if normalized:
            data["score_normalized"] = True
        return data


class StageTrace(BaseModel):
    """Observable record of a single stage execution in the baseline pipeline."""

    stage_name: str = Field(..., description="Name of the pipeline stage (e.g. 'Planner', 'Research', 'Analysis')")
    agent_name: str = Field(..., description="Name of the agent executing the stage")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    status: Literal["SUCCESS", "FAILED"] = Field(default="SUCCESS", description="Execution status")
    summary: Optional[str] = Field(default=None, description="Brief summary of output or error")


class TaskExecutionTrace(BaseModel):
    """Lightweight audit record for an attempted or blocked task."""

    task_id: str
    capability: str
    agent_name: Optional[str] = None
    started_at: str
    ended_at: str
    duration_ms: float
    status: Any
    success: bool
    output: Any = None
    error: Optional[str] = None


class DocumentPayload(BaseModel):
    """Payload representing an attached document for RAG ingestion."""

    filename: str = Field(..., description="Document filename or title")
    content: str = Field(..., description="Full text content of the document")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class ChatMessagePayload(BaseModel):
    """Previous conversation turn for multi-turn chat memory."""

    role: Literal["user", "assistant"] = Field(..., description="Message sender role")
    content: str = Field(..., description="Message text content")


class RunRequest(BaseModel):
    """API Request payload to execute a user task."""

    task: str = Field(
        ...,
        min_length=1,
        description="High-level user task to be decomposed and solved"
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional pre-assigned run ID to enable early SSE subscription prior to run completion"
    )
    documents: Optional[List[DocumentPayload]] = Field(
        default=None,
        description="Optional list of attached documents to ingest into RAG knowledge base for context grounding"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of pre-ingested document IDs to ground retrieval on"
    )
    async_run: Optional[bool] = Field(
        default=False,
        description="When True, pipeline executes asynchronously and returns immediately with run_id"
    )
    history: Optional[List[ChatMessagePayload]] = Field(
        default=None,
        description="Optional conversation history turns for multi-turn chat context"
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model identifier override (e.g. 'nvidia/nemotron-3-super-120b-a12b', 'gemini-3.1-flash-lite')"
    )


class AsyncRunResponse(BaseModel):
    """Response returned when run_pipeline is invoked with async_run=True."""

    run_id: str = Field(..., description="Unique UUID for this pipeline execution run")
    status: str = Field(default="started", description="Current status of the asynchronous run")
    message: str = Field(default="Pipeline execution started asynchronously", description="Status message")


class RunResponse(BaseModel):
    """API Response containing complete execution results and observable trace."""

    run_id: str = Field(..., description="Unique UUID for this pipeline execution run")
    user_task: str = Field(..., description="Original user task")
    subtasks: List[SubTask] = Field(..., description="Decomposed subtasks from Planner Agent")
    research_results: List[ResearchResult] = Field(..., description="Findings from Research Agents")
    analysis: str = Field(..., description="Synthesized analysis from Analyst Agent")
    evaluation: EvaluationResult = Field(..., description="Quality assessment from Evaluator Agent")
    final_answer: str = Field(..., description="Final user-facing response from Synthesizer Agent")
    execution_trace: List[StageTrace] = Field(..., description="Observable trace with per-stage measured latencies")
    created_at: str = Field(..., description="ISO 8601 timestamp of execution")

    # Stage 2-5 Extensions:
    execution_strategy: Optional[str] = Field(default="MULTI_AGENT", description="Selected execution strategy: DIRECT, SINGLE_AGENT, or MULTI_AGENT")
    replanning_count: int = Field(default=0, description="Number of targeted replanning iterations performed")
    task_graph: Optional[TaskGraph] = Field(default=None, description="Executed TaskGraph with runtime statuses and dependencies")
    retrieved_chunks: List[Any] = Field(default_factory=list, description="All retrieved knowledge chunks with provenance")
    task_execution_traces: List[TaskExecutionTrace] = Field(default_factory=list, description="Per-task execution traces from Orchestrator")
    evidence_ledger: Optional[Dict[str, Any]] = Field(default=None, description="Shared canonical Evidence Ledger for the run")
    verification_result: Optional[Dict[str, Any]] = Field(default=None, description="Programmatic grounding verification metrics from verifier")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Per-run resource, token, latency, and cost metrics")


class SystemStatusResponse(BaseModel):
    """API Response reporting the backend configuration and provider status."""

    configured: bool = Field(..., description="True if LLM provider has valid credentials, False otherwise")
    provider: str = Field(..., description="Configured LLM provider name")
    model: str = Field(..., description="Configured LLM model name")
    evaluator_threshold: int = Field(..., description="Evaluator pass threshold")
    message: str = Field(..., description="Human-readable status summary")
    available_models: List[str] = Field(default_factory=list, description="List of supported model identifiers")


class ModelInfo(BaseModel):
    """Metadata describing an available LLM model."""

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="User-facing display name")
    provider: str = Field(..., description="Provider name (nvidia, gemini, mock)")
    description: Optional[str] = Field(default=None, description="Brief description")
    recommended: bool = Field(default=False, description="Whether recommended by default")
    badge: Optional[str] = Field(default=None, description="Short tag or badge text")


class ModelListResponse(BaseModel):
    """Response returned by GET /api/models listing all available models."""

    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model ID")
