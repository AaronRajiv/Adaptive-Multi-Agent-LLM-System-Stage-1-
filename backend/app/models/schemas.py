"""
Data models and schemas for Stage 1 Baseline Multi-Agent System.

Provides strict Pydantic models for agent inputs, outputs, validation,
and API request/response contracts.
"""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field

from app.models.task_graph import TaskGraph, TaskNode


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


class EvaluationResult(BaseModel):
    """Structured output emitted by the Evaluator Agent."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Evaluation quality score from 0 to 100"
    )
    status: Literal["PASS", "FAIL"] = Field(
        ...,
        description="PASS if score >= threshold, otherwise FAIL"
    )
    feedback: str = Field(
        ...,
        description="Detailed constructive critique evaluating correctness, completeness, relevance, and clarity"
    )


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


class RunRequest(BaseModel):
    """API Request payload to execute a user task."""

    task: str = Field(
        ...,
        min_length=3,
        description="High-level user task to be decomposed and solved"
    )


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
    task_graph: Optional[TaskGraph] = Field(default=None, description="Executed TaskGraph with runtime statuses and dependencies")
    retrieved_chunks: List[Any] = Field(default_factory=list, description="All retrieved knowledge chunks with provenance")
    task_execution_traces: List[TaskExecutionTrace] = Field(default_factory=list, description="Per-task execution traces from Orchestrator")


class SystemStatusResponse(BaseModel):
    """API Response reporting the backend configuration and provider status."""

    configured: bool = Field(..., description="True if LLM provider has valid credentials, False otherwise")
    provider: str = Field(..., description="Configured LLM provider name")
    model: str = Field(..., description="Configured LLM model name")
    evaluator_threshold: int = Field(..., description="Evaluator pass threshold")
    message: str = Field(..., description="Human-readable status summary")
