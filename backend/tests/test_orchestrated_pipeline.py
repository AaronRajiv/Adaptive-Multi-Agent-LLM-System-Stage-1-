"""Integration and regression tests for OrchestratedPipeline (Stages 2–5)."""

import asyncio
from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from app.agents.analyst import AnalystAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.config import settings
from app.knowledge.chunking import FixedSizeTextChunker
from app.knowledge.embeddings import MockEmbeddingProvider
from app.knowledge.ingestion import DocumentIngestionService
from app.knowledge.models import KnowledgeDocument, RetrievedChunk, RetrievedContext
from app.knowledge.repository import InMemoryKnowledgeRepository
from app.knowledge.retrieval import RetrievalService
from app.llm.base import LLMProvider
from app.main import app
from app.models.schemas import EvaluationResult, PlannerOutput, ResearchResult, RunResponse
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus, TaskType
from app.orchestration.agents import (
    AnalysisTaskAgent,
    CapabilityRegistry,
    ResearchTaskAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


class CapturePromptLLMProvider(LLMProvider):
    """Mock LLM Provider that records all generated prompts for inspection."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.prompts: List[str] = []
        self.responses = responses or {}

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        sys_lower = (system_prompt or "").lower()
        if "synthesizer" in sys_lower:
            return "# Final Comprehensive Response\n\nExecutive summary and findings."
        if "analyst" in sys_lower:
            return "Analytical synthesis covering core dimensions and trade-offs."
        if "research" in sys_lower:
            return "Research findings on topic."
        return "Generic mock generation."

    async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
        self.prompts.append(prompt)
        if schema == PlannerOutput:
            return PlannerOutput(
                tasks=[
                    TaskNode(id="T1", description="Research cost and capital requirements", type=TaskType.RESEARCH),
                    TaskNode(id="T2", description="Research operational maintenance", type=TaskType.RESEARCH),
                    TaskNode(
                        id="T3",
                        description="Synthesize trade-offs and recommendations",
                        type=TaskType.ANALYSIS,
                        dependencies=["T1", "T2"],
                    ),
                ]
            )
        if schema == EvaluationResult:
            return EvaluationResult(
                score=92,
                status="PASS",
                feedback="Clear, coherent, and highly structured analytical synthesis.",
            )
        if schema.__name__ == "StrategyDecision":
            from app.orchestration.strategy import ExecutionStrategy, StrategyDecision
            req_rag = "with knowledge base" in prompt.lower()
            return StrategyDecision(
                strategy=ExecutionStrategy.MULTI_AGENT,
                reasoning="CapturePrompt mock strategy",
                estimated_complexity="HIGH",
                requires_rag=req_rag,
            )
        raise ValueError(f"Unexpected schema {schema}")


class DelayTrackingAgent(ResearchTaskAgent):
    """Tracks start and completion timestamps for concurrency verification."""

    def __init__(self, agent: ResearchAgent, delay_s: float = 0.05):
        super().__init__(agent)
        self.delay_s = delay_s
        self.events: List[tuple[str, str, float]] = []

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        loop = asyncio.get_running_loop()
        start = loop.time()
        self.events.append((context.task.id, "start", start))
        await asyncio.sleep(self.delay_s)
        end = loop.time()
        self.events.append((context.task.id, "end", end))
        return await super().execute(context)


@pytest.mark.asyncio
async def test_orchestrated_pipeline_executes_through_task_graph_and_orchestrator():
    """1. /api/run pipeline executes through Planner -> TaskGraph -> Orchestrator."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Analyze renewable energy vs fossil fuels")

    assert response.run_id is not None
    assert len(response.subtasks) == 3
    assert response.task_graph is not None
    assert len(response.task_graph.tasks) == 3
    # Check that task statuses were executed to completion
    assert all(t.status == TaskStatus.COMPLETED for t in response.subtasks)
    assert len(response.research_results) == 2
    assert "Analytical synthesis" in response.analysis
    assert response.evaluation.score == 92
    assert response.evaluation.status == "PASS"
    assert "Executive summary" in response.final_answer
    assert len(response.task_execution_traces) == 3


@pytest.mark.asyncio
async def test_orchestrated_pipeline_preserves_task_dependencies():
    """2. Dependencies between tasks are preserved and reflected in TaskGraph."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Compare EV and Petrol cars")

    analysis_task = next(t for t in response.subtasks if t.id == "T3")
    assert "T1" in analysis_task.dependencies
    assert "T2" in analysis_task.dependencies
    assert response.task_graph.get_task("T3").dependencies == ["T1", "T2"]


@pytest.mark.asyncio
async def test_orchestrated_pipeline_executes_independent_tasks_concurrently():
    """3. Independent tasks (T1 and T2) execute concurrently within Orchestrator."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings, max_concurrency=4)

    tracker = DelayTrackingAgent(pipeline.researcher, delay_s=0.08)
    pipeline.registry.register("research", tracker)

    await pipeline.execute("Evaluate battery storage technologies")

    t1_start = next(t for task_id, event, t in tracker.events if task_id == "T1" and event == "start")
    t2_start = next(t for task_id, event, t in tracker.events if task_id == "T2" and event == "start")
    t1_end = next(t for task_id, event, t in tracker.events if task_id == "T1" and event == "end")

    # T2 should have started before T1 ended (overlapping / concurrent execution)
    assert t2_start < t1_end


@pytest.mark.asyncio
async def test_orchestrated_pipeline_dependent_tasks_wait_for_prerequisites():
    """4. Dependent tasks (T3) wait for prerequisite tasks (T1, T2) to complete."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Analyze infrastructure dependencies")

    t1_trace = next(t for t in response.task_execution_traces if t.task_id == "T1")
    t2_trace = next(t for t in response.task_execution_traces if t.task_id == "T2")
    t3_trace = next(t for t in response.task_execution_traces if t.task_id == "T3")

    assert t3_trace.started_at >= t1_trace.ended_at or t3_trace.started_at >= t2_trace.ended_at


@pytest.mark.asyncio
async def test_orchestrated_pipeline_failed_task_propagation():
    """5. Failed prerequisite marks dependents BLOCKED in execution trace."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    # Replace research agent with a failing agent
    class FailingResearchAgent(ResearchTaskAgent):
        async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
            if context.task.id == "T1":
                raise RuntimeError("Simulated API failure on T1")
            return await super().execute(context)

    pipeline.registry.register("research", FailingResearchAgent(pipeline.researcher))

    response = await pipeline.execute("Analyze resilient architectures")

    t1_trace = next(t for t in response.task_execution_traces if t.task_id == "T1")
    t3_trace = next(t for t in response.task_execution_traces if t.task_id == "T3")

    assert t1_trace.status == TaskStatus.FAILED
    assert t3_trace.status == TaskStatus.BLOCKED


@pytest.mark.asyncio
async def test_orchestrated_pipeline_task_results_are_retained():
    """6. All completed task outputs are retained in research_results and execution traces."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Analyze solar energy feasibility")

    assert len(response.research_results) == 2
    assert response.research_results[0].subtask_id == "T1"
    assert response.research_results[0].findings == "Research findings on topic."
    assert len(response.task_execution_traces) == 3
    assert all(t.output is not None or t.error is not None for t in response.task_execution_traces)


@pytest.mark.asyncio
async def test_orchestrated_pipeline_optional_rag_invoked_only_when_requested():
    """7 & 8 & 9. RAG is invoked only when task has retrieval_query, grounds agent, retains provenance."""
    llm = CapturePromptLLMProvider()
    embedding_provider = MockEmbeddingProvider(dimensions=16)
    repository = InMemoryKnowledgeRepository()
    ingestion = DocumentIngestionService(FixedSizeTextChunker(chunk_size=100), embedding_provider, repository)
    await ingestion.ingest(
        KnowledgeDocument(
            id="kb-doc-1",
            source="solar_panel_data.pdf",
            content="Bifacial solar panels provide 15% higher energy yield in high-albedo environments.",
            metadata={"topic": "solar", "verified": True},
        )
    )
    retrieval_service = RetrievalService(embedding_provider, repository)

    class RAGPlannerLLMProvider(CapturePromptLLMProvider):
        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            self.prompts.append(prompt)
            if schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        TaskNode(
                            id="T1",
                            description="Investigate bifacial solar efficiency",
                            type=TaskType.RESEARCH,
                            metadata={"retrieval_query": "bifacial solar panels energy yield", "retrieval_top_k": 2},
                        ),
                        TaskNode(
                            id="T2",
                            description="Evaluate installation labor costs (parametric)",
                            type=TaskType.RESEARCH,
                        ),
                    ]
                )
            return await super().generate_structured(prompt, schema, system_prompt)

    rag_llm = RAGPlannerLLMProvider()
    pipeline = OrchestratedPipeline(
        llm_provider=rag_llm,
        settings=settings,
        retrieval_service=retrieval_service,
    )

    response = await pipeline.execute("Analyze solar efficiency with knowledge base")

    # 7: Provenance retained in response.retrieved_chunks
    assert len(response.retrieved_chunks) > 0
    chunk = response.retrieved_chunks[0]
    assert chunk.document_id == "kb-doc-1"
    assert chunk.source == "solar_panel_data.pdf"
    assert "Bifacial solar panels" in chunk.content

    # 8: Grounding reached agent prompt for T1 (which had retrieval_query)
    t1_prompts = [p for p in rag_llm.prompts if "Assigned Subtask ID: T1" in p]
    assert len(t1_prompts) == 1
    assert "KNOWLEDGE BASE CONTEXT:" in t1_prompts[0]
    assert "Bifacial solar panels provide 15% higher energy yield" in t1_prompts[0]

    # Grounding NOT in prompt for T2 (which had no retrieval_query)
    t2_prompts = [p for p in rag_llm.prompts if "Assigned Subtask ID: T2" in p]
    assert len(t2_prompts) == 1
    assert "KNOWLEDGE BASE CONTEXT:" not in t2_prompts[0]


@pytest.mark.asyncio
async def test_orchestrated_pipeline_final_synthesis_incorporates_outputs():
    """10. Final synthesis incorporates research findings, analysis, and evaluation."""
    llm = CapturePromptLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Synthesize strategic roadmap")

    synth_prompts = [p for p in llm.prompts if "Evaluator Score: 92/100" in p]
    assert len(synth_prompts) == 1
    assert "Collected Research Subtasks & Findings:" in synth_prompts[0]
    assert "Analyst Agent Findings" in synth_prompts[0]
    assert response.final_answer.startswith("# Final Comprehensive Response")


def test_api_run_endpoint_returns_extended_orchestrated_response(monkeypatch):
    """11. POST /api/run exposes the full Stage 2-5 response contract."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    client = TestClient(app)

    payload = {"task": "Compare wind vs solar power systems"}
    res = client.post("/api/run", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "run_id" in data
    assert "task_graph" in data
    assert data["task_graph"] is not None
    assert "tasks" in data["task_graph"]
    assert "task_execution_traces" in data
    assert len(data["task_execution_traces"]) > 0
    assert "subtasks" in data
    assert len(data["subtasks"]) > 0
    assert data["subtasks"][0]["status"] == "COMPLETED"
    assert "research_results" in data
    assert "analysis" in data
    assert "evaluation" in data
    assert data["evaluation"]["score"] is not None
    assert "final_answer" in data
    assert "execution_trace" in data


@pytest.mark.asyncio
async def test_evaluator_triggered_replanning_invokes_correct_research_interface():
    """Regression test: evaluator failure (<80) triggers replanning using execute_subtask without AttributeError."""
    class ReplanningLLMProvider(LLMProvider):
        def __init__(self):
            self.prompts: List[str] = []
            self.evaluation_calls = 0
            self.research_prompts: List[str] = []

        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            self.prompts.append(prompt)
            sys_lower = (system_prompt or "").lower()
            if "synthesizer" in sys_lower:
                return "# Final Response\n\nSynthesized answer including revised findings."
            if "analyst" in sys_lower:
                if "Revised research findings" in prompt:
                    return "Revised analysis synthesis including deep cost breakdown."
                return "Initial analytical synthesis."
            if "research" in sys_lower:
                self.research_prompts.append(prompt)
                if "Critique: Missing cost breakdown" in prompt:
                    return "Revised research findings: Deep cost breakdown addressing critique."
                return "Initial research findings on topic."
            return "Generic response."

        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            self.prompts.append(prompt)
            if schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        TaskNode(id="T1", description="Research cost analysis", type=TaskType.RESEARCH),
                        TaskNode(id="T2", description="Synthesize cost conclusions", type=TaskType.ANALYSIS, dependencies=["T1"]),
                    ]
                )
            if schema == EvaluationResult:
                self.evaluation_calls += 1
                if self.evaluation_calls == 1:
                    return EvaluationResult(
                        score=65,
                        status="FAIL",
                        feedback="Critique: Missing cost breakdown.",
                    )
                return EvaluationResult(
                    score=88,
                    status="PASS",
                    feedback="All requirements satisfied after replanning.",
                )
            if schema.__name__ == "StrategyDecision":
                from app.orchestration.strategy import ExecutionStrategy, StrategyDecision
                return StrategyDecision(
                    strategy=ExecutionStrategy.MULTI_AGENT,
                    reasoning="Complex multi-agent requirement",
                    estimated_complexity="HIGH",
                    requires_rag=False,
                )
            raise ValueError(f"Unexpected schema: {schema}")

    provider = ReplanningLLMProvider()
    pipeline = OrchestratedPipeline(llm_provider=provider, settings=settings)

    response = await pipeline.execute("Analyze enterprise cloud migration costs")

    # 1. Verify replanning triggered and completed successfully without AttributeError
    assert response.replanning_count == 1
    assert provider.evaluation_calls == 2

    # 2. Verify research prompt in replanning included evaluator critique
    replan_research_prompts = [p for p in provider.research_prompts if "Critique: Missing cost breakdown" in p]
    assert len(replan_research_prompts) == 1
    assert "Assigned Subtask ID: T1" in replan_research_prompts[0]

    # 3. Verify revised research findings were propagated to analyst, evaluation, and response
    assert response.research_results[0].findings == "Revised research findings: Deep cost breakdown addressing critique."
    assert "Revised analysis synthesis including deep cost breakdown." in response.analysis
    assert response.evaluation.score == 88
    assert response.evaluation.status == "PASS"

    # 4. Verify stage trace records TargetedReplanning stage
    replan_traces = [t for t in response.execution_trace if t.stage_name == "TargetedReplanning"]
    assert len(replan_traces) == 1
    assert replan_traces[0].status == "SUCCESS"
    assert "Revised evaluation score: 88/100" in replan_traces[0].summary

