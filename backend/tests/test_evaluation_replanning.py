"""
Phase 6 Tests: Grounding Evaluator and Replanning.

Verifies:
1. Programmatic Verifier (No LLM):
   - Extracts citations [doc, art X.Y.Z, p.N] from text.
   - Validates existence in EvidenceLedger.
   - Validates numeric quantities against cited chunks.
   - Identifies unsupported claims and computes citation_validity %.
2. Evaluator Agent Rubric:
   - Evaluates the FINAL answer against 4 dimensions:
     Groundedness (25), Coverage (25), Distinction (25), Honesty (25).
   - Emits sub-scores and identifies uncovered domains.
   - Integrates programmatic verifier audit findings into evaluation prompt.
3. Targeted Iterative Replanning:
   - Triggers when evaluator status == FAIL or verifier fails.
   - Identifies weak/uncovered domains.
   - Refines retrieval queries, pulls new chunks into EvidenceLedger.
   - Re-executes only targeted research subtasks.
   - Re-synthesizes and re-evaluates.
   - Respects configurable max_replanning_rounds (default 2).
   - Emits REPLANNING_STARTED, REPLAN_STARTED, REPLANNING_COMPLETED, REPLAN_COMPLETED.
   - Tracks whether score improved.
"""

import pytest
from typing import Any, Dict, List, Optional

from app.agents.evaluator import EvaluatorAgent
from app.config import Settings, settings
from app.events import ExecutionEvent, ExecutionEventType, RunEventBus
from app.knowledge.embeddings import MockEmbeddingProvider
from app.knowledge.models import RetrievedChunk, RetrievedContext
from app.llm.base import LLMProvider
from app.llm.provider import MockLLMProvider
from app.models.schemas import EvaluationResult, PlannerOutput, ResearchResult, RunResponse
from app.models.task_graph import TaskGraph, TaskNode, TaskType
from app.orchestration.evidence_ledger import EvidenceItem, EvidenceLedger
from app.orchestration.grounding_verifier import GroundingVerificationResult, GroundingVerifier
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


# ==============================================================================
# 1. Programmatic Verifier Tests (No LLM)
# ==============================================================================

def test_grounding_verifier_valid_citations_and_numeric_grounding():
    """Verify that verifier succeeds when citations exist in ledger and numbers match."""
    ledger = EvidenceLedger()
    chunk1 = RetrievedChunk(
        id="c1",
        document_id="doc-fia",
        source="fia_2025.pdf",
        content="Article 3.4.1 Overall Width: No part of the car may be wider than 2000mm.",
        similarity_score=0.95,
        chunk_index=0,
        metadata={"article": "3.4.1", "label": "FIA 2025"},
    )
    chunk2 = RetrievedChunk(
        id="c2",
        document_id="doc-fia",
        source="fia_2025.pdf",
        content="Article 4.1 Minimum Mass: The mass of the car without fuel must not be less than 798kg.",
        similarity_score=0.92,
        chunk_index=1,
        metadata={"article": "4.1", "label": "FIA 2025"},
    )
    ledger.add_chunks([chunk1, chunk2])

    text = (
        "### Explicit Regulatory Requirements\n"
        "- The maximum car width is strictly limited to 2000mm [FIA 2025, art 3.4.1, p.1].\n"
        "- The minimum car mass is 798kg [FIA 2025, art 4.1, p.1].\n"
    )

    result = GroundingVerifier.verify(text, ledger)
    assert result.total_citations == 2
    assert result.valid_citations == 2
    assert result.citation_validity == 100.0
    assert len(result.unsupported_claims) == 0
    assert result.passed is True


def test_grounding_verifier_detects_missing_citation_and_numeric_hallucination():
    """Verify that verifier catches citations missing from ledger and mismatched numbers."""
    ledger = EvidenceLedger()
    chunk1 = RetrievedChunk(
        id="c1",
        document_id="doc-fia",
        source="fia_2025.pdf",
        content="Article 3.4.1 Overall Width: No part of the car may be wider than 2000mm.",
        similarity_score=0.95,
        chunk_index=0,
        metadata={"article": "3.4.1", "label": "FIA 2025"},
    )
    ledger.add_chunks([chunk1])

    text = (
        "### Regulatory Summary\n"
        "- The car width must not exceed 2500mm [FIA 2025, art 3.4.1, p.1].\n"  # 2500mm not in chunk
        "- Front wing endplates must be 150mm tall [FIA 2025, art 3.9, p.12].\n"  # art 3.9 not in ledger
    )

    result = GroundingVerifier.verify(text, ledger)
    assert result.total_citations == 2
    assert result.valid_citations == 0
    assert result.citation_validity == 0.0
    assert len(result.unsupported_claims) == 2
    assert result.passed is False
    assert any("not found in Evidence Ledger" in c for c in result.unsupported_claims)
    assert any("2500mm" in c for c in result.unsupported_claims)


def test_grounding_verifier_empty_text_or_no_ledger():
    """Verify behavior for empty text or runs without document evidence."""
    res_empty = GroundingVerifier.verify("", None)
    assert res_empty.passed is True
    assert res_empty.total_citations == 0

    res_no_docs = GroundingVerifier.verify("Simple parametric response.", None)
    assert res_no_docs.passed is True
    assert res_no_docs.total_citations == 0


# ==============================================================================
# 2. Evaluator Agent Rubric Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_evaluator_agent_4_dimension_rubric_audits_final_answer():
    """Verify EvaluatorAgent passes final answer to LLM and parses 4 rubric scores."""
    captured_prompts = []

    class MockEvalLLM(LLMProvider):
        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            return ""

        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            captured_prompts.append(prompt)
            return EvaluationResult(
                score=90,
                status="PASS",
                feedback="All 4 dimensions well addressed with high citation grounding.",
                groundedness_score=24,
                coverage_score=22,
                distinction_score=22,
                honesty_score=22,
                uncovered_domains=[],
            )

    llm = MockEvalLLM()
    evaluator = EvaluatorAgent(llm, pass_threshold=80)

    ver_res = GroundingVerificationResult(
        total_citations=2,
        valid_citations=2,
        citation_validity=100.0,
        unsupported_claims=[],
        passed=True,
    )

    eval_result = await evaluator.evaluate(
        user_task="Explain 2025 width and mass rules",
        analysis="Intermediate analysis findings",
        final_answer="Final user-facing report with explicit requirements and citations.",
        verification_result=ver_res,
    )

    assert eval_result.score == 90
    assert eval_result.status == "PASS"
    assert eval_result.groundedness_score == 24
    assert eval_result.coverage_score == 22
    assert eval_result.distinction_score == 22
    assert eval_result.honesty_score == 22
    assert eval_result.verification_result is not None
    assert eval_result.verification_result["citation_validity"] == 100.0

    # Ensure evaluation prompt targeted Final User-Facing Answer
    assert len(captured_prompts) == 1
    assert "Final User-Facing Answer to Evaluate:" in captured_prompts[0]
    assert "Programmatic Grounding Verifier Audit:" in captured_prompts[0]
    assert "Citation Validity: 100.0%" in captured_prompts[0]


# ==============================================================================
# 3. Targeted Iterative Replanning Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_targeted_replanning_triggers_refines_queries_and_tracks_score_improvement():
    """Verify full targeted replanning loop: identifies uncovered domain, emits events, tracks score improvement."""
    events_emitted: List[ExecutionEvent] = []

    class ReplanningTestLLM(LLMProvider):
        def __init__(self):
            self.eval_calls = 0
            self.research_prompts = []

        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            sys = (system_prompt or "").lower()
            if "synthesizer" in sys:
                if self.eval_calls >= 1:
                    return (
                        "### Explicit Regulatory Requirements\n"
                        "- Dimensional envelope max width 2000mm [FIA 2025, art 3.4.1, p.1].\n"
                        "- Minimum car weight 798kg [FIA 2025, art 4.1, p.1].\n\n"
                        "### Engineering Interpretation / Inference\n"
                        "- Packaging cooling in sidepods.\n\n"
                        "### Limitations and Unknowns\n"
                        "- Thermal limits dependent on ambient conditions."
                    )
                return "Initial incomplete synthesis lacking mass regulations."
            if "analyst" in sys:
                return "Analytical synthesis across active subtasks."
            if "research" in sys:
                self.research_prompts.append(prompt)
                return "Revised research findings with mass regulations."
            return "Generic generation."

        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            if schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        TaskNode(
                            id="T1",
                            description="Research car overall width boundaries",
                            type=TaskType.RESEARCH,
                            metadata={"domain": "aerodynamics", "retrieval_query": "overall width Article 3.4.1"},
                        ),
                        TaskNode(
                            id="T2",
                            description="Research minimum vehicle mass and ballast",
                            type=TaskType.RESEARCH,
                            metadata={"domain": "mass", "retrieval_query": "minimum mass Article 4.1"},
                        ),
                    ]
                )
            if schema == EvaluationResult:
                self.eval_calls += 1
                if self.eval_calls == 1:
                    # Initial evaluation fails due to uncovered mass domain
                    return EvaluationResult(
                        score=60,
                        status="FAIL",
                        feedback="Critique: The mass domain was insufficiently addressed.",
                        groundedness_score=15,
                        coverage_score=10,
                        distinction_score=18,
                        honesty_score=17,
                        uncovered_domains=["mass"],
                    )
                # Second evaluation passes
                return EvaluationResult(
                    score=92,
                    status="PASS",
                    feedback="Mass domain thoroughly integrated with verified citations.",
                    groundedness_score=24,
                    coverage_score=23,
                    distinction_score=23,
                    honesty_score=22,
                    uncovered_domains=[],
                )
            if schema.__name__ == "StrategyDecision":
                from app.orchestration.strategy import ExecutionStrategy, StrategyDecision
                return StrategyDecision(
                    strategy=ExecutionStrategy.MULTI_AGENT,
                    reasoning="Complex test run",
                    estimated_complexity="HIGH",
                    requires_rag=False,
                )
            raise ValueError(f"Unexpected schema {schema}")

    provider = ReplanningTestLLM()
    test_settings = Settings(llm_provider="mock", max_replanning_rounds=2, evaluator_pass_threshold=80)
    bus = RunEventBus(run_id="test-replan-run")
    queue = await bus.subscribe()
    pipeline = OrchestratedPipeline(llm_provider=provider, settings=test_settings, event_bus=bus)

    response = await pipeline.execute("Analyze 2025 width and mass rules")

    while not queue.empty():
        evt = queue.get_nowait()
        if evt is not None:
            events_emitted.append(evt)

    # 1. Replanning triggered exactly once and resolved to PASS
    assert response.replanning_count == 1
    assert provider.eval_calls == 2
    assert response.evaluation.score == 92
    assert response.evaluation.status == "PASS"

    # 2. Targeted subtask T2 (mass) received the critique
    assert any("Critique: The mass domain was insufficiently addressed." in p for p in provider.research_prompts)

    # 3. Events emitted: REPLANNING_STARTED, REPLAN_STARTED, REPLANNING_COMPLETED, REPLAN_COMPLETED
    event_types = [e.event_type for e in events_emitted]
    assert ExecutionEventType.REPLANNING_STARTED in event_types
    assert ExecutionEventType.REPLAN_STARTED in event_types
    assert ExecutionEventType.REPLANNING_COMPLETED in event_types
    assert ExecutionEventType.REPLAN_COMPLETED in event_types

    # 4. Check REPLANNING_COMPLETED event metadata tracked score improvement
    comp_events = [e for e in events_emitted if e.event_type == ExecutionEventType.REPLANNING_COMPLETED]
    assert len(comp_events) == 1
    meta = comp_events[0].metadata
    assert meta["initial_score"] == 60
    assert meta["revised_score"] == 92
    assert meta["score_improved"] is True

    # 5. Execution trace contains TargetedReplanning stage
    replan_traces = [t for t in response.execution_trace if t.stage_name == "TargetedReplanning"]
    assert len(replan_traces) == 1
    assert replan_traces[0].status == "SUCCESS"
    assert "Revised evaluation score: 92/100" in replan_traces[0].summary


@pytest.mark.asyncio
async def test_targeted_replanning_respects_max_replanning_rounds():
    """Verify that replanning halts after max_replanning_rounds even if score remains below pass threshold."""
    class AlwaysFailingLLM(LLMProvider):
        def __init__(self):
            self.eval_calls = 0

        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            return "Attempt output."

        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            if schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        TaskNode(id="T1", description="Research aerodynamics", type=TaskType.RESEARCH),
                    ]
                )
            if schema == EvaluationResult:
                self.eval_calls += 1
                return EvaluationResult(
                    score=50,
                    status="FAIL",
                    feedback="Persistently failing critique.",
                )
            if schema.__name__ == "StrategyDecision":
                from app.orchestration.strategy import ExecutionStrategy, StrategyDecision
                return StrategyDecision(
                    strategy=ExecutionStrategy.MULTI_AGENT,
                    reasoning="Always failing test",
                    estimated_complexity="HIGH",
                    requires_rag=False,
                )
            raise ValueError(f"Unexpected schema {schema}")

    provider = AlwaysFailingLLM()
    # Configure max 2 rounds
    test_settings = Settings(llm_provider="mock", max_replanning_rounds=2, evaluator_pass_threshold=80)
    pipeline = OrchestratedPipeline(llm_provider=provider, settings=test_settings)

    response = await pipeline.execute("Analyze complex topic")

    # Initial eval + 2 replan rounds = 3 total evaluation calls
    assert provider.eval_calls == 3
    assert response.replanning_count == 2
    assert response.evaluation.score == 50
    assert response.evaluation.status == "FAIL"

    replan_traces = [t for t in response.execution_trace if t.stage_name == "TargetedReplanning"]
    assert len(replan_traces) == 2
