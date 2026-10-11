"""
Phase W4 Unit and Integration Tests:
1. StrategySelector needs_web and web_rationale classification.
2. ResourceBudget web fields and AtomicWebBudgetTracker integration.
3. ResearchTaskAgent capability execution with web search and fetch_page.
4. Replanner targeted web queries for missing evidence and budget termination.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.config import Settings
from app.llm.base import LLMProvider
from app.models.schemas import ChatMessagePayload, ResearchResult, SubTask
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus, TaskType
from app.orchestration.agents import CapabilityRegistry, ResearchTaskAgent, TaskExecutionContext
from app.orchestration.evidence_ledger import EvidenceLedger
from app.orchestration.grounding_verifier import GroundingVerifier, GroundingVerificationResult
from app.orchestration.resource_budget import (
    AtomicWebBudgetTracker,
    BudgetTracker,
    ResourceBudget,
    ResourceBudgetExceededError,
    WebBudget,
    get_budget_for_strategy,
)
from app.orchestration.strategy import ExecutionStrategy, StrategyDecision, StrategySelector
from app.tools.fetch_page import FetchPageInput, FetchPageTool
from app.tools.search_provider import MockSearchProvider
from app.tools.web_cache import WebCacheManager
from app.tools.web_search import WebSearchInput, WebSearchTool


class DummyLLM(LLMProvider):
    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "Dummy generated response with citation [web:a1b2c3d4] [EXPLICIT]."

    async def generate_structured(self, prompt: str, schema: type, system_prompt: str | None = None):
        return schema(
            strategy=ExecutionStrategy.MULTI_AGENT,
            reasoning="Multi-domain query requiring external web search.",
            estimated_complexity="HIGH",
            requires_rag=False,
            domains=["aerodynamics", "power unit"],
            cross_document_reasoning=False,
            needs_web=True,
            web_rationale="Requires latest 2026 rule revisions from external web.",
        )


@pytest.mark.asyncio
async def test_strategy_selector_needs_web_with_llm():
    """Verify that StrategySelector extracts needs_web and web_rationale via LLM."""
    selector = StrategySelector(DummyLLM())
    decision = await selector.select_strategy("What are the latest 2026 F1 engine rules?")
    assert decision.needs_web is True
    assert decision.web_rationale is not None
    assert "2026" in decision.web_rationale or "web" in decision.web_rationale.lower()


@pytest.mark.asyncio
async def test_strategy_selector_fallback_heuristics_web_cues():
    """Verify fallback rule-based heuristics detect web cues when LLM is absent."""
    selector = StrategySelector(None)
    # 1. Query with web cue -> needs_web=True
    dec_web = await selector.select_strategy("search the web for latest aerodynamic updates and chassis news")
    assert dec_web.needs_web is True
    assert "web" in dec_web.web_rationale.lower()

    # 2. Greeting -> needs_web=False
    dec_greet = await selector.select_strategy("Hello there")
    assert dec_greet.needs_web is False
    assert dec_greet.strategy == ExecutionStrategy.DIRECT

    # 3. Simple math -> needs_web=False
    dec_math = await selector.select_strategy("calculate 45 * 20")
    assert dec_math.needs_web is False
    assert dec_math.strategy == ExecutionStrategy.SINGLE_AGENT


def test_resource_budget_and_tracker_web_integration():
    """Verify ResourceBudget to_web_budget and BudgetTracker atomic tracker."""
    budget = ResourceBudget(
        strategy=ExecutionStrategy.MULTI_AGENT,
        max_llm_calls=10,
        max_tokens=40000,
        max_wall_time_seconds=60.0,
        max_web_searches=4,
        max_web_fetches=4,
        max_web_bytes=1024 * 1024,
        max_web_wall_time_seconds=30.0,
    )
    web_b = budget.to_web_budget()
    assert web_b.max_searches == 4
    assert web_b.max_fetches == 4
    assert web_b.max_bytes == 1024 * 1024

    tracker = BudgetTracker(budget)
    assert isinstance(tracker.web_tracker, AtomicWebBudgetTracker)
    summary = tracker.to_dict()
    assert "web_budget" in summary
    assert summary["web_budget"]["max_searches"] == 4


@pytest.mark.asyncio
async def test_atomic_web_budget_enforcement_concurrency():
    """Verify AtomicWebBudgetTracker enforces limits under concurrent reservations."""
    web_budget = WebBudget(max_searches=3, max_fetches=3, max_bytes=1000, max_wall_time_seconds=10.0)
    tracker = AtomicWebBudgetTracker(web_budget)

    # Launch 5 concurrent search reservations (only 3 should succeed)
    results = []
    async def try_reserve():
        try:
            return await tracker.reserve_search()
        except ResourceBudgetExceededError:
            return "EXCEEDED"

    tasks = [try_reserve() for _ in range(5)]
    res = await asyncio.gather(*tasks)

    successes = [r for r in res if isinstance(r, int)]
    exceeded = [r for r in res if r == "EXCEEDED"]

    assert len(successes) == 3
    assert len(exceeded) == 2
    assert tracker.searches == 3


@pytest.mark.asyncio
async def test_research_task_agent_web_execution(tmp_path):
    """Verify ResearchTaskAgent invokes web search and page fetch, populating EvidenceLedger."""
    ledger = EvidenceLedger()

    # Setup mock web search and fetch tools
    cfg = Settings(web_cache_mode="live", web_cache_dir=str(tmp_path / "cache"))
    cache_mgr = WebCacheManager(custom_settings=cfg)

    mock_provider = MockSearchProvider()
    search_tool = WebSearchTool(custom_settings=cfg, provider=mock_provider, cache_manager=cache_mgr)

    # Mock fetch page tool that returns synthetic page
    fetch_tool = MagicMock(spec=FetchPageTool)
    fetch_tool.name = "fetch_page"
    async def fake_fetch(inp):
        from app.orchestration.tools import ToolResult
        return ToolResult(
            tool_name="fetch_page",
            success=True,
            output={
                "url": inp.url,
                "final_url": inp.url,
                "title": "2026 Engine Rules Official Page",
                "text": "The 2026 engine regulations mandate 100% sustainable fuels and 350kW MGU-K output.",
                "retrieval_timestamp": "2026-10-11T12:00:00Z",
                "content_hash": "a1b2c3d4e5f67890",
            },
            metadata={"from_mock": True},
        )
    fetch_tool.execute = AsyncMock(side_effect=fake_fetch)

    from app.agents.researcher import ResearchAgent
    research_agent = ResearchAgent(DummyLLM())

    task_agent = ResearchTaskAgent(
        agent=research_agent,
        web_search_tool=search_tool,
        fetch_page_tool=fetch_tool,
    )

    task_node = TaskNode(
        id="T1",
        description="Investigate 2026 MGU-K power output specifications",
        type=TaskType.RESEARCH,
        metadata={"needs_web": True, "web_search_query": "2026 F1 engine regulations MGU-K"},
    )
    exec_ctx = TaskExecutionContext(
        task=task_node,
        input_context={"user_task": "2026 rules analysis"},
        evidence_ledger=ledger,
    )

    result = await task_agent.execute(exec_ctx)
    assert result.success is True
    assert "web_research" in result.metadata
    assert len(ledger.items) > 0

    # Verify web citation in ledger
    web_items = [item for item in ledger.items.values() if item.source_type == "web"]
    assert len(web_items) > 0
    assert web_items[0].citation.startswith("[web:")
    assert "350kW" in web_items[0].text


@pytest.mark.asyncio
async def test_capability_registry_web_research_resolution():
    """Verify CapabilityRegistry resolves web_research capability."""
    registry = CapabilityRegistry()
    research_agent = MagicMock(spec=ResearchTaskAgent)
    registry.register("research", research_agent)
    registry.register("web_research", research_agent)

    assert registry.resolve("research") == research_agent
    assert registry.resolve("web_research") == research_agent
    assert registry.resolve("WEB_RESEARCH") == research_agent


@pytest.mark.asyncio
async def test_replanner_emits_targeted_web_queries_for_missing_evidence(tmp_path):
    """
    Verify that when evaluation fails due to missing evidence, the replanner
    emits targeted web queries for the gap and adds web pages to the EvidenceLedger.
    """
    from app.pipeline.orchestrated_pipeline import OrchestratedPipeline

    cfg = Settings(
        llm_provider="mock",
        evaluator_pass_threshold=70.0,
        max_replanning_rounds=2,
        web_cache_mode="live",
        web_cache_dir=str(tmp_path / "cache"),
    )

    call_count = {"eval": 0}

    class ReplannerMockLLM(LLMProvider):
        async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            if "RESEARCH AGENT" in (system_prompt or ""):
                if "Critique" in prompt:
                    return "Revised research findings: 2026 MGU-K power is 350kW [web:11223344] [EXPLICIT]."
                return "Initial research findings. MISSING EVIDENCE: 2026 MGU-K power limits."
            elif "ANALYST AGENT" in (system_prompt or ""):
                return "Analysis: 2026 power units produce 350kW from MGU-K [web:11223344]."
            elif "SYNTHESIZER AGENT" in (system_prompt or ""):
                return "Final synthesized report: The 2026 engine regulations mandate 350kW MGU-K [web:11223344]."
            return "Standard response"

        async def generate_structured(self, prompt: str, schema: type, system_prompt: str | None = None):
            from app.models.schemas import EvaluationResult, PlannerOutput, SubTask

            if schema == StrategyDecision:
                return StrategyDecision(
                    strategy=ExecutionStrategy.MULTI_AGENT,
                    reasoning="Complex query needing web search",
                    estimated_complexity="HIGH",
                    requires_rag=False,
                    domains=["power unit"],
                    cross_document_reasoning=False,
                    needs_web=True,
                    web_rationale="Requires fresh 2026 regulatory updates",
                )
            elif schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        SubTask(id="T1", description="Investigate 2026 MGU-K limits", type="research", dependencies=[]),
                    ]
                )
            elif schema == EvaluationResult:
                call_count["eval"] += 1
                if call_count["eval"] == 1:
                    return EvaluationResult(
                        score=50,
                        status="FAIL",
                        feedback="Missing empirical evidence on 2026 MGU-K power limits.",
                        uncovered_domains=["power unit"],
                    )
                else:
                    return EvaluationResult(
                        score=90,
                        status="PASS",
                        feedback="Thoroughly evidenced with 2026 regulatory citations.",
                        uncovered_domains=[],
                    )
            return schema()

    mock_llm = ReplannerMockLLM()
    pipeline = OrchestratedPipeline(llm_provider=mock_llm, settings=cfg)

    # Mock web search and fetch tools
    pipeline.web_search_tool._provider = MockSearchProvider()
    async def mock_fetch_exec(inp):
        from app.orchestration.tools import ToolResult
        return ToolResult(
            tool_name="fetch_page",
            success=True,
            output={
                "url": inp.url,
                "final_url": inp.url,
                "title": "2026 Technical Regulations Engine Specs",
                "text": "The MGU-K maximum power output for 2026 is increased to 350kW.",
                "retrieval_timestamp": "2026-10-11T12:00:00Z",
                "content_hash": "1122334455667788",
            },
            metadata={"from_mock": True},
        )
    pipeline.fetch_page_tool.execute = AsyncMock(side_effect=mock_fetch_exec)

    resp = await pipeline.execute(user_task="Analyze 2026 F1 engine regulations updates")
    assert resp.execution_strategy == "MULTI_AGENT"
    assert resp.replanning_count >= 1
    assert resp.evaluation.status == "PASS"
    assert "[web:11223344]" in resp.final_answer


@pytest.mark.asyncio
async def test_document_only_run_never_triggers_web_calls(tmp_path):
    """
    Verify that a document-only run (needs_web=False) never triggers web search
    or page fetch calls, even when evaluation fails and missing evidence is reported during replanning.
    """
    from app.pipeline.orchestrated_pipeline import OrchestratedPipeline

    cfg = Settings(
        llm_provider="mock",
        evaluator_pass_threshold=70.0,
        max_replanning_rounds=2,
        web_cache_mode="live",
        web_cache_dir=str(tmp_path / "cache"),
    )

    eval_count = {"calls": 0}

    class DocOnlyMockLLM(LLMProvider):
        async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            if "RESEARCH AGENT" in (system_prompt or ""):
                if "Critique" in prompt:
                    return "Revised research findings from documents: Mass is 798kg [doc-1] [EXPLICIT]."
                return "Initial findings from documents. MISSING EVIDENCE: Specific ballast limits."
            elif "ANALYST AGENT" in (system_prompt or ""):
                return "Analysis: Minimum mass is 798kg [doc-1]."
            elif "SYNTHESIZER AGENT" in (system_prompt or ""):
                return "Synthesized answer: The minimum car mass is 798kg [doc-1]."
            return "Generic response"

        async def generate_structured(self, prompt: str, schema: type, system_prompt: str | None = None):
            from app.models.schemas import EvaluationResult, PlannerOutput, SubTask

            if schema == StrategyDecision:
                return StrategyDecision(
                    strategy=ExecutionStrategy.MULTI_AGENT,
                    reasoning="Document-grounded regulatory query",
                    estimated_complexity="HIGH",
                    requires_rag=True,
                    domains=["chassis"],
                    cross_document_reasoning=False,
                    needs_web=False,  # STRICTLY DOCUMENT-ONLY
                    web_rationale="All information is expected in local documents",
                )
            elif schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        SubTask(id="T1", description="Examine document chassis limits", type="research", dependencies=[]),
                    ]
                )
            elif schema == EvaluationResult:
                eval_count["calls"] += 1
                if eval_count["calls"] == 1:
                    return EvaluationResult(
                        score=45,
                        status="FAIL",
                        feedback="Missing ballast distribution details from documents.",
                        uncovered_domains=["chassis"],
                    )
                else:
                    return EvaluationResult(
                        score=88,
                        status="PASS",
                        feedback="Sufficient document evidence provided.",
                        uncovered_domains=[],
                    )
            return schema()

    mock_llm = DocOnlyMockLLM()
    pipeline = OrchestratedPipeline(llm_provider=mock_llm, settings=cfg)

    # Attach spies to web tools
    pipeline.web_search_tool.execute = AsyncMock(wraps=pipeline.web_search_tool.execute)
    pipeline.fetch_page_tool.execute = AsyncMock(wraps=pipeline.fetch_page_tool.execute)

    resp = await pipeline.execute(user_task="Analyze document ballast regulations")
    assert resp.execution_strategy == "MULTI_AGENT"
    assert resp.replanning_count >= 1
    assert resp.evaluation.status == "PASS"

    # CRITICAL INVARIANT: Zero web search or fetch tool executions for document-only runs
    assert pipeline.web_search_tool.execute.call_count == 0
    assert pipeline.fetch_page_tool.execute.call_count == 0


