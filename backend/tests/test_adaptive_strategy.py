"""
Behavioral and Unit Tests for Adaptive Strategy Selection & Resource-Aware Orchestration.
"""

import pytest
from app.config import Settings
from app.llm.provider import MockLLMProvider
from app.orchestration.strategy import ExecutionStrategy, StrategySelector
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline
from app.models.schemas import ChatMessagePayload, DocumentPayload


@pytest.mark.asyncio
async def test_unseen_casual_greeting_strategy():
    selector = StrategySelector()
    decision = await selector.select_strategy("Hey, how's it going?")
    assert decision.strategy == ExecutionStrategy.DIRECT
    assert decision.requires_rag is False


@pytest.mark.asyncio
async def test_unseen_coding_problem_strategy():
    llm = MockLLMProvider()
    selector = StrategySelector(llm)
    decision = await selector.select_strategy("Write a Python function to check if a string is a palindrome.")
    assert decision.strategy in (ExecutionStrategy.SINGLE_AGENT, ExecutionStrategy.DIRECT)


@pytest.mark.asyncio
async def test_unseen_writing_task_strategy():
    llm = MockLLMProvider()
    selector = StrategySelector(llm)
    decision = await selector.select_strategy("Draft a polite email declining a job offer.")
    assert decision.strategy in (ExecutionStrategy.SINGLE_AGENT, ExecutionStrategy.DIRECT)


@pytest.mark.asyncio
async def test_unseen_complex_reasoning_strategy():
    selector = StrategySelector()
    decision = await selector.select_strategy(
        "Analyze the interactions between quantum entanglement, thermal noise, and cryptographic latency in distributed satellite networks."
    )
    assert decision.strategy == ExecutionStrategy.MULTI_AGENT


@pytest.mark.asyncio
async def test_direct_greeting_pipeline_execution():
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    result = await pipeline.execute("Hello there!")

    assert result.execution_strategy == "DIRECT"
    assert len(result.subtasks) == 0
    assert result.evaluation.status == "NOT_APPLICABLE"
    assert result.evaluation.score is None


@pytest.mark.asyncio
async def test_single_agent_calculation_pipeline_execution():
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    result = await pipeline.execute("Calculate 45 * 55")

    assert result.execution_strategy == "SINGLE_AGENT"
    assert len(result.subtasks) == 1
    assert result.evaluation.status == "SKIPPED"
    assert result.evaluation.score is None


@pytest.mark.asyncio
async def test_multi_agent_complex_pipeline_execution():
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    result = await pipeline.execute(
        "Compare electric and petrol vehicles across lifecycle emissions, total cost of ownership, and long-term adoption."
    )

    assert result.execution_strategy == "MULTI_AGENT"
    assert len(result.subtasks) >= 3
    assert result.evaluation.status in ("PASS", "FAIL")
    assert result.evaluation.score is not None


@pytest.mark.asyncio
async def test_document_attached_triggers_rag_on_relevant_query():
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    doc = DocumentPayload(filename="report.pdf", content="Solar energy revenue grew by 34% in Q3 2026 reaching $4.2B.")
    result = await pipeline.execute("What was the solar tech revenue growth?", documents=[doc])

    assert result.execution_strategy in ("SINGLE_AGENT", "MULTI_AGENT")
