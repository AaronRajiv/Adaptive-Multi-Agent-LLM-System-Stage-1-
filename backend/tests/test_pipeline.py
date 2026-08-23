"""
Integration tests for Fixed Baseline Pipeline using MockLLMProvider.
"""

import pytest
from app.config import Settings
from app.llm.provider import MockLLMProvider
from app.pipeline.baseline_pipeline import BaselinePipeline


@pytest.mark.asyncio
async def test_baseline_pipeline_execution():
    test_settings = Settings(
        llm_provider="mock",
        evaluator_pass_threshold=80
    )
    provider = MockLLMProvider()
    pipeline = BaselinePipeline(llm_provider=provider, settings=test_settings)

    user_task = "Compare electric vehicles and petrol vehicles for a college student."
    result = await pipeline.execute(user_task)

    assert result.run_id is not None
    assert result.user_task == user_task
    assert len(result.subtasks) >= 3
    assert len(result.research_results) >= 1
    assert result.analysis is not None and len(result.analysis) > 0
    assert result.evaluation.score >= 0
    assert result.evaluation.status in ["PASS", "FAIL"]
    assert result.final_answer is not None and len(result.final_answer) > 0

    # Verify execution trace captured all 5 stages
    stage_names = [t.stage_name for t in result.execution_trace]
    assert "Planner" in stage_names
    assert "Research" in stage_names
    assert "Analysis" in stage_names
    assert "Evaluation" in stage_names
    assert "Synthesis" in stage_names

    for trace_item in result.execution_trace:
        assert trace_item.status == "SUCCESS"
        assert trace_item.duration_ms >= 0.0
