"""
Tests for Phase 8 Experimental Benchmark and Comparative Harness.

Verifies:
1. Benchmark dataset contains >= 30 tasks across 4 categories and includes held-out tasks.
2. ExperimentRunner executes tasks across all 4 systems (Direct LLM, Naive RAG, Fixed Multi-Agent, Adaptive System).
3. Evaluator rubric and programmatic verifier produce quantitative quality and citation scores.
4. Summary metrics (tokens, latency, cost, success rate, replanning count) are properly aggregated.
5. CSV export and Markdown summary reporting succeed.
"""

import os
import tempfile
import pytest

from app.config import Settings
from app.experiments.benchmark_dataset import BENCHMARK_TASKS, BenchmarkTask, get_benchmark_tasks
from app.experiments.harness import (
    FIA_REFERENCE_CHUNKS,
    ExperimentRunner,
    create_experiment_retrieval_service,
)
from app.experiments.run_experiments import HUMAN_SCORED_SAMPLE
from app.llm.provider import MockLLMProvider


def test_benchmark_dataset_integrity():
    """Verify benchmark has >= 30 tasks, all 4 categories, and held-out tasks."""
    assert len(BENCHMARK_TASKS) >= 30
    categories = {t.category for t in BENCHMARK_TASKS}
    assert "SIMPLE_FACT" in categories
    assert "SINGLE_DOC_LOOKUP" in categories
    assert "MULTI_DOMAIN_ANALYSIS" in categories
    assert "CONVERSATIONAL_FOLLOWUP" in categories

    held_out = [t for t in BENCHMARK_TASKS if t.is_held_out]
    assert len(held_out) >= 5, "Must include held-out tasks not used during initial tuning"


def test_benchmark_dataset_filtering():
    """Verify get_benchmark_tasks filters by category, held_out status, and sample size."""
    mda_tasks = get_benchmark_tasks(category="MULTI_DOMAIN_ANALYSIS")
    assert len(mda_tasks) >= 8
    assert all(t.category == "MULTI_DOMAIN_ANALYSIS" for t in mda_tasks)

    held_out_tasks = get_benchmark_tasks(held_out_only=True)
    assert all(t.is_held_out for t in held_out_tasks)

    sample = get_benchmark_tasks(sample_size=3)
    assert len(sample) == 3


@pytest.mark.asyncio
async def test_experiment_retrieval_service_seeding():
    """Verify experiment retrieval service is pre-seeded with FIA reference chunks."""
    retrieval_service = create_experiment_retrieval_service()
    ctx = await retrieval_service.retrieve("minimum mass", top_k=2)
    assert len(ctx.chunks) > 0
    assert any("798kg" in c.content for c in ctx.chunks)


@pytest.mark.asyncio
async def test_experiment_runner_executes_sample_across_4_systems():
    """Verify ExperimentRunner executes sample tasks across all 4 systems."""
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock", max_replanning_rounds=2)
    runner = ExperimentRunner(llm_provider=llm, settings=settings)

    sample_tasks = [
        BENCHMARK_TASKS[0],  # SF-01: Greeting
        BENCHMARK_TASKS[8],  # SDL-01: Mass lookup
        BENCHMARK_TASKS[16], # MDA-01: Aero vs suspension
    ]

    records = await runner.run_all(sample_tasks)
    # 3 tasks * 4 systems = 12 execution records
    assert len(records) == 12

    system_names = {r.system_name for r in records}
    assert system_names == {"Direct LLM", "Naive RAG", "Fixed Multi-Agent", "Adaptive System"}

    for r in records:
        assert r.task_success is True
        assert r.quality_score > 0
        assert r.total_tokens > 0
        assert r.cost_usd >= 0.0

    # Verify Adaptive System token efficiency on simple greeting
    greeting_direct = next(r for r in records if r.task_id == "SF-01" and r.system_name == "Direct LLM")
    greeting_fixed = next(r for r in records if r.task_id == "SF-01" and r.system_name == "Fixed Multi-Agent")
    greeting_adapt = next(r for r in records if r.task_id == "SF-01" and r.system_name == "Adaptive System")

    assert greeting_adapt.strategy_used == "DIRECT"
    # Adaptive token consumption on greeting must be drastically lower than fixed multi-agent
    assert greeting_adapt.total_tokens < greeting_fixed.total_tokens


@pytest.mark.asyncio
async def test_experiment_summary_computation_and_csv_export():
    """Verify aggregated summary metrics calculation and CSV export functionality."""
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    runner = ExperimentRunner(llm_provider=llm, settings=settings)

    sample_tasks = BENCHMARK_TASKS[:2]
    records = await runner.run_all(sample_tasks)
    summaries = runner.compute_summaries(records)

    assert len(summaries) == 4
    for sys_name, s in summaries.items():
        assert s.total_tasks == 2
        assert s.success_rate_pct == 100.0
        assert s.avg_quality_score > 0
        assert s.avg_tokens > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_results.csv")
        runner.export_csv(records, csv_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 100

        md_table = runner.generate_markdown_table(summaries)
        assert "Direct LLM" in md_table
        assert "Adaptive System" in md_table
        assert "Fixed Multi-Agent" in md_table


def test_human_scored_sample_structure():
    """Verify human-scored sample includes representative tasks and multi-system scores."""
    assert len(HUMAN_SCORED_SAMPLE) >= 5
    for item in HUMAN_SCORED_SAMPLE:
        assert "task_id" in item
        assert "prompt" in item
        assert "direct_llm" in item
        assert "naive_rag" in item
        assert "fixed_multi_agent" in item
        assert "adaptive_system" in item
        assert 0 <= item["adaptive_system"]["score"] <= 100
