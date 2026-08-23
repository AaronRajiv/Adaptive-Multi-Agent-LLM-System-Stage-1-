"""
Tests for Pydantic Schema Validation.
"""

import pytest
from pydantic import ValidationError
from app.models.schemas import (
    SubTask,
    PlannerOutput,
    ResearchResult,
    EvaluationResult,
    RunRequest,
    RunResponse,
    StageTrace,
)


def test_subtask_valid():
    task = SubTask(id="T1", description="Analyze EV vs Petrol costs", type="research")
    assert task.id == "T1"
    assert task.type == "research"


def test_subtask_invalid_type():
    with pytest.raises(ValidationError):
        SubTask(id="T1", description="Invalid type subtask", type="invalid_type")  # type: ignore


def test_planner_output_valid():
    output = PlannerOutput(
        tasks=[
            SubTask(id="T1", description="Research cost", type="research"),
            SubTask(id="T2", description="Analyze environmental impact", type="analysis"),
        ]
    )
    assert len(output.tasks) == 2


def test_planner_output_empty_tasks():
    with pytest.raises(ValidationError):
        PlannerOutput(tasks=[])


def test_evaluation_result_valid():
    eval_res = EvaluationResult(
        score=85,
        status="PASS",
        feedback="Clear and comprehensive analysis covering all core points.",
    )
    assert eval_res.score == 85
    assert eval_res.status == "PASS"


def test_evaluation_result_invalid_score_bounds():
    with pytest.raises(ValidationError):
        EvaluationResult(score=105, status="PASS", feedback="Too high")

    with pytest.raises(ValidationError):
        EvaluationResult(score=-5, status="FAIL", feedback="Too low")


def test_run_request_too_short():
    with pytest.raises(ValidationError):
        RunRequest(task="ab")
