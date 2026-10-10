"""
Phase 8 Benchmark Dataset for Adaptive Multi-Agent Evaluation.

Contains 32 diverse tasks across 4 core operational categories:
1. SIMPLE_FACT: Conversational greetings, quick arithmetic, basic definitions (8 tasks)
2. SINGLE_DOC_LOOKUP: Targeted single-document queries grounded in specific regulations (8 tasks)
3. MULTI_DOMAIN_ANALYSIS: Multi-constraint cross-domain engineering & policy trade-offs (10 tasks)
4. CONVERSATIONAL_FOLLOWUP: Multi-turn continuations, clarifications, and summarizations (6 tasks)

Includes held-out benchmark tasks marked with is_held_out=True.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """Specification of an evaluation benchmark task."""

    id: str = Field(..., description="Unique task identifier (e.g. SF-01, SDL-01, MDA-01, CFU-01)")
    category: str = Field(..., description="Task category: SIMPLE_FACT, SINGLE_DOC_LOOKUP, MULTI_DOMAIN_ANALYSIS, CONVERSATIONAL_FOLLOWUP")
    prompt: str = Field(..., description="User prompt text")
    has_documents: bool = Field(default=False, description="Whether reference documents are attached")
    document_ids: List[str] = Field(default_factory=list, description="IDs of relevant reference documents")
    expected_domains: List[str] = Field(default_factory=list, description="Domains expected to be identified")
    requires_cross_doc: bool = Field(default=False, description="Whether cross-document or cross-section reasoning is required")
    is_held_out: bool = Field(default=False, description="Whether this task was held out from tuning")
    ground_truth_keywords: List[str] = Field(default_factory=list, description="Key facts, numbers, or terms expected in high-quality answer")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior conversation history for follow-up tasks")


BENCHMARK_TASKS: List[BenchmarkTask] = [
    # --------------------------------------------------------------------------
    # Category 1: Simple Facts & Conversational Greetings (8 tasks)
    # --------------------------------------------------------------------------
    BenchmarkTask(
        id="SF-01",
        category="SIMPLE_FACT",
        prompt="Hello there! How are you doing today?",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["hello", "assistant", "help"],
    ),
    BenchmarkTask(
        id="SF-02",
        category="SIMPLE_FACT",
        prompt="Calculate 145 * 12.",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["1740"],
    ),
    BenchmarkTask(
        id="SF-03",
        category="SIMPLE_FACT",
        prompt="Who are you and what are your capabilities?",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["assistant", "multi-agent", "orchestration"],
    ),
    BenchmarkTask(
        id="SF-04",
        category="SIMPLE_FACT",
        prompt="What is the speed of light in meters per second?",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["299,792,458", "300,000", "m/s"],
    ),
    BenchmarkTask(
        id="SF-05",
        category="SIMPLE_FACT",
        prompt="Draft a polite two-sentence email thanking the engineering team for the status review.",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["thank", "team", "review"],
    ),
    BenchmarkTask(
        id="SF-06",
        category="SIMPLE_FACT",
        prompt="Convert 798 kg to pounds.",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["1759", "1,759", "lbs", "pounds"],
    ),
    BenchmarkTask(
        id="SF-07",
        category="SIMPLE_FACT",
        prompt="What is the capital city of Australia?",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["Canberra"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="SF-08",
        category="SIMPLE_FACT",
        prompt="What is the difference between kinetic energy and potential energy in classical mechanics?",
        has_documents=False,
        expected_domains=[],
        ground_truth_keywords=["motion", "position", "velocity", "height"],
        is_held_out=True,
    ),

    # --------------------------------------------------------------------------
    # Category 2: Single-Document Lookups (8 tasks)
    # --------------------------------------------------------------------------
    BenchmarkTask(
        id="SDL-01",
        category="SINGLE_DOC_LOOKUP",
        prompt="According to the 2025 FIA technical regulations, what is the minimum mass of the car without fuel?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["mass", "regulations"],
        ground_truth_keywords=["798kg", "798 kg", "Article 4.1"],
    ),
    BenchmarkTask(
        id="SDL-02",
        category="SINGLE_DOC_LOOKUP",
        prompt="What is the maximum overall width permitted for the car under Article 3.4.1?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["dimensions", "regulations"],
        ground_truth_keywords=["2000mm", "2000 mm", "Article 3.4.1"],
    ),
    BenchmarkTask(
        id="SDL-03",
        category="SINGLE_DOC_LOOKUP",
        prompt="What is the maximum fuel flow mass rate limit specified under Article 5.1?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["power unit", "regulations"],
        ground_truth_keywords=["100 kg/h", "100kg/h", "Article 5.1"],
    ),
    BenchmarkTask(
        id="SDL-04",
        category="SINGLE_DOC_LOOKUP",
        prompt="Under Article 4.6, what is the minimum weight requirement for the driver including seat and equipment?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["mass", "safety"],
        ground_truth_keywords=["80kg", "80 kg", "Article 4.6"],
    ),
    BenchmarkTask(
        id="SDL-05",
        category="SINGLE_DOC_LOOKUP",
        prompt="What is the maximum allowable rear wing height relative to the reference plane under Article 3.10?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["aerodynamics", "regulations"],
        ground_truth_keywords=["910mm", "910 mm", "Article 3.10"],
    ),
    BenchmarkTask(
        id="SDL-06",
        category="SINGLE_DOC_LOOKUP",
        prompt="According to Article 10.2, what are the restrictions on active suspension systems?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["suspension", "regulations"],
        ground_truth_keywords=["prohibited", "passive", "Article 10.2"],
    ),
    BenchmarkTask(
        id="SDL-07",
        category="SINGLE_DOC_LOOKUP",
        prompt="What is the mandatory survival cell front impact deceleration limit under Article 13.3?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["safety", "chassis"],
        ground_truth_keywords=["40g", "Article 13.3"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="SDL-08",
        category="SINGLE_DOC_LOOKUP",
        prompt="Under Article 12.1 of the Financial Regulations, what is the baseline Cost Cap limit in USD?",
        has_documents=True,
        document_ids=["doc-fia-cost-cap"],
        expected_domains=["financial", "cost", "regulations"],
        ground_truth_keywords=["$135,000,000", "$135M", "135 million", "Article 12.1"],
        is_held_out=True,
    ),

    # --------------------------------------------------------------------------
    # Category 3: Multi-Domain FIA Analysis (10 tasks)
    # --------------------------------------------------------------------------
    BenchmarkTask(
        id="MDA-01",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Analyze the trade-offs between aerodynamic downforce limits under Article 3.4.1 and suspension stiffness under Article 10.2.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["aerodynamics", "suspension", "regulations"],
        requires_cross_doc=True,
        ground_truth_keywords=["2000mm", "Article 3.4.1", "Article 10.2", "ride height", "stiffness"],
    ),
    BenchmarkTask(
        id="MDA-02",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Evaluate the interactions between minimum vehicle mass (Article 4.1), ballast placement, and center of gravity height under braking.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["mass", "chassis", "dynamics"],
        requires_cross_doc=True,
        ground_truth_keywords=["798kg", "Article 4.1", "ballast", "weight distribution"],
    ),
    BenchmarkTask(
        id="MDA-03",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Compare the impact of the Financial Cost Cap (Article 12.1) on aerodynamic wind tunnel testing hours vs CFD compute allocations.",
        has_documents=True,
        document_ids=["doc-fia-2025", "doc-fia-cost-cap"],
        expected_domains=["financial", "aerodynamics", "regulations"],
        requires_cross_doc=True,
        ground_truth_keywords=["cost cap", "wind tunnel", "CFD", "Article 12.1", "aero testing"],
    ),
    BenchmarkTask(
        id="MDA-04",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Analyze the cross-system constraints between Power Unit fuel flow (Article 5.1), cooling air duct dimensions, and rear diffuser aerodynamic efficiency.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["power unit", "thermal", "aerodynamics"],
        requires_cross_doc=True,
        ground_truth_keywords=["fuel flow", "100 kg/h", "cooling", "diffuser", "drag"],
    ),
    BenchmarkTask(
        id="MDA-05",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Investigate how tire pressure limits and camber angles interact with brake thermal dissipation and front wing wake management.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["tires", "brakes", "thermal", "aerodynamics"],
        requires_cross_doc=True,
        ground_truth_keywords=["camber", "thermal", "brake duct", "wake"],
    ),
    BenchmarkTask(
        id="MDA-06",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Cross-reference the 2025 chassis dimensional envelope (Article 3.4.1) against driver survival cell crash structure requirements (Article 13.3).",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["dimensions", "safety", "chassis"],
        requires_cross_doc=True,
        ground_truth_keywords=["2000mm", "Article 3.4.1", "survival cell", "Article 13.3"],
    ),
    BenchmarkTask(
        id="MDA-07",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Analyze the feasibility and engineering trade-offs of lightweight composite materials for unsprung mass reduction under budget cap limits.",
        has_documents=True,
        document_ids=["doc-fia-2025", "doc-fia-cost-cap"],
        expected_domains=["mass", "chassis", "financial"],
        requires_cross_doc=True,
        ground_truth_keywords=["unsprung mass", "composites", "cost cap", "titanium"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="MDA-08",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Examine the trade-off between DRS rear wing slot gap dimensions (Article 3.10) and wake turbulence generation for trailing cars.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["aerodynamics", "regulations"],
        requires_cross_doc=True,
        ground_truth_keywords=["DRS", "slot gap", "Article 3.10", "downforce", "overtaking"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="MDA-09",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Evaluate the systemic impact of MGU-K electrical recovery deployment limits on brake bias stability and battery thermal degradation.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["power unit", "brakes", "thermal"],
        requires_cross_doc=True,
        ground_truth_keywords=["MGU-K", "brake-by-wire", "energy recovery", "thermal"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="MDA-010",
        category="MULTI_DOMAIN_ANALYSIS",
        prompt="Assess the strategic trade-offs between qualifying-optimized low downforce setup versus race-pace tire degradation management across a high-temperature grand prix.",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["aerodynamics", "tires", "thermal", "strategy"],
        requires_cross_doc=True,
        ground_truth_keywords=["tire degradation", "qualifying", "downforce", "thermal degradation"],
        is_held_out=True,
    ),

    # --------------------------------------------------------------------------
    # Category 4: Conversational Follow-ups (6 tasks)
    # --------------------------------------------------------------------------
    BenchmarkTask(
        id="CFU-01",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="Can you summarize that in 3 concise bullet points?",
        has_documents=False,
        expected_domains=[],
        history=[
            {"role": "user", "content": "Explain Article 4.1 minimum mass rules."},
            {"role": "assistant", "content": "Article 4.1 prescribes that the minimum mass without fuel is 798kg, including driver ballast where required."},
        ],
        ground_truth_keywords=["798kg", "mass", "fuel"],
    ),
    BenchmarkTask(
        id="CFU-02",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="What was the exact article number for the width limit you mentioned?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["regulations"],
        history=[
            {"role": "user", "content": "What is the maximum car width?"},
            {"role": "assistant", "content": "The maximum overall width permitted is 2000mm."},
        ],
        ground_truth_keywords=["3.4.1", "Article 3.4.1"],
    ),
    BenchmarkTask(
        id="CFU-03",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="How does that specific 798kg mass threshold affect tire wear in high-speed corners?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["mass", "tires"],
        history=[
            {"role": "user", "content": "Explain the minimum vehicle weight under Article 4.1."},
            {"role": "assistant", "content": "Under Article 4.1, the car must weigh at least 798kg without fuel."},
        ],
        ground_truth_keywords=["lateral load", "tire wear", "degradation", "mass"],
    ),
    BenchmarkTask(
        id="CFU-04",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="Thanks, that makes sense. Is the driver weighed with or without their helmet?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["safety", "mass"],
        history=[
            {"role": "user", "content": "What is the driver weight rule?"},
            {"role": "assistant", "content": "Under Article 4.6, the minimum driver weight is 80kg including race gear."},
        ],
        ground_truth_keywords=["with", "gear", "helmet", "equipment", "80kg"],
    ),
    BenchmarkTask(
        id="CFU-05",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="Could you clarify if the active suspension ban also applies to mechanical inerters?",
        has_documents=True,
        document_ids=["doc-fia-2025"],
        expected_domains=["suspension"],
        history=[
            {"role": "user", "content": "What does Article 10.2 say about active suspension?"},
            {"role": "assistant", "content": "Article 10.2 mandates purely passive systems and bans active suspension."},
        ],
        ground_truth_keywords=["inerter", "hydraulic", "passive", "Article 10.2"],
        is_held_out=True,
    ),
    BenchmarkTask(
        id="CFU-06",
        category="CONVERSATIONAL_FOLLOWUP",
        prompt="Great. Can you contrast that with the 2021 financial regulation grandfathering clause?",
        has_documents=True,
        document_ids=["doc-fia-cost-cap"],
        expected_domains=["financial", "regulations"],
        history=[
            {"role": "user", "content": "What is the budget cap under Article 12.1?"},
            {"role": "assistant", "content": "The baseline cost cap is $135M under Article 12.1."},
        ],
        ground_truth_keywords=["cost cap", "Article 12.1", "financial regulations"],
        is_held_out=True,
    ),
]


def get_benchmark_tasks(
    category: Optional[str] = None,
    held_out_only: bool = False,
    sample_size: Optional[int] = None,
) -> List[BenchmarkTask]:
    """Filter and retrieve benchmark tasks."""
    tasks = BENCHMARK_TASKS
    if category:
        tasks = [t for t in tasks if t.category.upper() == category.upper()]
    if held_out_only:
        tasks = [t for t in tasks if t.is_held_out]
    if sample_size and sample_size > 0:
        tasks = tasks[:sample_size]
    return tasks
