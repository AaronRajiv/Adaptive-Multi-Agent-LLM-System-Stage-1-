"""
Phase 5 Tests: Agents and Evidence Flow.

Verifies:
1. Planner: Dynamic subtask graph without 3-5 cap, domain subtasks, 1-4 retrieval queries, structured schema.
2. Researcher: Answers ONLY from supplied chunks, citations [doc, art X.Y.Z, p.N], labels EXPLICIT | INFERENCE | NOT_IN_SOURCE, missing evidence extraction.
3. Evidence Ledger: Canonical indexing, standard citations, prompt formatting, lookup by article.
4. Analyst: Reasons from evidence ledger + findings, dual-sided citations for cross-system interactions.
5. Synthesizer: Preserves citations, mandatory 3 sections, no ungrounded lap-time gains, no definitive wording.
6. GeminiProvider: Clean schema conversion (pydantic_to_gemini_schema) without $defs/$ref, native systemInstruction and responseSchema.
7. Pipeline E2E: Full run with evidence ledger flow through research, analysis, and synthesis.
"""

import pytest
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agents.analyst import AnalystAgent
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
from app.llm.provider import MockLLMProvider, pydantic_to_gemini_schema
from app.models.schemas import EvaluationResult, PlannerOutput, ResearchResult, RunResponse
from app.models.task_graph import TaskGraph, TaskNode, TaskType
from app.orchestration.evidence_ledger import EvidenceLedger, format_citation
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


# ==============================================================================
# 1. Planner Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_planner_dynamic_domains_and_retrieval_queries():
    """Verify Planner generates domain subtasks with 1-4 retrieval queries without 3-5 cap."""
    mock_tasks = [
        TaskNode(
            id="T1",
            description="Investigate front wing width and endplate regulations",
            type=TaskType.RESEARCH,
            metadata={
                "domain": "aerodynamics",
                "retrieval_queries": ["Article 3.4 overall width", "Article 3.9 front wing", "endplate geometry"],
                "retrieval_query": "Article 3.4 overall width",
            },
        ),
        TaskNode(
            id="T2",
            description="Investigate minimum car weight and ballast distribution",
            type=TaskType.RESEARCH,
            metadata={
                "domain": "mass",
                "retrieval_queries": ["Article 4.1 minimum mass", "ballast distribution"],
                "retrieval_query": "Article 4.1 minimum mass",
            },
        ),
        TaskNode(
            id="T3",
            description="Investigate suspension kinematics and ride height travel",
            type=TaskType.RESEARCH,
            metadata={
                "domain": "suspension",
                "retrieval_queries": ["Article 10.1 suspension travel", "Article 10.2 ride height"],
                "retrieval_query": "Article 10.1 suspension travel",
            },
        ),
        TaskNode(
            id="T4",
            description="Investigate power unit heat rejection and radiator ducting",
            type=TaskType.RESEARCH,
            metadata={
                "domain": "cooling",
                "retrieval_queries": ["power unit heat exchangers cooling"],
                "retrieval_query": "power unit heat exchangers cooling",
            },
        ),
        TaskNode(
            id="T5",
            description="Cross-system interaction: Aerodynamic downforce vs suspension stiffness trade-off",
            type=TaskType.ANALYSIS,
            dependencies=["T1", "T3"],
        ),
        TaskNode(
            id="T6",
            description="Cross-system interaction: Sidepod aerodynamic packaging vs power unit thermal cooling",
            type=TaskType.ANALYSIS,
            dependencies=["T1", "T4"],
        ),
        TaskNode(
            id="T7",
            description="Synthesize comprehensive technical regulation compliance and vehicle dynamics report",
            type=TaskType.ANALYSIS,
            dependencies=["T2", "T5", "T6"],
        ),
    ]

    llm = MockLLMProvider(custom_responses={"structured": {"tasks": [t.model_dump() for t in mock_tasks]}})
    planner = PlannerAgent(llm)

    output = await planner.plan(
        user_task="Analyze aerodynamic, mass, suspension, and cooling interactions for 2025 regulations",
        attached_documents=["fia_2025_technical_regulations.pdf"],
    )

    # Assert subtask count exceeds the old 3-5 cap
    assert len(output.tasks) == 7
    research_tasks = [t for t in output.tasks if t.type == TaskType.RESEARCH]
    analysis_tasks = [t for t in output.tasks if t.type == TaskType.ANALYSIS]
    assert len(research_tasks) == 4
    assert len(analysis_tasks) == 3

    # Assert 1 to 4 retrieval queries per research subtask
    for rt in research_tasks:
        assert rt.metadata is not None
        queries = rt.metadata.get("retrieval_queries", [])
        assert 1 <= len(queries) <= 4
        assert rt.metadata.get("domain") is not None

    # Assert dependency edges exist for cross-system analysis
    t5 = next(t for t in output.tasks if t.id == "T5")
    assert t5.dependencies == ["T1", "T3"]


# ==============================================================================
# 2. Evidence Ledger Tests
# ==============================================================================

def test_evidence_ledger_formatting_and_citations():
    """Verify EvidenceLedger standardizes citations [doc, art X.Y.Z, p.N] and formats prompts."""
    ledger = EvidenceLedger()

    chunk1 = RetrievedChunk(
        id="chunk-1",
        document_id="fia_2025_regulations.pdf",
        chunk_index=0,
        source="fia_2025_regulations.pdf",
        content="Overall car width must not exceed 2000mm outboard of Y=1000.",
        similarity_score=0.91,
        score=0.032,
        metadata={
            "document_label": "FIA 2025",
            "article_path": "3.4.1",
            "page_start": 24,
            "page_end": 24,
            "heading": "Overall Width",
        },
    )

    chunk2 = RetrievedChunk(
        id="chunk-2",
        document_id="fia_2025_regulations.pdf",
        chunk_index=1,
        source="fia_2025_regulations.pdf",
        content="The minimum car mass at all times during competition is 798 kg.",
        similarity_score=0.88,
        score=0.029,
        metadata={
            "document_label": "FIA 2025",
            "article_path": "4.1",
            "page_start": 38,
            "page_end": 39,
            "heading": "Minimum Mass",
        },
    )

    ledger.add_chunks([chunk1, chunk2])
    assert len(ledger) == 2

    # Check citation formatting
    item1 = ledger.get("chunk-1")
    assert item1 is not None
    assert item1.citation == "[FIA 2025, art 3.4.1, p.24]"

    item2 = ledger.get("chunk-2")
    assert item2 is not None
    assert item2.citation == "[FIA 2025, art 4.1, p.38-39]"

    # Test lookup by article
    found = ledger.find_by_article("3.4.1")
    assert len(found) == 1
    assert found[0].chunk_id == "chunk-1"

    # Test prompt formatting
    formatted = ledger.format_for_prompt()
    assert "[Evidence #1 | ID: chunk-1 | Citation: [FIA 2025, art 3.4.1, p.24]]" in formatted
    assert "Overall car width must not exceed 2000mm" in formatted


# ==============================================================================
# 3. Research Agent Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_research_agent_strict_grounding_citations_and_labels():
    """Verify ResearchAgent uses citations [doc, art X.Y.Z, p.N] and labels EXPLICIT/INFERENCE/NOT_IN_SOURCE."""
    findings_text = (
        "- [EXPLICIT] The maximum overall car width is capped at 2000mm [FIA 2025, art 3.4.1, p.24].\n"
        "- [INFERENCE] Reducing endplate curvature lowers downforce penalty in crosswinds [FIA 2025, art 3.9.2, p.31].\n"
        "- [NOT_IN_SOURCE] Teams commonly utilize carbon composite weave for torsional stiffness.\n\n"
        "MISSING EVIDENCE: Specific wing flap gurney tab height limits are not defined in the provided chunk."
    )

    llm = MockLLMProvider(custom_responses={"generate": findings_text})
    agent = ResearchAgent(llm)

    chunk = RetrievedChunk(
        id="c1",
        document_id="fia_2025.pdf",
        chunk_index=0,
        source="fia_2025.pdf",
        content="Overall width: 2000mm.",
        similarity_score=0.9,
        metadata={"article_path": "3.4.1", "page_start": 24, "page_end": 24, "label": "FIA 2025"},
    )
    ctx = RetrievedContext(query="front wing width", chunks=[chunk])
    task = TaskNode(id="T1", description="Research front wing dimensional boundaries", type=TaskType.RESEARCH)

    result = await agent.execute_subtask("Analyze wing dimensions", task, retrieved_context=ctx)

    assert result.subtask_id == "T1"
    assert "[EXPLICIT]" in result.findings
    assert "[INFERENCE]" in result.findings
    assert "[NOT_IN_SOURCE]" in result.findings
    assert len(result.citations) >= 1
    assert any("art 3.4.1" in cit for cit in result.citations)
    assert result.missing_evidence is not None
    assert "Specific wing flap gurney tab" in result.missing_evidence


# ==============================================================================
# 4. Analyst Agent Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_analyst_agent_dual_sided_cross_system_citations():
    """Verify AnalystAgent reasons from evidence ledger and provides dual-sided citations for interactions."""
    analysis_response = (
        "### Key Dimensions & Constraints\n"
        "- Aerodynamic envelope is strictly bounded by 2000mm overall width [FIA 2025, art 3.4.1, p.24].\n\n"
        "### Cross-System Interactions & Trade-offs\n"
        "- Downforce vs Suspension: Increasing rear downforce through wing angle [FIA 2025, art 3.10, p.45] "
        "imposes higher vertical wheel loads that demand stiffer bump rubbers under suspension rules [FIA 2025, art 10.2, p.112].\n"
        "- Cooling vs Aero: Narrow sidepod cutouts for aerodynamic floor feeding [FIA 2025, art 3.5, p.28] "
        "constrain radiator heat exchanger packaging [FIA 2025, art 5.4, p.62].\n\n"
        "### Risk & Packaging Assessment\n"
        "- High risk of ride height stalling if aerodynamic elasticity deflects beyond tolerances."
    )

    captured_prompts: List[str] = []

    class CapturingLLM(MockLLMProvider):
        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            captured_prompts.append(prompt)
            return analysis_response

    llm = CapturingLLM()
    analyst = AnalystAgent(llm)

    ledger = EvidenceLedger()
    ledger.add_chunk(
        RetrievedChunk(
            id="c1",
            document_id="fia.pdf",
            chunk_index=0,
            source="fia.pdf",
            content="Article 3.10 rear wing envelope.",
            similarity_score=0.9,
            metadata={"article_path": "3.10", "page_start": 45, "label": "FIA 2025"},
        )
    )
    ledger.add_chunk(
        RetrievedChunk(
            id="c2",
            document_id="fia.pdf",
            chunk_index=1,
            source="fia.pdf",
            content="Article 10.2 suspension kinematics.",
            similarity_score=0.85,
            metadata={"article_path": "10.2", "page_start": 112, "label": "FIA 2025"},
        )
    )

    research_results = [
        ResearchResult(subtask_id="T1", subtask_description="Aero", findings="Rear wing findings [FIA 2025, art 3.10, p.45]"),
        ResearchResult(subtask_id="T2", subtask_description="Suspension", findings="Suspension findings [FIA 2025, art 10.2, p.112]"),
    ]

    report = await analyst.analyze(
        user_task="Evaluate aero vs suspension interaction",
        research_results=research_results,
        evidence_ledger=ledger,
    )

    # Verify prompt contains canonical evidence ledger
    assert len(captured_prompts) == 1
    assert "CANONICAL EVIDENCE LEDGER:" in captured_prompts[0]
    assert "FIA 2025, art 3.10, p.45" in captured_prompts[0]
    assert "FIA 2025, art 10.2, p.112" in captured_prompts[0]

    # Verify dual-sided citations in analysis
    assert "[FIA 2025, art 3.10, p.45]" in report
    assert "[FIA 2025, art 10.2, p.112]" in report


# ==============================================================================
# 5. Synthesizer Agent Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_synthesizer_agent_mandatory_sections_and_no_lap_times():
    """Verify Synthesizer produces the 3 mandatory sections, preserves citations, and rejects ungrounded lap times."""
    captured_prompts: List[str] = []

    class CapturingLLM(MockLLMProvider):
        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            captured_prompts.append(prompt)
            return (
                "### Explicit Regulatory Requirements\n"
                "- Overall car width is fixed at 2000mm [FIA 2025, art 3.4.1, p.24].\n"
                "- Minimum mass is 798 kg [FIA 2025, art 4.1, p.38].\n\n"
                "### Engineering Interpretation / Inference\n"
                "- Tighter rear packaging promotes cleaner floor airflow toward the diffuser [INFERENCE].\n"
                "- Aerodynamic balance shifts dynamically under braking, requiring mechanical balance adjustment [INFERENCE].\n\n"
                "### Limitations and Unknowns\n"
                "- Track-specific yaw angles and crosswind tolerances are not defined in the regulations."
            )

    llm = CapturingLLM()
    synth = SynthesizerAgent(llm)

    ledger = EvidenceLedger()
    ledger.add_chunk(
        RetrievedChunk(
            id="c1",
            document_id="fia.pdf",
            chunk_index=0,
            source="fia.pdf",
            content="Article 3.4.1 width",
            similarity_score=0.9,
            metadata={"article_path": "3.4.1", "page_start": 24, "label": "FIA 2025"},
        )
    )

    evaluation = EvaluationResult(score=90, status="PASS", feedback="Comprehensive and accurate.")
    research = [ResearchResult(subtask_id="T1", subtask_description="Width", findings="2000mm [FIA 2025, art 3.4.1, p.24]")]
    analysis = "Analysis report [FIA 2025, art 3.4.1, p.24]"

    final = await synth.synthesize(
        user_task="Synthesize vehicle compliance",
        research_results=research,
        analysis=analysis,
        evaluation=evaluation,
        evidence_ledger=ledger,
    )

    # Check mandatory sections
    assert "### Explicit Regulatory Requirements" in final
    assert "### Engineering Interpretation / Inference" in final
    assert "### Limitations and Unknowns" in final

    # Check preserved citation
    assert "[FIA 2025, art 3.4.1, p.24]" in final

    # Check no ungrounded lap time claims like "0.3 seconds per lap"
    assert "seconds per lap" not in final
    assert "tenths per lap" not in final

    # Check no authoritative / definitive words
    assert "definitive" not in final.lower()
    assert "authoritative" not in final.lower()


# ==============================================================================
# 6. Schema Conversion (Gemini responseSchema Compatibility)
# ==============================================================================

def test_pydantic_to_gemini_schema_clean_inlined_openapi():
    """Verify pydantic_to_gemini_schema eliminates $defs, $ref, anyOf and converts types to uppercase."""
    class SubItem(BaseModel):
        name: str
        count: Optional[int] = None

    class ComplexModel(BaseModel):
        title: str
        items: List[SubItem]
        tags: Optional[List[str]] = None

    pydantic_schema = ComplexModel.model_json_schema()
    assert "$defs" in pydantic_schema

    gemini_schema = pydantic_to_gemini_schema(pydantic_schema)

    # Disallowed keys must not be present
    assert "$defs" not in gemini_schema
    assert "$ref" not in str(gemini_schema)
    assert "anyOf" not in str(gemini_schema)
    assert "title" not in gemini_schema

    # Types must be uppercase
    assert gemini_schema["type"] == "OBJECT"
    assert gemini_schema["properties"]["items"]["type"] == "ARRAY"
    assert gemini_schema["properties"]["items"]["items"]["type"] == "OBJECT"
    assert gemini_schema["properties"]["items"]["items"]["properties"]["name"]["type"] == "STRING"
    assert gemini_schema["properties"]["items"]["items"]["properties"]["count"]["type"] == "INTEGER"


# ==============================================================================
# 7. End-to-End Orchestrated Pipeline with Evidence Ledger Flow
# ==============================================================================

@pytest.mark.asyncio
async def test_orchestrated_pipeline_full_evidence_ledger_flow():
    """Verify that an end-to-end multi-agent run populates EvidenceLedger and passes it to Analyst and Synthesizer."""
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider(dimensions=16)
    ingestion = DocumentIngestionService(FixedSizeTextChunker(chunk_size=120), embedder, repo)

    doc = KnowledgeDocument(
        id="doc-fia-2025",
        source="fia_technical_regulations.pdf",
        content=(
            "Article 3.4.1 Overall Width: No part of the car may be wider than 2000mm. "
            "Article 4.1 Minimum Mass: The mass of the car without fuel must not be less than 798kg."
        ),
        metadata={"label": "FIA 2025"},
    )
    await ingestion.ingest(doc)

    retrieval_service = RetrievalService(embedder, repo)

    class CustomPhase5LLM(MockLLMProvider):
        async def generate_structured(self, prompt: str, schema: Any, system_prompt: Optional[str] = None) -> Any:
            if schema == PlannerOutput:
                return PlannerOutput(
                    tasks=[
                        TaskNode(
                            id="T1",
                            description="Research car overall width boundaries",
                            type=TaskType.RESEARCH,
                            metadata={"retrieval_query": "Article 3.4.1 Overall Width", "retrieval_top_k": 2},
                        ),
                        TaskNode(
                            id="T2",
                            description="Research car minimum mass requirements",
                            type=TaskType.RESEARCH,
                            metadata={"retrieval_query": "Article 4.1 Minimum Mass", "retrieval_top_k": 2},
                        ),
                    ]
                )
            return await super().generate_structured(prompt, schema, system_prompt)

    llm = CustomPhase5LLM()
    pipeline = OrchestratedPipeline(
        llm_provider=llm,
        settings=settings,
        retrieval_service=retrieval_service,
    )

    response = await pipeline.execute(
        user_task="Analyze overall width and minimum mass for 2025 with knowledge base",
        document_ids=["doc-fia-2025"],
    )

    assert response.execution_strategy == "MULTI_AGENT"
    assert len(response.subtasks) == 2
    assert len(response.research_results) == 2

    # Evidence ledger was populated and returned in response
    assert response.evidence_ledger is not None
    assert "items" in response.evidence_ledger
    assert len(response.evidence_ledger["items"]) > 0

    # Final answer has the 3 required sections
    assert "### Explicit Regulatory Requirements" in response.final_answer
    assert "### Engineering Interpretation / Inference" in response.final_answer
    assert "### Limitations and Unknowns" in response.final_answer
