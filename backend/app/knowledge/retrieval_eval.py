"""
Retrieval Evaluation Harness & Benchmark for Regulation Question-Article Pairs.
Evaluates Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
"""

from typing import Any, Dict, List, Optional

from app.knowledge.models import RetrievedContext
from app.knowledge.retrieval import RetrievalService
from app.logging_config import get_logger

logger = get_logger("app.knowledge.retrieval_eval")

# 32 Curated FIA Formula 1 Technical Regulations Question -> Expected Article Pairs
FIA_EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "id": 1,
        "question": "What is the maximum allowed car width limit outboard of Y=1000?",
        "expected_article": "3.4.1",
        "topic": "Aerodynamics / Overall Width",
    },
    {
        "id": 2,
        "question": "What is the maximum wheelbase distance between planes XF=0 and XR=0?",
        "expected_article": "3.4.2",
        "topic": "Aerodynamics / Wheelbase",
    },
    {
        "id": 3,
        "question": "What are the rules and constraints governing floor bodywork and diffusers?",
        "expected_article": "3.5",
        "topic": "Aerodynamics / Floor",
    },
    {
        "id": 4,
        "question": "What are the dimensional restrictions and geometry for the rear wing assembly?",
        "expected_article": "3.10",
        "topic": "Aerodynamics / Rear Wing",
    },
    {
        "id": 5,
        "question": "What are the rules for wheel bodywork and wheel covers?",
        "expected_article": "3.13",
        "topic": "Aerodynamics / Wheel Bodywork",
    },
    {
        "id": 6,
        "question": "What are the limitations on suspension fairings and aerodynamic shrouds?",
        "expected_article": "3.14",
        "topic": "Aerodynamics / Suspension Fairings",
    },
    {
        "id": 7,
        "question": "What are the general principles governing aerodynamic component flexibility and deflection?",
        "expected_article": "3.15",
        "topic": "Aerodynamics / Component Flexibility",
    },
    {
        "id": 8,
        "question": "What is the absolute minimum mass of the car without fuel during competition?",
        "expected_article": "4.1",
        "topic": "Mass / Minimum Weight",
    },
    {
        "id": 9,
        "question": "What are the requirements for mass distribution and ballast placement on the car?",
        "expected_article": "4.2",
        "topic": "Mass / Mass Distribution",
    },
    {
        "id": 10,
        "question": "What is the minimum weight requirement for the driver and seat assembly?",
        "expected_article": "4.6",
        "topic": "Mass / Driver Mass",
    },
    {
        "id": 11,
        "question": "What are the engine cylinder arrangement, displacement, and internal combustion specifications?",
        "expected_article": "5.1",
        "topic": "Power Unit / Engine Specifications",
    },
    {
        "id": 12,
        "question": "Are other means of vehicle propulsion or thrust permitted besides the approved power unit?",
        "expected_article": "5.2",
        "topic": "Power Unit / Propulsion",
    },
    {
        "id": 13,
        "question": "What is the minimum mass and centre of gravity height requirement for the power unit?",
        "expected_article": "5.5",
        "topic": "Power Unit / Mass & Centre of Gravity",
    },
    {
        "id": 14,
        "question": "What restrictions apply to variable geometry systems on the engine and intake?",
        "expected_article": "5.10",
        "topic": "Power Unit / Variable Geometry",
    },
    {
        "id": 15,
        "question": "What are the operational constraints and energy limits for the Energy Recovery System (ERS)?",
        "expected_article": "5.13",
        "topic": "Power Unit / Energy Recovery System",
    },
    {
        "id": 16,
        "question": "What are the rules regarding oil and coolant systems and charge air cooling?",
        "expected_article": "5.24",
        "topic": "Power Unit / Coolant Systems",
    },
    {
        "id": 17,
        "question": "What general electrical safety and high voltage insulation requirements must be met?",
        "expected_article": "5.25",
        "topic": "Power Unit / Electrical Safety",
    },
    {
        "id": 18,
        "question": "What standard specifications must the fuel bladder and rubber fuel cell comply with?",
        "expected_article": "6.1",
        "topic": "Fuel System / Fuel Tanks",
    },
    {
        "id": 19,
        "question": "What are the fuel flow limit requirements and fuel pump rules?",
        "expected_article": "6.2",
        "topic": "Fuel System / Fuel Flow Limit",
    },
    {
        "id": 20,
        "question": "What are the cockpit and external master switch shutoff requirements?",
        "expected_article": "8.8",
        "topic": "Electrical Systems / Master Switch",
    },
    {
        "id": 21,
        "question": "What accident data recorder and accident analysis sensors must be fitted?",
        "expected_article": "8.10",
        "topic": "Electrical Systems / Accident Data",
    },
    {
        "id": 22,
        "question": "What sensor signals and telemetry must be shared with the standard ECU?",
        "expected_article": "8.21",
        "topic": "Electrical Systems / Sensor Signals",
    },
    {
        "id": 23,
        "question": "What are the gearbox casing dimensions and transmission constraints?",
        "expected_article": "9.6",
        "topic": "Transmission / Gearbox Dimensions",
    },
    {
        "id": 24,
        "question": "What torque transfer systems, differentials, and traction control rules apply?",
        "expected_article": "9.10",
        "topic": "Transmission / Torque Transfer Systems",
    },
    {
        "id": 25,
        "question": "What dimensions and material restrictions apply to wheel rims?",
        "expected_article": "10.7",
        "topic": "Wheels / Wheel Rims",
    },
    {
        "id": 26,
        "question": "What regulations govern tyres and tyre supply?",
        "expected_article": "10.8",
        "topic": "Wheels / Tyres",
    },
    {
        "id": 27,
        "question": "What safety tether and retention systems are required for wheel attachment?",
        "expected_article": "10.9",
        "topic": "Wheels / Wheel Attachment & Tethers",
    },
    {
        "id": 28,
        "question": "How many separate hydraulic circuits are required for the brake system?",
        "expected_article": "11.1",
        "topic": "Brake System / Hydraulic Circuits",
    },
    {
        "id": 29,
        "question": "What are the dimensional restrictions for brake discs and brake pads?",
        "expected_article": "11.3",
        "topic": "Brake System / Brake Discs",
    },
    {
        "id": 30,
        "question": "What are the rules and controls for the rear brake control system (brake-by-wire)?",
        "expected_article": "11.6",
        "topic": "Brake System / Rear Brake Control",
    },
    {
        "id": 31,
        "question": "What are the mandatory dimensions and templates for the cockpit specification?",
        "expected_article": "12.5",
        "topic": "Chassis / Cockpit Specification",
    },
    {
        "id": 32,
        "question": "What static load and dynamic crash tests must the front impact structure pass?",
        "expected_article": "13.6",
        "topic": "Safety Structures / Front Impact Structure",
    },
]


def _article_matches(expected_article: str, retrieved_chunk) -> bool:
    """Checks whether a retrieved chunk matches the expected article."""
    expected_norm = expected_article.strip().lower()
    chunk_art = (retrieved_chunk.article_path or retrieved_chunk.metadata.get("article_path", "")).strip().lower()

    if not chunk_art:
        # Check in content or heading if article path was not isolated
        heading = (retrieved_chunk.heading or retrieved_chunk.metadata.get("heading", "")).strip().lower()
        content = retrieved_chunk.content.lower()
        if expected_norm in heading or expected_norm in content:
            return True
        return False

    # Exact or prefix match (e.g. expected 3.2 matches chunk 3.2 or 3.2.1)
    if chunk_art == expected_norm:
        return True
    if chunk_art.startswith(f"{expected_norm}.") or expected_norm.startswith(f"{chunk_art}."):
        return True
    return False


async def evaluate_retrieval_dataset(
    retrieval_service: RetrievalService,
    dataset: Optional[List[Dict[str, Any]]] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Executes benchmark queries against the retrieval engine and computes
    Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
    """
    eval_set = dataset or FIA_EVALUATION_DATASET
    total_queries = len(eval_set)
    if total_queries == 0:
        return {"total_queries": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "details": []}

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: List[float] = []
    details: List[Dict[str, Any]] = []

    for item in eval_set:
        qid = item["id"]
        q_text = item["question"]
        expected_art = item["expected_article"]

        ctx: RetrievedContext = await retrieval_service.retrieve(
            query=q_text,
            top_k=top_k,
            document_ids=document_ids,
        )

        found_rank: Optional[int] = None
        retrieved_articles = []

        for rank, chunk in enumerate(ctx.chunks):
            art = chunk.article_path or chunk.metadata.get("article_path", "unknown")
            retrieved_articles.append(art)
            if found_rank is None and _article_matches(expected_art, chunk):
                found_rank = rank + 1  # 1-indexed rank

        if found_rank is not None:
            if found_rank <= 5:
                hits_at_5 += 1
            if found_rank <= 10:
                hits_at_10 += 1
            rr = 1.0 / found_rank
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)
        details.append(
            {
                "id": qid,
                "question": q_text,
                "expected_article": expected_art,
                "found_rank": found_rank,
                "rr": round(rr, 4),
                "top_retrieved": retrieved_articles[:5],
            }
        )

    recall_at_5 = round(hits_at_5 / total_queries, 4)
    recall_at_10 = round(hits_at_10 / total_queries, 4)
    mrr = round(sum(reciprocal_ranks) / total_queries, 4)

    return {
        "total_queries": total_queries,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "details": details,
    }
