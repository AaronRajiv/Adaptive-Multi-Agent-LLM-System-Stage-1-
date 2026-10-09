"""
Adaptive Execution Strategy Selector for Resource-Aware Orchestration.

Determines whether a user task warrants:
1. DIRECT: Natural conversational response without graph overhead.
2. SINGLE_AGENT: Direct execution for single-step questions, math, or content drafting.
3. SINGLE_AGENT_RAG: Focused single-turn document lookup and context grounding.
4. MULTI_AGENT: Dynamic task-graph decomposition, parallel execution, cross-document reasoning, evaluation, and synthesis.

Evaluates 4 core dimensions:
1. Task length / structure
2. Number of technical/operational domains
3. Whether documents are attached
4. Need for cross-document or cross-constraint reasoning
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.llm.base import LLMProvider
from app.models.schemas import ChatMessagePayload


class ExecutionStrategy(str, Enum):
    DIRECT = "DIRECT"
    SINGLE_AGENT = "SINGLE_AGENT"
    SINGLE_AGENT_RAG = "SINGLE_AGENT_RAG"
    MULTI_AGENT = "MULTI_AGENT"


class StrategyDecision(BaseModel):
    """Result of adaptive strategy classification."""

    strategy: ExecutionStrategy = Field(
        ...,
        description="Selected execution strategy: DIRECT, SINGLE_AGENT, SINGLE_AGENT_RAG, or MULTI_AGENT",
    )
    reasoning: str = Field(
        ...,
        description="Justification for the strategy decision based on task criteria",
    )
    estimated_complexity: str = Field(
        default="LOW",
        description="Estimated task complexity: LOW, MEDIUM, or HIGH",
    )
    requires_rag: bool = Field(
        default=False,
        description="Whether knowledge base retrieval is expected",
    )
    domains: List[str] = Field(
        default_factory=list,
        description="Identified technical, operational, or business domains involved",
    )
    cross_document_reasoning: bool = Field(
        default=False,
        description="Whether cross-document or multi-section comparative synthesis is needed",
    )
    budget: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resource budget constraints (max_llm_calls, max_tokens, max_wall_time) allocated",
    )


STRATEGY_SELECTOR_PROMPT = """You are an adaptive Execution Strategy Selector in a resource-aware multi-agent AI system.
Your job is to evaluate incoming user requests and select the most appropriate, resource-efficient execution strategy.

Evaluate the request across 4 explicit criteria:
1. Task Length and Structure:
   - Simple conversational greetings/pleasantries -> DIRECT.
   - Focused, single-step questions or drafting -> SINGLE_AGENT or SINGLE_AGENT_RAG.
   - Multi-step, structured analyses, trade-off comparisons -> MULTI_AGENT.
2. Number of Domains:
   - 0 or 1 domain (e.g. only mass, or only basic arithmetic) -> DIRECT, SINGLE_AGENT, or SINGLE_AGENT_RAG.
   - >= 2 distinct interacting domains (e.g. aerodynamics AND suspension AND cost) -> MULTI_AGENT.
3. Attached Documents:
   - If documents are attached and the task requires referencing them for a single topic -> SINGLE_AGENT_RAG.
   - If documents are attached and the task requires cross-document/cross-section comparative synthesis -> MULTI_AGENT with requires_rag=true.
   - If documents are attached but the user asks a greeting or unrelated math -> requires_rag=false.
4. Need for Cross-Document Reasoning:
   - Comparing multiple documents, regulations, or distinct sections -> cross_document_reasoning=true -> MULTI_AGENT.

Execution Strategies:
- DIRECT: Conversational greetings, pleasantries, acknowledgments.
- SINGLE_AGENT: Single-step questions, calculations, explanations, code snippets, or drafting without attached documents.
- SINGLE_AGENT_RAG: Focused single-step query grounded in attached documents.
- MULTI_AGENT: Complex, multi-domain analytical tasks requiring decomposed research, cross-document reasoning, evaluation, and synthesis.

Return your decision strictly conforming to the JSON schema.
"""


class StrategySelector:
    """Evaluates task requirements and selects the optimal resource-aware execution strategy."""

    KNOWN_DOMAINS = [
        "aerodynamics", "aero", "chassis", "suspension", "powertrain", "power unit",
        "engine", "battery", "thermal", "cooling", "mass", "weight", "dimensions",
        "regulations", "compliance", "safety", "cost", "financial", "cap", "tires",
        "brakes", "telemetry", "software", "logistics", "strategy"
    ]

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider

    async def select_strategy(
        self,
        user_task: str,
        history: Optional[List[ChatMessagePayload]] = None,
        has_documents: bool = False,
    ) -> StrategyDecision:
        cleaned = user_task.strip().lower()

        # -------------------------------------------------------------
        # 1. Instant Conversational Pre-Checks (0 LLM overhead)
        # -------------------------------------------------------------
        pure_greeting_patterns = [
            r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))[\s!,.]*$",
            r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))[\s,]+(there|everyone|all|friend|assistant|bot)[\s!,.]*$",
            r"^(how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|sup)[\s?!.]*$",
            r"^(hi|hello|hey|greetings|howdy)(?:[\s,]+(?:there|everyone|all|friend|assistant|bot))?[\s,]+(how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|sup)[\s?!.]*$",
            r"^(thanks|thank\s+you|thanks\s+a\s+lot|thank\s+you\s+very\s+much|ok|okay|got\s+it|cool|understood)[\s!.?]*$",
            r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)[\s!.?]*$",
            r"^(tell\s+me\s+a\s+joke|say\s+something\s+funny)[\s!.?]*$",
        ]
        if len(cleaned) <= 80:
            for pattern in pure_greeting_patterns:
                if re.match(pattern, cleaned):
                    return StrategyDecision(
                        strategy=ExecutionStrategy.DIRECT,
                        reasoning="Task is a conversational greeting or simple pleasantry.",
                        estimated_complexity="LOW",
                        requires_rag=False,
                        domains=[],
                        cross_document_reasoning=False,
                    )

        # -------------------------------------------------------------
        # 2. Extract Domains & Cross-Document Signals
        # -------------------------------------------------------------
        detected_domains = [d for d in self.KNOWN_DOMAINS if re.search(r"\b" + re.escape(d) + r"\b", cleaned)]
        has_cross_doc_cue = bool(
            re.search(r"\b(cross-reference|compare documents|between articles|across sections|multiple regulations|trade-off|tradeoff)\b", cleaned)
        )

        # -------------------------------------------------------------
        # 3. Primary Classifier: Structured LLM Call
        # -------------------------------------------------------------
        if self.llm:
            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-3:]]
                history_text = "Recent Conversation History:\n" + "\n".join(turns) + "\n\n"

            prompt = (
                f"{history_text}"
                f"Attached Documents Present: {has_documents}\n"
                f"User Task:\n\"{user_task.strip()}\"\n\n"
                f"Classify the optimal execution strategy (DIRECT, SINGLE_AGENT, SINGLE_AGENT_RAG, or MULTI_AGENT). "
                f"Identify the number and names of domains, whether cross-document reasoning is needed, "
                f"and whether RAG retrieval is required."
            )

            try:
                decision = await self.llm.generate_structured(
                    prompt=prompt,
                    schema=StrategyDecision,
                    system_prompt=STRATEGY_SELECTOR_PROMPT,
                )
                if not decision.domains and detected_domains:
                    decision.domains = detected_domains
                return decision
            except Exception:
                # Fallback to rules if LLM provider structured output fails
                pass

        # -------------------------------------------------------------
        # 4. Fallback Rule-Based Heuristics
        # -------------------------------------------------------------
        # Multi-domain or cross-document reasoning -> MULTI_AGENT
        multi_keywords = [
            "analyze", "analyse", "interactions", "compare", "comparative", "tradeoff", "trade-off",
            "tradeoffs", "trade-offs", "lifecycle", "architecture", "deconstruct", "investigate",
            "multi-criteria", "feasibility", "roadmap", "regulations", "framework", "distributed"
        ]
        is_multi_domain = len(detected_domains) >= 2
        is_multi_structure = any(kw in cleaned for kw in multi_keywords) and (
            " and " in cleaned or " vs " in cleaned or " versus " in cleaned or "," in cleaned or "between" in cleaned
        )

        if is_multi_domain or is_multi_structure or has_cross_doc_cue:
            return StrategyDecision(
                strategy=ExecutionStrategy.MULTI_AGENT,
                reasoning=(
                    f"Task requires multi-faceted analysis across {len(detected_domains)} domain(s) "
                    f"({'cross-document reasoning' if has_cross_doc_cue else 'structural decomposition'})."
                ),
                estimated_complexity="HIGH" if has_documents else "MEDIUM",
                requires_rag=has_documents,
                domains=detected_domains,
                cross_document_reasoning=has_cross_doc_cue or (has_documents and is_multi_domain),
            )

        # Single-topic document query
        if has_documents and any(w in cleaned for w in ["document", "file", "pdf", "according to", "based on", "regulations", "article", "clause", "rule"]):
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT_RAG,
                reasoning="Focused single-step question grounded in attached document.",
                estimated_complexity="MEDIUM",
                requires_rag=True,
                domains=detected_domains,
                cross_document_reasoning=False,
            )

        # Calculation or single math problem
        math_keywords = ["calc", "calculate", "solve", "+", "-", "*", "/", "%", "math", "sum", "multiply"]
        if any(kw in cleaned for kw in math_keywords) and len(cleaned) < 80:
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT,
                reasoning="Direct calculation or mathematical reasoning.",
                estimated_complexity="LOW",
                requires_rag=False,
                domains=[],
                cross_document_reasoning=False,
            )

        # Writing or drafting request
        writing_keywords = ["write", "draft", "compose", "letter", "email", "memo", "essay", "article", "story"]
        if any(kw in cleaned for kw in writing_keywords):
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT,
                reasoning="Writing or drafting task suitable for single-agent execution.",
                estimated_complexity="MEDIUM",
                requires_rag=False,
                domains=detected_domains,
                cross_document_reasoning=False,
            )

        return StrategyDecision(
            strategy=ExecutionStrategy.SINGLE_AGENT,
            reasoning="Default single-agent execution for standard query.",
            estimated_complexity="LOW" if len(cleaned) < 100 else "MEDIUM",
            requires_rag=False,
            domains=detected_domains,
            cross_document_reasoning=False,
        )
