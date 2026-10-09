"""
Adaptive Execution Strategy Selector for Resource-Aware Orchestration.

Determines whether a user task warrants:
1. DIRECT: Natural conversational response without graph orchestration.
2. SINGLE_AGENT: Direct execution for single-step questions, math, or content drafting.
3. MULTI_AGENT: Dynamic task-graph decomposition, parallel execution, evaluation, and synthesis for complex analytical requests.
"""

import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.llm.base import LLMProvider
from app.models.schemas import ChatMessagePayload


class ExecutionStrategy(str, Enum):
    DIRECT = "DIRECT"
    SINGLE_AGENT = "SINGLE_AGENT"
    MULTI_AGENT = "MULTI_AGENT"


class StrategyDecision(BaseModel):
    """Result of adaptive strategy selection."""

    strategy: ExecutionStrategy = Field(
        ...,
        description="Selected execution strategy: DIRECT, SINGLE_AGENT, or MULTI_AGENT"
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for the strategy decision"
    )
    estimated_complexity: str = Field(
        default="LOW",
        description="Estimated task complexity: LOW, MEDIUM, or HIGH"
    )
    requires_rag: bool = Field(
        default=False,
        description="Whether knowledge base retrieval is expected"
    )


STRATEGY_SELECTOR_PROMPT = """You are an adaptive Execution Strategy Selector in a resource-aware multi-agent AI system.
Your job is to evaluate incoming user requests and select the most appropriate, resource-efficient execution strategy.

Execution Strategies:
1. DIRECT:
   - Simple greetings ("hi", "hello"), pleasantries, acknowledgments ("thanks", "got it"), conversational identity ("who are you"), and short conversational follow-ups.
   - Requires 0 subtasks, 0 graph overhead, and bypasses formal evaluation.

2. SINGLE_AGENT:
   - Single-step factual questions, calculations, explanations, code snippets/debugging, and straightforward writing or content drafting (even long documents like drafting an essay, letter, or memo where multi-agent decomposition is unnecessary).
   - Also used for single-step document-grounded questions where a single agent with retrieved context suffices.
   - Formal multi-agent evaluation is skipped.

3. MULTI_AGENT:
   - Complex, multi-faceted analytical tasks requiring decomposed research, comparative engineering trade-offs, multi-stage investigations, architectural designs, or multi-constraint problem solving where independent subtasks, peer evaluation, and synthesis are genuinely needed.
   - Full task graph decomposition, bounded parallel execution, evaluation audit, and consensus synthesis are executed.

Decision Rules:
- Do NOT classify based purely on message length. A long but straightforward writing/drafting task is SINGLE_AGENT. A short problem with complex interdependent constraints, trade-offs, or multi-criteria analysis is MULTI_AGENT.
- Attached Documents: Set requires_rag=true ONLY if the user's request actually requires querying or referencing the attached document. If the user asks a greeting, a general math problem, or an unrelated question while documents are attached, set requires_rag=false and choose DIRECT or SINGLE_AGENT accordingly.
- Follow-ups: If conversation history is present, brief conversational continuations, clarifications, or summaries are DIRECT or SINGLE_AGENT. Only trigger MULTI_AGENT if the follow-up asks for a new comprehensive multi-part investigation.
- Return your decision strictly matching the schema.
"""


class StrategySelector:
    """Evaluates task requirements and selects the optimal resource-aware execution strategy."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider

    async def select_strategy(
        self,
        user_task: str,
        history: Optional[List[ChatMessagePayload]] = None,
        has_documents: bool = False,
    ) -> StrategyDecision:
        cleaned = user_task.strip().lower()

        # 1. Instant Conversational Pre-Checks for pure greetings & pleasantries
        # ONLY match if the message is purely a greeting or pleasantry without a substantive task
        pure_greeting_patterns = [
            r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))[\s!,.]*$",
            r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))[\s,]+(there|everyone|all|friend|assistant|bot)[\s!,.]*$",
            r"^(how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|sup)[\s?!.]*$",
            r"^(hi|hello|hey)[\s,]+(how\s+are\s+you|how's\s+it\s+going|what's\s+up)[\s?!.]*$",
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
                    )

        # 2. Primary Mechanism: Semantic LLM Classification if LLM Provider is available
        if self.llm:
            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-3:]]
                history_text = "Recent Conversation History:\n" + "\n".join(turns) + "\n\n"

            prompt = (
                f"{history_text}"
                f"Attached Documents Present: {has_documents}\n"
                f"User Task:\n\"{user_task.strip()}\"\n\n"
                f"Evaluate the task requirements, select the optimal strategy (DIRECT, SINGLE_AGENT, or MULTI_AGENT), specify if requires_rag is needed, and provide clear reasoning."
            )

            try:
                decision = await self.llm.generate_structured(
                    prompt=prompt,
                    schema=StrategyDecision,
                    system_prompt=STRATEGY_SELECTOR_PROMPT,
                )
                return decision
            except Exception:
                # Catch invalid structured classification or provider failures cleanly
                pass

        # 3. Robust Fallback Heuristics when LLM provider is unavailable or fails
        writing_keywords = ["write", "draft", "compose", "letter", "email", "memo", "essay", "article", "story"]
        math_keywords = ["calc", "calculate", "solve", "+", "-", "*", "/", "%", "math", "sum", "multiply"]
        multi_keywords = [
            "analyze", "analyse", "interactions", "compare", "comparative", "tradeoff", "trade-off",
            "tradeoffs", "trade-offs", "lifecycle", "architecture", "deconstruct", "investigate",
            "multi-criteria", "feasibility", "roadmap", "regulations", "framework", "distributed"
        ]

        # Multi-facet analytical signals (multiple dimensions, constraints, or comparative elements)
        is_multi = any(kw in cleaned for kw in multi_keywords) and (
            " and " in cleaned or " vs " in cleaned or " versus " in cleaned or "," in cleaned or "between" in cleaned
        )

        if is_multi:
            return StrategyDecision(
                strategy=ExecutionStrategy.MULTI_AGENT,
                reasoning="Fallback: task features indicate multi-faceted comparative analysis.",
                estimated_complexity="HIGH" if has_documents else "MEDIUM",
                requires_rag=has_documents,
            )

        if any(kw in cleaned for kw in writing_keywords):
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT,
                reasoning="Fallback: writing and content drafting request.",
                estimated_complexity="MEDIUM",
                requires_rag=False,
            )

        if any(kw in cleaned for kw in math_keywords) and len(cleaned) < 80:
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT,
                reasoning="Fallback: calculation or mathematical reasoning.",
                estimated_complexity="LOW",
                requires_rag=False,
            )

        if has_documents and any(w in cleaned for w in ["document", "file", "pdf", "according to", "based on", "regulations", "section", "clause"]):
            return StrategyDecision(
                strategy=ExecutionStrategy.SINGLE_AGENT,
                reasoning="Fallback: document-grounded query.",
                estimated_complexity="MEDIUM",
                requires_rag=True,
            )

        return StrategyDecision(
            strategy=ExecutionStrategy.SINGLE_AGENT,
            reasoning="Fallback: default single-agent execution for standard query.",
            estimated_complexity="MEDIUM",
            requires_rag=False,
        )
