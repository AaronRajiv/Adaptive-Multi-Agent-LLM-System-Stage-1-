"""
Evaluator Agent with Groundedness, Domain Coverage, Distinction, and Honesty Rubric.

Responsible for critically evaluating the final user-facing answer and analytical synthesis
against the original user task across 4 dimensions:
1. Groundedness & Source Fidelity (25 pts)
2. Multi-Domain Task Coverage (25 pts)
3. Correct Labelling (Explicit vs Inferred) (25 pts)
4. Honesty About Limits & Unknowns (25 pts)

Emits a structured score, PASS/FAIL status, component sub-scores, uncovered domains,
and detailed constructive feedback for targeted replanning.
"""

from typing import Any, Dict, Optional
from app.llm.base import LLMProvider, LLMStructuredOutputError
from app.models.schemas import EvaluationResult
from app.orchestration.evidence_ledger import EvidenceLedger
from app.orchestration.grounding_verifier import GroundingVerificationResult


EVALUATOR_SYSTEM_PROMPT = """You are the EVALUATOR AGENT performing an objective, rigorous quality audit of the system's output.

Evaluation Rubric (100 total points):
1. Groundedness & Source Fidelity (25 pts):
   - Are all factual assertions strictly grounded in cited source chunks?
   - Zero tolerance for hallucinated or ungrounded claims (e.g. ungrounded lap-time gains or fabricated parameters).
   - Verify every cited article exists and key numbers match source evidence.
2. Multi-Domain Task Coverage (25 pts):
   - Did the output thoroughly cover every domain and technical dimension asked in the user task?
   - If any domains are omitted or weak, enumerate them in 'uncovered_domains'.
3. Correct Labelling (Explicit vs Inferred) (25 pts):
   - Are explicit regulatory rules cleanly separated from engineering interpretations or inferences?
   - Are sections structured into '### Explicit Regulatory Requirements' vs '### Engineering Interpretation / Inference'?
4. Honesty About Limits & Unknowns (25 pts):
   - Does the output honestly acknowledge unknowns, ambiguities, and missing data in '### Limitations and Unknowns'?
   - Does it avoid deceptive "definitive" or "authoritative" language when evidence is incomplete?

Scoring Rules:
1. Provide an overall integer score between 0 and 100 (sum of sub-scores).
2. If overall score >= {threshold}, status MUST be "PASS". Otherwise status MUST be "FAIL".
3. Provide component sub-scores (CRITICAL: each must strictly be an integer between 0 and 25, NOT 0-100, do NOT use percentages; 25 represents perfect score for each category):
   - groundedness_score (0-25 pts max)
   - coverage_score (0-25 pts max)
   - distinction_score (0-25 pts max)
   - honesty_score (0-25 pts max)
4. List any weak or uncovered domains in 'uncovered_domains'.
5. Provide constructive feedback detailing specific gaps to guide targeted replanning.
6. Return your evaluation strictly in the required JSON schema.
"""


class EvaluatorAgent:
    """Agent that evaluates analytical quality and outputs structured score and feedback."""

    def __init__(self, llm_provider: LLMProvider, pass_threshold: int = 80):
        self.llm = llm_provider
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        user_task: str,
        analysis: str,
        final_answer: Optional[str] = None,
        evidence_ledger: Optional[EvidenceLedger] = None,
        verification_result: Optional[GroundingVerificationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate the output against the original user task using the 4-dimension rubric.

        Args:
            user_task: The original user task.
            analysis: The analytical synthesis produced by the Analyst Agent.
            final_answer: Optional final user-facing response from Synthesizer Agent.
            evidence_ledger: Optional Evidence Ledger containing canonical chunks.
            verification_result: Optional deterministic verifier metrics.

        Returns:
            EvaluationResult with score, status, sub-scores, uncovered domains, and feedback.
        """
        system_prompt = EVALUATOR_SYSTEM_PROMPT.format(threshold=self.pass_threshold)

        eval_target = final_answer if final_answer and final_answer.strip() else analysis
        target_label = "Final User-Facing Answer" if final_answer and final_answer.strip() else "Analyst Agent Output"

        prompt_parts = [
            f"Original User Task:\n\"{user_task.strip()}\"\n",
            f"{target_label} to Evaluate:\n{eval_target}\n",
        ]

        if final_answer and analysis and analysis.strip() != final_answer.strip():
            prompt_parts.append(f"Intermediate Analyst Synthesis (Context):\n{analysis}\n")

        if verification_result:
            prompt_parts.append(
                f"Programmatic Grounding Verifier Audit:\n"
                f"- Citation Validity: {verification_result.citation_validity}%\n"
                f"- Valid Citations: {verification_result.valid_citations}/{verification_result.total_citations}\n"
                f"- Unsupported Claims: {verification_result.unsupported_claims}\n"
            )

        if evidence_ledger and evidence_ledger.items:
            summary = evidence_ledger.get_ledger_summary()
            prompt_parts.append(
                f"Evidence Ledger Summary:\n"
                f"- Total chunks: {summary['total_chunks']}\n"
                f"- Indexed articles: {summary['indexed_articles']}\n"
            )
            chunk_snippets = []
            for item in list(evidence_ledger.items.values())[:15]:
                chunk_snippets.append(
                    f"[{item.citation}] (Doc: {item.document_id}, Source: {item.document_label or item.document_id})\n{item.text}"
                )
            prompt_parts.append(
                "Evidence Ledger Source Chunks (Ground Truth Evidence):\n"
                + "\n\n".join(chunk_snippets)
                + "\n"
            )

        prompt_parts.append(
            f"Pass Threshold: {self.pass_threshold} / 100\n\n"
            f"Score this output against all 4 rubric criteria and provide constructive evaluation feedback."
        )

        prompt = "\n".join(prompt_parts)

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=EvaluationResult,
                system_prompt=system_prompt,
            )

            # Attach verification result metadata if available
            if verification_result:
                result.verification_result = verification_result.model_dump()

            # 1. Compute total score in code as sum of validated subscores (ignore model-reported total)
            subscores = [
                result.groundedness_score,
                result.coverage_score,
                result.distinction_score,
                result.honesty_score,
            ]
            if any(s is not None for s in subscores):
                result.score = sum(s or 0 for s in subscores)

            # 2. Groundedness Gate: fail if groundedness < 15/25 or verifier citation_validity < threshold
            groundedness_gated = (result.groundedness_score is not None and result.groundedness_score < 15)
            verifier_gated = bool(verification_result and verification_result.citation_validity < float(self.pass_threshold))

            if groundedness_gated or verifier_gated:
                result.status = "FAIL"
                gate_reasons = []
                if groundedness_gated:
                    gate_reasons.append(f"Groundedness score ({result.groundedness_score}/25) failed minimum gate threshold (15/25)")
                if verifier_gated:
                    gate_reasons.append(f"Citation validity ({verification_result.citation_validity}%) below threshold ({self.pass_threshold}%)")
                result.feedback += f" [GATE FAILED: {'; '.join(gate_reasons)}]"
            else:
                result.status = "PASS" if (result.score or 0) >= self.pass_threshold else "FAIL"

            return result
        except Exception as e:
            if isinstance(e, LLMStructuredOutputError):
                raise
            raise LLMStructuredOutputError(f"Evaluator Agent execution failed: {str(e)}") from e
