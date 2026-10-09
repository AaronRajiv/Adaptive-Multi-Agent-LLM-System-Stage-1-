"""
Programmatic Grounding Verifier (No LLM).

Verifies citation validity and groundedness between the final answer
and the shared canonical Evidence Ledger:
1. Extracts all citations matching `[doc, art X.Y.Z, p.N]` or `[art X.Y.Z]`.
2. Checks that every cited article/page exists in the EvidenceLedger.
3. Checks that key numeric quantities in the claim sentence actually appear
   in the cited chunk text.
4. Computes citation_validity % and compiles unsupported_claims list.
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.orchestration.evidence_ledger import EvidenceLedger


class GroundingVerificationResult(BaseModel):
    """Result of programmatic citation and grounding verification."""

    total_citations: int = Field(default=0, description="Total number of citations extracted from text")
    valid_citations: int = Field(default=0, description="Number of citations matching evidence ledger items")
    citation_validity: float = Field(default=100.0, description="Percentage of valid citations (0.0 - 100.0)")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unverified claims or missing ledger references")
    passed: bool = Field(default=True, description="True if citation_validity >= threshold and no ungrounded citations")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed diagnostic breakdown")


class GroundingVerifier:
    """Programmatic, deterministic verifier that requires no LLM calls."""

    # Matches [FIA 2025, art 3.4.1, p.24], [doc, art X.Y, p.N], [Article 3.4.1], [art 4.1]
    CITATION_REGEX = re.compile(
        r"\[([^\]]*(?:art|article|p\.|page)[^\]]*)\]",
        re.IGNORECASE,
    )
    # Extracts specific article numbers like "3.4.1", "10.2", "4.1"
    ARTICLE_NUM_REGEX = re.compile(r"(?:art|article)\.?\s*([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)

    # Extracts numeric values with units or standalone numbers: e.g. 2000mm, 798kg, 60%, 150,000, $100
    NUMERIC_QUANTITY_REGEX = re.compile(
        r"\b(?:\$)?\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:mm|cm|m|km|kg|g|%|s|ms|min|h|kwh|mwh|kw|mw|usd))?\b",
        re.IGNORECASE,
    )

    @classmethod
    def verify(
        cls,
        text: str,
        evidence_ledger: Optional[EvidenceLedger] = None,
        min_validity_threshold: float = 80.0,
    ) -> GroundingVerificationResult:
        """
        Extract citations from text and verify against the evidence ledger.
        """
        if not text or not text.strip():
            return GroundingVerificationResult(
                total_citations=0,
                valid_citations=0,
                citation_validity=100.0,
                unsupported_claims=[],
                passed=True,
                details={"reason": "Empty text provided"},
            )

        citations_found = cls.CITATION_REGEX.findall(text)
        total_citations = len(citations_found)

        if total_citations == 0:
            return GroundingVerificationResult(
                total_citations=0,
                valid_citations=0,
                citation_validity=100.0,
                unsupported_claims=[],
                passed=True,
                details={"reason": "No citations extracted from text"},
            )

        if evidence_ledger is None or len(evidence_ledger.items) == 0:
            return GroundingVerificationResult(
                total_citations=total_citations,
                valid_citations=total_citations,
                citation_validity=100.0,
                unsupported_claims=[],
                passed=True,
                details={"reason": "No evidence ledger chunks provided for run"},
            )

        valid_citations = 0
        unsupported_claims: List[str] = []
        details_list: List[Dict[str, Any]] = []

        # Split text into sentence/line segments for claim context
        segments = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]

        for raw_citation in citations_found:
            full_cit_bracket = f"[{raw_citation}]"
            matching_segment = next((s for s in segments if raw_citation in s), full_cit_bracket)

            # Check existence in evidence ledger
            matched_items = []
            # 1. Exact or substring match in citation strings
            matched_items = [
                item for item in evidence_ledger.items.values()
                if raw_citation.lower() in item.citation.lower()
                or item.citation.lower() in raw_citation.lower()
            ]

            # 2. Match via article number if specified in citation
            art_match = cls.ARTICLE_NUM_REGEX.search(raw_citation)
            if art_match:
                art_num = art_match.group(1)
                matched_items = evidence_ledger.lookup_by_article(art_num)
            elif not matched_items:
                # No article specified in citation: match via document label or source
                for item in evidence_ledger.items.values():
                    if item.document_label and item.document_label.lower() in raw_citation.lower():
                        matched_items.append(item)

            if not matched_items:
                unsupported_claims.append(
                    f"Citation '{full_cit_bracket}' in claim '{matching_segment[:120]}...' not found in Evidence Ledger."
                )
                details_list.append({
                    "citation": full_cit_bracket,
                    "valid": False,
                    "reason": "Not found in Evidence Ledger",
                })
                continue

            # Check numeric quantities in the claim sentence (excluding numbers inside citation brackets)
            clean_segment = cls.CITATION_REGEX.sub("", matching_segment)
            numbers_in_claim = cls.NUMERIC_QUANTITY_REGEX.findall(clean_segment)
            meaningful_numbers = [
                n.strip() for n in numbers_in_claim
                if len(n.strip()) > 1 or re.search(r"[a-z%]", n.lower())
            ]

            combined_evidence_text = " ".join(item.text for item in matched_items)
            missing_numbers = []
            for num in meaningful_numbers:
                clean_num = re.sub(r"[^\d.]", "", num)
                if clean_num and clean_num not in combined_evidence_text:
                    missing_numbers.append(num)

            if missing_numbers:
                unsupported_claims.append(
                    f"Claim '{matching_segment[:120]}...' cited '{full_cit_bracket}' but numeric value(s) {missing_numbers} not found in cited evidence."
                )
                details_list.append({
                    "citation": full_cit_bracket,
                    "valid": False,
                    "missing_numbers": missing_numbers,
                    "reason": "Numeric values ungrounded in cited chunk",
                })
            else:
                valid_citations += 1
                details_list.append({
                    "citation": full_cit_bracket,
                    "valid": True,
                    "matched_chunks": [item.chunk_id for item in matched_items],
                })

        citation_validity = round((valid_citations / total_citations) * 100.0, 1)
        passed = citation_validity >= min_validity_threshold and len(unsupported_claims) == 0

        return GroundingVerificationResult(
            total_citations=total_citations,
            valid_citations=valid_citations,
            citation_validity=citation_validity,
            unsupported_claims=unsupported_claims,
            passed=passed,
            details={"verifications": details_list},
        )
