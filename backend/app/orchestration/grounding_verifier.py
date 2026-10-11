"""
Programmatic Grounding Verifier (No LLM).

Verifies citation validity and groundedness between the final answer
and the shared canonical Evidence Ledger:
1. Extracts all web citations matching `[web:<8-char hash>]` FIRST.
2. Extracts all document citations matching `[doc, art X.Y.Z, p.N]` or `[art X.Y.Z]`.
3. Checks that every cited article/page/web-hash exists in the EvidenceLedger.
4. Checks that key numeric quantities in the claim sentence actually appear
   in the cited chunk text.
5. Computes citation_validity % and compiles unsupported_claims list.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.orchestration.evidence_ledger import ConflictRecord, EvidenceLedger


class GroundingVerificationResult(BaseModel):
    """Result of programmatic citation and grounding verification."""

    total_citations: int = Field(default=0, description="Total number of citations extracted from text")
    valid_citations: int = Field(default=0, description="Number of citations matching evidence ledger items")
    citation_validity: float = Field(default=100.0, description="Percentage of valid citations (0.0 - 100.0)")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unverified claims or missing ledger references")
    claim_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Claim-by-claim classification by source type ('document', 'web', 'hybrid')")
    source_distribution: Dict[str, int] = Field(default_factory=dict, description="Counts of claims by source type: document, web, hybrid")
    conflicts: List[ConflictRecord] = Field(default_factory=list, description="Detected conflicts between document and web evidence")
    conflicts_flagged_in_text: bool = Field(default=True, description="True if detected conflicts are flagged in text")
    inferred_with_citations: List[str] = Field(default_factory=list, description="Citations attached to engineering inference or trade-offs")
    uncited_factual_claims: List[str] = Field(default_factory=list, description="Factual metric claims lacking citations")
    claim_support_failures: List[str] = Field(default_factory=list, description="Claims marked EXPLICIT lacking lexical overlap with cited chunks")
    passed: bool = Field(default=True, description="True if citation_validity >= threshold, no ungrounded citations, and conflicts flagged")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed diagnostic breakdown")


class GroundingVerifier:
    """Programmatic, deterministic verifier that requires no LLM calls."""

    # 1. Separate web-citation pattern checked before document pattern
    WEB_CITATION_REGEX = re.compile(r"\[web:([a-f0-9]{8})\]", re.IGNORECASE)

    # 2. Document citation pattern: matches [FIA 2025, art 3.4.1, p.24], [doc, art X.Y, p.N], [Article 3.4.1], [art 4.1]
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
    def _extract_local_numbers_for_citation(
        cls, segment: str, target_citation: str
    ) -> List[str]:
        """
        Extract numeric quantities belonging to a specific citation within a sentence.
        If the sentence contains multiple citations (e.g. comparative or conflict sentences),
        partitions the text between adjacent citations so numbers in other clauses are not
        falsely expected from this citation's evidence.
        """
        all_spans: List[Tuple[int, int, str]] = []
        for m in cls.WEB_CITATION_REGEX.finditer(segment):
            all_spans.append((m.start(), m.end(), m.group(0)))
        for m in cls.CITATION_REGEX.finditer(segment):
            if not any(s <= m.start() and m.end() <= e for s, e, _ in all_spans):
                all_spans.append((m.start(), m.end(), m.group(0)))

        all_spans.sort(key=lambda x: x[0])

        if len(all_spans) <= 1:
            clean = cls.WEB_CITATION_REGEX.sub("", segment)
            clean = cls.CITATION_REGEX.sub("", clean)
            nums = cls.NUMERIC_QUANTITY_REGEX.findall(clean)
            return [n.strip() for n in nums if len(n.strip()) > 1 or re.search(r"[a-z%]", n.lower())]

        target_idx = -1
        for idx, (s, e, cit_text) in enumerate(all_spans):
            if cit_text.lower() == target_citation.lower():
                target_idx = idx
                break

        if target_idx == -1:
            clean = cls.WEB_CITATION_REGEX.sub("", segment)
            clean = cls.CITATION_REGEX.sub("", clean)
            nums = cls.NUMERIC_QUANTITY_REGEX.findall(clean)
            return [n.strip() for n in nums if len(n.strip()) > 1 or re.search(r"[a-z%]", n.lower())]

        win_start = all_spans[target_idx - 1][1] if target_idx > 0 else 0
        win_end = all_spans[target_idx + 1][0] if target_idx < len(all_spans) - 1 else len(segment)

        local_window = segment[win_start:win_end]
        clean_local = cls.WEB_CITATION_REGEX.sub("", local_window)
        clean_local = cls.CITATION_REGEX.sub("", clean_local)
        nums = cls.NUMERIC_QUANTITY_REGEX.findall(clean_local)
        return [n.strip() for n in nums if len(n.strip()) > 1 or re.search(r"[a-z%]", n.lower())]

    @classmethod
    def verify(
        cls,
        text: str,
        evidence_ledger: Optional[EvidenceLedger] = None,
        min_validity_threshold: float = 80.0,
    ) -> GroundingVerificationResult:
        """
        Extract citations from text and verify against the evidence ledger.
        Web citations are checked first and excluded from document citation matching.
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

        # Split text into sentence/line segments for claim context
        segments = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]

        # STEP 1: Extract all Web Citations first
        web_matches = cls.WEB_CITATION_REGEX.findall(text)
        web_citations = [f"[web:{h.lower()}]" for h in web_matches]

        # STEP 2: Mask web citations before searching for document citations
        # This guarantees web citations (or text containing "art" / "page") are never parsed as document citations
        text_without_web = cls.WEB_CITATION_REGEX.sub(" ", text)
        doc_matches = cls.CITATION_REGEX.findall(text_without_web)
        doc_citations = [f"[{raw}]" for raw in doc_matches]

        total_citations = len(web_citations) + len(doc_citations)

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

        # --- A. Verify Web Citations ---
        for full_cit in web_citations:
            hash8 = full_cit[5:13].lower()
            matching_segment = next((s for s in segments if full_cit in s), full_cit)

            # Look up in evidence ledger
            matched_items = [
                item for item in evidence_ledger.items.values()
                if item.source_type == "web"
                and (
                    item.citation.lower() == full_cit
                    or (item.content_hash and item.content_hash[:8].lower() == hash8)
                    or item.chunk_id == f"web-{hash8}"
                )
            ]

            if not matched_items:
                unsupported_claims.append(
                    f"Web citation '{full_cit}' in claim '{matching_segment[:120]}...' not found in Evidence Ledger."
                )
                details_list.append({
                    "citation": full_cit,
                    "citation_type": "web",
                    "valid": False,
                    "reason": "Web hash not found in Evidence Ledger",
                })
                continue

            # Verify numeric quantities against fetched text (local to this citation clause)
            meaningful_numbers = cls._extract_local_numbers_for_citation(matching_segment, full_cit)

            combined_evidence_text = " ".join(item.text for item in matched_items)
            missing_numbers = []
            for num in meaningful_numbers:
                clean_num = re.sub(r"[^\d.]", "", num)
                if clean_num and clean_num not in combined_evidence_text:
                    missing_numbers.append(num)

            if missing_numbers:
                unsupported_claims.append(
                    f"Claim '{matching_segment[:120]}...' cited '{full_cit}' but numeric value(s) {missing_numbers} not found in cited web content."
                )
                details_list.append({
                    "citation": full_cit,
                    "citation_type": "web",
                    "valid": False,
                    "missing_numbers": missing_numbers,
                    "reason": "Numeric values ungrounded in cited web content",
                })
            else:
                valid_citations += 1
                details_list.append({
                    "citation": full_cit,
                    "citation_type": "web",
                    "valid": True,
                    "matched_chunks": [item.chunk_id for item in matched_items],
                })

        # --- B. Verify Document Citations ---
        for full_cit_bracket in doc_citations:
            raw_citation = full_cit_bracket[1:-1]
            matching_segment = next((s for s in segments if raw_citation in s), full_cit_bracket)

            # Check existence in evidence ledger
            matched_items = [
                item for item in evidence_ledger.items.values()
                if item.source_type == "document"
                and (
                    raw_citation.lower() in item.citation.lower()
                    or item.citation.lower() in raw_citation.lower()
                )
            ]

            # Match via article number if specified in citation
            art_match = cls.ARTICLE_NUM_REGEX.search(raw_citation)
            if art_match:
                art_num = art_match.group(1)
                matched_items = evidence_ledger.lookup_by_article(art_num)
            elif not matched_items:
                for item in evidence_ledger.items.values():
                    if item.source_type == "document" and item.document_label and item.document_label.lower() in raw_citation.lower():
                        matched_items.append(item)

            if not matched_items:
                unsupported_claims.append(
                    f"Citation '{full_cit_bracket}' in claim '{matching_segment[:120]}...' not found in Evidence Ledger."
                )
                details_list.append({
                    "citation": full_cit_bracket,
                    "citation_type": "document",
                    "valid": False,
                    "reason": "Not found in Evidence Ledger",
                })
                continue

            # Check numeric quantities in claim sentence (local to this citation clause)
            meaningful_numbers = cls._extract_local_numbers_for_citation(matching_segment, full_cit_bracket)

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
                    "citation_type": "document",
                    "valid": False,
                    "missing_numbers": missing_numbers,
                    "reason": "Numeric values ungrounded in cited chunk",
                })
            else:
                valid_citations += 1
                details_list.append({
                    "citation": full_cit_bracket,
                    "citation_type": "document",
                    "valid": True,
                    "matched_chunks": [item.chunk_id for item in matched_items],
                })

        citation_validity = round((valid_citations / total_citations) * 100.0, 1)

        # Classify each claim/sentence by source type
        claim_sources: List[Dict[str, Any]] = []
        doc_claim_count = 0
        web_claim_count = 0
        hybrid_claim_count = 0

        for seg in segments:
            seg_web = [f"[web:{h.lower()}]" for h in cls.WEB_CITATION_REGEX.findall(seg)]
            seg_without_web = cls.WEB_CITATION_REGEX.sub(" ", seg)
            seg_doc = [f"[{raw}]" for raw in cls.CITATION_REGEX.findall(seg_without_web)]

            if seg_web and seg_doc:
                src_type = "hybrid"
                hybrid_claim_count += 1
            elif seg_web:
                src_type = "web"
                web_claim_count += 1
            elif seg_doc:
                src_type = "document"
                doc_claim_count += 1
            else:
                src_type = "uncited"

            if seg_web or seg_doc:
                claim_sources.append({
                    "claim": seg,
                    "citations": seg_web + seg_doc,
                    "source_type": src_type,
                    "web_citations": seg_web,
                    "doc_citations": seg_doc,
                })

        source_distribution = {
            "document": doc_claim_count,
            "web": web_claim_count,
            "hybrid": hybrid_claim_count,
            "total_cited_claims": len(claim_sources),
        }

        # Check document-vs-web conflicts
        conflicts = evidence_ledger.detect_conflicts() if evidence_ledger else []
        conflicts_flagged = True
        if conflicts:
            text_lower = text.lower()
            has_conflict_indication = any(
                term in text_lower for term in ["conflict", "discrepanc", "mismatch", "contradic", "dispute", "differ"]
            )
            topics_mentioned = any(c.topic.lower() in text_lower for c in conflicts)
            conflicts_flagged = has_conflict_indication and topics_mentioned
            if not conflicts_flagged:
                unsupported_claims.append(
                    f"Evidence ledger detected {len(conflicts)} conflict(s) between document and web evidence "
                    f"that are not flagged in text: {[c.description for c in conflicts]}."
                )

        # Check claim support (lexical overlap for EXPLICIT claims, citations on inference, uncited factual metrics)
        claim_support_data = cls.verify_claim_support(text=text, evidence_ledger=evidence_ledger)
        inferred_with_citations = claim_support_data.get("inferred_with_citations", [])
        uncited_factual_claims = claim_support_data.get("uncited_factual_claims", [])
        claim_support_failures = claim_support_data.get("claim_support_failures", [])

        passed = (
            citation_validity >= min_validity_threshold
            and len(unsupported_claims) == 0
            and len(claim_support_failures) == 0
        )

        return GroundingVerificationResult(
            total_citations=total_citations,
            valid_citations=valid_citations,
            citation_validity=citation_validity,
            unsupported_claims=unsupported_claims,
            claim_sources=claim_sources,
            source_distribution=source_distribution,
            conflicts=conflicts,
            conflicts_flagged_in_text=conflicts_flagged,
            inferred_with_citations=inferred_with_citations,
            uncited_factual_claims=uncited_factual_claims,
            claim_support_failures=claim_support_failures,
            passed=passed,
            details={"verifications": details_list, "claim_support": claim_support_data},
        )

    @classmethod
    def verify_claim_support(
        cls,
        text: str,
        evidence_ledger: Optional[EvidenceLedger] = None,
    ) -> Dict[str, Any]:
        """
        Verify claim support across three dimensions:
        1. For each claim labelled EXPLICIT with a citation: verify lexical overlap with the cited chunk.
        2. Flag citations attached to engineering inference or trade-offs.
        3. Flag uncited factual claims in final answer (containing metrics/numbers without citations).
        """
        inferred_with_citations: List[str] = []
        uncited_factual_claims: List[str] = []
        claim_support_failures: List[str] = []

        stopwords = {
            "this", "that", "with", "from", "must", "than", "more", "less", "have", "been",
            "will", "when", "explicit", "source", "document", "article", "there", "these",
            "those", "under", "over", "into", "also", "each", "both", "their", "where"
        }

        # Numeric metric pattern (e.g. 798 kg, 100 kg/h, 2000 mm, 350 kW, $135M)
        metric_regex = re.compile(
            r"\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:kg|g|mm|cm|m|km|kw|mw|kwh|mwh|%|usd|million|billion|h|s|ms|min|rpm|bar)\b",
            re.IGNORECASE,
        )

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        current_section = "general"

        for line in lines:
            line_lower = line.lower()
            if "### explicit" in line_lower or "#### 1." in line_lower or "regulatory requirements" in line_lower:
                current_section = "explicit"
            elif "### engineering interpretation" in line_lower or "#### 2." in line_lower or "interactions & trade-offs" in line_lower or "trade-off" in line_lower:
                current_section = "inference"
            elif "### limitations" in line_lower or "#### 3." in line_lower:
                current_section = "limitations"

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            for s in sentences:
                s_lower = s.lower()
                cits = cls.CITATION_REGEX.findall(s) + [f"web:{h}" for h in cls.WEB_CITATION_REGEX.findall(s)]
                has_cit = len(cits) > 0
                is_explicit = "[explicit]" in s_lower or current_section == "explicit"
                is_inference = (
                    "[inferred]" in s_lower
                    or current_section == "inference"
                    or any(k in s_lower for k in ["trade-off", "tradeoff", "inference", "extrapolation", "suggests", "implies"])
                )

                # 1. Flag citations attached to engineering inference
                if has_cit and is_inference and not is_explicit:
                    inferred_with_citations.append(
                        f"Citation(s) {cits} attached to engineering inference: '{s[:120]}...'"
                    )

                # 2. Check lexical overlap for claims labelled EXPLICIT
                if is_explicit and has_cit and evidence_ledger and evidence_ledger.items:
                    # Clean words from sentence
                    clean_s = re.sub(r"\[[^\]]+\]", " ", s)
                    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", clean_s) if w.lower() not in stopwords]
                    if words:
                        # Find matching ledger chunk
                        matched_chunks = []
                        for cit in cits:
                            cit_clean = re.sub(r"[\[\]]", "", cit).strip()
                            for item in evidence_ledger.items.values():
                                if cit_clean.lower() in item.citation.lower() or item.citation.lower() in cit_clean.lower():
                                    matched_chunks.append(item.text.lower())
                        if matched_chunks:
                            chunk_text = " ".join(matched_chunks)
                            overlap = sum(1 for w in words if w in chunk_text)
                            if overlap == 0:
                                claim_support_failures.append(
                                    f"Claim marked EXPLICIT has zero lexical overlap with cited chunk: '{s[:120]}...'"
                                )

                # 3. Flag uncited factual claims containing specific metrics
                if not has_cit:
                    metrics_found = metric_regex.findall(s)
                    if metrics_found:
                        uncited_factual_claims.append(
                            f"Factual metrics {metrics_found} asserted without citation: '{s[:120]}...'"
                        )

        return {
            "inferred_with_citations": inferred_with_citations,
            "uncited_factual_claims": uncited_factual_claims,
            "claim_support_failures": claim_support_failures,
        }

