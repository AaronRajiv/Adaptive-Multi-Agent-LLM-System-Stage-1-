"""
Document Text Extraction Module with Multi-Page PDF and Structure Parsing Support.
"""

import base64
import io
import re
from typing import List, Tuple, Union

from app.logging_config import get_logger

logger = get_logger("app.knowledge.pdf_extractor")


def _structure_tables_in_text(text: str) -> str:
    """
    Detect tabular row patterns and format them cleanly with markdown pipes,
    or preserve text structure if already structured.
    """
    if not text:
        return ""

    lines = text.split("\n")
    formatted_lines: List[str] = []
    has_table_patterns = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        # Check if line has tab-separated values or multi-column spacing (>= 3 spaces between terms)
        if "\t" in stripped:
            cells = [c.strip() for c in stripped.split("\t") if c.strip()]
            if len(cells) >= 2:
                formatted_lines.append("| " + " | ".join(cells) + " |")
                has_table_patterns = True
                continue
        elif re.search(r"\S\s{3,}\S", stripped):
            cells = [c.strip() for c in re.split(r"\s{3,}", stripped) if c.strip()]
            if len(cells) >= 2:
                formatted_lines.append("| " + " | ".join(cells) + " |")
                has_table_patterns = True
                continue

        formatted_lines.append(line)

    result = "\n".join(formatted_lines).strip()
    if not has_table_patterns and ("table" in text.lower() or "---" in text):
        logger.debug("Tables in text block flattened to text without structural delimiters.")

    return result


def extract_pages_from_document(
    filename: str, content: Union[str, bytes]
) -> List[Tuple[int, str]]:
    """
    Extract readable text per page from attached document content.
    Returns a list of tuples: (page_number, page_text) with 1-indexed page numbers.
    """
    if not content:
        return []

    fname_lower = filename.lower()
    pdf_bytes: bytes | None = None

    if isinstance(content, bytes):
        if fname_lower.endswith(".pdf") or content.startswith(b"%PDF"):
            pdf_bytes = content
        else:
            try:
                raw_text = content.decode("utf-8").strip()
            except UnicodeDecodeError:
                raw_text = content.decode("latin1", errors="ignore").strip()
            return _split_non_pdf_pages(raw_text)
    else:
        if not content.strip():
            return []
        is_pdf = (
            fname_lower.endswith(".pdf")
            or content.startswith("%PDF")
            or "data:application/pdf" in content
            or content.startswith("JVBERi")
        )
        if is_pdf:
            clean_data = content
            if "base64," in content:
                clean_data = content.split("base64,", 1)[1]
            try:
                pdf_bytes = base64.b64decode(clean_data.strip())
            except Exception:
                try:
                    pdf_bytes = content.encode("latin1", errors="ignore")
                except Exception:
                    pdf_bytes = None
        else:
            return _split_non_pdf_pages(content.strip())

    if pdf_bytes and pdf_bytes.startswith(b"%PDF"):
        pages: List[Tuple[int, str]] = []

        # Strategy 1: pdfminer.six page layout iteration
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer

            for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
                page_num = getattr(page_layout, "pageid", len(pages) + 1)
                text_parts = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text_parts.append(element.get_text())
                page_text = _structure_tables_in_text("".join(text_parts).strip())
                if page_text:
                    pages.append((page_num, page_text))
        except Exception as e:
            logger.warning("pdfminer.six page extraction failed for '%s': %s", filename, str(e))
            pages = []

        # Strategy 2: pypdf fallback if pdfminer extracted nothing
        if not pages:
            try:
                import pypdf

                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                for idx, page in enumerate(reader.pages):
                    raw = page.extract_text() or ""
                    page_text = _structure_tables_in_text(raw.strip())
                    if page_text:
                        pages.append((idx + 1, page_text))
            except Exception as e:
                logger.warning("pypdf page extraction failed for '%s': %s", filename, str(e))

        if pages:
            logger.info("Successfully extracted %d page(s) from PDF '%s'", len(pages), filename)
            return pages

    # Fallback for plain text or failed PDF decode
    fallback_text = content.decode("utf-8", errors="replace").strip() if isinstance(content, bytes) else content.strip()
    return _split_non_pdf_pages(fallback_text)


def _split_non_pdf_pages(text: str) -> List[Tuple[int, str]]:
    """Split non-PDF text on standard form-feed characters if present, else return single page 1."""
    if not text:
        return []
    if "\x0c" in text:
        parts = [p.strip() for p in text.split("\x0c") if p.strip()]
        if parts:
            return [(idx + 1, _structure_tables_in_text(p)) for idx, p in enumerate(parts)]
    return [(1, _structure_tables_in_text(text))]


def extract_text_from_document(filename: str, content: Union[str, bytes]) -> str:
    """
    Extract readable plain text from attached document content.
    Backward-compatible wrapper that joins all extracted pages.
    """
    pages = extract_pages_from_document(filename, content)
    if not pages:
        return ""
    return "\n\n".join(page_text for _, page_text in pages).strip()
