"""
Document Text Extraction Module with PDF and Base64 Parsing Support.
"""

import base64
import io
import re


def extract_text_from_document(filename: str, content: str) -> str:
    """
    Extract readable plain text from attached document content.
    Handles plain text, markdown, JSON, and PDF files (Base64 or binary strings).
    """
    if not content or not content.strip():
        return ""

    fname_lower = filename.lower()
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

        pdf_bytes = None
        try:
            pdf_bytes = base64.b64decode(clean_data.strip())
        except Exception:
            try:
                pdf_bytes = content.encode("latin1", errors="ignore")
            except Exception:
                pdf_bytes = None

        if pdf_bytes and pdf_bytes.startswith(b"%PDF"):
            extracted_text = ""
            # Try 1: pdfminer.six (robust stream extractor)
            try:
                from pdfminer.high_level import extract_text
                extracted_text = extract_text(io.BytesIO(pdf_bytes))
            except Exception:
                extracted_text = ""

            # Try 2: pypdf if pdfminer returned empty
            if not extracted_text or not extracted_text.strip():
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
                    extracted_text = "\n\n".join(pages)
                except Exception:
                    pass

            if extracted_text and extracted_text.strip():
                return extracted_text.strip()

    # Return plain text or cleaned text
    return content.strip()
