from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def _validate_pdf_path(pdf_path: str) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
    return path


def _extract_pages_to_markdown(reader: PdfReader, *, start_page: int, end_page: int) -> list[str]:
    page_blocks: list[str] = []
    for idx, page in enumerate(reader.pages[start_page:end_page], start=start_page + 1):
        extracted = page.extract_text() or ""
        normalized = "\n".join(line.rstrip() for line in extracted.splitlines())
        if normalized.strip():
            page_blocks.append(f"## Trang {idx}\n{normalized.strip()}")
    return page_blocks


def extract_pdf_page_range(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extract a 1-based inclusive page range from PDF into markdown text."""
    path = _validate_pdf_path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise RuntimeError("PDF has no readable pages.")

    normalized_start = max(1, start_page)
    normalized_end = min(total_pages, end_page)
    if normalized_start > normalized_end:
        return ""

    page_blocks = _extract_pages_to_markdown(
        reader,
        start_page=normalized_start - 1,
        end_page=normalized_end,
    )
    markdown = f"# NOI DUNG PDF (Pages {normalized_start}-{normalized_end})\n\n" + "\n\n".join(page_blocks)
    logger.info(
        "Extracted page range %d-%d from %s",
        normalized_start,
        normalized_end,
        pdf_path,
    )
    return markdown


def parse_pdf_first_pages(pdf_path: str, num_pages: int) -> str:
    """Extract only the first N pages from PDF into markdown text."""
    path = _validate_pdf_path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    page_blocks = _extract_pages_to_markdown(reader, start_page=0, end_page=num_pages)

    markdown = f"# NOI DUNG PDF (First {num_pages} pages)\n\n" + "\n\n".join(page_blocks)
    logger.info("Extracted first %d page(s) from %s", min(num_pages, len(reader.pages)), pdf_path)
    return markdown


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parse PDF into markdown-like plain text for downstream extraction.

    This function uses text extraction from each PDF page, then concatenates
    page content into a markdown document. It is lightweight and works for
    digital PDFs that contain selectable text.
    """
    path = _validate_pdf_path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    page_blocks = _extract_pages_to_markdown(reader, start_page=0, end_page=len(reader.pages))

    if not page_blocks:
        raise RuntimeError(
            "Could not extract text from PDF. The file may be scanned and require OCR/VLM parsing."
        )

    markdown = "# NOI DUNG PDF\n\n" + "\n\n".join(page_blocks)
    logger.info("Extracted text from %d page(s) in %s", len(page_blocks), pdf_path)
    return markdown
