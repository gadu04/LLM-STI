from __future__ import annotations

import argparse
import logging
from pathlib import Path

from output_writer import write_markdown_output
from product_tracker import extract_information_v2


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(pdf_path: str) -> None:
    logger = logging.getLogger("main")
    logger.info("Starting IE pipeline for: %s", pdf_path)

    validated = extract_information_v2(pdf_path)
    logger.info("Extraction result: %s", validated.model_dump_json(indent=2))

    pdf_stem = Path(pdf_path).stem
    output_path = Path("data") / f"final_extraction_{pdf_stem}.md"
    write_markdown_output(validated, output_path)
    logger.info("Wrote markdown output to: %s", output_path)


def _resolve_input_pdfs(pdf_arg: str | None) -> list[Path]:
    if pdf_arg:
        input_pdf = Path(pdf_arg)
        if not input_pdf.exists() or input_pdf.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Input PDF not found or invalid: {input_pdf}")
        return [input_pdf]

    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("Data folder not found: data")

    # Batch mode: process every PDF in data/ (recursive) so Streamlit is unnecessary.
    pdf_files = sorted(p for p in data_dir.rglob("*.pdf") if p.is_file())
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data folder.")
    return pdf_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IE pipeline on one PDF or all PDFs in data folder.")
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        default=None,
        help="Path to a single input PDF. If omitted, process all PDFs in data/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    input_pdfs = _resolve_input_pdfs(args.pdf_path)

    try:
        for pdf_path in input_pdfs:
            run_pipeline(str(pdf_path))
    except Exception as exc:  # pragma: no cover - entrypoint safety
        logging.getLogger("main").exception("Pipeline failed: %s", exc)
