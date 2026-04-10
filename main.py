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

    output_path = Path("data/final_extraction.md")
    write_markdown_output(validated, output_path)
    logger.info("Wrote markdown output to: %s", output_path)


def _resolve_input_pdf(pdf_arg: str | None) -> Path:
    default_pdf = Path("data/STI_Ver2_FileNVKHCN_6628e4a1c78ae.pdf")
    if not default_pdf.exists():
        default_pdf = Path("data/sample_report.pdf")

    input_pdf = Path(pdf_arg) if pdf_arg else default_pdf
    if not input_pdf.exists():
        input_pdf.parent.mkdir(parents=True, exist_ok=True)
        input_pdf.touch()
    return input_pdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IE pipeline on a PDF file.")
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        default=None,
        help="Path to input PDF. If omitted, use default file in data folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    input_pdf = _resolve_input_pdf(args.pdf_path)

    try:
        run_pipeline(str(input_pdf))
    except Exception as exc:  # pragma: no cover - entrypoint safety
        logging.getLogger("main").exception("Pipeline failed: %s", exc)
