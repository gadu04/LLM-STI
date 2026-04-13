from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = [
	"Tên file PDF",
	"Tên nhiệm vụ",
	"Chủ nhiệm",
	"Tổ chức chủ trì",
	"Sản phẩm KH&CN đã tạo ra",
]


def extract_metadata_value(content: str, label: str) -> str:
	pattern = rf"^-\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*(.+?)\s*$"
	match = re.search(pattern, content, flags=re.MULTILINE | re.IGNORECASE)
	return match.group(1).strip() if match else ""


def parse_products(content: str) -> str:
	section_match = re.search(
		r"^##\s*Sản phẩm KH&CN đã tạo ra\s*$([\s\S]*?)(?:^##\s+|\Z)",
		content,
		flags=re.MULTILINE,
	)
	if not section_match:
		return ""

	section = section_match.group(1)
	if "không tìm thấy sản phẩm kh&cn" in section.lower():
		return ""

	# In current output format, each product name is a level-4 markdown heading.
	product_titles = re.findall(r"^####\s+(.+?)\s*$", section, flags=re.MULTILINE)
	cleaned = []
	for title in product_titles:
		value = re.sub(r"^\d+\.\s*", "", title).strip()
		if value:
			cleaned.append(value)

	return " | ".join(cleaned)


def pdf_name_from_markdown(md_path: Path) -> str:
	stem = md_path.stem
	prefix = "final_extraction_"
	if stem.startswith(prefix):
		return f"{stem[len(prefix):]}.pdf"
	return f"{stem}.pdf"


def build_rows(data_dir: Path, pattern: str) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []

	for md_file in sorted(data_dir.glob(pattern)):
		content = md_file.read_text(encoding="utf-8")
		row = {
			"Tên file PDF": pdf_name_from_markdown(md_file),
			"Tên nhiệm vụ": extract_metadata_value(content, "Tên nhiệm vụ"),
			"Chủ nhiệm": extract_metadata_value(content, "Chủ nhiệm"),
			"Tổ chức chủ trì": extract_metadata_value(content, "Tổ chức chủ trì"),
			"Sản phẩm KH&CN đã tạo ra": parse_products(content),
		}
		rows.append(row)

	return rows


def export_csv(rows: list[dict[str, str]], output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8-sig", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export thông tin từ các file final_extraction_bcth_*.md ra CSV"
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("data"),
		help="Thư mục chứa các file markdown đã bóc tách",
	)
	parser.add_argument(
		"--pattern",
		default="final_extraction_bcth_*.md",
		help="Mẫu tên file markdown cần xử lý",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data") / "export.csv",
		help="Đường dẫn file CSV đầu ra",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	rows = build_rows(args.data_dir, args.pattern)
	export_csv(rows, args.output)
	print(f"Da xuat {len(rows)} dong vao: {args.output}")


if __name__ == "__main__":
	main()
