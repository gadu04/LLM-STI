from __future__ import annotations

from datetime import datetime
from pathlib import Path

from schemas import ScientificReportSchema


def _md_escape(value: str) -> str:
    """Escape markdown table separators in free text values."""
    return value.replace("|", "\\|").strip()


def write_markdown_output(result: ScientificReportSchema, output_path: Path) -> None:
    """Write final validated extraction to a markdown summary file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Kết quả bóc tách thông tin báo cáo",
        "",
        f"- Thời gian tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Tên nhiệm vụ: {_md_escape(result.ten_nhiem_vu)}",
        f"- Chủ nhiệm: {_md_escape(result.chu_nhiem)}",
        f"- Tổ chức chủ trì: {_md_escape(result.to_chuc_chu_tri or 'Chưa xác định')}",
        "",
        "## Sản phẩm KH&CN đã tạo ra",
    ]

    if result.san_pham_khcn:
        lines.extend(["", "### Danh sách chi tiết"])
        for idx, item in enumerate(result.san_pham_khcn, start=1):
            lines.extend(
                [
                    "",
                    f"#### {idx}. {_md_escape(item.ten_san_pham)}",
                    f"- Loại sản phẩm: {_md_escape(item.loai_san_pham)}",
                    f"- Số lượng: {_md_escape(item.so_luong)}",
                    f"- Mô tả chi tiết: {_md_escape(item.mo_ta_chi_tiet or 'Chưa có')}",
                    f"- Kết quả chính: {_md_escape(item.ket_qua_chinh or 'Chưa có')}",
                    f"- Ghi chú: {_md_escape(item.ghi_chu or 'Chưa có')}",
                ]
            )
    else:
        lines.extend(["", "_Không tìm thấy sản phẩm KH&CN trong tài liệu._"])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
