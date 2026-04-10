from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from output_writer import write_markdown_output
from product_tracker import extract_information_v2

logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = Path("data/uploads")
FINAL_OUTPUT = Path("data/final_extraction.md")


def _save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_summary(metadata_box, metadata: dict[str, Any]) -> None:
    metadata_box.markdown(
        f"""
        **Tên nhiệm vụ:** {metadata.get('ten_nhiem_vu', 'Chưa có')}  
        **Chủ nhiệm:** {metadata.get('chu_nhiem', 'Chưa có')}  
        **Tổ chức chủ trì:** {metadata.get('to_chuc_chu_tri') or 'Chưa xác định'}
        """
    )


def _render_products_table(products_box, products: list[dict[str, Any]]) -> None:
    if not products:
        products_box.info("Chưa có sản phẩm nào được xử lý.")
        return

    rows = []
    for idx, item in enumerate(products, start=1):
        rows.append(
            {
                "STT": idx,
                "Tên sản phẩm": item.get("ten_san_pham", ""),
                "Loại": item.get("loai_san_pham", ""),
                "Số lượng": item.get("so_luong", ""),
                "Mô tả": item.get("mo_ta_chi_tiet") or "",
                "Kết quả": item.get("ket_qua_chinh") or "",
                "Ghi chú": item.get("ghi_chu") or "",
            }
        )
    df = pd.DataFrame(rows)
    styled = df.style.set_properties(
        **{
            "white-space": "normal",
            "text-align": "left",
        }
    )
    products_box.dataframe(styled, use_container_width=True, hide_index=True)


st.set_page_config(page_title="Trích xuất tài liệu STI", page_icon="📄", layout="wide")
st.title("Trích xuất thông tin PDF")
st.caption("Tải PDF lên, theo dõi tiến trình xử lý và xem kết quả ngay trên giao diện")

uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])

run_clicked = st.button("Bắt đầu", type="primary", disabled=uploaded_file is None)

if run_clicked and uploaded_file is not None:
    pdf_path = _save_uploaded_file(uploaded_file)

    st.write(f"Đã tải lên: `{pdf_path}`")

    progress_bar = st.progress(0)
    status_box = st.empty()
    metadata_box = st.empty()
    products_box = st.empty()
    top_hint_box = st.empty()

    def progress_callback(percent: int, message: str) -> None:
        progress_bar.progress(max(0, min(100, percent)))
        status_box.info(message)

    def event_callback(event_name: str, payload: dict[str, Any]) -> None:
        if event_name == "phase1_complete":
            _render_summary(metadata_box, payload)
            top_hint_box.info("Đã trích xuất xong thông tin tổng quan")

        elif event_name == "phase2_complete":
            top_hint_box.info(f"Đã nhận diện {len(payload.get('products', []))} sản phẩm, đang xử lý chi tiết...")

        elif event_name == "product_complete":
            product = payload.get("product", {})
            current_products = st.session_state.get("stream_products", [])
            current_products = list(current_products)
            current_products.append(product)
            st.session_state["stream_products"] = current_products
            _render_products_table(products_box, current_products)

        elif event_name == "complete":
            top_hint_box.success("Hoàn tất xử lý")

    try:
        with st.spinner("Đang xử lý..."):
            st.session_state["stream_products"] = []
            result = extract_information_v2(
                str(pdf_path),
                progress_callback=progress_callback,
                event_callback=event_callback,
            )

        write_markdown_output(result, FINAL_OUTPUT)

        st.success("Xử lý hoàn tất")
        _render_summary(metadata_box, result.model_dump())
        _render_products_table(products_box, [item.model_dump() for item in result.san_pham_khcn])

        markdown_content = FINAL_OUTPUT.read_text(encoding="utf-8")
        st.download_button(
            label="Tải final_extraction.md",
            data=markdown_content,
            file_name="final_extraction.md",
            mime="text/markdown",
        )

        with st.expander("Xem JSON", expanded=False):
            st.code(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), language="json")

    except Exception as exc:
        st.error(f"Pipeline lỗi: {exc}")
