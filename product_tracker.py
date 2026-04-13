"""3-phase extraction: Phase 1 metadata, Phase 2 product names, Phase 3 details."""
from __future__ import annotations

import logging
import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from pydantic import BaseModel

from config import get_llm_config
from llm_client import create_llm_client
from schemas import SanPhamKHCN, ScientificReportSchema
from vlm_parser import parse_pdf_first_pages, parse_pdf_to_markdown

logger = logging.getLogger(__name__)

NUM_WORKERS = 5
PHASE1_PAGES = 4
PHASE2_MIN_SECTION_LEN = 50
PHASE2_MAX_TOKENS = 1200
PHASE3_MAX_TOKENS = 700
SEMANTIC_QUERY = "Các kết quả nghiên cứu hoặc sản phẩm khoa học và công nghệ của đề tài này là gì?"
SEMANTIC_TOP_K = 5
MOCK_EMBED_DIM = 256
PARAGRAPH_OVERLAP_CHARS = 120
ProgressCallback = Callable[[int, str], None]
EventCallback = Callable[[str, dict[str, Any]], None]

_BGE_M3_MODEL = None


class ProductCandidate(BaseModel):
    ten_san_pham: str
    dang_san_pham: Optional[str] = None


class MetadataSchema(BaseModel):
    ten_nhiem_vu: str
    chu_nhiem: str
    to_chuc_chu_tri: Optional[str] = None


class ProductListSchema(BaseModel):
    products: list[ProductCandidate]


class DetailSchema(BaseModel):
    loai_san_pham: Optional[str] = None
    so_luong: Optional[str] = None
    mo_ta_chi_tiet: Optional[str] = None
    ket_qua_chinh: Optional[str] = None
    ghi_chu: Optional[str] = None


def _notify(progress_callback: ProgressCallback | None, percent: int, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(percent, message)
    except Exception as exc:  # pragma: no cover - callback safety
        logger.debug("Progress callback error: %s", exc)


def _emit(event_callback: EventCallback | None, event_name: str, payload: dict[str, Any]) -> None:
    if event_callback is None:
        return
    try:
        event_callback(event_name, payload)
    except Exception as exc:  # pragma: no cover - callback safety
        logger.debug("Event callback error: %s", exc)


def _build_phase1_prompts(first_pages: str) -> tuple[str, str]:
    system_prompt = (
        "Bạn là hệ thống trích xuất metadata từ biên bản nghiệm thu. "
        "Giữ tiếng Việt có dấu. Chỉ trả JSON hợp lệ."
    )
    user_prompt = (
        "Trích xuất 3 trường:\n"
        "- ten_nhiem_vu\n"
        "- chu_nhiem\n"
        "- to_chuc_chu_tri\n\n"
        f"Từ trang đầu:\n{first_pages}"
    )
    return system_prompt, user_prompt


def _build_phase2_prompts(section_text: str) -> tuple[str, str]:
    system_prompt = (
        "Bạn là hệ thống trích xuất sản phẩm KH&CN từ tài liệu tiếng Việt. "
        "Nhiệm vụ: lấy danh sách sản phẩm từ section KH&CN và gán loại sản phẩm (Dạng I/II/III nếu có). "
        "Nếu tài liệu không ghi rõ Dạng, suy luận theo nội dung: vật chất/chế phẩm ~ Dạng I; "
        "quy trình/công nghệ/phương pháp ~ Dạng II; bài báo/sáng chế/công bố ~ Dạng III. "
        "Trả JSON hợp lệ, không thêm giải thích."
    )
    user_prompt = (
        "Từ section KH&CN bên dưới, trích xuất danh sách sản phẩm.\n"
        "Mỗi sản phẩm gồm:\n"
        "- ten_san_pham: tên sạch từ cột tên sản phẩm\n"
        "- dang_san_pham: Dạng I / Dạng II / Dạng III (nếu nhận diện được, nếu không thì null)\n\n"
        "Yêu cầu làm sạch tên:\n"
        "- Bỏ STT, đơn vị đo, cột trạng thái, dữ liệu số lượng\n"
        "- Giữ nguyên tiếng Việt, không thêm bớt nghĩa\n\n"
        'Trả: {"products": [{"ten_san_pham": "...", "dang_san_pham": "Dạng II"}, ...]}\n\n'
        f"Section KH&CN:\n{section_text}"
    )
    return system_prompt, user_prompt


def _build_phase3_prompts(product_name: str, full_text: str, type_hint: str | None) -> tuple[str, str]:
    system_prompt = (
        "Bạn là chuyên gia trích xuất dữ liệu từ bảng KH&CN. "
        "Tìm sản phẩm trong tài liệu và trích xuất CHÍNH XÁC dữ liệu từ bảng. "
        "Quy tắc: "
        "1. so_luong PHẢI là giá trị từ cột 'Thực tế đạt được' (ví dụ: '2-3 chủng', '5 kg...', '01 bài báo'). Không được dùng giá trị của cột khác. "
        "2. loai_san_pham: Xác định đúng (Dạng I = vật chất, Dạng II = quy trình, Dạng III = báo cáo/sáng chế). "
        "3. mo_ta_chi_tiet: Từ cột 'Chỉ tiêu chất lượng' hoặc 'Yêu cầu'. "
        "4. Trả JSON chính xác, không thêm giải thích, giữ tiếng Việt."
    )
    user_prompt = (
        f"Sản phẩm cần tìm: '{product_name}'\n\n"
        f"Gợi ý loại từ bước trước: {type_hint or 'không có'}\n\n"
        "Yêu cầu:\n"
        "1. Tìm dòng bảng chính xác chứa tên sản phẩm này\n"
        "2. Trích xuất từ cột 'Thực tế đạt được' → so_luong (BẮT BUỘC)\n"
        "3. Xác định Dạng (I/II/III) → loai_san_pham; ưu tiên khớp gợi ý loại nếu hợp lý\n"
        "4. Mô tả chỉ tiêu → mo_ta_chi_tiet\n"
        "5. Kết quả đạt được → ket_qua_chinh\n\n"
        "Ví dụ đúng:\n"
        "- Sản phẩm 'Tập hợp chủng vi sinh vật...': so_luong='3 chủng vi khuẩn, 2 chủng nấm men'\n"
        "- Sản phẩm 'Chế phẩm sinh học': so_luong='5 kg chế phẩm ở dạng viên và dạng tấm 10 m2'\n"
        "- Sản phẩm 'Bài báo khoa học': so_luong='01 bài báo'\n\n"
        f"Tài liệu đầy đủ:\n{full_text}"
    )
    return system_prompt, user_prompt


def _call_llm_json(model: type[BaseModel], *, system_prompt: str, user_prompt: str, max_tokens: int) -> BaseModel | None:
    llm_config = get_llm_config()
    client = create_llm_client()
    response = client.chat.completions.create(
        model=llm_config.model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__.lower(),
                "strict": True,
                "schema": model.model_json_schema(),
            },
        },
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        return None
    return model.model_validate_json(content)


def _canonical_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

    normalized_chars: list[str] = []
    for ch in no_accents:
        if ch.isalnum() or ch.isspace():
            normalized_chars.append(ch)
        else:
            normalized_chars.append(" ")
    return " ".join("".join(normalized_chars).split())


def _token_set(value: str) -> set[str]:
    stop = {"va", "cua", "cho", "voi", "tu", "trong", "da", "duoc", "co", "la"}
    return {t for t in _canonical_text(value).split() if len(t) > 2 and t not in stop}


def _normalize_type_by_name(name: str, hint: str | None) -> str | None:
    text = _canonical_text(name)
    if any(k in text for k in ["bai bao", "sang che", "bang sang che", "cong bo"]):
        return "Dạng III"
    if any(k in text for k in ["quy trinh", "cong nghe", "phuong phap"]):
        return "Dạng II"
    if any(k in text for k in ["che pham", "tap hop", "chung vi sinh", "vat lieu", "thiet bi"]):
        return "Dạng I"
    return hint


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Chunk by paragraphs and preserve overlap for better semantic recall."""
    if not text:
        return []

    safe_chunk_size = max(100, chunk_size)
    safe_overlap = max(0, min(overlap, safe_chunk_size - 1))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= safe_chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap_tail = current[-safe_overlap:] if safe_overlap else ""
            current = f"{overlap_tail}\n\n{paragraph}".strip() if overlap_tail else paragraph
        else:
            chunks.append(paragraph[:safe_chunk_size])
            current = paragraph[max(0, safe_chunk_size - safe_overlap):]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _mock_embed_texts(texts: list[str], dim: int = MOCK_EMBED_DIM) -> list[list[float]]:
    """Deterministic lightweight embedding fallback so pipeline still runs without external model."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in _token_set(text):
            idx = hash(token) % dim
            vec[idx] += 1.0
        vectors.append(vec)
    return vectors


def _get_bge_m3_model():
    global _BGE_M3_MODEL
    if _BGE_M3_MODEL is None:
        from FlagEmbedding import BGEM3FlagModel

        _BGE_M3_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _BGE_M3_MODEL


def embed_texts(list_of_strings: list[str]) -> list[list[float]]:
    """Embed texts with BGEM3; fall back to mock vectors if model is unavailable."""
    if not list_of_strings:
        return []

    try:
        model = _get_bge_m3_model()
        encoded = model.encode(
            list_of_strings,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = encoded.get("dense_vecs")
        if dense_vectors is None:
            raise RuntimeError("BGEM3 did not return dense_vecs")
        return [list(map(float, row)) for row in dense_vectors]
    except Exception as exc:
        logger.warning("BGEM3 unavailable, using mock embeddings: %s", exc)
        return _mock_embed_texts(list_of_strings)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity for two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_retrieve(chunks: list[str], query: str, top_k: int = 5) -> list[str]:
    """Return top-k chunks most semantically similar to the query."""
    if not chunks or not query.strip():
        return []

    embeddings = embed_texts(chunks + [query])
    if not embeddings or len(embeddings) != len(chunks) + 1:
        return []

    query_vec = embeddings[-1]
    ranked = []
    for idx, chunk_vec in enumerate(embeddings[:-1]):
        score = cosine_similarity(chunk_vec, query_vec)
        ranked.append((score, idx))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected_idx = [idx for _, idx in ranked[: max(1, top_k)]]
    selected_idx.sort()
    return [chunks[idx] for idx in selected_idx]


def _phase2a_build_semantic_context(full_text: str) -> str:
    """Build Phase 2 context via semantic retrieval instead of brittle heading regex.

    Regex heading detection often fails because products can appear across many sections
    (mô hình, giải pháp, đóng góp, kết quả), not only one fixed title.
    """
    chunks = chunk_text(full_text, chunk_size=800, overlap=PARAGRAPH_OVERLAP_CHARS)
    if not chunks:
        return ""

    top_chunks = semantic_retrieve(chunks, SEMANTIC_QUERY, top_k=SEMANTIC_TOP_K)
    if top_chunks:
        return "\n\n".join(top_chunks).strip()

    # Semantic fallback keeps pipeline running even when embedding backend is unavailable.
    return "\n\n".join(chunks[:SEMANTIC_TOP_K]).strip()


def _dedupe_candidates(items: list[ProductCandidate]) -> list[ProductCandidate]:
    """Remove near-duplicate names and keep more informative candidate."""
    result: list[ProductCandidate] = []
    for item in items:
        cur = item.ten_san_pham.strip()
        cur_norm = _canonical_text(cur)
        if not cur_norm:
            continue

        replaced = False
        skip = False
        for i, kept in enumerate(result):
            kept_norm = _canonical_text(kept.ten_san_pham)
            if cur_norm in kept_norm:
                skip = True
                break
            if kept_norm in cur_norm:
                result[i] = item
                replaced = True
                break

            cur_tokens = _token_set(cur)
            kept_tokens = _token_set(kept.ten_san_pham)
            cur_type = item.dang_san_pham
            kept_type = kept.dang_san_pham
            same_or_unknown_type = (not cur_type or not kept_type or cur_type == kept_type)
            if cur_tokens and kept_tokens:
                overlap = len(cur_tokens & kept_tokens) / min(len(cur_tokens), len(kept_tokens))
                if same_or_unknown_type and overlap >= 0.8:
                    # Giữ tên giàu thông tin hơn (nhiều token hơn)
                    if len(cur_tokens) > len(kept_tokens):
                        result[i] = item
                    skip = True
                    break

        if not skip and not replaced:
            result.append(item)
    return result


def _phase1_extract_metadata(pdf_path: str, num_pages: int = 4) -> tuple[str, str, str | None]:
    """Phase 1: Extract metadata from first N pages."""
    logger.info("Phase 1: Extracting metadata from first %d pages", num_pages)
    first_pages = parse_pdf_first_pages(pdf_path, num_pages)
    if not first_pages.strip():
        raise ValueError("Could not extract first pages.")

    system_prompt, user_prompt = _build_phase1_prompts(first_pages)
    data = _call_llm_json(
        MetadataSchema,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=500,
    )
    if data is None:
        raise RuntimeError("LLM returned empty metadata.")
    logger.info("Phase 1 complete: %s / %s", data.ten_nhiem_vu[:50], data.chu_nhiem)
    return data.ten_nhiem_vu, data.chu_nhiem, data.to_chuc_chu_tri


def _phase2b_extract_products(khcn_section: str) -> list[ProductCandidate]:
    """
    Phase 2b: Extract product list and type hints from KH&CN section.

    Không dùng regex theo Dạng I/II/III; để LLM tự nhận diện theo ngữ cảnh bảng.
    """
    if not khcn_section or len(khcn_section) < PHASE2_MIN_SECTION_LEN:
        logger.warning("Phase 2b: Section too short")
        return []

    llm_config = get_llm_config()
    safe_input = khcn_section[: llm_config.input_max_chars]

    system_prompt, user_prompt = _build_phase2_prompts(safe_input)

    try:
        data = _call_llm_json(
            ProductListSchema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=PHASE2_MAX_TOKENS,
        )
        if data is None:
            return []

        cleaned: list[ProductCandidate] = []
        for item in data.products:
            clean_name = item.ten_san_pham.strip().lstrip(":").strip('"').strip("'").strip()
            if clean_name:
                normalized_type = _normalize_type_by_name(clean_name, item.dang_san_pham)
                cleaned.append(
                    ProductCandidate(
                        ten_san_pham=clean_name,
                        dang_san_pham=normalized_type.strip() if normalized_type else None,
                    )
                )
        deduped = _dedupe_candidates(cleaned)
        logger.info("Phase 2b: Extracted %d products (with type hints)", len(deduped))
        return deduped
    except Exception as exc:
        logger.warning("Phase 2b failed: %s", exc)
        return []


def _phase3_track_product(product_name: str, full_text: str, type_hint: str | None = None) -> SanPhamKHCN:
    """
    Phase 3: Extract detailed product info from table row (worker task).
    
    Maps table columns:
    - so_luong: Từ "Thực tế đạt được" 
    - loai_san_pham: Dạng I/II/III
    - mo_ta_chi_tiet: Từ "Chỉ tiêu" hoặc "Yêu cầu"
    - ket_qua_chinh: Kết quả đạt được
    - ghi_chu: Ghi chú nếu có
    """
    llm_config = get_llm_config()
    safe_input = full_text[: llm_config.input_max_chars]
    system_prompt, user_prompt = _build_phase3_prompts(product_name, safe_input, type_hint)

    try:
        detail = _call_llm_json(
            DetailSchema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=PHASE3_MAX_TOKENS,
        )
        if detail is not None:
            normalized_type = _normalize_type_by_name(product_name, detail.loai_san_pham or type_hint)
            return SanPhamKHCN(
                ten_san_pham=product_name,
                loai_san_pham=normalized_type or "Chưa xác định",
                so_luong=detail.so_luong or "Chưa xác định",
                mo_ta_chi_tiet=detail.mo_ta_chi_tiet,
                ket_qua_chinh=detail.ket_qua_chinh,
                ghi_chu=detail.ghi_chu,
            )
    except Exception as exc:
        logger.warning("Detail extraction failed for %s: %s", product_name[:50], exc)

    return SanPhamKHCN(
        ten_san_pham=product_name,
        loai_san_pham=_normalize_type_by_name(product_name, type_hint) or "Chưa xác định",
        so_luong="Chưa xác định",
    )


def extract_information_v2(
    pdf_path: str,
    progress_callback: ProgressCallback | None = None,
    event_callback: EventCallback | None = None,
) -> ScientificReportSchema:
    """Execute 3-phase extraction with parallel product detail extraction."""
    logger.info("=== Starting 3-Phase Extraction (NUM_WORKERS=%d) ===", NUM_WORKERS)
    _notify(progress_callback, 2, "Khởi tạo pipeline bóc tách...")
    _notify(progress_callback, 8, "Phase 1: Đọc metadata từ các trang đầu...")

    ten_nhiem_vu, chu_nhiem, to_chuc_chu_tri = _phase1_extract_metadata(pdf_path, num_pages=PHASE1_PAGES)
    _emit(
        event_callback,
        "phase1_complete",
        {
            "ten_nhiem_vu": ten_nhiem_vu,
            "chu_nhiem": chu_nhiem,
            "to_chuc_chu_tri": to_chuc_chu_tri,
        },
    )
    _notify(progress_callback, 25, "Phase 1 hoàn tất")

    _notify(progress_callback, 32, "Phase 2: Đọc nội dung tài liệu...")
    # Giữ đúng yêu cầu: chỉ OCR/đọc trang đầu cho metadata (Phase 1),
    # còn sản phẩm KH&CN thì extract full text (không OCR full) + semantic retrieval.
    full_text = parse_pdf_to_markdown(pdf_path)
    _notify(progress_callback, 42, "Phase 2a: Semantic retrieve nội dung sản phẩm KH&CN...")
    khcn_section = _phase2a_build_semantic_context(full_text)

    _notify(progress_callback, 52, "Phase 2b: Nhận diện danh sách sản phẩm...")
    products = _phase2b_extract_products(khcn_section)
    _notify(progress_callback, 62, f"Đã nhận diện {len(products)} sản phẩm")
    _emit(
        event_callback,
        "phase2_complete",
        {
            "products": [
                {"ten_san_pham": item.ten_san_pham, "dang_san_pham": item.dang_san_pham}
                for item in products
            ]
        },
    )

    if not products:
        logger.warning("No product names found")
        _notify(progress_callback, 100, "Hoàn tất (không tìm thấy sản phẩm KH&CN)")
        return ScientificReportSchema(
            ten_nhiem_vu=ten_nhiem_vu,
            chu_nhiem=chu_nhiem,
            to_chuc_chu_tri=to_chuc_chu_tri,
            san_pham_khcn=[],
        )

    logger.info("Phase 3: Tracking %d products with %d workers (parallel)", len(products), NUM_WORKERS)
    _notify(progress_callback, 68, "Phase 3: Bóc tách chi tiết từng sản phẩm...")
    
    # Xử lý chi tiết sản phẩm song song
    san_pham_khcn = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit tất cả task vào executor
        future_to_index = {
            executor.submit(_phase3_track_product, item.ten_san_pham, full_text, item.dang_san_pham): i
            for i, item in enumerate(products)
        }
        
        # Nhận kết quả khi hoàn thành
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                product = future.result()
                san_pham_khcn.append((idx, product))
                completed_count += 1
                ratio = completed_count / max(1, len(products))
                percent = 68 + int(ratio * 30)
                _notify(
                    progress_callback,
                    min(percent, 98),
                    f"Phase 3: Hoàn thành {completed_count}/{len(products)} sản phẩm",
                )
                _emit(
                    event_callback,
                    "product_complete",
                    {
                        "index": idx + 1,
                        "total": len(products),
                        "completed": completed_count,
                        "product": product.model_dump(),
                    },
                )
                logger.info("✓ Product %d/%d completed: %s", idx + 1, len(products), product.ten_san_pham[:50])
            except Exception as exc:
                logger.error("Product %d failed: %s", idx + 1, exc)
    
    # Sắp xếp lại theo thứ tự ban đầu
    san_pham_khcn.sort(key=lambda x: x[0])
    san_pham_khcn = [p for _, p in san_pham_khcn]

    logger.info("=== 3-Phase Extraction Complete: %d products ===", len(san_pham_khcn))
    _notify(progress_callback, 100, "Hoàn tất bóc tách")

    result = ScientificReportSchema(
        ten_nhiem_vu=ten_nhiem_vu,
        chu_nhiem=chu_nhiem,
        to_chuc_chu_tri=to_chuc_chu_tri,
        san_pham_khcn=san_pham_khcn,
    )
    _emit(event_callback, "complete", {"result": result.model_dump()})
    return result
