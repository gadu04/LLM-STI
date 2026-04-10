from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SanPhamKHCN(BaseModel):
    """Schema for each KHCN product item."""

    model_config = ConfigDict(strict=True, extra="forbid")

    ten_san_pham: str = Field(..., min_length=1)
    loai_san_pham: str = Field(..., min_length=1)
    so_luong: str = Field(..., min_length=1)
    mo_ta_chi_tiet: Optional[str] = None
    ket_qua_chinh: Optional[str] = None
    ghi_chu: Optional[str] = None


class ScientificReportSchema(BaseModel):
    """Top-level schema for Vietnamese scientific acceptance reports."""

    model_config = ConfigDict(strict=True, extra="forbid")

    ten_nhiem_vu: str = Field(..., min_length=1)
    chu_nhiem: str = Field(..., min_length=1)
    to_chuc_chu_tri: Optional[str] = None
    san_pham_khcn: List[SanPhamKHCN] = Field(default_factory=list)
