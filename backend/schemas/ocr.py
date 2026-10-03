from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    document_id: str
    page_index: Optional[int] = None


class OCRResult(BaseModel):
    document_id: str
    status: str
    text: str = ""
    filename: Optional[str] = None
    pages: int = 0
    error: Optional[str] = None
    confidence: Optional[float] = None
    page_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
