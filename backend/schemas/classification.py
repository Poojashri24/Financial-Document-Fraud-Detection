from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    document_type: str
    confidence: float
    classification_method: str = "heuristic_ocr_rules"
    evidence: List[str] = Field(default_factory=list)


class FieldExtractionResult(BaseModel):
    document_type: str
    fields: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    extraction_method: str = "heuristic_ocr_rules"
