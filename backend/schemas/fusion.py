from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MultimodalEvidenceItem(BaseModel):
    source: str
    document_ids: List[str] = Field(default_factory=list)
    summary: str
    severity: str = "INFO"
    details: Dict[str, Any] = Field(default_factory=dict)
    region: Optional[List[int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalDocumentInput(BaseModel):
    document_id: str
    document_type: str = "Unknown"
    ocr: Optional[Dict[str, Any]] = None
    structured_fields: Dict[str, Any] = Field(default_factory=dict)
    cross_document: Optional[Dict[str, Any]] = None
    tampering: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalAnalysisRequest(BaseModel):
    documents: List[MultimodalDocumentInput] = Field(default_factory=list)


class MultimodalModalitySummary(BaseModel):
    documents: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    count: int = 0


class MultimodalFusionSummary(BaseModel):
    documents_analyzed: int
    evidence_count: int
    inconsistencies_detected: int
    insufficient_data_count: int
    modality_count: int
