from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RiskEvidenceEntry(BaseModel):
    source: str
    summary: str
    severity: str = "INFO"
    document_ids: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    regions: List[List[int]] = Field(default_factory=list)
    traceability: Dict[str, Any] = Field(default_factory=dict)


class RiskAssessmentRequest(BaseModel):
    reasoning_id: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None


class RiskAssessment(BaseModel):
    analysis_id: str
    reasoning_id: str
    risk_score: int
    risk_level: str
    risk_explanation: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    contributing_evidence: List[RiskEvidenceEntry] = Field(default_factory=list)
    supporting_documents: List[str] = Field(default_factory=list)
    inconsistencies: List[str] = Field(default_factory=list)
    tampering_observations: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_summary: str
    limitations: List[str] = Field(default_factory=list)
    methodology: Dict[str, Any] = Field(default_factory=dict)
    generated_at: str
