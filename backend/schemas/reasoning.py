from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReasoningFinding(BaseModel):
    finding_id: str
    summary: str
    document_ids: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    regions: List[List[int]] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    severity: str = "INFO"


class ReasoningRequest(BaseModel):
    analysis_id: Optional[str] = None
    multimodal_analysis: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None


class ReasoningResult(BaseModel):
    reasoning_id: str
    analysis_id: str
    reasoning_mode: str
    observed_findings: List[ReasoningFinding] = Field(default_factory=list)
    evidence_support: Dict[str, List[str]] = Field(default_factory=dict)
    cross_document_inconsistencies: List[str] = Field(default_factory=list)
    tampering_related_observations: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: str
    limitations: List[str] = Field(default_factory=list)
    traceability: Dict[str, Any] = Field(default_factory=dict)
    generated_at: str
