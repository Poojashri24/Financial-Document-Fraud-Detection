from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TamperingFinding(BaseModel):
    detector: str
    finding: str
    severity: str
    region: Optional[List[int]] = None
    explanation: str


class TamperingAnalysis(BaseModel):
    document_id: str
    analysis_id: str
    detector_type: str
    tampering_detected: bool
    risk_score: int = 0
    findings: List[TamperingFinding] = Field(default_factory=list)
    suspicious_regions: List[List[int]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
