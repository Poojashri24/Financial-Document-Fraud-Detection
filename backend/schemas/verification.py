from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationDocumentInput(BaseModel):
    document_id: str
    document_type: str = "Unknown"
    fields: Dict[str, Any] = Field(default_factory=dict)


class VerificationRuleResult(BaseModel):
    rule_id: str
    rule_name: str
    status: str
    documents: List[str] = Field(default_factory=list)
    fields_compared: List[str] = Field(default_factory=list)
    values: Dict[str, Any] = Field(default_factory=dict)
    explanation: str
    evidence: List[str] = Field(default_factory=list)


class VerificationSummary(BaseModel):
    number_of_rules_checked: int
    passed_rules: int
    failed_rules: int
    warnings: int
    insufficient_data_checks: int
    overall_inconsistency_count: int


class CrossDocumentVerificationAnalysis(BaseModel):
    analysis_id: str
    document_count: int
    results: List[VerificationRuleResult] = Field(default_factory=list)
    summary: VerificationSummary


class CrossDocumentVerificationRequest(BaseModel):
    document_ids: List[str] = Field(default_factory=list)
    documents: List[VerificationDocumentInput] = Field(default_factory=list)
