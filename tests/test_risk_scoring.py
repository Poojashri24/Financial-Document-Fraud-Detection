import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_reasoning_service import LLMReasoningService
from backend.services.multimodal_fusion_service import MultimodalFusionEngine
from backend.services.risk_scoring_service import RiskScoringService


@pytest.fixture
def clean_reasoning():
    analysis = MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-1",
            "document_type": "Invoice",
            "ocr": {"status": "success", "text": "Invoice Total 100.00", "pages": 1},
            "structured_fields": {"invoice_number": "INV-1", "total": 100.0},
            "metadata": {"page_count": 1},
        }
    ])
    return LLMReasoningService().reason_about_analysis(analysis)


def test_no_suspicious_evidence(low_risk_reasoning=None):
    reasoning = {
        "analysis_id": "a-1",
        "reasoning_id": "r-1",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [{
            "finding_id": "f-1",
            "summary": "Invoice total recorded with matching fields.",
            "document_ids": ["doc-1"],
            "pages": [1],
            "fields": ["total"],
            "regions": [],
            "evidence": ["Invoice total 100.00"],
            "supporting_evidence": ["Invoice total 100.00"],
            "severity": "INFO",
        }],
        "evidence_support": {"structured_field": ["Invoice total 100.00"]},
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [],
        "reasoning": "Evidence is internally consistent.",
        "limitations": [],
        "traceability": {"document_count": 1, "modalities_present": ["ocr", "structured_fields"], "source_evidence_count": 2},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_level"] in {"LOW", "MODERATE"}
    assert assessment["risk_score"] <= 40


def test_one_inconsistency():
    reasoning = {
        "analysis_id": "a-2",
        "reasoning_id": "r-2",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [{
            "summary": "Invoice total differs from payment amount.",
            "document_ids": ["doc-2", "doc-3"],
            "pages": [1],
            "fields": ["total", "credits"],
            "regions": [],
            "evidence": ["doc-2: total=1000.0", "doc-3: credits=850.0"],
            "supporting_evidence": ["doc-2: total=1000.0", "doc-3: credits=850.0"],
            "severity": "MEDIUM",
        }],
        "evidence_support": {"cross_document": ["doc-2: total=1000.0", "doc-3: credits=850.0"]},
        "cross_document_inconsistencies": ["Invoice total differs from payment amount."],
        "tampering_related_observations": [],
        "reasoning": "One cross-document inconsistency was detected.",
        "limitations": [],
        "traceability": {"document_count": 2, "modalities_present": ["cross_document"], "source_evidence_count": 2},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] >= 30
    assert assessment["risk_level"] in {"MODERATE", "HIGH"}


def test_multiple_inconsistencies():
    reasoning = {
        "analysis_id": "a-3",
        "reasoning_id": "r-3",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [
            {"summary": "Invoice mismatch", "document_ids": ["doc-4", "doc-5"], "pages": [1], "fields": ["total", "credits"], "regions": [], "evidence": ["e1"], "supporting_evidence": ["e1"], "severity": "MEDIUM"},
            {"summary": "Name mismatch", "document_ids": ["doc-4", "doc-6"], "pages": [1], "fields": ["name", "account_holder"], "regions": [], "evidence": ["e2"], "supporting_evidence": ["e2"], "severity": "HIGH"},
        ],
        "evidence_support": {"cross_document": ["e1", "e2"]},
        "cross_document_inconsistencies": ["Invoice mismatch", "Name mismatch"],
        "tampering_related_observations": [],
        "reasoning": "Multiple inconsistencies were detected.",
        "limitations": [],
        "traceability": {"document_count": 3, "modalities_present": ["cross_document"], "source_evidence_count": 2},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] >= 50
    assert assessment["risk_level"] in {"HIGH", "CRITICAL"}


def test_tampering_evidence_increases_risk():
    reasoning = {
        "analysis_id": "a-4",
        "reasoning_id": "r-4",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [{
            "summary": "Tampering regions detected.",
            "document_ids": ["doc-7"],
            "pages": [1],
            "fields": [],
            "regions": [[80, 90, 180, 190]],
            "evidence": ["noise inconsistency"],
            "supporting_evidence": ["noise inconsistency"],
            "severity": "HIGH",
        }],
        "evidence_support": {"tampering": ["noise inconsistency"]},
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [{
            "document_ids": ["doc-7"],
            "summary": "Tampering region identified.",
            "severity": "HIGH",
            "details": {"tampering_detected": True},
            "region": [80, 90, 180, 190],
        }],
        "reasoning": "Visible tampering region exists.",
        "limitations": ["Heuristic only."],
        "traceability": {"document_count": 1, "modalities_present": ["tampering"], "source_evidence_count": 1},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] >= 45
    assert any(item["source"] == "tampering" for item in assessment["contributing_evidence"])


def test_combined_evidence_raises_score():
    reasoning = {
        "analysis_id": "a-5",
        "reasoning_id": "r-5",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [
            {"summary": "Invoice mismatch", "document_ids": ["doc-8", "doc-9"], "pages": [1], "fields": ["total", "credits"], "regions": [], "evidence": ["e1"], "supporting_evidence": ["e1"], "severity": "MEDIUM"},
            {"summary": "Tampering region", "document_ids": ["doc-8"], "pages": [1], "fields": [], "regions": [[10, 10, 50, 60]], "evidence": ["e2"], "supporting_evidence": ["e2"], "severity": "HIGH"},
        ],
        "evidence_support": {"cross_document": ["e1"], "tampering": ["e2"]},
        "cross_document_inconsistencies": ["Invoice mismatch"],
        "tampering_related_observations": [{"document_ids": ["doc-8"], "summary": "Tampering region", "severity": "HIGH", "details": {}, "region": [10, 10, 50, 60]}],
        "reasoning": "Multiple suspicious indicators are present.",
        "limitations": [],
        "traceability": {"document_count": 2, "modalities_present": ["cross_document", "tampering"], "source_evidence_count": 3},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] >= 60
    assert assessment["risk_level"] in {"HIGH", "CRITICAL"}


def test_missing_evidence_increases_limitations():
    reasoning = {
        "analysis_id": "a-6",
        "reasoning_id": "r-6",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [{"summary": "No strong evidence available.", "document_ids": ["doc-10"], "pages": [], "fields": [], "regions": [], "evidence": [], "supporting_evidence": [], "severity": "INFO"}],
        "evidence_support": {},
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [],
        "reasoning": "No supporting evidence was supplied.",
        "limitations": ["Context is sparse."],
        "traceability": {"document_count": 1, "modalities_present": [], "source_evidence_count": 0},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] <= 25
    assert assessment["limitations"]


def test_insufficient_data_sets_low_score():
    reasoning = {
        "analysis_id": "a-7",
        "reasoning_id": "r-7",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [],
        "evidence_support": {},
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [],
        "reasoning": "Insufficient evidence was supplied.",
        "limitations": ["Not enough evidence available."],
        "traceability": {"document_count": 1, "modalities_present": [], "source_evidence_count": 0},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_score"] <= 20
    assert assessment["risk_level"] == "LOW"


def test_multiple_documents_and_traceability():
    reasoning = {
        "analysis_id": "a-8",
        "reasoning_id": "r-8",
        "reasoning_mode": "heuristic_rule_based",
        "observed_findings": [
            {"summary": "Document mismatch", "document_ids": ["doc-11", "doc-12"], "pages": [1, 2], "fields": ["total", "credits"], "regions": [[1, 2, 3, 4]], "evidence": ["mismatch"], "supporting_evidence": ["mismatch"], "severity": "MEDIUM"},
        ],
        "evidence_support": {"cross_document": ["mismatch"]},
        "cross_document_inconsistencies": ["Document mismatch"],
        "tampering_related_observations": [],
        "reasoning": "Multiple documents were compared.",
        "limitations": [],
        "traceability": {"document_count": 2, "modalities_present": ["cross_document"], "source_evidence_count": 1},
        "generated_at": "2026-09-21T00:00:00",
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert {"doc-11", "doc-12"}.issubset(set(assessment["supporting_documents"]))
    assert assessment["contributing_evidence"][0]["traceability"]["document_ids"]


def test_score_boundary_cases():
    low = RiskScoringService().score_reasoning({"analysis_id": "a-9", "reasoning_id": "r-9", "reasoning_mode": "heuristic_rule_based", "observed_findings": [], "evidence_support": {}, "cross_document_inconsistencies": [], "tampering_related_observations": [], "reasoning": "Nothing suspicious.", "limitations": ["No evidence"], "traceability": {"document_count": 1, "modalities_present": [], "source_evidence_count": 0}, "generated_at": "2026-09-21T00:00:00"})
    high = RiskScoringService().score_reasoning({"analysis_id": "a-10", "reasoning_id": "r-10", "reasoning_mode": "heuristic_rule_based", "observed_findings": [{"summary": "High-risk mismatch", "document_ids": ["d1", "d2"], "pages": [1], "fields": ["total", "credits"], "regions": [[1,1,5,5]], "evidence": ["x"], "supporting_evidence": ["x"], "severity": "HIGH"}], "evidence_support": {"cross_document": ["x"], "tampering": ["y"]}, "cross_document_inconsistencies": ["High-risk mismatch"], "tampering_related_observations": [{"document_ids": ["d1"], "summary": "Tampering", "severity": "HIGH", "details": {"tampering_detected": True}, "region": [1,1,5,5]}], "reasoning": "Multiple issues.", "limitations": [], "traceability": {"document_count": 2, "modalities_present": ["cross_document", "tampering"], "source_evidence_count": 2}, "generated_at": "2026-09-21T00:00:00"})
    assert low["risk_score"] == 0
    assert high["risk_score"] >= 70


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        RiskScoringService().score_reasoning(None)


def test_api_risk_endpoint():
    client = TestClient(app)
    reasoning = LLMReasoningService().reason_about_analysis(MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-13", "document_type": "Invoice", "ocr": {"status": "success", "text": "Invoice Number INV-13\nTotal 250.00", "pages": 1}, "structured_fields": {"invoice_number": "INV-13", "total": 250.0}, "cross_document": {"rule_id": "invoice_total_vs_payment_amount", "rule_name": "Invoice total vs payment amount", "status": "FAIL", "documents": ["doc-13", "doc-14"], "fields_compared": ["total", "credits"], "values": {"invoice_total": 250.0, "payment_amount": 180.0}, "explanation": "Mismatch", "evidence": ["doc-13: total=250.0", "doc-14: credits=180.0"]}, "metadata": {"file_type": "png"}},
        {"document_id": "doc-14", "document_type": "Bank Statement", "structured_fields": {"credits": 180.0}}
    ]))
    response = client.post("/api/analysis/risk", json={"reasoning_id": reasoning["reasoning_id"]})
    assert response.status_code == 200
    body = response.json()
    assert body["risk_score"] >= 0
    assert body["risk_level"]


def test_get_risk_endpoint():
    client = TestClient(app)
    reasoning = LLMReasoningService().reason_about_analysis(MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-15", "document_type": "Invoice", "structured_fields": {"total": 1200.0}}
    ]))
    create_response = client.post("/api/analysis/risk", json={"reasoning_id": reasoning["reasoning_id"]})
    assessment_id = create_response.json()["analysis_id"]
    response = client.get(f"/api/analysis/risk/{assessment_id}")
    assert response.status_code == 200
    assert response.json()["analysis_id"] == assessment_id


def test_regression_previous_phases_still_work():
    analysis = MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-16", "document_type": "Invoice", "structured_fields": {"invoice_number": "INV-16", "total": 300.0}, "ocr": {"status": "success", "text": "INV-16 300.00", "pages": 1}},
        {"document_id": "doc-17", "document_type": "Bank Statement", "structured_fields": {"credits": 300.0}}
    ])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    risk = RiskScoringService().score_reasoning(reasoning)
    assert reasoning["reasoning_mode"] == "heuristic_rule_based"
    assert risk["methodology"]["score_type"] == "heuristic_baseline"
