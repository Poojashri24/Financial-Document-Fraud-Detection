import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_reasoning_service import LLMReasoningService
from backend.services.multimodal_fusion_service import MultimodalFusionEngine


@pytest.fixture
def sample_multimodal_analysis():
    return MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-a",
            "document_type": "Invoice",
            "ocr": {
                "status": "success",
                "text": "Invoice Number INV-1042\nTotal 1650.00\nCurrency USD",
                "pages": 1,
            },
            "structured_fields": {"invoice_number": "INV-1042", "total": 1650.0, "currency": "USD"},
            "cross_document": {
                "rule_id": "invoice_total_vs_payment_amount",
                "rule_name": "Invoice total vs payment amount",
                "status": "FAIL",
                "documents": ["doc-a", "doc-b"],
                "fields_compared": ["total", "credits"],
                "values": {"invoice_total": 1650.0, "payment_amount": 1200.0},
                "explanation": "The amounts differ.",
                "evidence": ["doc-a: total=1650.0", "doc-b: credits=1200.0"],
            },
            "tampering": {
                "document_id": "doc-a",
                "detector_type": "baseline_forensic",
                "tampering_detected": False,
                "findings": [],
                "suspicious_regions": [],
                "limitations": ["No anomalies found"],
            },
            "metadata": {"page_count": 1, "file_type": "png"},
        },
        {
            "document_id": "doc-b",
            "document_type": "Bank Statement",
            "structured_fields": {"account_holder": "Alice Smith", "credits": 1200.0, "currency": "USD"},
            "metadata": {"page_count": 1, "file_type": "png"},
        },
    ])


def test_reasoning_for_normal_multimodal_evidence(sample_multimodal_analysis):
    reasoning = LLMReasoningService().reason_about_analysis(sample_multimodal_analysis)

    assert reasoning["observed_findings"]
    assert reasoning["cross_document_inconsistencies"]
    assert reasoning["reasoning_mode"] == "heuristic_rule_based"
    assert any(item["document_ids"] for item in reasoning["observed_findings"])


def test_reasoning_when_multiple_inconsistencies_exist():
    analysis = MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-c",
            "document_type": "Salary Slip",
            "structured_fields": {"employee_name": "Mary", "net_salary": 4000.0},
            "cross_document": {
                "rule_id": "salary_vs_bank_credit",
                "rule_name": "Salary vs bank credit",
                "status": "FAIL",
                "documents": ["doc-c", "doc-d"],
                "fields_compared": ["net_salary", "credits"],
                "values": {"net_salary": 4000.0, "bank_credit": 2100.0},
                "explanation": "Mismatch detected",
                "evidence": ["doc-c: net_salary=4000.0", "doc-d: credits=2100.0"],
            },
        },
        {
            "document_id": "doc-d",
            "document_type": "Bank Statement",
            "structured_fields": {"account_holder": "Mary", "credits": 2100.0},
        },
    ])

    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert len(reasoning["cross_document_inconsistencies"]) >= 1
    assert reasoning["observed_findings"][0]["summary"]


def test_reasoning_with_tampering_evidence():
    analysis = MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-e",
            "document_type": "Identity Document",
            "ocr": {"status": "success", "text": "Name: Jane Doe\nDocument Number: AB1234567", "pages": 1},
            "tampering": {
                "document_id": "doc-e",
                "detector_type": "baseline_forensic",
                "tampering_detected": True,
                "findings": [{
                    "detector": "baseline_noise",
                    "finding": "local noise inconsistency",
                    "severity": "MEDIUM",
                    "region": [80, 90, 180, 190],
                    "explanation": "Noise is inconsistent.",
                }],
                "suspicious_regions": [[80, 90, 180, 190]],
                "limitations": ["Heuristic only"],
            },
        }
    ])

    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["tampering_related_observations"]
    assert reasoning["tampering_related_observations"][0]["region"] == [80, 90, 180, 190]


def test_reasoning_handles_missing_modalities():
    analysis = MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-f", "document_type": "Unknown", "metadata": {"file_type": "png"}}
    ])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["limitations"]
    assert reasoning["observed_findings"]


def test_reasoning_handles_insufficient_evidence():
    analysis = MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-g", "document_type": "Salary Slip", "structured_fields": {"employee_name": "Alice"}}
    ])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["limitations"]
    assert reasoning["reasoning"]


def test_reasoning_keeps_traceability_across_multiple_documents():
    analysis = MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-h",
            "document_type": "Invoice",
            "ocr": {"status": "success", "text": "Invoice Number INV-1\nTotal 100.00", "pages": [1]},
            "structured_fields": {"invoice_number": "INV-1", "total": 100.0},
        },
        {
            "document_id": "doc-i",
            "document_type": "Bank Statement",
            "ocr": {"status": "success", "text": "Statement\nCredits 100\nAccount Holder Alice", "pages": [1]},
            "structured_fields": {"credits": 100.0, "account_holder": "Alice"},
        },
    ])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    ids = {doc_id for item in reasoning["observed_findings"] for doc_id in item["document_ids"]}
    assert {"doc-h", "doc-i"}.issubset(ids)


def test_invalid_reasoning_input_raises():
    with pytest.raises(ValueError):
        LLMReasoningService().reason_about_analysis(None)


def test_fallback_mode_is_explicitly_labeled():
    reasoning = LLMReasoningService(provider="", api_key="").reason_about_analysis({
        "analysis_id": "fallback-test",
        "documents": [{"document_id": "doc-j", "document_type": "Invoice", "structured_fields": {"total": 500.0}}],
        "summary": {"documents_analyzed": 1, "evidence_count": 1, "inconsistencies_detected": 0},
        "evidence": [{"source": "structured_field", "document_ids": ["doc-j"], "summary": "Total field present.", "details": {"fields": ["total"]}}],
    })
    assert reasoning["reasoning_mode"] == "heuristic_rule_based"
    assert "heuristic" in reasoning["reasoning"].lower()


def test_api_reason_route_uses_multimodal_analysis():
    client = TestClient(app)
    analysis = MultimodalFusionEngine().fuse_documents([
        {
            "document_id": "doc-k",
            "document_type": "Invoice",
            "ocr": {"status": "success", "text": "Invoice Total 200.00", "pages": 1},
            "structured_fields": {"invoice_number": "INV-200", "total": 200.0},
        }
    ])
    response = client.post("/api/analysis/reason", json={"analysis_id": analysis["analysis_id"]})
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == analysis["analysis_id"]
    assert body["observed_findings"]


def test_api_reason_route_invalid_input():
    client = TestClient(app)
    response = client.post("/api/analysis/reason", json={"analysis": {}})
    assert response.status_code == 400


def test_get_reason_route():
    analysis = MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-l", "document_type": "Bank Statement", "structured_fields": {"credits": 250.0}}
    ])
    client = TestClient(app)
    response = client.post("/api/analysis/reason", json={"analysis_id": analysis["analysis_id"]})
    reasoning_id = response.json()["reasoning_id"]
    get_response = client.get(f"/api/analysis/reason/{reasoning_id}")
    assert get_response.status_code == 200
    assert get_response.json()["reasoning_id"] == reasoning_id
