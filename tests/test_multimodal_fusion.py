from fastapi.testclient import TestClient

from backend.main import app
from backend.services.multimodal_fusion_service import MultimodalFusionEngine


def test_multimodal_ocr_and_structured_fields():
    payload = {
        "documents": [
            {
                "document_id": "doc-1",
                "document_type": "Invoice",
                "ocr": {"status": "success", "text": "INVOICE\nInvoice Number INV-1042\nTotal 1650.00\nCurrency USD", "pages": 1},
                "structured_fields": {"invoice_number": "INV-1042", "total": 1650.0, "currency": "USD"},
                "metadata": {"page_count": 1, "file_type": "png"},
            }
        ]
    }
    analysis = MultimodalFusionEngine().fuse_documents(payload["documents"])

    assert analysis["modalities"]["ocr"]["documents"] == ["doc-1"]
    assert analysis["modalities"]["structured_fields"]["document_ids"] == ["doc-1"]
    assert analysis["summary"]["documents_analyzed"] == 1
    assert any(item["source"] == "ocr" for item in analysis["evidence"])


def test_multimodal_structured_fields_plus_cross_document_result():
    payload = {
        "documents": [
            {
                "document_id": "doc-2",
                "document_type": "Salary Slip",
                "structured_fields": {"employee_name": "John Smith", "net_salary": 4300.0},
                "cross_document": {
                    "rule_id": "salary_vs_bank_credit",
                    "rule_name": "Salary vs bank credit",
                    "status": "FAIL",
                    "documents": ["doc-2", "doc-3"],
                    "fields_compared": ["net_salary", "credits"],
                    "values": {"net_salary": 4300.0, "bank_credit": 3500.0},
                    "explanation": "The salary differs from the bank credit amount.",
                    "evidence": ["doc-2: net_salary=4300.0", "doc-3: credits=3500.0"],
                },
            },
            {
                "document_id": "doc-3",
                "document_type": "Bank Statement",
                "structured_fields": {"account_holder": "John Smith", "credits": 3500.0},
            },
        ]
    }

    analysis = MultimodalFusionEngine().fuse_documents(payload["documents"])
    assert any(item["source"] == "cross_document" for item in analysis["evidence"])
    assert analysis["summary"]["inconsistencies_detected"] >= 1


def test_multimodal_tampering_and_ocr_evidence():
    payload = {
        "documents": [
            {
                "document_id": "doc-4",
                "document_type": "Identity Document",
                "ocr": {"status": "success", "text": "IDENTITY DOCUMENT\nName Mary Jane Watson\nDocument Number AB1234567", "pages": [1]},
                "tampering": {
                    "document_id": "doc-4",
                    "detector_type": "baseline_forensic",
                    "tampering_detected": True,
                    "findings": [{
                        "detector": "baseline_noise",
                        "finding": "local noise inconsistency",
                        "severity": "MEDIUM",
                        "region": [80, 90, 180, 190],
                        "explanation": "Noise looks inconsistent in the central region.",
                    }],
                    "suspicious_regions": [[80, 90, 180, 190]],
                    "limitations": ["heuristic only"],
                },
            }
        ]
    }

    analysis = MultimodalFusionEngine().fuse_documents(payload["documents"])
    assert analysis["modalities"]["tampering"]["documents"] == ["doc-4"]
    assert any(item["source"] == "tampering" for item in analysis["evidence"])
    assert any(item.get("region") is not None for item in analysis["evidence"] if item["source"] == "tampering")
    assert analysis["summary"]["inconsistencies_detected"] == 0
    assert analysis["summary"]["tampering_observations_detected"] == 1


def test_all_modalities_together():
    documents = [
        {
            "document_id": "doc-5",
            "document_type": "Invoice",
            "ocr": {"status": "success", "text": "INVOICE\nTotal 1200.00\nCurrency USD", "pages": 1},
            "structured_fields": {"invoice_number": "INV-88", "total": 1200.0, "currency": "USD"},
            "cross_document": {
                "rule_id": "invoice_total_vs_payment_amount",
                "rule_name": "Invoice total vs payment amount",
                "status": "PASS",
                "documents": ["doc-5", "doc-6"],
                "fields_compared": ["total", "credits"],
                "values": {"invoice_total": 1200.0, "payment_amount": 1200.0},
                "explanation": "The amounts match.",
                "evidence": ["doc-5: total=1200.0", "doc-6: credits=1200.0"],
            },
            "tampering": {
                "document_id": "doc-5",
                "detector_type": "baseline_forensic",
                "tampering_detected": False,
                "findings": [],
                "suspicious_regions": [],
                "limitations": ["no heuristic anomalies"],
            },
            "metadata": {"file_type": "pdf", "page_count": 1},
        },
        {
            "document_id": "doc-6",
            "document_type": "Bank Statement",
            "structured_fields": {"account_holder": "Jane Doe", "credits": 1200.0, "currency": "USD"},
        },
    ]

    analysis = MultimodalFusionEngine().fuse_documents(documents)
    assert analysis["modalities"]["visual"]["documents"] == ["doc-5"]
    assert analysis["modalities"]["cross_document"]["documents"] == ["doc-5", "doc-6"]
    assert analysis["summary"]["evidence_count"] >= 4


def test_missing_modality_is_handled_gracefully():
    documents = [{"document_id": "doc-7", "document_type": "Unknown", "metadata": {"file_type": "png"}}]
    analysis = MultimodalFusionEngine().fuse_documents(documents)
    assert analysis["summary"]["documents_analyzed"] == 1
    assert analysis["summary"]["evidence_count"] >= 1


def test_insufficient_data_is_preserved():
    analysis = MultimodalFusionEngine().fuse_documents([
        {"document_id": "doc-8", "document_type": "Salary Slip", "structured_fields": {"employee_name": "Alice"}}
    ])
    assert analysis["summary"]["insufficient_data_count"] >= 0
    assert any(item["source"] in {"structured_field", "metadata"} for item in analysis["evidence"])


def test_multiple_documents_keep_traceability():
    documents = [
        {"document_id": "doc-9", "document_type": "Invoice", "structured_fields": {"invoice_number": "INV-1", "total": 100.0}, "ocr": {"status": "success", "text": "Invoice Number INV-1", "pages": [1]}},
        {"document_id": "doc-10", "document_type": "Bank Statement", "structured_fields": {"account_holder": "Alice", "credits": 100.0}, "ocr": {"status": "success", "text": "Account Holder Alice\nCredits 100.00", "pages": [1]}},
    ]
    analysis = MultimodalFusionEngine().fuse_documents(documents)
    assert all(item["document_ids"] for item in analysis["evidence"])
    assert any(item["source"] == "ocr" for item in analysis["evidence"])


def test_invalid_analysis_input_raises():
    try:
        MultimodalFusionEngine().fuse_documents(None)
    except ValueError:
        return
    assert False, "Expected ValueError for invalid input"


def test_api_multimodal_endpoint():
    client = TestClient(app)
    payload = {
        "documents": [
            {
                "document_id": "api-doc-1",
                "document_type": "Invoice",
                "ocr": {"status": "success", "text": "INVOICE\nTotal 300.00", "pages": 1},
                "structured_fields": {"invoice_number": "INV-API", "total": 300.0},
            }
        ]
    }
    response = client.post("/api/analysis/multimodal", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["documents"][0]["document_id"] == "api-doc-1"
    assert body["summary"]["documents_analyzed"] == 1


def test_get_multipart_analysis_endpoint():
    engine = MultimodalFusionEngine()
    analysis = engine.fuse_documents([
        {"document_id": "doc-11", "document_type": "Invoice", "structured_fields": {"invoice_number": "INV-11", "total": 99.0}}
    ])
    client = TestClient(app)
    response = client.get(f"/api/analysis/multimodal/{analysis['analysis_id']}")
    assert response.status_code == 200
    assert response.json()["analysis_id"] == analysis["analysis_id"]
