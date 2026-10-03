from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from frontend.integration import is_supported_file, upload_documents


class DummyUpload:
    def __init__(self, filename: str, payload: bytes):
        self.name = filename
        self._payload = payload

    def read(self):
        return self._payload


def test_supported_file_validation_accepts_real_document_formats():
    assert is_supported_file("invoice.pdf") is True
    assert is_supported_file("scan.png") is True
    assert is_supported_file("photo.jpg") is True
    assert is_supported_file("photo.jpeg") is True
    assert is_supported_file("notes.txt") is False


def test_upload_helper_rejects_unsupported_files(monkeypatch):
    dummy = DummyUpload("bad.txt", b"hello")

    with pytest.raises(ValueError):
        upload_documents([dummy])


def test_upload_helper_builds_valid_multipart_payload(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"count": 1, "documents": [{"document_id": "doc-123", "filename": "sample.pdf", "status": "uploaded"}]}

    def fake_post(url, files=None, timeout=None):
        calls["url"] = url
        calls["files"] = files
        calls["timeout"] = timeout
        return FakeResponse()

    import frontend.integration as integration

    monkeypatch.setattr(integration.httpx, "post", fake_post)
    payload = upload_documents([DummyUpload("sample.pdf", b"%PDF-1.4")])

    assert payload["count"] == 1
    assert calls["url"].endswith("/api/documents/upload")
    assert calls["files"][0][1][0] == "sample.pdf"


def test_pdf_pipeline_routes_traverse_real_outputs_for_synthetic_documents():
    client = TestClient(app)
    pdf_paths = {
        "salary": next(Path("data/uploads").rglob("test_salary_slip.pdf")),
        "bank": next(Path("data/uploads").rglob("test_bank_statement_matching.pdf")),
    }

    uploaded = {}
    for label, pdf_path in pdf_paths.items():
        response = client.post("/api/documents/upload", files=[("files", (pdf_path.name, pdf_path.read_bytes(), "application/pdf"))])
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["count"] == 1
        uploaded[label] = payload["documents"][0]["document_id"]

    salary_id = uploaded["salary"]
    bank_id = uploaded["bank"]

    process_response = client.post(f"/api/documents/{salary_id}/process")
    assert process_response.status_code == 200, process_response.text
    salary_process = process_response.json()
    assert salary_process["status"] == "processed"
    assert salary_process["page_count"] == 1
    assert len(salary_process["page_images"]) == 1
    assert Path(salary_process["page_images"][0]).exists()

    ocr_response = client.post(f"/api/documents/{salary_id}/ocr")
    assert ocr_response.status_code == 200, ocr_response.text
    salary_ocr = ocr_response.json()
    assert salary_ocr["status"] == "success"
    assert len(salary_ocr["text"].strip()) > 50

    classify_response = client.post(f"/api/documents/{salary_id}/classify", json={"ocr_text": salary_ocr["text"]})
    assert classify_response.status_code == 200, classify_response.text
    salary_type = classify_response.json()["document_type"]
    assert salary_type == "Salary Slip"

    extract_response = client.post(f"/api/documents/{salary_id}/extract", json={"document_type": salary_type, "ocr_text": salary_ocr["text"]})
    assert extract_response.status_code == 200, extract_response.text
    salary_fields = extract_response.json()
    assert "employee_name" in salary_fields or "employer" in salary_fields

    register_response = client.post("/api/verify/cross-document/register", json={"document_id": salary_id, "document_type": salary_type, "fields": salary_fields})
    assert register_response.status_code == 200, register_response.text
    assert register_response.json()["record"]["document_id"] == salary_id
    assert register_response.json()["record"]["fields"]["employee_name"] == salary_fields["employee_name"]

    bank_process = client.post(f"/api/documents/{bank_id}/process")
    assert bank_process.status_code == 200, bank_process.text
    bank_process_payload = bank_process.json()
    assert bank_process_payload["status"] == "processed"
    assert len(bank_process_payload["page_images"]) == 1
    assert Path(bank_process_payload["page_images"][0]).exists()

    bank_ocr_response = client.post(f"/api/documents/{bank_id}/ocr")
    assert bank_ocr_response.status_code == 200, bank_ocr_response.text
    bank_ocr = bank_ocr_response.json()
    assert bank_ocr["status"] == "success"
    assert len(bank_ocr["text"].strip()) > 50

    bank_classify_response = client.post(f"/api/documents/{bank_id}/classify", json={"ocr_text": bank_ocr["text"]})
    assert bank_classify_response.status_code == 200, bank_classify_response.text
    bank_type = bank_classify_response.json()["document_type"]
    assert bank_type == "Bank Statement"

    bank_extract_response = client.post(f"/api/documents/{bank_id}/extract", json={"document_type": bank_type, "ocr_text": bank_ocr["text"]})
    assert bank_extract_response.status_code == 200, bank_extract_response.text
    bank_fields = bank_extract_response.json()
    assert "account_holder" in bank_fields or "statement_period" in bank_fields or "closing_balance" in bank_fields

    register_bank = client.post("/api/verify/cross-document/register", json={"document_id": bank_id, "document_type": bank_type, "fields": bank_fields})
    assert register_bank.status_code == 200, register_bank.text
    assert register_bank.json()["record"]["document_id"] == bank_id
    assert client.post("/api/verify/cross-document", json={"document_ids": [salary_id, bank_id]}).json()["document_count"] == 2

    tampering_salary = client.post(f"/api/documents/{salary_id}/tampering")
    assert tampering_salary.status_code == 200, tampering_salary.text
    tampering_payload = tampering_salary.json()
    assert "tampering_detected" in tampering_payload
    assert isinstance(tampering_payload["findings"], list)
    assert tampering_payload["detector_type"] == "baseline_forensic"
    assert tampering_payload["heuristic_evidence"]["evidence_source"] == "baseline_forensic_heuristic"
    assert tampering_payload["ml_evidence"]["evidence_source"] == "trained_visual_model"
    assert tampering_payload["ml_evidence"]["prediction"] in {"authentic", "forged"}

    verification_response = client.post("/api/verify/cross-document", json={"document_ids": [salary_id, bank_id]})
    assert verification_response.status_code == 200, verification_response.text
    verification_payload = verification_response.json()
    assert verification_payload["document_count"] == 2
    assert verification_payload["results"]
    salary_credit_result = next(item for item in verification_payload["results"] if item["rule_id"] == "salary_vs_bank_credit")
    assert salary_credit_result["status"] == "PASS"
    assert salary_id in salary_credit_result["documents"]
    assert bank_id in salary_credit_result["documents"]

    multimodal_response = client.post(
        "/api/analysis/multimodal",
        json={
            "documents": [
                {
                    "document_id": salary_id,
                    "document_type": salary_type,
                    "ocr": salary_ocr,
                    "structured_fields": {k: v for k, v in salary_fields.items() if k != "document_type"},
                    "cross_document": verification_payload["results"][0],
                    "tampering": tampering_payload,
                    "metadata": {"file_name": pdf_paths["salary"].name, "page_count": 1},
                },
                {
                    "document_id": bank_id,
                    "document_type": bank_type,
                    "ocr": bank_ocr,
                    "structured_fields": {k: v for k, v in bank_fields.items() if k != "document_type"},
                    "cross_document": verification_payload["results"][0],
                    "tampering": tampering_salary.json(),
                    "metadata": {"file_name": pdf_paths["bank"].name, "page_count": 1},
                },
            ]
        },
    )
    assert multimodal_response.status_code == 200, multimodal_response.text
    multimodal = multimodal_response.json()
    assert multimodal["summary"]["documents_analyzed"] == 2
    assert multimodal["summary"]["evidence_count"] >= 2
    evidence_sources = {item["source"] for item in multimodal["evidence"]}
    assert {"ocr", "structured_field", "cross_document", "tampering", "metadata"}.issubset(evidence_sources)
    assert "trained_visual_model" in evidence_sources

    reasoning_response = client.post("/api/analysis/reason", json={"analysis_id": multimodal["analysis_id"]})
    assert reasoning_response.status_code == 200, reasoning_response.text
    reasoning = reasoning_response.json()
    assert reasoning["reasoning_mode"] == "heuristic_rule_based"
    assert "ocr" in reasoning["evidence_support"]
    assert "structured_field" in reasoning["evidence_support"]
    assert "cross_document" in reasoning["evidence_support"]
    assert reasoning["trained_visual_model_observations"]

    risk_response = client.post("/api/analysis/risk", json={"reasoning": reasoning})
    assert risk_response.status_code == 200, risk_response.text
    risk = risk_response.json()
    assert risk["risk_score"] >= 0
    assert risk["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert not any(item["source"] == "cross_document" for item in risk["contributing_evidence"])
    assert risk["risk_level"] != "CRITICAL"
    assert any(item["source"] == "tampering" for item in risk["contributing_evidence"])
    assert any(item["source"] == "trained_visual_model" for item in risk["contributing_evidence"])


def test_pdf_pipeline_handles_invalid_and_missing_cases():
    client = TestClient(app)

    bad_pdf = client.post("/api/documents/upload", files=[("files", ("broken.pdf", b"not-a-real-pdf", "application/pdf"))])
    assert bad_pdf.status_code == 400

    unsupported = client.post("/api/documents/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert unsupported.status_code == 400

    missing_doc = client.post("/api/documents/missing-id/ocr")
    assert missing_doc.status_code == 404

    missing_verification = client.post("/api/verify/cross-document", json={"document_ids": ["missing-id"]})
    assert missing_verification.status_code == 404

    missing_render = client.post("/api/documents/does-not-exist/tampering")
    assert missing_render.status_code == 404
