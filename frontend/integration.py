from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence

import httpx

API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def is_supported_file(filename: str | None) -> bool:
    if not filename:
        return False
    return filename.lower().rsplit(".", 1)[-1] in {"pdf", "png", "jpg", "jpeg"} if "." in filename else False


def format_backend_error(error: Exception) -> str:
    detail = getattr(error, "response", None)
    if detail is not None:
        try:
            payload = detail.json()
            if isinstance(payload, dict) and "detail" in payload:
                return str(payload["detail"])
        except Exception:
            pass
        status = getattr(detail, "status_code", "unknown")
        return f"Backend request failed with status {status}: {error}"
    return str(error)


def upload_documents(files: Sequence[Any]) -> Dict[str, Any]:
    if not files:
        raise ValueError("No files were provided.")

    multipart = []
    for item in files:
        filename = getattr(item, "name", None) or getattr(item, "filename", "document")
        if not is_supported_file(filename):
            raise ValueError(f"Unsupported file format for {filename}. Supported formats: PDF, PNG, JPG, JPEG.")
        content = item.read() if hasattr(item, "read") else item.getvalue()
        multipart.append(("files", (filename, content, "application/octet-stream")))

    response = httpx.post(f"{API_BASE_URL}/api/documents/upload", files=multipart, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def process_document(document_id: str) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/documents/{document_id}/process", timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def run_ocr(document_id: str) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/documents/{document_id}/ocr", timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def classify_document(document_id: str, ocr_text: str) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/documents/{document_id}/classify", json={"ocr_text": ocr_text}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def extract_document_fields(document_id: str, document_type: str, ocr_text: str) -> Dict[str, Any]:
    response = httpx.post(
        f"{API_BASE_URL}/api/documents/{document_id}/extract",
        json={"document_type": document_type, "ocr_text": ocr_text},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def analyze_tampering(document_id: str) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/documents/{document_id}/tampering", timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def register_document_fields(document_id: str, document_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    response = httpx.post(
        f"{API_BASE_URL}/api/verify/cross-document/register",
        json={"document_id": document_id, "document_type": document_type, "fields": fields},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def verify_cross_document(document_ids: Sequence[str]) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/verify/cross-document", json={"document_ids": list(document_ids)}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def analyze_multimodal(documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}/api/analysis/multimodal", json={"documents": list(documents)}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def generate_reasoning(analysis_id: str | None = None, multimodal_analysis: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if analysis_id:
        payload["analysis_id"] = analysis_id
    if multimodal_analysis is not None:
        payload["multimodal_analysis"] = multimodal_analysis
    response = httpx.post(f"{API_BASE_URL}/api/analysis/reason", json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def assess_risk(reasoning_id: str | None = None, reasoning: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if reasoning_id:
        payload["reasoning_id"] = reasoning_id
    if reasoning is not None:
        payload["reasoning"] = reasoning
    response = httpx.post(f"{API_BASE_URL}/api/analysis/risk", json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(format_backend_error(Exception(response.text)))
    return response.json()


def build_multimodal_document_payload(document: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "document_id": document["document_id"],
        "document_type": document.get("document_type", "Unknown"),
        "ocr": document.get("ocr") or {},
        "structured_fields": document.get("structured_fields") or {},
        "cross_document": document.get("cross_document") or {},
        "tampering": document.get("tampering") or {},
        "metadata": document.get("metadata") or {},
    }
    return payload


def extract_evidence_items(evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in evidence or []:
        items.append({
            "source": item.get("source", "unknown"),
            "document_ids": item.get("document_ids", []),
            "summary": item.get("summary", "No summary available."),
            "severity": item.get("severity", "INFO"),
            "details": item.get("details", {}),
            "region": item.get("region"),
        })
    return items


def build_consistency_graph(verification: Dict[str, Any], documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare explainability nodes and relationships from existing verification results."""
    labels = {
        str(document.get("document_id")): f"{document.get('document_type', 'Document')}\n{document.get('metadata', {}).get('file_name', document.get('document_id'))}"
        for document in documents
    }
    relationships = []
    for result in verification.get("results", []):
        document_ids = [str(item) for item in result.get("documents", [])]
        relationships.append({
            "rule_id": result.get("rule_id"),
            "label": result.get("rule_name", "Comparison"),
            "status": result.get("status", "INSUFFICIENT_DATA"),
            "documents": document_ids,
            "left": labels.get(document_ids[0], document_ids[0]) if document_ids else "Unavailable",
            "right": labels.get(document_ids[1], document_ids[1]) if len(document_ids) > 1 else "Unavailable",
            "values": result.get("values", {}),
            "explanation": result.get("explanation", "No explanation supplied."),
        })
    return {"nodes": [{"document_id": key, "label": value} for key, value in labels.items()], "relationships": relationships}


def format_verification_values(rule_id: str, values: Dict[str, Any]) -> str:
    """Format rule values for readable UI without changing their source/status."""
    money_fields = {"net_salary", "bank_credit", "invoice_total", "payment_amount", "total", "credits", "debits", "closing_balance", "transaction_amount"}

    def display(key: str, value: Any) -> str:
        if isinstance(value, (int, float)) and key in money_fields:
            return f"₹{value:,.0f}" if float(value).is_integer() else f"₹{value:,.2f}"
        if isinstance(value, list):
            return ", ".join(str(part) for part in value)
        return str(value)

    if rule_id == "salary_vs_bank_credit":
        salary = values.get("net_salary")
        credit = values.get("bank_credit")
        if salary is not None and credit is not None:
            return f"Salary: {display('net_salary', salary)} ↔ Bank credit: {display('bank_credit', credit)}"
    if rule_id == "invoice_total_vs_payment_amount":
        invoice = values.get("invoice_total")
        payment = values.get("payment_amount")
        if invoice is not None and payment is not None:
            return f"Invoice: {display('invoice_total', invoice)} ↔ Payment: {display('payment_amount', payment)}"
    if not values:
        return "Not enough comparable values"
    return " · ".join(f"{key.replace('_', ' ').title()}: {display(key, value)}" for key, value in values.items())
