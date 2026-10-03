from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from backend.schemas.verification import CrossDocumentVerificationRequest, VerificationDocumentInput
from backend.services.cross_document_verification_service import CrossDocumentVerificationService
from backend.services.cross_document_gnn_service import CrossDocumentGNNService, GraphConsistencyConfig

router = APIRouter(prefix="/api/verify", tags=["verification"])


@router.post("/cross-document")
async def verify_cross_document(payload: CrossDocumentVerificationRequest) -> dict:
    if not payload.documents and not payload.document_ids:
        raise HTTPException(status_code=400, detail="Provide either document_ids or a documents payload for cross-document verification.")

    documents: List[dict] = [document.model_dump() for document in payload.documents]
    if payload.document_ids:
        registry_docs = CrossDocumentVerificationService.fetch_documents_by_ids(payload.document_ids)
        if not registry_docs:
            raise HTTPException(status_code=404, detail="One or more requested document IDs were not found in the verification registry.")
        documents = registry_docs

    analysis = CrossDocumentVerificationService.verify_documents(documents)
    checkpoint = GraphConsistencyConfig().checkpoint_path
    if Path(checkpoint).exists():
        try:
            analysis["gnn_evidence"] = CrossDocumentGNNService().predict(documents, analysis["results"])
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            analysis["gnn_evidence"] = {"status": "unavailable", "reason": str(exc), "predictions": []}
    else:
        analysis["gnn_evidence"] = {"status": "not_trained", "predictions": []}
    return analysis


@router.get("/cross-document/{analysis_id}")
async def get_cross_document_analysis(analysis_id: str) -> dict:
    analysis = CrossDocumentVerificationService.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Cross-document verification analysis not found.")
    return analysis


@router.post("/cross-document/register")
async def register_document_fields(payload: VerificationDocumentInput) -> dict:
    record = CrossDocumentVerificationService.register_document_fields(
        payload.document_id,
        payload.document_type,
        payload.fields,
    )
    return {"document_id": payload.document_id, "status": "registered", "record": record}
