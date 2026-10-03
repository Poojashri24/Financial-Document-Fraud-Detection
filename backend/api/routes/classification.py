from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.classification_service import ClassificationService
from backend.services.extraction_service import ExtractionService

router = APIRouter(prefix="/api/documents", tags=["classification"])


class ClassificationRequest(BaseModel):
    ocr_text: str


class ExtractionRequest(BaseModel):
    document_type: str
    ocr_text: str


@router.post("/{document_id}/classify")
async def classify_document(document_id: str, payload: ClassificationRequest) -> dict:
    if not payload.ocr_text or not payload.ocr_text.strip():
        raise HTTPException(status_code=400, detail="OCR text is required for classification.")
    result = ClassificationService.classify_text(payload.ocr_text)
    return {"document_id": document_id, **result}


@router.post("/{document_id}/extract")
async def extract_document_fields(document_id: str, payload: ExtractionRequest) -> dict:
    if not payload.ocr_text or not payload.ocr_text.strip():
        raise HTTPException(status_code=400, detail="OCR text is required for extraction.")
    fields = ExtractionService.extract_fields(payload.document_type, payload.ocr_text)
    return {"document_id": document_id, "document_type": payload.document_type, **fields}


@router.get("/{document_id}/fields")
async def get_document_fields(document_id: str) -> dict:
    raise HTTPException(status_code=404, detail="Document not found. This baseline endpoint expects a stored document record in a later phase.")
