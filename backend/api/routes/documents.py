from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.config import settings
from backend.schemas.document import ProcessingResult, UploadedDocument
from backend.services.document_service import DocumentService
from backend.services.ocr_service import OCRService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    results: List[UploadedDocument] = []
    for uploaded_file in files:
        file_bytes = await uploaded_file.read()
        document_id = DocumentService.generate_document_id()
        try:
            document = DocumentService.build_uploaded_document(document_id, uploaded_file.filename or "document", file_bytes)
            results.append(document)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{uploaded_file.filename}: {str(exc)}") from exc

    return {
        "count": len(results),
        "documents": [document.model_dump() for document in results],
    }


@router.get("/{document_id}")
async def get_document(document_id: str) -> dict:
    document = DocumentService.get_document_record(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document.model_dump()


@router.post("/{document_id}/process")
async def process_document(document_id: str) -> ProcessingResult:
    result = DocumentService.process_document(document_id)
    if result.status == "error" and result.errors:
        raise HTTPException(status_code=404 if result.errors[0].code == "not_found" else 400, detail=result.errors[0].message)
    return result


@router.post("/{document_id}/ocr")
async def run_ocr(document_id: str) -> dict:
    document = DocumentService.get_document_record(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if document.storage_path is None:
        raise HTTPException(status_code=400, detail="Document storage path is missing.")

    file_path = Path(document.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored document file not found.")

    result = OCRService.extract_text_from_document(document.model_dump())
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "OCR failed."))

    return {
        "document_id": document_id,
        "filename": document.filename,
        "status": result["status"],
        "text": result["text"],
        "pages": result.get("pages", 1),
    }
