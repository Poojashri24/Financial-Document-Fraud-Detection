from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    page_count: Optional[int] = None
    pages: List[int] = Field(default_factory=list)
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    source_path: Optional[str] = None
    extracted_at: Optional[str] = None


class UploadedDocument(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    file_type: str
    size_bytes: int
    page_count: int = 1
    status: str = "uploaded"
    storage_path: Optional[str] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class ProcessingError(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    document_id: str
    status: str
    page_count: int
    pages_processed: int
    page_images: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[ProcessingError] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str
    filename: str
    file_type: str
    pages: List[int] = Field(default_factory=list)
    page_images: List[str] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
