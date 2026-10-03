from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from PIL import Image, UnidentifiedImageError

from backend.core.config import settings
from backend.schemas.document import Document, DocumentMetadata, ProcessingError, ProcessingResult, UploadedDocument


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
}

ALLOWED_MIME_TYPES = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpg": {"image/jpeg", "image/jpg"},
    "jpeg": {"image/jpeg", "image/jpg"},
}


class DocumentService:
    @staticmethod
    def generate_document_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def safe_filename(filename: str) -> str:
        name = os.path.basename(filename or "document")
        name = name.strip()
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        if not name or name in {".", ".."}:
            name = "document"
        return name

    @staticmethod
    def ensure_storage_dirs() -> Dict[str, str]:
        root = Path(settings.storage_root)
        root.mkdir(parents=True, exist_ok=True)
        uploads = root / "uploads"
        processed = root / "processed"
        temp = root / "temp"
        for path in (uploads, processed, temp):
            path.mkdir(parents=True, exist_ok=True)
        return {
            "root": str(root),
            "uploads": str(uploads),
            "processed": str(processed),
            "temp": str(temp),
        }

    @staticmethod
    def validate_document_file(filename: str, file_bytes: bytes) -> str:
        if not filename:
            raise ValueError("Filename is required.")

        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}")

        DocumentService.validate_file_size(file_bytes, filename)

        if extension == ".pdf":
            if not file_bytes.startswith(b"%PDF"):
                raise ValueError("Corrupted or invalid PDF file.")
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if doc.page_count <= 0:
                        raise ValueError("PDF has no pages.")
            except Exception as exc:
                raise ValueError(f"Corrupted or invalid PDF file: {exc}") from exc
        else:
            try:
                with Image.open(BytesIO(file_bytes)) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError(f"Corrupted or invalid image file: {exc}") from exc

        return SUPPORTED_EXTENSIONS[extension]

    @staticmethod
    def validate_file_size(file_bytes: bytes, filename: str) -> None:
        max_bytes = int(settings.max_upload_size_mb * 1024 * 1024)
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"File '{filename}' exceeds the maximum upload size of {settings.max_upload_size_mb} MB."
            )

    @staticmethod
    def get_pdf_page_count_bytes(file_bytes: bytes) -> int:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return doc.page_count
        except Exception as exc:  # pragma: no cover - kept for validation safety
            raise ValueError(f"Unable to read PDF page count: {exc}") from exc

    @staticmethod
    def extract_image_metadata_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            with Image.open(BytesIO(file_bytes)) as img:
                img.load()
                return {
                    "format": img.format or Path(filename).suffix.upper().lstrip("."),
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                }
        except Exception as exc:  # pragma: no cover - safety path
            raise ValueError(f"Unable to read image metadata for '{filename}': {exc}") from exc

    @staticmethod
    def compute_sha256(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def pdf_to_images(file_bytes: bytes, output_dir: str, document_id: str) -> List[str]:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image_paths: List[str] = []

        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(matrix=fitz.Matrix(settings.pdf_scale, settings.pdf_scale))
                    image_path = output_dir_path / f"{document_id}_page_{page_index + 1}.png"
                    pix.save(str(image_path))
                    image_paths.append(str(image_path))
                    pix = None
        except Exception as exc:
            raise ValueError(f"Failed to convert PDF to images: {exc}") from exc

        return image_paths

    @staticmethod
    def store_uploaded_file(document_id: str, filename: str, file_bytes: bytes) -> str:
        storage_dirs = DocumentService.ensure_storage_dirs()
        safe_name = DocumentService.safe_filename(filename)
        target_dir = Path(storage_dirs["uploads"]) / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        storage_path = target_dir / safe_name
        with open(storage_path, "wb") as file_handle:
            file_handle.write(file_bytes)
        return str(storage_path)

    @staticmethod
    def build_uploaded_document(document_id: str, filename: str, file_bytes: bytes) -> UploadedDocument:
        file_type = DocumentService.validate_document_file(filename, file_bytes)
        size_bytes = len(file_bytes)
        page_count = 1
        if file_type == "pdf":
            page_count = DocumentService.get_pdf_page_count_bytes(file_bytes)
        metadata = DocumentMetadata(
            page_count=page_count,
            content_type=file_type,
            sha256=DocumentService.compute_sha256(file_bytes),
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

        if file_type in {"png", "jpg", "jpeg"}:
            image_meta = DocumentService.extract_image_metadata_bytes(file_bytes, filename)
            metadata.width = image_meta["width"]
            metadata.height = image_meta["height"]
            metadata.format = image_meta["format"]

        storage_path = DocumentService.store_uploaded_file(document_id, filename, file_bytes)
        metadata.source_path = storage_path

        return UploadedDocument(
            document_id=document_id,
            filename=DocumentService.safe_filename(filename),
            original_filename=filename,
            file_type=file_type,
            size_bytes=size_bytes,
            page_count=page_count,
            status="uploaded",
            storage_path=storage_path,
            metadata=metadata,
        )

    @staticmethod
    def get_document_record(document_id: str) -> Optional[UploadedDocument]:
        root = Path(settings.storage_root)
        uploads_root = root / "uploads"
        if not uploads_root.exists():
            return None
        for item in uploads_root.rglob("*"):
            if item.is_file() and item.parent.name == document_id:
                file_bytes = item.read_bytes()
                original_name = item.name
                doc = DocumentService.build_uploaded_document(document_id, original_name, file_bytes)
                return doc
        return None

    @staticmethod
    def process_document(document_id: str) -> ProcessingResult:
        document = DocumentService.get_document_record(document_id)
        if document is None:
            return ProcessingResult(
                document_id=document_id,
                status="error",
                page_count=0,
                pages_processed=0,
                errors=[ProcessingError(code="not_found", message="Document not found.")],
            )

        page_count = document.page_count
        page_images: List[str] = []
        errors: List[ProcessingError] = []
        processed_dir = Path(settings.storage_root) / "processed" / document_id
        processed_dir.mkdir(parents=True, exist_ok=True)

        if document.file_type == "pdf":
            try:
                file_bytes = (Path(document.storage_path) if document.storage_path else Path(settings.storage_root) / "uploads" / document_id).read_bytes()
                page_images = DocumentService.pdf_to_images(file_bytes, str(processed_dir), document_id)
            except ValueError as exc:
                errors.append(ProcessingError(code="pdf_error", message=str(exc)))
        else:
            if document.storage_path:
                page_images = [document.storage_path]

        return ProcessingResult(
            document_id=document_id,
            status="processed" if not errors else "error",
            page_count=page_count,
            pages_processed=len(page_images),
            page_images=page_images,
            metadata={
                "file_type": document.file_type,
                "filename": document.filename,
                "size_bytes": document.size_bytes,
            },
            errors=errors,
        )

    @staticmethod
    def create_document_model(document: UploadedDocument) -> Document:
        page_images = []
        if document.file_type == "pdf" and document.storage_path:
            processed_dir = Path(settings.storage_root) / "processed" / document.document_id
            page_images = [str(path) for path in sorted(processed_dir.glob("*.png"))]

        return Document(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            pages=list(range(1, document.page_count + 1)) if document.page_count else [],
            page_images=page_images,
            metadata=document.metadata,
        )
