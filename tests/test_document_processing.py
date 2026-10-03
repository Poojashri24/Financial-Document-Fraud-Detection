from io import BytesIO

import fitz
from PIL import Image
from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.main import app
from backend.services.document_service import DocumentService


def make_pdf_bytes(page_count: int = 2) -> bytes:
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice Test")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_png_bytes(size=(64, 64), color=(255, 0, 0)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg_bytes(size=(64, 64), color=(0, 255, 0)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_document_id_generation_is_unique():
    ids = {DocumentService.generate_document_id() for _ in range(20)}
    assert len(ids) == 20


def test_valid_pdf_page_count():
    pdf_bytes = make_pdf_bytes(3)
    page_count = DocumentService.get_pdf_page_count_bytes(pdf_bytes)
    assert page_count == 3


def test_valid_png_metadata():
    png_bytes = make_png_bytes()
    metadata = DocumentService.extract_image_metadata_bytes(png_bytes, "sample.png")
    assert metadata["format"] == "PNG"
    assert metadata["width"] == 64
    assert metadata["height"] == 64


def test_invalid_extension_is_rejected():
    try:
        DocumentService.validate_document_file("notes.txt", b"hello")
        raise AssertionError("Expected ValueError for invalid extension")
    except ValueError:
        pass


def test_oversized_file_is_rejected():
    original_limit = settings.max_upload_size_mb
    settings.max_upload_size_mb = 0.000001
    try:
        try:
            DocumentService.validate_file_size(b"x" * 50, "sample.pdf")
            raise AssertionError("Expected ValueError for oversized file")
        except ValueError:
            pass
    finally:
        settings.max_upload_size_mb = original_limit


def test_corrupted_file_is_rejected():
    try:
        DocumentService.validate_document_file("broken.pdf", b"not a real pdf file")
        raise AssertionError("Expected ValueError for corrupted PDF")
    except ValueError:
        pass


def test_upload_api_accepts_multiple_documents():
    client = TestClient(app)
    response = client.post(
        "/api/documents/upload",
        files=[
            ("files", ("sample.pdf", make_pdf_bytes(2), "application/pdf")),
            ("files", ("sample.png", make_png_bytes(), "image/png")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["file_type"] in {"pdf", "png"}


def test_processing_api_for_pdf_converts_pages_to_images():
    client = TestClient(app)
    upload_response = client.post(
        "/api/documents/upload",
        files=[("files", ("demo.pdf", make_pdf_bytes(2), "application/pdf"))],
    )
    document_id = upload_response.json()["documents"][0]["document_id"]

    response = client.post(f"/api/documents/{document_id}/process")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "processed"
    assert payload["page_count"] == 2
    assert payload["pages_processed"] == 2
    assert len(payload["page_images"]) == 2


def test_get_document_metadata_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/documents/upload",
        files=[("files", ("sample.jpg", make_jpeg_bytes(), "image/jpeg"))],
    )
    document_id = response.json()["documents"][0]["document_id"]

    detail_response = client.get(f"/api/documents/{document_id}")
    assert detail_response.status_code == 200
    payload = detail_response.json()
    assert payload["document_id"] == document_id
    assert payload["file_type"] == "jpg"
