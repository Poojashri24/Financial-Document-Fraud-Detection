from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.ocr_service import OCRService


def make_text_image_bytes(text: str = "INVOICE 1234") -> bytes:
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 60), text, fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_ocr_service_extracts_text_from_image_bytes():
    image_bytes = make_text_image_bytes("INVOICE 1234")
    result = OCRService.extract_text_from_image_bytes(image_bytes, "sample.png")
    assert result["status"] == "success"
    assert "INVOICE" in result["text"].upper()


def test_ocr_endpoint_for_uploaded_image():
    client = TestClient(app)
    upload_response = client.post(
        "/api/documents/upload",
        files=[("files", ("invoice.png", make_text_image_bytes("INVOICE 1234"), "image/png"))],
    )
    document_id = upload_response.json()["documents"][0]["document_id"]

    response = client.post(f"/api/documents/{document_id}/ocr")
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == document_id
    assert payload["status"] == "success"
    assert "INVOICE" in payload["text"].upper()
