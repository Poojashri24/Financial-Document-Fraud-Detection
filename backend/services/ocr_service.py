from __future__ import annotations

import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from backend.core.config import settings


class OCRService:
    @staticmethod
    def extract_text_from_image_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            with Image.open(BytesIO(file_bytes)) as img:
                img.load()
                with tempfile.TemporaryDirectory() as temp_dir:
                    source_path = Path(temp_dir) / (Path(filename).stem + ".png")
                    img.save(source_path)
                    result = subprocess.run(
                        [
                            "tesseract",
                            str(source_path),
                            "stdout",
                            "--psm",
                            "6",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode != 0:
                        return {
                            "status": "error",
                            "text": "",
                            "error": result.stderr.strip() or "OCR process failed.",
                        }
                    return {
                        "status": "success",
                        "text": result.stdout.strip(),
                        "filename": filename,
                    }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "error",
                "text": "",
                "error": str(exc),
            }

    @staticmethod
    def extract_text_from_document(document: Dict[str, Any]) -> Dict[str, Any]:
        if document.get("file_type") == "pdf":
            image_paths = document.get("page_images") or []
            document_id = document.get("document_id")
            if not image_paths and document_id:
                processed_dir = Path(settings.storage_root) / "processed" / str(document_id)
                if processed_dir.exists():
                    image_paths = [str(path) for path in sorted(processed_dir.glob("*.png"))]

            texts: List[str] = []
            for image_path in image_paths:
                path = Path(image_path)
                if path.exists():
                    result = subprocess.run(
                        [
                            "tesseract",
                            str(path),
                            "stdout",
                            "--psm",
                            "6",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        texts.append(result.stdout.strip())
            if texts:
                return {
                    "status": "success",
                    "text": "\n".join(texts),
                    "pages": len(image_paths),
                }
            return {
                "status": "error",
                "text": "",
                "error": "No OCR text could be extracted from the rendered PDF pages.",
            }

        if document.get("storage_path"):
            path = Path(document["storage_path"])
            if path.exists():
                with open(path, "rb") as handle:
                    return OCRService.extract_text_from_image_bytes(handle.read(), path.name)

        return {
            "status": "error",
            "text": "",
            "error": "No readable document content available for OCR.",
        }
