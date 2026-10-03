from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.core.config import settings
from backend.services.document_service import DocumentService
from backend.services.tampering_feature_baseline_service import TamperingFeatureBaselineService
from backend.services.tampering_detection_service import TamperingDetectionService

router = APIRouter(prefix="/api/documents", tags=["tampering"])


def _find_document_images(document_id: str) -> list[str]:
    candidates: list[str] = []
    locations = [
        Path(settings.storage_root) / "processed" / document_id,
        Path(settings.storage_root) / "uploads" / document_id,
    ]
    for location in locations:
        if not location.exists():
            continue
        for candidate in location.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                candidates.append(str(candidate))
    return sorted(candidates)


def _add_trained_visual_evidence(result: dict, image_path: str) -> dict:
    classifier_path = Path("models/tampering_feature_baseline/logistic_regression.pkl")
    result["heuristic_evidence"] = {
        "detector_type": result["detector_type"],
        "tampering_detected": result["tampering_detected"],
        "findings": result["findings"],
        "risk_score": result["risk_score"],
        "limitations": result["limitations"],
        "evidence_source": "baseline_forensic_heuristic",
    }
    if classifier_path.exists():
        try:
            result["ml_evidence"] = TamperingFeatureBaselineService().predict_image(image_path, classifier_path)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            result["ml_evidence"] = {
                "status": "unavailable",
                "evidence_source": "trained_visual_model",
                "reason": f"Trained visual model could not be loaded in this runtime: {exc}",
            }
    else:
        result["ml_evidence"] = {
            "status": "unavailable",
            "evidence_source": "trained_visual_model",
            "reason": "The trained classifier artifact is not available.",
        }
    return result


@router.post("/{document_id}/tampering")
async def analyze_tampering(document_id: str) -> dict:
    document = DocumentService.get_document_record(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    image_candidates = _find_document_images(document_id)
    if not image_candidates:
        raise HTTPException(status_code=400, detail="No valid image was found for tampering analysis.")

    service = TamperingDetectionService(detector_type="baseline_forensic")
    result = service.analyze_document(document_id, image_path=image_candidates[0])
    return _add_trained_visual_evidence(result, image_candidates[0])


@router.get("/{document_id}/tampering")
async def get_tampering_analysis(document_id: str) -> dict:
    document = DocumentService.get_document_record(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    image_candidates = _find_document_images(document_id)
    if not image_candidates:
        raise HTTPException(status_code=404, detail="No valid image is available for this document.")

    service = TamperingDetectionService(detector_type="baseline_forensic")
    result = service.analyze_document(document_id, image_path=image_candidates[0])
    return _add_trained_visual_evidence(result, image_candidates[0])
