from pathlib import Path

import numpy as np
from PIL import Image

from backend.services.tampering_detection_service import BaselineTamperingDetector, DocTamperDatasetAdapter, TamperingDetectionService


def _make_image(path: Path, size=(600, 800), noise=False):
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 245
    if noise:
        rng = np.random.default_rng(7)
        noise_map = rng.integers(0, 30, size=image.shape, dtype=np.uint8)
        image = np.clip(image + noise_map, 0, 255).astype(np.uint8)
    image = Image.fromarray(image, mode="RGB")
    image.save(path)


def test_baseline_detector_on_normal_document_image(tmp_path):
    image_path = tmp_path / "normal.png"
    _make_image(image_path)
    findings = BaselineTamperingDetector.detect_document(str(image_path))
    assert isinstance(findings, list)


def test_baseline_detector_with_local_modification(tmp_path):
    image_path = tmp_path / "modified.png"
    _make_image(image_path)
    image = Image.open(image_path).convert("RGB")
    modified = image.copy()
    for x in range(150, 260):
        for y in range(150, 260):
            modified.putpixel((x, y), (40, 40, 40))
    modified.save(image_path)

    findings = BaselineTamperingDetector.detect_document(str(image_path))
    assert isinstance(findings, list)


def test_baseline_detector_on_noisy_image(tmp_path):
    image_path = tmp_path / "noisy.png"
    _make_image(image_path, noise=True)
    findings = BaselineTamperingDetector.detect_document(str(image_path))
    assert isinstance(findings, list)


def test_baseline_detector_on_blank_image(tmp_path):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(image_path)
    findings = BaselineTamperingDetector.detect_document(str(image_path))
    assert isinstance(findings, list)


def test_baseline_detector_on_corrupted_image(tmp_path):
    image_path = tmp_path / "corrupted.png"
    image_path.write_bytes(b"not-a-valid-image")
    findings = BaselineTamperingDetector.detect_document(str(image_path))
    assert findings == []


def test_tampering_service_returns_structured_response(tmp_path):
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    document_dir = Path("data") / "uploads" / "tamper-doc-1"
    document_dir.mkdir(parents=True, exist_ok=True)
    image_path2 = document_dir / "page_1.png"
    image_path2.write_bytes(image_path.read_bytes())

    result = TamperingDetectionService(detector_type="baseline_forensic").analyze_document("tamper-doc-1", image_path=str(image_path2))
    assert result["document_id"] == "tamper-doc-1"
    assert result["detector_type"] == "baseline_forensic"
    assert "findings" in result
    assert "suspicious_regions" in result
    assert isinstance(result["tampering_detected"], bool)
    assert isinstance(result["limitations"], list)


def test_tampering_detector_finding_schema():
    image_path = Path("data") / "tmp_schema_test.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _make_image(image_path)
    findings = BaselineTamperingDetector.detect_document(str(image_path))
    if findings:
        item = findings[0]
        assert "detector" in item
        assert "finding" in item
        assert "severity" in item
        assert "explanation" in item
        if item.get("region") is not None:
            assert len(item["region"]) == 4
    image_path.unlink(missing_ok=True)


def test_doctamper_unavailable_behavior():
    adapter = DocTamperDatasetAdapter(root=str(Path("data/raw")))
    assert adapter.available() in {True, False}
    summary = adapter.describe()
    assert "available" in summary
    assert "structure" in summary


def test_tampering_service_missing_document_raises():
    try:
        TamperingDetectionService().analyze_document("missing-document-id", image_path="missing.png")
    except FileNotFoundError:
        return
    assert False, "Expected FileNotFoundError for missing document image" 
