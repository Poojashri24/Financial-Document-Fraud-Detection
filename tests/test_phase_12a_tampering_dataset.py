import json
from pathlib import Path

import pytest
from PIL import Image

from backend.services.tampering_dataset_service import DatasetSpec, TamperingDatasetService


def make_image(path: Path, color: int = 255, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=color).save(path)


def make_service(root: Path, *, expects_masks: bool = True, official_splits: bool = False) -> TamperingDatasetService:
    return TamperingDatasetService([
        DatasetSpec(
            name="local-tampering",
            dataset_type="document_tampering",
            local_path=str(root),
            image_directory="images",
            mask_directory="masks",
            label_format="binary image and mask",
            expects_masks=expects_masks,
            official_splits=official_splits,
            suitable_for_training=True,
            suitable_for_evaluation=True,
        )
    ])


def test_unavailable_dataset_is_reported_without_download(tmp_path):
    service = TamperingDatasetService()
    status = service.status("DocTamper")
    assert status["status"] == "NOT_AVAILABLE"
    assert "local dataset path" in status["reason"]


def test_default_registry_distinguishes_sroie_from_tampering():
    status = TamperingDatasetService().status("SROIE")
    assert status["dataset_type"] == "document_understanding"
    assert status["tampering_dataset"] is False
    assert status["fraud_labels"] is False


def test_valid_dataset_discovery_and_validation(tmp_path):
    make_image(tmp_path / "images" / "doc-1.png")
    make_image(tmp_path / "masks" / "doc-1.png", color=0)
    result = make_service(tmp_path).validate("local-tampering")
    assert result["valid"] is True
    assert result["records"][0]["document_id"] == "doc-1"
    assert result["records"][0]["mask_path"]


def test_invalid_path_is_reported(tmp_path):
    service = make_service(tmp_path / "missing")
    result = service.validate("local-tampering")
    assert result["valid"] is False
    assert result["errors"]


def test_image_mask_mismatch_is_reported(tmp_path):
    make_image(tmp_path / "images" / "doc-1.png")
    make_image(tmp_path / "masks" / "other.png", color=0)
    result = make_service(tmp_path).validate("local-tampering")
    assert result["valid"] is False
    assert any("image/mask mismatch" in error for error in result["errors"])


def test_duplicate_identifiers_are_reported(tmp_path):
    make_image(tmp_path / "images" / "doc-1.png")
    make_image(tmp_path / "images" / "doc-1.jpg")
    make_image(tmp_path / "masks" / "doc-1.png", color=0)
    result = make_service(tmp_path).validate("local-tampering")
    assert result["valid"] is False
    assert any("duplicate identifier" in error for error in result["errors"])


def test_unsupported_image_format_is_reported(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "images" / "doc-1.gif").write_bytes(b"not an accepted dataset image")
    result = make_service(tmp_path).validate("local-tampering")
    assert result["valid"] is False
    assert any("unsupported image formats" in error for error in result["errors"])


def test_invalid_mask_values_are_reported(tmp_path):
    make_image(tmp_path / "images" / "doc-1.png")
    make_image(tmp_path / "masks" / "doc-1.png", color=128)
    result = make_service(tmp_path).validate("local-tampering")
    assert result["valid"] is False
    assert any("unexpected values" in error for error in result["errors"])


def test_deterministic_splits_and_no_overlap(tmp_path):
    records = [{"document_id": f"doc-{index}", "path": f"doc-{index}.png", "label": index % 2} for index in range(10)]
    service = make_service(tmp_path, expects_masks=False)
    first = service.prepare_splits("local-tampering", records=records, random_seed=17)
    second = service.prepare_splits("local-tampering", records=records, random_seed=17)
    assert first["splits"] == second["splits"]
    assert first["no_document_leakage"] is True
    assert set(first["splits"]["train"]).isdisjoint(first["splits"]["test"])


def test_split_information_can_be_persisted(tmp_path):
    records = [{"document_id": f"doc-{index}", "path": f"doc-{index}.png", "label": index % 2} for index in range(5)]
    output = tmp_path / "splits.json"
    prepared = make_service(tmp_path, expects_masks=False).prepare_splits("local-tampering", records, output_path=str(output))
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["splits"] == prepared["splits"]


def test_official_split_overlap_is_reported(tmp_path):
    make_image(tmp_path / "train" / "doc-1.png")
    make_image(tmp_path / "test" / "doc-1.png")
    service = make_service(tmp_path, expects_masks=False, official_splits=True)
    result = service.validate("local-tampering")
    assert result["valid"] is False
    assert any("split overlap" in error for error in result["errors"])


def test_dataset_statistics_include_masks_and_labels(tmp_path):
    records = [
        {"document_id": "auth", "path": "auth.png", "label": 0, "mask_path": None, "split": "train"},
        {"document_id": "tampered", "path": "tampered.png", "label": 1, "mask_path": str(tmp_path / "mask.png"), "split": "test"},
    ]
    make_image(tmp_path / "mask.png", color=255)
    stats = make_service(tmp_path, expects_masks=False).statistics("local-tampering", records)
    assert stats["total_images"] == 2
    assert stats["authentic_count"] == 1
    assert stats["tampered_count"] == 1
    assert stats["train_count"] == 1
    assert stats["test_count"] == 1
    assert stats["mask_availability"] == 1
    assert stats["tampered_pixel_fraction_mean"] == 1.0
