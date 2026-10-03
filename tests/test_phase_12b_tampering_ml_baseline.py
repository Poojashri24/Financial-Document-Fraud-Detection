import json
from pathlib import Path

import pytest

from backend.services.tampering_ml_baseline_service import (
    LABEL_MAP,
    TamperingBaselineService,
    TamperingTrainingConfig,
)


MANIFEST = Path("reports/phase12b1_train_validation_split.json")


def test_manifest_loading_and_label_mapping():
    service = TamperingBaselineService()
    manifest = service.load_manifest(MANIFEST)
    counts = service.validate_manifest_records(manifest)
    assert LABEL_MAP == {"authentic": 0, "forged": 1}
    assert counts["total"] == 3248
    assert counts["train_authentic"] == 1299
    assert counts["train_forged"] == 1299
    assert service.label_for_record({"label": "authentic"}) == 0
    assert service.label_for_record({"label": "forged"}) == 1


def test_manifest_records_resolve_to_images_and_masks():
    service = TamperingBaselineService()
    records = service.resolve_records(service.load_manifest(MANIFEST))
    assert len(records) == 3248
    authentic = next(item for item in records if item["label"] == "authentic")
    forged = next(item for item in records if item["label"] == "forged")
    assert authentic["label_id"] == 0
    assert authentic.get("mask_path") is None
    assert forged["label_id"] == 1
    assert Path(forged["image_path"]).exists()
    assert Path(forged["mask_path"]).exists()


def test_dataset_construction_uses_manifest_split():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    service = TamperingBaselineService()
    records = service.resolve_records(service.load_manifest(MANIFEST))
    train_dataset = service.build_dataset(records, "train", image_size=32)
    validation_dataset = service.build_dataset(records, "validation", image_size=32)
    assert len(train_dataset) == 2598
    assert len(validation_dataset) == 650
    image, label, path = train_dataset[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert label in {0, 1}
    assert path.startswith("TrainingSet/")


def test_model_construction_freezes_backbone_and_has_two_classes():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    service = TamperingBaselineService()
    model = service.build_model(pretrained=False)
    assert model.classifier[-1].out_features == 2
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())


def test_checkpoint_path_and_configuration_are_explicit():
    config = TamperingTrainingConfig()
    assert config.manifest_path == "reports/phase12b1_train_validation_split.json"
    assert config.model_dir.startswith("models/")
    assert config.backbone == "mobilenet_v3_small"
    assert config.epochs == 2


def test_metric_calculation_returns_binary_metrics_and_confusion_matrix():
    result = TamperingBaselineService.calculate_metrics(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        probabilities=[0.1, 0.8, 0.7, 0.9],
    )
    assert result["accuracy"] == 0.75
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["f1_score"] == pytest.approx(0.8)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["roc_auc"] is not None


def test_invalid_manifest_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dataset": "SROIE", "records": [{"label": "forged"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        TamperingBaselineService.load_manifest(path)
