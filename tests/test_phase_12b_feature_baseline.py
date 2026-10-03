from pathlib import Path

import pytest

from backend.services.tampering_feature_baseline_service import (
    FeatureBaselineConfig,
    TamperingFeatureBaselineService,
)


MANIFEST = Path("reports/phase12b1_train_validation_split.json")


def test_feature_manifest_records_and_label_mapping():
    base = TamperingFeatureBaselineService()
    manifest = base.load_manifest(MANIFEST)
    records = __import__("backend.services.tampering_ml_baseline_service", fromlist=["TamperingBaselineService"]).TamperingBaselineService.resolve_records(manifest)
    assert len(records) == 3248
    assert {record["label_id"] for record in records} == {0, 1}
    assert all(record.get("mask_path") for record in records if record["label_id"] == 1)
    assert all(record.get("mask_path") is None for record in records if record["label_id"] == 0)


def test_feature_dataset_construction_uses_existing_manifest():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    service = TamperingFeatureBaselineService(FeatureBaselineConfig(image_size=32))
    manifest = service.load_manifest(MANIFEST)
    records = __import__("backend.services.tampering_ml_baseline_service", fromlist=["TamperingBaselineService"]).TamperingBaselineService.resolve_records(manifest)
    train = service.build_dataset(records, "train", image_size=32)
    validation = service.build_dataset(records, "validation", image_size=32)
    assert len(train) == 2598
    assert len(validation) == 650
    image, label, path = train[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert label in {0, 1}
    assert path.startswith("TrainingSet/")


def test_frozen_feature_extractor_shape_and_freezing():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    service = TamperingFeatureBaselineService()
    extractor = service.build_feature_extractor(pretrained=False)
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    import torch
    output = extractor(torch.zeros(2, 3, 32, 32))
    assert output.ndim == 2
    assert output.shape[0] == 2


def test_classifier_and_metrics_are_realistic():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import torch
    service = TamperingFeatureBaselineService()
    classifier = service.train_classifier(torch.tensor([[0.0], [0.1], [1.0], [1.1]]), [0, 0, 1, 1])
    probabilities = classifier.predict_proba([[0.0], [0.1], [1.0], [1.1]])[:, 1].tolist()
    predictions = classifier.predict([[0.0], [0.1], [1.0], [1.1]]).tolist()
    metrics = service.calculate_metrics([0, 0, 1, 1], predictions, probabilities)
    assert metrics["sample_count"] == 4
    assert set(metrics["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert metrics["roc_auc"] is not None


def test_feature_artifact_paths_are_explicit():
    config = FeatureBaselineConfig()
    assert config.manifest_path == "reports/phase12b1_train_validation_split.json"
    assert config.artifact_dir.startswith("models/")
    assert config.seed == 42
