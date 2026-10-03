import json
from pathlib import Path

import pytest

from backend.services.tampering_feature_baseline_service import (
    FeatureBaselineConfig,
    TamperingFeatureBaselineService,
)


class FakeClassifier:
    def predict_proba(self, features):
        import numpy as np
        return np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

    def predict(self, features):
        import numpy as np
        return np.asarray([0, 1, 0, 1])


def write_testing_manifest(path: Path) -> None:
    path.write_text(json.dumps({
        "dataset": "AIForge-Doc-v1",
        "source_split": "TestingSet",
        "source_root": "data/raw/AIForge-Doc-v1-testing",
        "records": [
            {"split": "test", "label": "authentic", "image": "TestingSet/authentic/a.png", "mask": None},
            {"split": "test", "label": "forged", "image": "TestingSet/images/1.png", "mask": "TestingSet/masks/1.png"},
            {"split": "test", "label": "authentic", "image": "TestingSet/authentic/b.png", "mask": None},
            {"split": "test", "label": "forged", "image": "TestingSet/images/2.png", "mask": "TestingSet/masks/2.png"},
        ],
    }), encoding="utf-8")


def test_testing_manifest_requires_testing_set_source(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"dataset": "AIForge-Doc-v1", "source_split": "TrainingSet only", "records": [{"label": "authentic"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="TestingSet"):
        TamperingFeatureBaselineService._load_evaluation_manifest(path)


def test_evaluate_manifest_uses_saved_classifier_without_fitting(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import torch
    import backend.services.tampering_feature_baseline_service as module

    manifest_path = tmp_path / "testing.json"
    classifier_path = tmp_path / "classifier.pkl"
    report_path = tmp_path / "testing_report.json"
    write_testing_manifest(manifest_path)
    classifier_path.write_bytes(b"saved-classifier")

    service = TamperingFeatureBaselineService(FeatureBaselineConfig(batch_size=64, image_size=64))
    monkeypatch.setattr(service, "build_dataset", lambda records, split, image_size: object())
    monkeypatch.setattr(service, "build_feature_extractor", lambda pretrained: object())
    monkeypatch.setattr(service, "extract_features", lambda extractor, dataset, cache_path: {
        "features": torch.zeros((4, 576)),
        "labels": [0, 1, 0, 1],
        "image_paths": ["a", "b", "c", "d"],
    })
    monkeypatch.setattr(module.pickle, "load", lambda handle: FakeClassifier())

    report = service.evaluate_manifest(manifest_path, classifier_path, report_path=report_path)

    assert report["dataset"] == "AIForge-Doc-v1"
    assert report["split"] == "TestingSet"
    assert report["classifier_fitted"] is False
    assert report["preprocessing"]["image_size"] == [64, 64]
    assert report["preprocessing"]["batch_size"] == 64
    assert report["sample_count"] == 4
    assert report["metrics"]["accuracy"] == 1.0
    assert report_path.exists()
