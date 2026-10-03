from pathlib import Path

import pytest

from backend.services.dataset_preparation_service import DatasetPreparationService, SROIEDatasetAdapter
from backend.services.evaluation_service import EvaluationService


@pytest.fixture
def sample_records():
    return [
        {"document_id": "doc-1", "label": 1, "source": "sroie", "path": "a1.png", "metadata": {"page_count": 1}},
        {"document_id": "doc-2", "label": 0, "source": "sroie", "path": "a2.png", "metadata": {"page_count": 1}},
        {"document_id": "doc-3", "label": 1, "source": "sroie", "path": "a3.png", "metadata": {"page_count": 1}},
        {"document_id": "doc-4", "label": 0, "source": "sroie", "path": "a4.png", "metadata": {"page_count": 1}},
    ]


def test_sroie_dataset_adapter_detects_available_or_unavailable_status():
    adapter = SROIEDatasetAdapter(root=str(Path("data/raw/sroie")))
    status = adapter.describe()
    assert "status" in status
    assert status["dataset_name"] == "SROIE"


def test_dataset_preparation_pipeline_loads_samples(sample_records):
    prepared = DatasetPreparationService.prepare_records(sample_records, dataset_name="demo_dataset", random_seed=12)
    assert prepared["sample_count"] == 4
    assert prepared["split_sizes"]["train"] >= 1
    assert prepared["reproducibility"]["random_seed"] == 12


def test_dataset_preparation_rejects_missing_or_malformed_samples():
    with pytest.raises(ValueError):
        DatasetPreparationService.prepare_records([{"document_id": "bad"}], dataset_name="demo_dataset")


def test_split_has_no_overlap(sample_records):
    prepared = DatasetPreparationService.prepare_records(sample_records, dataset_name="demo_dataset", random_seed=7)
    train_ids = set(prepared["splits"]["train"])
    val_ids = set(prepared["splits"]["validation"])
    test_ids = set(prepared["splits"]["test"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_metric_calculation_works_for_binary_labels():
    metrics = EvaluationService.evaluate_predictions(
        labels=[1, 0, 1, 0],
        predictions=[1, 0, 0, 0],
        probabilities=[0.9, 0.2, 0.4, 0.1],
    )
    assert metrics["status"] == "evaluated"
    assert "accuracy" in metrics["metrics"]
    assert "precision" in metrics["metrics"]
    assert "recall" in metrics["metrics"]
    assert "f1_score" in metrics["metrics"]
    assert "roc_auc" in metrics["metrics"]


def test_no_positive_samples_returns_not_evaluated():
    metrics = EvaluationService.evaluate_predictions(labels=[0, 0, 0], predictions=[0, 0, 0])
    assert metrics["status"] == "not_evaluated"
    assert "No positive samples" in metrics["message"]


def test_invalid_predictions_raise_value_error():
    with pytest.raises(ValueError):
        EvaluationService.evaluate_predictions(labels=[0, 1], predictions=[0])


def test_doc_tamper_unavailable_status_is_clear():
    from backend.services.tampering_detection_service import DocTamperDatasetAdapter

    adapter = DocTamperDatasetAdapter(root=str(Path("data/raw")))
    status = adapter.describe()
    assert status["available"] is False
    assert "not found locally" in status["structure"].lower()


def test_reproducibility_metadata_present():
    prepared = DatasetPreparationService.prepare_records([
        {"document_id": "doc-a", "label": 1, "source": "synthetic", "path": "a.png", "metadata": {"page_count": 1}},
        {"document_id": "doc-b", "label": 0, "source": "synthetic", "path": "b.png", "metadata": {"page_count": 1}},
        {"document_id": "doc-c", "label": 1, "source": "synthetic", "path": "c.png", "metadata": {"page_count": 1}},
    ], dataset_name="synthetic", random_seed=13)
    assert prepared["reproducibility"]["dataset_name"] == "synthetic"
    assert prepared["reproducibility"]["random_seed"] == 13
    assert prepared["reproducibility"]["preprocessing"]


def test_dataset_status_distinguishes_available_and_unavailable_sources():
    status = DatasetPreparationService.dataset_status(root=str(Path("data/raw")))
    assert "status" in status
    assert "dataset_name" in status
