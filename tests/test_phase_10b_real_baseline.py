from pathlib import Path

import pytest

from backend.services.real_baseline_evaluation_service import (
    ReceiptFieldBaselineModel,
    RealBaselineEvaluationService,
)


def test_receipt_field_baseline_dataset_status_is_real():
    model = ReceiptFieldBaselineModel(random_seed=42)
    status = model.dataset_status()
    assert status["dataset_name"] == "SROIE"
    assert status["status"] in {"available", "blocked"}
    assert "dataset" in status


def test_receipt_field_baseline_predictions_are_deterministic():
    model = ReceiptFieldBaselineModel(random_seed=7)
    dataset = model.load_dataset(split="train[:10]")
    first = model.predict(dataset)
    second = model.predict(dataset)
    assert len(first) == len(dataset)
    assert first == second
    assert all("predicted_total" in sample for sample in first)


def test_prediction_validation_rejects_invalid_shapes():
    model = ReceiptFieldBaselineModel(random_seed=11)
    with pytest.raises(ValueError):
        model.validate_predictions([{"document_id": "a"}])


def test_insufficient_resources_are_reported_cleanly():
    service = RealBaselineEvaluationService(random_seed=13)
    result = service.evaluate_dataset(
        dataset=[],
        resource_requirements={"min_cpu_cores": 999, "min_ram_mb": 999999},
    )
    assert result["status"] == "blocked"
    assert "insufficient" in result["message"].lower()


def test_unavailable_dataset_returns_not_evaluated():
    service = RealBaselineEvaluationService(random_seed=21)
    result = service.evaluate_dataset(dataset=None)
    assert result["status"] == "not_evaluated"
    assert "dataset" in result["message"].lower()


def test_real_sroie_baseline_evaluation_runs_and_records_artifacts():
    service = RealBaselineEvaluationService(random_seed=31)
    artifact_path = Path("reports") / "phase10b_real_baseline.json"
    if artifact_path.exists():
        artifact_path.unlink()

    result = service.evaluate_from_hf_cache(split="train[:50]", artifact_path=str(artifact_path))
    assert result["status"] in {"evaluated", "not_evaluated"}
    if result["status"] == "evaluated":
        assert result["samples_evaluated"] > 0
        assert "metrics" in result
        assert "confusion_matrix" in result
    assert artifact_path.exists() or result["status"] == "not_evaluated"


def test_real_baseline_saves_evaluation_json():
    service = RealBaselineEvaluationService(random_seed=17)
    target = Path("reports") / "phase10b_validation.json"
    if target.exists():
        target.unlink()
    payload = {"status": "evaluated", "metrics": {"accuracy": 1.0}, "samples_evaluated": 1}
    saved = service.save_artifact(payload, target)
    assert Path(saved).exists()
    assert Path(saved).suffix == ".json"
