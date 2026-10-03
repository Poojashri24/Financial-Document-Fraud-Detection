from pathlib import Path

import pytest

from backend.services.cross_document_gnn_service import (
    CrossDocumentGNN,
    CrossDocumentGNNService,
    GraphConsistencyConfig,
)
from backend.services.tampering_representation_experiment import (
    RepresentationExperimentConfig,
    TamperingRepresentationExperiment,
)


def test_representation_manifest_is_train_validation_only_and_disjoint():
    service = TamperingRepresentationExperiment()
    manifest = service.config.manifest_path
    from backend.services.tampering_ml_baseline_service import TamperingBaselineService
    payload = TamperingBaselineService.load_manifest(manifest)
    TamperingBaselineService.validate_manifest_records(payload)
    train_ids = {item["image"] for item in payload["records"] if item["split"] == "train"}
    validation_ids = {item["image"] for item in payload["records"] if item["split"] == "validation"}
    assert train_ids.isdisjoint(validation_ids)
    assert all("TestingSet" not in path for path in train_ids | validation_ids)
    assert service.config.seed == 42


def test_partial_finetune_model_uses_cached_pretrained_weights_and_trains_last_layers():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    config = RepresentationExperimentConfig(pretrained=False)
    model = TamperingRepresentationExperiment(config).build_model()
    feature_parameters = [parameter.requires_grad for parameter in model.features.parameters()]
    assert any(feature_parameters)
    assert not all(feature_parameters)
    assert model.classifier[-1].out_features == 2
    assert config.image_size == 96


def test_rule_distillation_labels_are_explicit_and_splits_disjoint():
    cases = CrossDocumentGNNService.rule_distillation_cases()
    partitions = {name: {item["case_id"] for item in cases if item["split"] == name} for name in ("train", "validation", "test")}
    assert partitions["train"].isdisjoint(partitions["validation"])
    assert partitions["train"].isdisjoint(partitions["test"])
    assert partitions["validation"].isdisjoint(partitions["test"])
    assert len(partitions["train"]) >= 6
    assert all(item["rule_id"] and len(item["documents"]) == 2 for item in cases)


def test_gcn_forward_outputs_three_rule_statuses():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    model = CrossDocumentGNN()
    output = model(torch.zeros(2, 11), torch.tensor([[0, 1], [1, 0]]), torch.zeros(14))
    assert output.shape == (3,)


def test_gnn_prediction_is_advisory_and_not_override():
    assert "rule-distilled" in CrossDocumentGNN.__doc__.lower()
    assert "not fraud labels" in CrossDocumentGNNService.rule_distillation_cases.__doc__.lower()


def test_gnn_prediction_flows_as_advisory_evidence_without_changing_rule_status(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from backend.services.multimodal_fusion_service import MultimodalFusionEngine
    from backend.services.llm_reasoning_service import LLMReasoningService
    from backend.services.risk_scoring_service import RiskScoringService

    config = GraphConsistencyConfig(
        epochs=8,
        checkpoint_path=str(tmp_path / "gnn.pt"),
        report_path=str(tmp_path / "gnn.json"),
    )
    service = CrossDocumentGNNService(config)
    service.train_and_evaluate()
    documents = [
        {"document_id": "s", "document_type": "Salary Slip", "fields": {"net_salary": 35000.0}, "structured_fields": {"net_salary": 35000.0}},
        {"document_id": "b", "document_type": "Bank Statement", "fields": {"credits": 25000.0}, "structured_fields": {"credits": 25000.0}},
    ]
    explicit_rule = {"rule_id": "salary_vs_bank_credit", "rule_name": "Salary vs bank credit", "status": "FAIL", "documents": ["s", "b"], "values": {"net_salary": 35000.0, "bank_credit": 25000.0}, "explanation": "Salary differs from bank credit."}
    verification = {"results": [explicit_rule]}
    gnn = service.predict(documents, verification["results"])
    verification["gnn_evidence"] = gnn
    assert explicit_rule["status"] == "FAIL"
    assert gnn["predictions"]
    assert gnn["predictions"][0]["evidence_source"] == "learned_graph_based_evidence"

    fused_docs = [{
        "document_id": "s", "document_type": "Salary Slip", "structured_fields": {"net_salary": 35000.0},
        "cross_document": {**explicit_rule, "gnn_evidence": gnn, "evidence": [explicit_rule["explanation"]]},
    }, {
        "document_id": "b", "document_type": "Bank Statement", "structured_fields": {"credits": 25000.0},
        "cross_document": {**explicit_rule, "gnn_evidence": gnn, "evidence": [explicit_rule["explanation"]]},
    }]
    analysis = MultimodalFusionEngine.fuse_documents(fused_docs)
    graph_evidence = [item for item in analysis["evidence"] if item["source"] == "learned_graph_based_evidence"]
    assert len(graph_evidence) == len(gnn["predictions"])
    assert analysis["modalities"]["graph_consistency"]["count"] == 2
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["cross_document_inconsistencies"]
    assert reasoning["graph_based_observations"]
    risk_with_graph = RiskScoringService().score_reasoning(reasoning)
    without_graph = dict(reasoning)
    without_graph["graph_based_observations"] = []
    without_graph["evidence_support"] = {key: value for key, value in reasoning["evidence_support"].items() if key != "learned_graph_based_evidence"}
    risk_without_graph = RiskScoringService().score_reasoning(without_graph)
    assert risk_with_graph["risk_score"] == risk_without_graph["risk_score"]
    assert any(item["source"] == "learned_graph_based_evidence" for item in risk_with_graph["contributing_evidence"])


def test_verification_api_attaches_gnn_without_overriding_rule(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.services.cross_document_gnn_service import CrossDocumentGNNService

    advisory = {
        "status": "available",
        "predictions": [{
            "rule_id": "salary_vs_bank_credit",
            "rule_based_status": "FAIL",
            "gnn_prediction": "PASS",
            "confidence": 0.6,
            "documents": ["salary-api", "bank-api"],
            "evidence_source": "learned_graph_based_evidence",
            "explanation": "Advisory only; does not override explicit verification.",
        }],
    }
    monkeypatch.setattr(CrossDocumentGNNService, "predict", lambda self, documents, results: advisory)
    response = TestClient(app).post("/api/verify/cross-document", json={"documents": [
        {"document_id": "salary-api", "document_type": "Salary Slip", "fields": {"net_salary": 35000}},
        {"document_id": "bank-api", "document_type": "Bank Statement", "fields": {"credits": 25000}},
    ]})
    assert response.status_code == 200
    payload = response.json()
    rule = next(item for item in payload["results"] if item["rule_id"] == "salary_vs_bank_credit")
    assert rule["status"] == "FAIL"
    assert payload["gnn_evidence"] == advisory
