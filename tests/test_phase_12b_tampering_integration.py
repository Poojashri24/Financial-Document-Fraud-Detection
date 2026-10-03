from pathlib import Path

import pytest
from PIL import Image

from backend.services.llm_reasoning_service import LLMReasoningService
from backend.services.multimodal_fusion_service import MultimodalFusionEngine
from backend.services.risk_scoring_service import RiskScoringService
from backend.services.tampering_feature_baseline_service import TamperingFeatureBaselineService
from backend.services.cross_document_verification_service import CrossDocumentVerificationService
from frontend.integration import build_consistency_graph, build_multimodal_document_payload, format_verification_values


CLASSIFIER = Path("models/tampering_feature_baseline/logistic_regression.pkl")


def heuristic_result():
    return {
        "document_id": "doc-ml-1",
        "detector_type": "baseline_forensic",
        "tampering_detected": True,
        "risk_score": 2,
        "findings": [{"finding": "suspicious texture change", "severity": "MEDIUM", "region": None}],
        "limitations": ["Heuristic suspicion is not proof of fraud."],
    }


def ml_result():
    return {
        "prediction": "forged",
        "forged_probability": 0.81,
        "confidence": 0.81,
        "model_name": "AIForge-Doc-v1 MobileNetV3-Small feature baseline",
        "model_version": {"backbone": "mobilenet_v3_small", "image_size": 64},
        "evidence_source": "trained_visual_model",
        "calibration_note": "Model output is not proof of fraud.",
    }


def test_saved_classifier_loads_and_predicts_on_valid_image(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("sklearn")
    image_path = tmp_path / "document.png"
    Image.new("RGB", (96, 96), color="white").save(image_path)
    result = TamperingFeatureBaselineService().predict_image(image_path, CLASSIFIER)
    assert result["prediction"] in {"authentic", "forged"}
    assert 0.0 <= result["forged_probability"] <= 1.0
    assert result["evidence_source"] == "trained_visual_model"
    assert result["model_version"]["image_size"] == 64


def test_heuristic_and_ml_sources_are_preserved_in_payload(monkeypatch):
    import backend.api.routes.tampering as route

    result = heuristic_result()
    monkeypatch.setattr(route.TamperingFeatureBaselineService, "predict_image", lambda self, image, classifier: ml_result())
    enriched = route._add_trained_visual_evidence(result, "unused.png")
    assert enriched["tampering_detected"] is True
    assert enriched["heuristic_evidence"]["evidence_source"] == "baseline_forensic_heuristic"
    assert enriched["ml_evidence"]["evidence_source"] == "trained_visual_model"


def test_fusion_contains_both_tampering_sources():
    analysis = MultimodalFusionEngine.fuse_documents([{
        "document_id": "doc-ml-1",
        "document_type": "Invoice",
        "ocr": {"status": "success", "text": "Invoice"},
        "structured_fields": {"total": 10.0},
        "tampering": {**heuristic_result(), "ml_evidence": ml_result()},
        "metadata": {"page_count": 1},
    }])
    sources = {item["source"] for item in analysis["evidence"]}
    assert "tampering" in sources
    assert "trained_visual_model" in sources
    assert analysis["modalities"]["tampering"]["documents"] == ["doc-ml-1"]
    assert analysis["modalities"]["tampering_ml"]["documents"] == ["doc-ml-1"]


def test_reasoning_identifies_trained_model_evidence_separately():
    analysis = MultimodalFusionEngine.fuse_documents([{
        "document_id": "doc-ml-2",
        "ocr": {"status": "success", "text": "Document"},
        "tampering": {**heuristic_result(), "ml_evidence": ml_result()},
    }])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["trained_visual_model_observations"][0]["evidence_source"] == "trained_visual_model"
    assert "trained_visual_model" in reasoning["evidence_support"]
    assert "not a definitive fraud verdict" in reasoning["reasoning"]


def test_risk_accepts_model_evidence_without_treating_probability_as_fraud_probability():
    reasoning = {
        "analysis_id": "a-ml",
        "reasoning_id": "r-ml",
        "observed_findings": [],
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [],
        "trained_visual_model_observations": [{
            "document_ids": ["doc-ml-3"],
            "summary": "Trained model predicted forged.",
            "prediction": "forged",
            "forged_probability": 0.81,
            "evidence_source": "trained_visual_model",
        }],
        "evidence_support": {"trained_visual_model": ["Trained model predicted forged."]},
        "traceability": {"document_count": 1, "modalities_present": ["tampering_ml"], "source_evidence_count": 1},
        "reasoning": "The model output is evidence only.",
        "limitations": [],
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    model_items = [item for item in assessment["contributing_evidence"] if item["source"] == "trained_visual_model"]
    assert model_items
    assert model_items[0]["traceability"]["model_output"]["forged_probability"] == 0.81
    assert assessment["methodology"]["score_type"] == "heuristic_baseline"


def test_weak_heuristic_alone_does_not_become_critical():
    reasoning = {
        "analysis_id": "weak-heuristic",
        "reasoning_id": "weak-heuristic-r",
        "observed_findings": [{"summary": "Suspicious texture change", "document_ids": ["doc-weak"], "severity": "HIGH", "evidence": ["texture"]}],
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [{"document_ids": ["doc-weak"], "summary": "Heuristic texture signal", "severity": "MEDIUM", "details": {}}],
        "trained_visual_model_observations": [],
        "evidence_support": {"tampering": ["texture"]},
        "traceability": {"document_count": 1, "modalities_present": ["tampering"], "source_evidence_count": 1},
        "reasoning": "Heuristic suspicion only.",
        "limitations": [],
    }
    assessment = RiskScoringService().score_reasoning(reasoning)
    assert assessment["risk_level"] != "CRITICAL"
    assert assessment["methodology"]["tampering_note"]


def test_salary_bank_mismatch_reaches_fusion_reasoning_and_risk():
    verification = CrossDocumentVerificationService.verify_documents([
        {"document_id": "salary", "document_type": "Salary Slip", "fields": {"employee_name": "Arun Kumar", "net_salary": 35000.0}},
        {"document_id": "bank", "document_type": "Bank Statement", "fields": {"account_holder": "Arun Kumar", "credits": 25000.0}},
    ])
    salary_rule = next(item for item in verification["results"] if item["rule_id"] == "salary_vs_bank_credit")
    assert salary_rule["status"] == "FAIL"
    cross = {**salary_rule, "documents": ["salary", "bank"]}
    documents = [{
        "document_id": "salary",
        "document_type": "Salary Slip",
        "ocr": {"status": "success", "text": "Net Salary 35000"},
        "structured_fields": {"net_salary": 35000.0},
        "cross_document": cross,
        "tampering": {**heuristic_result(), "ml_evidence": ml_result()},
    }, {
        "document_id": "bank",
        "document_type": "Bank Statement",
        "ocr": {"status": "success", "text": "Salary Credit 25000"},
        "structured_fields": {"credits": 25000.0},
        "cross_document": cross,
        "tampering": {**heuristic_result(), "ml_evidence": ml_result()},
    }]
    analysis = MultimodalFusionEngine.fuse_documents(documents)
    assert any(
        "differs from" in item["details"].get("explanation", "")
        for item in analysis["evidence"]
        if item["source"] == "cross_document"
    )
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["cross_document_inconsistencies"]
    risk = RiskScoringService().score_reasoning(reasoning)
    assert risk["inconsistencies"]
    assert any(item["source"] == "cross_document" for item in risk["contributing_evidence"])


def test_frontend_payload_preserves_both_sources():
    payload = build_multimodal_document_payload({
        "document_id": "doc-ml-4",
        "document_type": "Invoice",
        "tampering": {**heuristic_result(), "ml_evidence": ml_result()},
    })
    assert payload["tampering"]["tampering_detected"] is True
    assert payload["tampering"]["ml_evidence"]["evidence_source"] == "trained_visual_model"


def test_matching_pass_is_not_recorded_as_cross_document_inconsistency():
    analysis = MultimodalFusionEngine.fuse_documents([{
        "document_id": "salary",
        "cross_document": {
            "rule_id": "salary_vs_bank_credit",
            "rule_name": "Salary vs bank credit",
            "status": "PASS",
            "documents": ["salary", "bank"],
            "values": {"net_salary": 35000.0, "bank_credit": 35000.0},
            "explanation": "The salary and bank credit match.",
        },
    }])
    reasoning = LLMReasoningService().reason_about_analysis(analysis)
    assert reasoning["cross_document_inconsistencies"] == []
    assert "cross_document" in reasoning["evidence_support"]
    risk = RiskScoringService().score_reasoning(reasoning)
    assert risk["risk_level"] != "CRITICAL"


def test_consistency_graph_preserves_pass_fail_and_insufficient_relationships():
    graph = build_consistency_graph({"results": [
        {"rule_id": "pass", "rule_name": "Name", "status": "PASS", "documents": ["a", "b"], "explanation": "Names match."},
        {"rule_id": "fail", "rule_name": "Salary", "status": "FAIL", "documents": ["a", "b"], "explanation": "Amounts differ."},
        {"rule_id": "missing", "rule_name": "Employer", "status": "INSUFFICIENT_DATA", "documents": [], "explanation": "Not enough fields."},
    ]}, [
        {"document_id": "a", "document_type": "Salary Slip", "metadata": {"file_name": "salary.pdf"}},
        {"document_id": "b", "document_type": "Bank Statement", "metadata": {"file_name": "bank.pdf"}},
    ])
    assert {item["status"] for item in graph["relationships"]} == {"PASS", "FAIL", "INSUFFICIENT_DATA"}
    assert graph["nodes"][0]["label"]


def test_frontend_status_uses_trained_baseline_wording():
    text = Path("frontend/app.py").read_text(encoding="utf-8")
    assert "Multimodal financial-document verification prototype with a trained visual tampering baseline" in text
    assert "Current status: heuristic baseline pipeline only" not in text


def _score_salary_bank_pair(bank_credit: float):
    verification = CrossDocumentVerificationService.verify_documents([
        {"document_id": "salary-risk", "document_type": "Salary Slip", "fields": {"employee_name": "Arun Kumar", "net_salary": 35000.0}},
        {"document_id": "bank-risk", "document_type": "Bank Statement", "fields": {"account_holder": "Arun Kumar", "credits": bank_credit}},
    ])
    salary_rule = next(item for item in verification["results"] if item["rule_id"] == "salary_vs_bank_credit")
    cross_payload = {
        **salary_rule,
        "documents": ["salary-risk", "bank-risk"],
    }
    heuristic = {
        "detector_type": "baseline_forensic",
        "tampering_detected": True,
        "findings": [{"detector": "baseline_texture", "severity": "MEDIUM", "finding": "suspicious texture change"}],
        "limitations": ["Weak heuristic; not proof of fraud."],
        "ml_evidence": {**ml_result(), "prediction": "authentic", "forged_probability": 0.39},
    }
    fused = MultimodalFusionEngine.fuse_documents([
        {"document_id": "salary-risk", "document_type": "Salary Slip", "structured_fields": {"net_salary": 35000.0}, "cross_document": cross_payload, "tampering": heuristic},
        {"document_id": "bank-risk", "document_type": "Bank Statement", "structured_fields": {"credits": bank_credit}, "cross_document": cross_payload, "tampering": heuristic},
    ])
    reasoning = LLMReasoningService().reason_about_analysis(fused)
    risk = RiskScoringService().score_reasoning(reasoning)
    return verification, fused, reasoning, risk


def test_matching_salary_bank_has_no_mismatch_and_weak_visual_only_risk():
    verification, fused, reasoning, risk = _score_salary_bank_pair(35000.0)
    rule = next(item for item in verification["results"] if item["rule_id"] == "salary_vs_bank_credit")
    assert rule["status"] == "PASS"
    assert verification["summary"]["overall_inconsistency_count"] == 0
    assert reasoning["cross_document_inconsistencies"] == []
    assert fused["summary"]["inconsistencies_detected"] == 0
    assert fused["summary"]["tampering_observations_detected"] == 2
    assert risk["risk_level"] not in {"HIGH", "CRITICAL"}
    assert risk["score_breakdown"]["cross_document_inconsistencies"]["count"] == 0
    assert risk["score_breakdown"]["trained_visual_model"]["prediction_points"] == 0
    assert risk["score_breakdown"]["gnn"]["contribution"] == 0
    assert "No cross-document inconsistency was detected" in risk["risk_explanation"]


def test_salary_bank_mismatch_remains_materially_higher_than_matching():
    matching = _score_salary_bank_pair(35000.0)
    mismatch = _score_salary_bank_pair(25000.0)
    verification, _fused, reasoning, risk = mismatch
    rule = next(item for item in verification["results"] if item["rule_id"] == "salary_vs_bank_credit")
    assert rule["status"] == "FAIL"
    assert rule["values"] == {"net_salary": 35000.0, "bank_credit": 25000.0}
    assert reasoning["cross_document_inconsistencies"]
    assert any("differs from the bank credit amount" in item for item in reasoning["cross_document_inconsistencies"])
    assert risk["score_breakdown"]["cross_document_inconsistencies"]["contribution"] == 24
    assert risk["risk_score"] > matching[3]["risk_score"]


def test_single_heuristic_observation_is_counted_once_not_as_cross_document_failure():
    reasoning = {
        "analysis_id": "single-weak",
        "reasoning_id": "single-weak-r",
        "observed_findings": [{"source": "tampering", "summary": "Tampering assessment reviewed", "document_ids": ["doc"], "severity": "HIGH", "evidence": ["one"]}],
        "cross_document_inconsistencies": [],
        "tampering_related_observations": [{"document_ids": ["doc"], "summary": "Texture signal", "severity": "HIGH", "evidence_source": "baseline_forensic_heuristic", "details": {"findings": [{"severity": "MEDIUM"}]}}],
        "trained_visual_model_observations": [],
        "graph_based_observations": [],
        "evidence_support": {"tampering": ["Texture signal"]},
        "traceability": {"source_evidence_count": 1},
        "reasoning": "Weak visual signal only.",
        "limitations": [],
    }
    risk = RiskScoringService().score_reasoning(reasoning)
    assert risk["inconsistencies"] == []
    assert risk["score_breakdown"]["cross_document_inconsistencies"]["contribution"] == 0
    assert risk["score_breakdown"]["heuristic_tampering"]["count"] == 1
    assert risk["score_breakdown"]["heuristic_tampering"]["contribution"] == 8
    assert risk["risk_level"] not in {"HIGH", "CRITICAL"}


def test_verification_values_format_matching_and_mismatch_amounts_in_rupees():
    matching = format_verification_values("salary_vs_bank_credit", {"net_salary": 35000.0, "bank_credit": 35000.0})
    mismatch = format_verification_values("salary_vs_bank_credit", {"net_salary": 35000.0, "bank_credit": 25000.0})
    assert matching == "Salary: ₹35,000 ↔ Bank credit: ₹35,000"
    assert mismatch == "Salary: ₹35,000 ↔ Bank credit: ₹25,000"
