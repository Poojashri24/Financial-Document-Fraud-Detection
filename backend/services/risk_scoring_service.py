from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence

from backend.services.llm_reasoning_service import LLMReasoningService


class RiskScoringService:
    _risk_store: Dict[str, Dict[str, Any]] = {}

    SCORE_RULES = {
        "cross_document_inconsistency": 24,
        "high_severity_inconsistency": 12,
        "tampering_observation": 30,
        "missing_evidence": 0,
        "insufficient_evidence": 0,
        "multiple_sources": 5,
        "low_severity": 0,
        "medium_severity": 8,
        "high_severity": 15,
        "critical_severity": 20,
        "document_count_bonus": 4,
        "max_score": 100,
    }

    @staticmethod
    def _severity_weight(severity: str) -> int:
        mapping = {
            "LOW": 2,
            "MEDIUM": 8,
            "HIGH": 15,
            "CRITICAL": 20,
            "INFO": 0,
            "WARN": 4,
        }
        return mapping.get(str(severity).upper(), 0)

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 80:
            return "CRITICAL"
        if score >= 60:
            return "HIGH"
        if score >= 30:
            return "MODERATE"
        return "LOW"

    @classmethod
    def _normalize_reasoning(cls, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(reasoning, dict):
            raise ValueError("Reasoning payload must be a dictionary.")
        return reasoning

    @classmethod
    def score_reasoning(cls, reasoning: Dict[str, Any] | None) -> Dict[str, Any]:
        if reasoning is None:
            raise ValueError("A reasoning payload is required for risk scoring.")

        reasoning = cls._normalize_reasoning(reasoning)
        analysis_id = str(reasoning.get("analysis_id") or "unknown-analysis")
        reasoning_id = str(reasoning.get("reasoning_id") or "unknown-reasoning")
        observed_findings = reasoning.get("observed_findings") or []
        cross_document = reasoning.get("cross_document_inconsistencies") or []
        tampering = reasoning.get("tampering_related_observations") or []
        trained_visual_model = reasoning.get("trained_visual_model_observations") or []
        graph_based = reasoning.get("graph_based_observations") or []
        limitations = reasoning.get("limitations") or []
        evidence_support = reasoning.get("evidence_support") or {}
        traceability = reasoning.get("traceability") or {}

        score = 0
        contributing_evidence: List[Dict[str, Any]] = []
        supporting_documents: set[str] = set()
        suspicious_documents: set[str] = set()
        inconsistencies: List[str] = []
        tampering_observations: List[Dict[str, Any]] = []
        legacy_observation_score = 0

        for finding in observed_findings:
            if not isinstance(finding, dict):
                continue
            summary = str(finding.get("summary") or "Finding")
            doc_ids = [str(item) for item in (finding.get("document_ids") or [])]
            for doc_id in doc_ids:
                supporting_documents.add(doc_id)
            severity = str(finding.get("severity", "INFO")).upper()
            source = finding.get("source")
            if source is None:
                finding_score = cls._severity_weight(severity)
                if severity in {"MEDIUM", "HIGH", "CRITICAL"}:
                    finding_score += 5
                legacy_observation_score += finding_score
                if severity not in {"INFO", "LOW"}:
                    suspicious_documents.update(doc_ids)
            elif source == "cross_document" and severity in {"WARN", "MEDIUM", "HIGH", "CRITICAL"}:
                suspicious_documents.update(doc_ids)
            contributing_evidence.append({
                "source": "observation",
                "summary": summary,
                "severity": severity,
                "document_ids": doc_ids,
                "pages": [int(page) for page in (finding.get("pages") or []) if isinstance(page, (int, float))],
                "fields": [str(field) for field in (finding.get("fields") or [])],
                "regions": [list(region) for region in (finding.get("regions") or []) if isinstance(region, (list, tuple))],
                "traceability": {
                    "document_ids": doc_ids,
                    "field_count": len([str(field) for field in (finding.get("fields") or [])]),
                    "source_count": len((finding.get("evidence") or []) or (finding.get("supporting_evidence") or [])),
                },
            })

        if cross_document:
            cross_document_score = len(cross_document) * cls.SCORE_RULES["cross_document_inconsistency"]
            score += cross_document_score
            inconsistencies.extend([str(item) for item in cross_document])
            for item in cross_document:
                contributing_evidence.append({
                    "source": "cross_document",
                    "summary": str(item),
                    "severity": "MEDIUM",
                    "document_ids": [str(doc_id) for doc_id in sorted({part for part in str(item).split() if part.startswith("doc-")})],
                    "pages": [],
                    "fields": [],
                    "regions": [],
                    "traceability": {"document_ids": [], "field_count": 0, "source_count": 1},
                })

        if tampering:
            heuristic_raw_score = 0
            heuristic_signal_count = 0
            for item in tampering:
                if not isinstance(item, dict):
                    continue
                tampering_observations.append(item)
                item_documents = [str(doc_id) for doc_id in (item.get("document_ids") or [])]
                supporting_documents.update(item_documents)
                details = item.get("details") or {}
                findings = details.get("findings") or []
                severities = [str(finding.get("severity", "INFO")).upper() for finding in findings if isinstance(finding, dict)]
                severity = max(severities, key=cls._severity_weight) if severities else str(item.get("severity", "MEDIUM")).upper()
                item_points = cls._severity_weight(severity)
                typed_heuristic = item.get("evidence_source") == "baseline_forensic_heuristic"
                if typed_heuristic:
                    heuristic_signal_count += 1
                    heuristic_raw_score += item_points
                    if item_points:
                        suspicious_documents.update(item_documents)
                else:
                    # Preserve compatibility for legacy direct risk payloads which have no source tag.
                    item_points = cls.SCORE_RULES["tampering_observation"]
                    score += item_points
                    if item_points:
                        suspicious_documents.update(item_documents)
                region = item.get("region")
                contributing_evidence.append({
                    "source": "tampering",
                    "summary": str(item.get("summary") or "Tampering observation"),
                    "severity": severity,
                    "document_ids": item_documents,
                    "pages": [],
                    "fields": [],
                    "regions": [list(region)] if isinstance(region, (list, tuple)) and len(region) == 4 else [],
                    "traceability": {
                        "document_ids": item_documents,
                        "field_count": 0,
                        "source_count": 1,
                        "heuristic_score_contribution": item_points if not typed_heuristic else None,
                    },
                })
            heuristic_cap = 12
            heuristic_score = min(heuristic_raw_score, heuristic_cap)
            score += heuristic_score
        else:
            heuristic_signal_count = 0
            heuristic_raw_score = 0
            heuristic_score = 0

        trained_visual_score = 0
        for item in trained_visual_model:
            if not isinstance(item, dict):
                continue
            document_ids = [str(doc_id) for doc_id in (item.get("document_ids") or [])]
            supporting_documents.update(document_ids)
            prediction = str(item.get("prediction", "")).lower()
            item_points = 4 if prediction == "forged" else 0
            trained_visual_score += item_points
            if item_points:
                suspicious_documents.update(document_ids)
                score += item_points
            contributing_evidence.append({
                "source": "trained_visual_model",
                "summary": str(item.get("summary") or "Trained visual model observation"),
                "severity": "WARN",
                "document_ids": document_ids,
                "pages": [],
                "fields": [],
                "regions": [],
                "traceability": {
                    "document_ids": document_ids,
                    "field_count": 0,
                    "source_count": 1,
                    "model_output": {
                        "prediction": item.get("prediction"),
                        "forged_probability": item.get("forged_probability"),
                        "risk_score_contribution": item_points,
                        "probability_is_calibrated": False,
                    },
                },
            })

        for item in graph_based:
            if not isinstance(item, dict):
                continue
            document_ids = [str(doc_id) for doc_id in (item.get("document_ids") or [])]
            supporting_documents.update(document_ids)
            contributing_evidence.append({
                "source": "learned_graph_based_evidence",
                "summary": str(item.get("summary") or "Learned graph-based consistency evidence"),
                "severity": "INFO",
                "document_ids": document_ids,
                "pages": [],
                "fields": [],
                "regions": [],
                "traceability": {
                    "document_ids": document_ids,
                    "field_count": 0,
                    "source_count": 1,
                    "rule_id": item.get("rule_id"),
                    "rule_based_status": item.get("rule_based_status"),
                    "gnn_prediction": item.get("gnn_prediction"),
                    "confidence": item.get("confidence"),
                    "advisory_only": True,
                },
            })

        score += legacy_observation_score
        if heuristic_signal_count:
            limitations.append("Heuristic tampering observations are visual warning signals, not proof of fraud; their aggregate risk contribution is capped.")
        if trained_visual_model:
            limitations.append("Trained visual predictions are model evidence, not proof of fraud; forged probability is not a calibrated fraud probability.")
        if graph_based:
            limitations.append("GNN outputs are advisory rule-distillation evidence and do not override deterministic verification or add direct risk points.")

        if not evidence_support:
            score += cls.SCORE_RULES["missing_evidence"]
            limitations.append("No supporting multimodal evidence was supplied for scoring.")

        if traceability.get("source_evidence_count", 0) == 0:
            score += cls.SCORE_RULES["insufficient_evidence"]
            limitations.append("The supplied reasoning includes insufficient evidence to support a robust assessment.")

        suspicious_sources = int(bool(cross_document)) + int(bool(heuristic_signal_count)) + int(any(str(item.get("prediction", "")).lower() == "forged" for item in trained_visual_model if isinstance(item, dict)))
        multiple_sources_score = min(max(suspicious_sources - 1, 0) * cls.SCORE_RULES["multiple_sources"], 10)
        score += multiple_sources_score

        if len(suspicious_documents) > 1:
            document_count_score = min((len(suspicious_documents) - 1) * cls.SCORE_RULES["document_count_bonus"], 8)
            score += document_count_score
        else:
            document_count_score = 0

        score = max(0, min(score, cls.SCORE_RULES["max_score"]))

        if score >= 80:
            risk_level = "CRITICAL"
        elif score >= 60:
            risk_level = "HIGH"
        elif score >= 30:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        methodology = {
            "score_type": "heuristic_baseline",
            "explanation": "This is a rule-based baseline score derived from observable evidence, not a calibrated probability of fraud.",
            "rules": {
                "cross_document_inconsistency_weight": cls.SCORE_RULES["cross_document_inconsistency"],
                "tampering_observation_weight": cls.SCORE_RULES["medium_severity"],
                "tampering_aggregate_cap": 12,
                "missing_evidence_weight": cls.SCORE_RULES["missing_evidence"],
                "insufficient_evidence_weight": cls.SCORE_RULES["insufficient_evidence"],
                "multiple_sources_bonus": cls.SCORE_RULES["multiple_sources"],
            },
            "note": "Risk score is a heuristic baseline score and is not a calibrated probability of fraud.",
            "tampering_note": "Heuristic tampering and trained visual-model outputs are evidence signals, not proof of fraud. Model probabilities are not calibrated fraud probabilities. GNN evidence is advisory and adds no direct points.",
        }

        if cross_document:
            explanation = f"{len(cross_document)} explicit cross-document inconsistency check(s) contributed {len(cross_document) * cls.SCORE_RULES['cross_document_inconsistency']} points."
        else:
            explanation = "No cross-document inconsistency was detected."
        if heuristic_signal_count:
            explanation += f" {heuristic_signal_count} heuristic visual tampering observation(s) contributed {heuristic_score} points (capped); these observations are not proof of fraud."
        if trained_visual_model:
            model_predictions = ", ".join(
                f"{str(item.get('prediction', 'unknown')).upper()} (forged probability {float(item.get('forged_probability', 0.0)):.3f})"
                for item in trained_visual_model if isinstance(item, dict)
            )
            explanation += f" Trained visual model evidence: {model_predictions}; probabilities are not calibrated fraud probabilities or proof."
        if graph_based:
            explanation += " GNN evidence is advisory rule-distillation output and does not override verification."

        assessment = {
            "analysis_id": analysis_id,
            "reasoning_id": reasoning_id,
            "risk_score": score,
            "risk_level": risk_level,
            "risk_explanation": explanation,
            "score_breakdown": {
                "cross_document_inconsistencies": {"count": len(cross_document), "contribution": len(cross_document) * cls.SCORE_RULES["cross_document_inconsistency"]},
                "heuristic_tampering": {"count": heuristic_signal_count, "raw_contribution": heuristic_raw_score, "cap": 12, "contribution": heuristic_score},
                "trained_visual_model": {"count": len(trained_visual_model), "prediction_points": trained_visual_score, "probability_is_calibrated": False},
                "gnn": {"count": len(graph_based), "contribution": 0, "advisory_only": True},
                "legacy_untagged_observations": {"contribution": legacy_observation_score},
                "multiple_suspicious_sources": {"count": suspicious_sources, "contribution": multiple_sources_score},
                "multiple_suspicious_documents": {"count": len(suspicious_documents), "contribution": document_count_score},
                "missing_evidence": {"present": not bool(evidence_support), "contribution": cls.SCORE_RULES["missing_evidence"]},
                "insufficient_evidence": {"present": traceability.get("source_evidence_count", 0) == 0, "contribution": cls.SCORE_RULES["insufficient_evidence"]},
                "final_heuristic_score": score,
            },
            "contributing_evidence": contributing_evidence,
            "supporting_documents": sorted(supporting_documents),
            "inconsistencies": inconsistencies,
            "tampering_observations": tampering_observations,
            "reasoning_summary": str(reasoning.get("reasoning") or "No reasoning narrative supplied."),
            "limitations": limitations or ["No material limitations were identified in the supplied evidence; however, this remains a heuristic baseline score."],
            "methodology": methodology,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        cls._risk_store[assessment["analysis_id"]] = assessment
        return assessment
