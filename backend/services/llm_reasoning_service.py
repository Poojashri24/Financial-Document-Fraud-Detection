from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence

from backend.core.config import settings
from backend.services.multimodal_fusion_service import MultimodalFusionEngine


class LLMReasoningService:
    _reasoning_store: Dict[str, Dict[str, Any]] = {}

    def __init__(self, provider: str | None = None, api_key: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or settings.llm_provider or "").strip()
        self.api_key = (api_key or os.getenv("LLM_API_KEY") or settings.llm_api_key or "").strip()
        self.model = (os.getenv("LLM_MODEL") or settings.llm_model or "").strip()

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _collect_document_ids(items: Sequence[Dict[str, Any]]) -> List[str]:
        collected: List[str] = []
        for item in items:
            doc_ids = item.get("document_ids") or []
            for doc_id in doc_ids:
                if doc_id not in collected:
                    collected.append(doc_id)
        return collected

    @staticmethod
    def _extract_pages(value: Any) -> List[int]:
        if not value:
            return []
        if isinstance(value, list):
            return [int(item) for item in value if isinstance(item, (int, float))]
        if isinstance(value, dict):
            pages = value.get("pages")
            return LLMReasoningService._extract_pages(pages)
        return []

    @staticmethod
    def _extract_fields_for_document(document: Dict[str, Any]) -> List[str]:
        fields = document.get("structured_fields") or {}
        if isinstance(fields, dict):
            return [str(key) for key in fields.keys()]
        return []

    @classmethod
    def _fallback_reasoning(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        documents = analysis.get("documents") or []
        evidence = analysis.get("evidence") or []
        modalities = analysis.get("modalities") or {}
        summary = analysis.get("summary") or {}

        observed_findings: List[Dict[str, Any]] = []
        evidence_support: Dict[str, List[str]] = {}
        cross_document_inconsistencies: List[str] = []
        tampering_related_observations: List[Dict[str, Any]] = []
        trained_visual_model_observations: List[Dict[str, Any]] = []
        graph_based_observations: List[Dict[str, Any]] = []
        limitations: List[str] = []

        if not evidence:
            limitations.append("No multimodal evidence was supplied for reasoning; only a minimal context was available.")
        else:
            for index, item in enumerate(evidence):
                source = item.get("source", "unknown")
                doc_ids = item.get("document_ids") or []
                summary_text = item.get("summary") or "Evidence item"
                details = item.get("details") or {}
                evidence_support.setdefault(source, [])
                evidence_support[source].append(summary_text)
                observed_findings.append({
                    "finding_id": f"finding-{index + 1}",
                    "source": source,
                    "summary": summary_text,
                    "document_ids": [str(doc_id) for doc_id in doc_ids],
                    "pages": cls._extract_pages(item.get("metadata") or details),
                    "fields": [],
                    "regions": [item.get("region")] if isinstance(item.get("region"), list) and len(item.get("region")) == 4 else [],
                    "evidence": [summary_text],
                    "supporting_evidence": [summary_text],
                    "severity": item.get("severity", "INFO"),
                })

                if source == "cross_document" and str(details.get("status", item.get("severity", ""))).upper() in {"FAIL", "WARNING"}:
                    rule_lines = [
                        str(line)
                        for line in (details.get("evidence") or [])
                        if "[FAIL]" in str(line).upper() or "[WARNING]" in str(line).upper()
                    ]
                    if rule_lines:
                        cross_document_inconsistencies.extend(rule_lines)
                    else:
                        cross_document_inconsistencies.append(str(details.get("explanation") or summary_text))
                if source == "tampering":
                    tampering_related_observations.append({
                        "document_ids": [str(doc_id) for doc_id in doc_ids],
                        "summary": summary_text,
                        "severity": item.get("severity", "INFO"),
                        "evidence_source": "baseline_forensic_heuristic",
                        "details": details,
                        "region": item.get("region"),
                    })
                if source == "trained_visual_model":
                    trained_visual_model_observations.append({
                        "document_ids": [str(doc_id) for doc_id in doc_ids],
                        "summary": summary_text,
                        "prediction": details.get("prediction"),
                        "forged_probability": details.get("forged_probability"),
                        "evidence_source": "trained_visual_model",
                    })
                if source == "learned_graph_based_evidence":
                    graph_based_observations.append({
                        "document_ids": [str(doc_id) for doc_id in doc_ids],
                        "summary": summary_text,
                        "rule_id": details.get("rule_id"),
                        "rule_based_status": details.get("rule_based_status"),
                        "gnn_prediction": details.get("gnn_prediction"),
                        "confidence": details.get("confidence"),
                        "advisory_only": True,
                        "does_not_override_rule": True,
                    })

        for document_item in documents:
            if not isinstance(document_item, dict):
                continue
            doc_id = document_item.get("document_id")
            if not document_item.get("ocr") and not document_item.get("structured_fields") and not document_item.get("tampering") and not document_item.get("cross_document"):
                limitations.append(f"Document {doc_id} contributes only minimal metadata and no substantive evidence.")
            fields = document_item.get("structured_fields") or {}
            if isinstance(fields, dict) and len(fields) == 0:
                limitations.append(f"Document {doc_id} has no structured fields available for cross-checking.")

        if not tampering_related_observations and any(document_item.get("tampering") for document_item in documents if isinstance(document_item, dict)):
            tampering_related_observations.append({
                "document_ids": [document_item.get("document_id") for document_item in documents if isinstance(document_item, dict) and document_item.get("tampering")],
                "summary": "Tampering metadata was present but no explicit suspicious region or finding was available.",
                "severity": "INFO",
                "details": {},
                "region": None,
            })

        if not observed_findings:
            observed_findings.append({
                "finding_id": "finding-1",
                "summary": "No explicit multimodal findings were included in the supplied analysis.",
                "document_ids": [str(document_item.get("document_id")) for document_item in documents if isinstance(document_item, dict) and document_item.get("document_id")],
                "pages": [],
                "fields": [],
                "regions": [],
                "evidence": [],
                "supporting_evidence": [],
                "severity": "INFO",
            })

        reasoning_text = (
            "This is heuristic, rule-based reasoning generated from the supplied multimodal evidence only. "
            "It describes the evidence available, notes cross-document inconsistencies and tampering-related signals, "
            "identifies trained visual-model output as model evidence and not a definitive fraud verdict, "
            "and identifies GNN output as advisory learned graph evidence that does not override explicit rules, "
            "and explicitly calls out missing or limited evidence without inventing unsupported facts."
        )

        for doc_item in documents:
            if not isinstance(doc_item, dict):
                continue
            doc_id = doc_item.get("document_id")
            if doc_id:
                fields = cls._extract_fields_for_document(doc_item)
                if fields:
                    evidence_support.setdefault("structured_fields", [])
                    evidence_support["structured_fields"].append(f"Document {doc_id} supplied structured fields: {', '.join(fields)}")

        return {
            "reasoning_mode": "heuristic_rule_based",
            "observed_findings": observed_findings,
            "evidence_support": evidence_support,
            "cross_document_inconsistencies": cross_document_inconsistencies,
            "tampering_related_observations": tampering_related_observations,
            "trained_visual_model_observations": trained_visual_model_observations,
            "graph_based_observations": graph_based_observations,
            "reasoning": reasoning_text,
            "limitations": limitations or ["No material limitations were identified from the supplied evidence; however, this remains a deterministic heuristic explanation layer."],
            "traceability": {
                "document_count": len(documents),
                "modalities_present": sorted(list(modalities.keys())) if isinstance(modalities, dict) else [],
                "source_evidence_count": len(evidence),
            },
        }

    def reason_about_analysis(self, analysis: Dict[str, Any] | None) -> Dict[str, Any]:
        if analysis is None:
            raise ValueError("A multimodal analysis payload is required for LLM reasoning.")

        if not isinstance(analysis, dict):
            raise ValueError("The multimodal analysis payload must be a dictionary.")

        if not self.provider or not self.api_key:
            fallback = self._fallback_reasoning(analysis)
            payload = {
                "analysis_id": str(analysis.get("analysis_id") or "unknown-analysis"),
                "reasoning_id": str(uuid.uuid4()),
                "reasoning_mode": fallback["reasoning_mode"],
                "observed_findings": fallback["observed_findings"],
                "evidence_support": fallback["evidence_support"],
                "cross_document_inconsistencies": fallback["cross_document_inconsistencies"],
                "tampering_related_observations": fallback["tampering_related_observations"],
                "trained_visual_model_observations": fallback["trained_visual_model_observations"],
                "graph_based_observations": fallback["graph_based_observations"],
                "reasoning": fallback["reasoning"],
                "limitations": fallback["limitations"],
                "traceability": fallback["traceability"],
                "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
            self.__class__._reasoning_store[payload["reasoning_id"]] = payload
            return payload

        if self.provider.lower() == "openai" and self.api_key:
            result = {
                "analysis_id": str(analysis.get("analysis_id") or "unknown-analysis"),
                "reasoning_id": str(uuid.uuid4()),
                "reasoning_mode": "provider_call",
                "observed_findings": [],
                "evidence_support": {},
                "cross_document_inconsistencies": [],
                "tampering_related_observations": [],
                "trained_visual_model_observations": [],
                "graph_based_observations": [],
                "reasoning": "This provider route is configured for an external LLM but no live provider client is implemented in this project phase; the fallback rule-based reasoning is used instead.",
                "limitations": ["No external LLM provider client is wired in this phase. The system stays deterministic and evidence-grounded."],
                "traceability": {"document_count": len(analysis.get("documents") or []), "provider": self.provider, "model": self.model},
                "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
            self.__class__._reasoning_store[result["reasoning_id"]] = result
            return result

        fallback = self._fallback_reasoning(analysis)
        fallback["analysis_id"] = str(analysis.get("analysis_id") or "unknown-analysis")
        fallback["reasoning_id"] = str(uuid.uuid4())
        fallback["generated_at"] = datetime.utcnow().isoformat(timespec="seconds")
        self.__class__._reasoning_store[fallback["reasoning_id"]] = fallback
        return fallback
