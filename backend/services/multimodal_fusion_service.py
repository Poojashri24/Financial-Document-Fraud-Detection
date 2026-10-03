from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence


class MultimodalFusionEngine:
    _analysis_store: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _safe_document_id(document_id: Any, fallback: str) -> str:
        if isinstance(document_id, str) and document_id.strip():
            return document_id.strip()
        return fallback

    @classmethod
    def _register_analysis(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        cls._analysis_store[analysis["analysis_id"]] = analysis
        return analysis

    @classmethod
    def get_analysis(cls, analysis_id: str) -> Dict[str, Any] | None:
        return cls._analysis_store.get(analysis_id)

    @classmethod
    def fuse_documents(cls, documents: Sequence[Dict[str, Any]] | None) -> Dict[str, Any]:
        if not documents:
            raise ValueError("At least one document payload is required for multimodal fusion.")

        normalized_documents: List[Dict[str, Any]] = []
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                continue
            document_id = cls._safe_document_id(document.get("document_id"), f"doc-{index + 1}")
            normalized_documents.append({
                "document_id": document_id,
                "document_type": document.get("document_type", "Unknown"),
                "ocr": document.get("ocr") or {},
                "structured_fields": document.get("structured_fields") or {},
                "cross_document": document.get("cross_document") or {},
                "tampering": document.get("tampering") or {},
                "metadata": document.get("metadata") or {},
            })

        if not normalized_documents:
            raise ValueError("No valid document payloads were provided for multimodal fusion.")

        modalities: Dict[str, Dict[str, Any]] = {
            "ocr": {"documents": [], "count": 0},
            "structured_fields": {"document_ids": [], "count": 0},
            "cross_document": {"documents": [], "count": 0},
            "graph_consistency": {"documents": [], "count": 0},
            "tampering": {"documents": [], "count": 0},
            "tampering_ml": {"documents": [], "count": 0},
            "visual": {"documents": [], "count": 0},
            "metadata": {"documents": [], "count": 0},
        }
        evidence: List[Dict[str, Any]] = []
        seen_graph_evidence: set[tuple[str, tuple[str, ...]]] = set()
        seen_cross_evidence: set[tuple[str, tuple[str, ...], str]] = set()
        inconsistencies_detected = 0
        tampering_observations_detected = 0
        insufficient_data_count = 0

        for doc in normalized_documents:
            document_id = doc["document_id"]
            has_any_modality = False

            if doc.get("ocr"):
                has_any_modality = True
                modalities["ocr"]["documents"].append(document_id)
                modalities["ocr"]["count"] += 1
                evidence.append({
                    "source": "ocr",
                    "document_ids": [document_id],
                    "summary": f"OCR text captured for {document_id}.",
                    "severity": "INFO",
                    "details": {"status": doc["ocr"].get("status", "unknown"), "text_excerpt": (doc["ocr"].get("text") or "")[:120]},
                })

            if doc.get("structured_fields"):
                has_any_modality = True
                modalities["structured_fields"]["document_ids"].append(document_id)
                modalities["structured_fields"]["count"] += 1
                fields = doc["structured_fields"]
                evidence.append({
                    "source": "structured_field",
                    "document_ids": [document_id],
                    "summary": f"Structured fields extracted for {document_id}.",
                    "severity": "INFO",
                    "details": {"fields": list(fields.keys())[:10], "field_count": len(fields)},
                })

            if doc.get("cross_document"):
                has_any_modality = True
                cross = doc["cross_document"]
                related_docs = list(cross.get("documents", [document_id]))
                modalities["cross_document"]["documents"].extend(related_docs)
                cross_status = str(cross.get("status", "")).upper()
                cross_key = (str(cross.get("rule_id", cross.get("rule_name", ""))), tuple(sorted(str(value) for value in related_docs)), cross_status)
                if cross_key not in seen_cross_evidence:
                    seen_cross_evidence.add(cross_key)
                    modalities["cross_document"]["count"] += 1
                    evidence.append({
                        "source": "cross_document",
                        "document_ids": related_docs,
                        "summary": cross.get("rule_name") or "Cross-document rule applied.",
                        "severity": "WARN" if cross_status in {"WARNING", "FAIL"} else "INFO",
                        "details": {"status": cross_status, "explanation": cross.get("explanation", ""), "evidence": cross.get("evidence", [])},
                    })
                    if cross_status in {"FAIL", "WARNING"}:
                        inconsistencies_detected += 1
                    elif cross_status == "INSUFFICIENT_DATA":
                        insufficient_data_count += 1
                graph_predictions = (cross.get("gnn_evidence") or {}).get("predictions") or []
                for prediction in graph_predictions:
                    graph_documents = prediction.get("documents", [])
                    graph_key = (str(prediction.get("rule_id", "")), tuple(str(value) for value in graph_documents))
                    if graph_key in seen_graph_evidence:
                        continue
                    seen_graph_evidence.add(graph_key)
                    modalities["graph_consistency"]["documents"].extend(graph_documents)
                    modalities["graph_consistency"]["count"] += 1
                    evidence.append({
                        "source": "learned_graph_based_evidence",
                        "document_ids": graph_documents,
                        "summary": f"GNN advisory prediction for {prediction.get('rule_id')}: {prediction.get('gnn_prediction')}.",
                        "severity": "INFO",
                        "details": {**prediction, "advisory_only": True, "does_not_override_rule": True},
                    })

            if doc.get("tampering"):
                has_any_modality = True
                tampering = doc["tampering"]
                modalities["tampering"]["documents"].append(document_id)
                modalities["tampering"]["count"] += 1
                findings = tampering.get("findings") or []
                region = None
                if findings:
                    first = findings[0]
                    region = first.get("region")
                evidence.append({
                    "source": "tampering",
                    "document_ids": [document_id],
                    "summary": "Tampering assessment reviewed for this document.",
                    "severity": "HIGH" if bool(tampering.get("tampering_detected")) else "INFO",
                    "details": {"tampering_detected": bool(tampering.get("tampering_detected")), "findings": findings, "limitations": tampering.get("limitations", [])},
                    "region": region,
                })
                if bool(tampering.get("tampering_detected")):
                    tampering_observations_detected += 1
                ml_evidence = tampering.get("ml_evidence") or {}
                if ml_evidence.get("prediction") in {"authentic", "forged"} and ml_evidence.get("evidence_source") == "trained_visual_model":
                    modalities["tampering_ml"]["documents"].append(document_id)
                    modalities["tampering_ml"]["count"] += 1
                    evidence.append({
                        "source": "trained_visual_model",
                        "document_ids": [document_id],
                        "summary": f"Trained visual model prediction for {document_id}: {ml_evidence.get('prediction', 'unknown')}.",
                        "severity": "WARN" if ml_evidence.get("prediction") == "forged" else "INFO",
                        "details": ml_evidence,
                    })

            metadata = doc.get("metadata") or {}
            if metadata:
                has_any_modality = True
                modalities["metadata"]["documents"].append(document_id)
                modalities["visual"]["documents"].append(document_id)
                modalities["visual"]["count"] += 1
                evidence.append({
                    "source": "metadata",
                    "document_ids": [document_id],
                    "summary": f"File metadata captured for {document_id}.",
                    "severity": "INFO",
                    "details": {"metadata": metadata},
                })

            if not has_any_modality:
                modalities["metadata"]["documents"].append(document_id)
                modalities["metadata"]["count"] += 1
                evidence.append({
                    "source": "metadata",
                    "document_ids": [document_id],
                    "summary": f"No additional forensic or OCR evidence was available for {document_id}; metadata only.",
                    "severity": "INFO",
                    "details": {"metadata": {}},
                })
                insufficient_data_count += 1

            if doc.get("metadata") or doc.get("ocr"):
                modalities["visual"]["documents"].append(document_id)
                modalities["visual"]["count"] += 1

        # Deduplicate modality lists while preserving order.
        for key in ("ocr", "cross_document", "graph_consistency", "tampering", "tampering_ml", "visual", "metadata"):
            if key in modalities:
                seen = set()
                ordered = []
                for value in modalities[key].get("documents", []):
                    if value not in seen:
                        seen.add(value)
                        ordered.append(value)
                modalities[key]["documents"] = ordered
                modalities[key]["count"] = len(ordered)
        if "structured_fields" in modalities:
            seen = set()
            ordered = []
            for value in modalities["structured_fields"].get("document_ids", []):
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
            modalities["structured_fields"]["document_ids"] = ordered
            modalities["structured_fields"]["count"] = len(ordered)

        analysis_id = str(uuid.uuid4())
        analysis = {
            "analysis_id": analysis_id,
            "documents": normalized_documents,
            "modalities": modalities,
            "evidence": evidence,
            "summary": {
                "documents_analyzed": len(normalized_documents),
                "evidence_count": len(evidence),
                "inconsistencies_detected": inconsistencies_detected,
                "tampering_observations_detected": tampering_observations_detected,
                "insufficient_data_count": insufficient_data_count,
                "modality_count": len([key for key, value in modalities.items() if value.get("documents") or value.get("document_ids")]),
            },
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return cls._register_analysis(analysis)
