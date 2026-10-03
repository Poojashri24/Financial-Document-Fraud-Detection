from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


STATUS_TO_ID = {"PASS": 0, "FAIL": 1, "INSUFFICIENT_DATA": 2}
ID_TO_STATUS = {value: key for key, value in STATUS_TO_ID.items()}
RULE_IDS = [
    "name_consistency",
    "employer_company_consistency",
    "employee_id_consistency",
    "salary_vs_bank_credit",
    "invoice_total_vs_payment_amount",
    "invoice_date_vs_payment_date",
    "account_holder_consistency",
    "currency_consistency",
    "duplicate_document_identifiers",
    "date_inconsistencies",
]
DOC_TYPES = ["Salary Slip", "Bank Statement", "Invoice", "Identity Document", "Unknown"]


@dataclass
class GraphConsistencyConfig:
    seed: int = 42
    hidden_channels: int = 16
    epochs: int = 120
    learning_rate: float = 0.02
    weight_decay: float = 0.001
    checkpoint_path: str = "models/cross_document_gnn/rule_distillation_gcn.pt"
    report_path: str = "reports/phase13_gnn_rule_distillation.json"


class CrossDocumentGNN:
    """Small PyG GCN edge classifier; supervision is explicitly rule-distilled."""

    def __init__(self, node_features: int = 11, edge_features: int = 14, hidden_channels: int = 16):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GCNConv
        except ImportError as exc:
            raise RuntimeError("PyTorch and PyTorch Geometric are required for the graph model.") from exc

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(node_features, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.edge_head = nn.Sequential(
                    nn.Linear(hidden_channels * 3 + edge_features, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, len(STATUS_TO_ID)),
                )

            def forward(self, x, edge_index, edge_features_tensor):
                x = torch.relu(self.conv1(x, edge_index))
                x = torch.relu(self.conv2(x, edge_index))
                left, right = x[0], x[1]
                combined = torch.cat([left, right, torch.abs(left - right), edge_features_tensor], dim=0)
                return self.edge_head(combined)

        self.network = Network()

    def __call__(self, x, edge_index, edge_features):
        return self.network(x, edge_index, edge_features)

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        return self.network.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self.network.train(mode)
        return self

    def eval(self):
        self.network.eval()
        return self


class CrossDocumentGNNService:
    def __init__(self, config: GraphConsistencyConfig | None = None):
        self.config = config or GraphConsistencyConfig()

    @staticmethod
    def _type_vector(document_type: str) -> List[float]:
        return [1.0 if document_type == value else 0.0 for value in DOC_TYPES]

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").lower().split())

    @classmethod
    def _case_features(cls, case: Dict[str, Any]) -> tuple[list[list[float]], list[float]]:
        documents = case["documents"]
        rule = case["rule_id"]
        fields = [document.get("fields", {}) for document in documents]
        values = case.get("values", {})
        numeric_values = [
            value for value in values.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        numeric_present = len(numeric_values) >= 2
        left_value = float(numeric_values[0]) if numeric_present else 0.0
        right_value = float(numeric_values[1]) if numeric_present else 0.0
        scale = max(abs(left_value), abs(right_value), 1.0)
        left_text = cls._normalize_text(next((value for value in fields[0].values() if isinstance(value, str) and value.strip()), ""))
        right_text = cls._normalize_text(next((value for value in fields[1].values() if isinstance(value, str) and value.strip()), ""))
        text_equal = float(bool(left_text) and left_text == right_text)
        nodes: List[List[float]] = []
        for index, document in enumerate(documents):
            field_values = fields[index]
            numeric = next((float(value) for value in field_values.values() if isinstance(value, (int, float)) and not isinstance(value, bool)), 0.0)
            nodes.append(
                cls._type_vector(document.get("document_type", "Unknown"))
                + [
                    min(abs(numeric) / scale, 1.0),
                    float(bool(field_values)),
                    float(any(key in field_values for key in ("name", "employee_name", "account_holder"))),
                    float(any(key in field_values for key in ("employer", "vendor", "company"))),
                    float("currency" in field_values),
                    float(bool(document.get("tampering_detected"))),
                ]
            )
        edge = [1.0 if rule == rule_id else 0.0 for rule_id in RULE_IDS]
        relative_difference = abs(left_value - right_value) / scale if numeric_present else 0.0
        edge.extend([
            min(relative_difference, 1.0),
            min(min(abs(left_value), abs(right_value)) / max(max(abs(left_value), abs(right_value)), 1.0), 1.0) if numeric_present else 0.0,
            text_equal,
            float(numeric_present),
        ])
        return nodes, edge

    @staticmethod
    def rule_distillation_cases() -> List[Dict[str, Any]]:
        """Explicit examples already represented by cross-document unit fixtures.

        Their targets are deterministic PASS/FAIL/INSUFFICIENT_DATA rule outcomes,
        not fraud labels and not independent ground truth.
        """
        return [
            {"case_id": "salary-match-train", "split": "train", "rule_id": "salary_vs_bank_credit", "documents": [{"document_id": "s1", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith", "net_salary": 4300.0, "currency": "USD"}}, {"document_id": "b1", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith", "credits": 4300.0, "currency": "USD"}}]},
            {"case_id": "salary-mismatch-train", "split": "train", "rule_id": "salary_vs_bank_credit", "documents": [{"document_id": "s2", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith", "net_salary": 4300.0}}, {"document_id": "b2", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith", "credits": 3500.0}}]},
            {"case_id": "name-match-train", "split": "train", "rule_id": "name_consistency", "documents": [{"document_id": "s3", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith"}}, {"document_id": "b3", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith"}}]},
            {"case_id": "name-mismatch-train", "split": "train", "rule_id": "name_consistency", "documents": [{"document_id": "s4", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith"}}, {"document_id": "b4", "document_type": "Bank Statement", "fields": {"account_holder": "Jane Smith"}}]},
            {"case_id": "employer-mismatch-train", "split": "train", "rule_id": "employer_company_consistency", "documents": [{"document_id": "s5", "document_type": "Salary Slip", "fields": {"employer": "Acme Ltd"}}, {"document_id": "i5", "document_type": "Invoice", "fields": {"vendor": "Beta Corp"}}]},
            {"case_id": "currency-mismatch-train", "split": "train", "rule_id": "currency_consistency", "documents": [{"document_id": "s6", "document_type": "Salary Slip", "fields": {"currency": "USD"}}, {"document_id": "b6", "document_type": "Bank Statement", "fields": {"currency": "EUR"}}]},
            {"case_id": "salary-insufficient-train", "split": "train", "rule_id": "salary_vs_bank_credit", "documents": [{"document_id": "s6b", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith"}}, {"document_id": "b6b", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith"}}]},
            {"case_id": "invoice-mismatch-validation", "split": "validation", "rule_id": "invoice_total_vs_payment_amount", "documents": [{"document_id": "i7", "document_type": "Invoice", "fields": {"total": 1650.0}}, {"document_id": "b7", "document_type": "Bank Statement", "fields": {"transaction_amount": 1500.0}}]},
            {"case_id": "salary-insufficient-validation", "split": "validation", "rule_id": "salary_vs_bank_credit", "documents": [{"document_id": "s8", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith"}}, {"document_id": "b8", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith"}}]},
            {"case_id": "account-name-match-test", "split": "test", "rule_id": "account_holder_consistency", "documents": [{"document_id": "s9", "document_type": "Salary Slip", "fields": {"employee_name": "John Smith"}}, {"document_id": "b9", "document_type": "Bank Statement", "fields": {"account_holder": "John Smith"}}]},
            {"case_id": "duplicate-id-test", "split": "test", "rule_id": "duplicate_document_identifiers", "documents": [{"document_id": "i10", "document_type": "Invoice", "fields": {"invoice_number": "INV-1042"}}, {"document_id": "i11", "document_type": "Invoice", "fields": {"invoice_number": "INV-1042"}}]},
        ]

    @staticmethod
    def _rule_result_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
        from backend.services.cross_document_verification_service import CrossDocumentVerificationService
        result = CrossDocumentVerificationService.verify_documents(case["documents"])
        selected = next(item for item in result["results"] if item["rule_id"] == case["rule_id"])
        case["values"] = selected.get("values", {})
        case["label"] = selected["status"]
        case["explanation"] = selected["explanation"]
        return case

    def train_and_evaluate(self) -> Dict[str, Any]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        cases = [self._rule_result_for_case(case) for case in self.rule_distillation_cases()]
        datasets = {split: [case for case in cases if case["split"] == split] for split in ("train", "validation", "test")}
        graph_rows = {}
        for split, split_cases in datasets.items():
            graph_rows[split] = []
            for case in split_cases:
                nodes, edge = self._case_features(case)
                graph_rows[split].append((Data(x=torch.tensor(nodes, dtype=torch.float32), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)), torch.tensor(edge, dtype=torch.float32), STATUS_TO_ID[case["label"]], case))
        model = CrossDocumentGNN(hidden_channels=self.config.hidden_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        train_rows = graph_rows["train"]
        label_counts = torch.bincount(torch.tensor([row[2] for row in train_rows]), minlength=3).float().clamp(min=1)
        class_weights = len(train_rows) / (3.0 * label_counts)
        started = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            random.shuffle(train_rows)
            for graph, edge_features, target, _case in train_rows:
                optimizer.zero_grad()
                logits = model(graph.x, graph.edge_index, edge_features)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]), weight=class_weights)
                loss.backward()
                optimizer.step()
        train_seconds = time.perf_counter() - started
        results = {}
        for split in ("train", "validation", "test"):
            model.eval()
            actual, predicted, probabilities = [], [], []
            with torch.inference_mode():
                for graph, edge_features, target, _case in graph_rows[split]:
                    probs = torch.softmax(model(graph.x, graph.edge_index, edge_features), dim=0)
                    actual.append(target)
                    predicted.append(int(probs.argmax()))
                    probabilities.append(probs.tolist())
            results[split] = self._metrics(actual, predicted, probabilities)
        checkpoint = Path(self.config.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": asdict(self.config), "status_map": ID_TO_STATUS, "label_source": "deterministic cross-document rule outcomes on synthetic test examples"}, checkpoint)
        report = {
            "status": "evaluated_rule_distillation_prototype",
            "architecture": "2-layer PyTorch Geometric GCNConv plus edge MLP",
            "node_features": ["document_type_one_hot", "normalized_numeric_value", "field_presence", "person_name_presence", "company_presence", "currency_presence", "heuristic_tampering_flag"],
            "edge_features": ["rule_type_one_hot", "normalized_numeric_difference", "numeric_ratio", "text_equality", "numeric_values_present"],
            "label_mapping": STATUS_TO_ID,
            "label_method": "PASS/FAIL/INSUFFICIENT_DATA labels are outputs of the existing deterministic cross-document verifier on documented synthetic software-test scenarios; not fraud labels or independent ground truth.",
            "counts": {split: len(rows) for split, rows in datasets.items()},
            "metrics": results,
            "training_seconds": round(train_seconds, 3),
            "seed": self.config.seed,
            "checkpoint": str(checkpoint),
            "limitation": "Tiny rule-distillation fixture set; metrics measure agreement with the existing rules only and do not establish learned generalization or fraud performance.",
        }
        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.model = model
        return report

    @staticmethod
    def _metrics(actual: Sequence[int], predicted: Sequence[int], probabilities: Sequence[Sequence[float]]) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        if not actual:
            return {"sample_count": 0, "accuracy": None, "precision_macro": None, "recall_macro": None, "f1_macro": None, "roc_auc_ovr_macro": None, "confusion_matrix": []}
        labels = [0, 1, 2]
        auc = None
        if len(set(actual)) == 3:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(actual, probabilities, multi_class="ovr", average="macro"))
        return {
            "sample_count": len(actual),
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_macro": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "roc_auc_ovr_macro": auc,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
            "class_order": [ID_TO_STATUS[index] for index in labels],
        }

    def predict(self, documents: Sequence[Dict[str, Any]], verification_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        if not hasattr(self, "model"):
            checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=False)
            self.model = CrossDocumentGNN(hidden_channels=self.config.hidden_channels)
            self.model.load_state_dict(checkpoint["state_dict"])
        from torch_geometric.data import Data
        doc_map = {str(item.get("document_id")): item for item in documents}
        output = []
        self.model.eval()
        with torch.inference_mode():
            for result in verification_results:
                ids = [str(item) for item in result.get("documents", [])]
                if len(ids) < 2 or ids[0] not in doc_map or ids[1] not in doc_map:
                    continue
                case = {"documents": [{"document_id": doc_map[doc_id].get("document_id"), "document_type": doc_map[doc_id].get("document_type", "Unknown"), "fields": doc_map[doc_id].get("fields", doc_map[doc_id].get("structured_fields", {}))} for doc_id in ids[:2]], "rule_id": result.get("rule_id"), "values": result.get("values", {})}
                nodes, edge = self._case_features(case)
                graph = Data(x=torch.tensor(nodes, dtype=torch.float32), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
                probs = torch.softmax(self.model(graph.x, graph.edge_index, torch.tensor(edge, dtype=torch.float32)), dim=0)
                status = ID_TO_STATUS[int(probs.argmax())]
                output.append({"rule_id": result.get("rule_id"), "rule_based_status": result.get("status"), "gnn_prediction": status, "confidence": float(probs.max()), "probabilities": {ID_TO_STATUS[index]: float(probs[index]) for index in range(len(ID_TO_STATUS))}, "documents": ids[:2], "evidence_source": "learned_graph_based_evidence", "explanation": "GNN output is rule-distilled graph evidence and does not override explicit verification results."})
        return {"status": "available", "model_name": "Cross-document rule-distillation GCN", "predictions": output}
