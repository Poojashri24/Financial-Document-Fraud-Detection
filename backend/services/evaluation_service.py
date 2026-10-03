from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


class EvaluationService:
    @staticmethod
    def _validate_binary_inputs(labels: Sequence[int | float], predictions: Sequence[int | float]) -> None:
        if len(labels) != len(predictions):
            raise ValueError("Labels and predictions must have the same length.")
        if len(labels) == 0:
            raise ValueError("No evaluation samples were provided.")

        for label in labels:
            if label not in {0, 1}:
                raise ValueError("Binary classification labels must be 0 or 1.")
        for pred in predictions:
            if pred not in {0, 1}:
                raise ValueError("Binary predictions must be 0 or 1.")

    @staticmethod
    def evaluate_predictions(
        labels: Sequence[int | float],
        predictions: Sequence[int | float],
        probabilities: Sequence[float] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if labels is None or predictions is None:
            raise ValueError("Labels and predictions are required for evaluation.")

        EvaluationService._validate_binary_inputs(labels, predictions)

        if probabilities is not None and len(probabilities) != len(labels):
            raise ValueError("Probabilities must align with labels and predictions.")

        num_samples = len(labels)
        true_positive = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
        false_positive = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
        true_negative = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
        false_negative = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)

        positives = sum(labels)
        if positives == 0:
            return {
                "status": "not_evaluated",
                "message": "No positive samples were available to compute positive-class metrics.",
                "metrics": {},
                "confusion_matrix": {
                    "tp": true_positive,
                    "fp": false_positive,
                    "tn": true_negative,
                    "fn": false_negative,
                },
                "reproducibility": metadata or {},
            }

        accuracy = (true_positive + true_negative) / num_samples if num_samples else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if probabilities is None:
            roc_auc = None
        else:
            prob_scores = list(probabilities)
            if len(prob_scores) != len(labels):
                raise ValueError("Probability scores must match labels and predictions length.")
            pos = [prob_scores[i] for i, label in enumerate(labels) if label == 1]
            neg = [prob_scores[i] for i, label in enumerate(labels) if label == 0]
            if not pos or not neg:
                roc_auc = None
            else:
                ranked_pairs = 0
                total = 0
                for i, p_i in enumerate(prob_scores):
                    for j, p_j in enumerate(prob_scores):
                        if i == j:
                            continue
                        total += 1
                        if labels[i] == 1 and labels[j] == 0 and p_i > p_j:
                            ranked_pairs += 1
                        elif labels[i] == 1 and labels[j] == 0 and p_i == p_j:
                            ranked_pairs += 0.5
                roc_auc = ranked_pairs / total if total else None

        result = {
            "status": "evaluated",
            "message": "Binary classification metrics were computed from real labels and predictions.",
            "metrics": {
                "accuracy": round(float(accuracy), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1_score": round(float(f1), 6),
                "roc_auc": round(float(roc_auc), 6) if roc_auc is not None else None,
            },
            "confusion_matrix": {
                "tp": true_positive,
                "fp": false_positive,
                "tn": true_negative,
                "fn": false_negative,
            },
            "reproducibility": metadata or {},
        }
        return result

    @staticmethod
    def evaluate_tampering_predictions(
        labels: Sequence[int | float],
        predictions: Sequence[int | float],
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if labels is None or predictions is None:
            raise ValueError("Tampering labels and predictions are required.")
        if len(labels) != len(predictions):
            raise ValueError("Tampering labels and predictions must have the same length.")
        if any(label not in {0, 1} for label in labels):
            raise ValueError("Tampering labels must be binary (0 or 1).")
        if any(pred not in {0, 1} for pred in predictions):
            raise ValueError("Tampering predictions must be binary (0 or 1).")

        if sum(labels) == 0:
            return {
                "status": "not_evaluated",
                "message": "No positive tampering labels were available, so no binary tampering evaluation was performed.",
                "metrics": {},
                "confusion_matrix": {},
                "reproducibility": metadata or {},
            }

        return EvaluationService.evaluate_predictions(labels, predictions, metadata=metadata)
