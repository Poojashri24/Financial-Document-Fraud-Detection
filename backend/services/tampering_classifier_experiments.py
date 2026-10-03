from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TamperingClassifierExperiments:
    """Small predeclared classifier comparison over cached existing split features."""

    def __init__(self, artifact_dir: str = "models/tampering_feature_baseline", reports_dir: str = "reports", seed: int = 42):
        self.artifact_dir = Path(artifact_dir)
        self.reports_dir = Path(reports_dir)
        self.seed = seed

    @staticmethod
    def _load_cache(path: str | Path):
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _metrics(labels, predictions, probabilities) -> Dict[str, Any]:
        return {
            "accuracy": round(float(accuracy_score(labels, predictions)), 6),
            "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
            "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
            "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        }

    def _candidates(self):
        old_classifier_path = self.artifact_dir / "logistic_regression.pkl"
        old = pickle.loads(old_classifier_path.read_bytes())
        return [
            ("Existing LogisticRegression", old, False),
            ("Standardized LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=self.seed)), True),
            ("Linear SVM", make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True, random_state=self.seed)), True),
            ("RBF SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=self.seed)), True),
            ("Random Forest", RandomForestClassifier(n_estimators=250, class_weight="balanced", max_features="sqrt", random_state=self.seed, n_jobs=4), True),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, l2_regularization=1.0, random_state=self.seed), True),
            ("MLP 128", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), alpha=0.01, max_iter=120, early_stopping=True, random_state=self.seed)), True),
        ]

    def compare_and_select(self) -> Dict[str, Any]:
        train = self._load_cache(self.artifact_dir / "train_features.pt")
        validation = self._load_cache(self.artifact_dir / "validation_features.pt")
        x_train = train["features"].numpy()
        y_train = np.asarray(train["labels"], dtype=int)
        x_validation = validation["features"].numpy()
        y_validation = np.asarray(validation["labels"], dtype=int)
        rows: List[Dict[str, Any]] = []
        selected = None
        for name, classifier, needs_fit in self._candidates():
            started = time.perf_counter()
            if needs_fit:
                classifier.fit(x_train, y_train)
            prediction = classifier.predict(x_validation)
            probability = classifier.predict_proba(x_validation)[:, 1]
            metrics = self._metrics(y_validation, prediction, probability)
            row = {
                "model": name,
                "features": "MobileNetV3-Small frozen, cached 576-dimensional features",
                "preprocessing": "Existing 64x64 RGB and ImageNet normalization",
                **metrics,
                "training_seconds": round(time.perf_counter() - started, 3),
                "validation_count": len(y_validation),
            }
            rows.append(row)
            if selected is None or (row["f1"], row["roc_auc"], row["accuracy"]) > (selected["row"]["f1"], selected["row"]["roc_auc"], selected["row"]["accuracy"]):
                selected = {"row": row, "classifier": classifier}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        selected_path = self.artifact_dir / "validation_selected_classifier.pkl"
        selected_path.write_bytes(pickle.dumps(selected["classifier"]))
        report = {
            "selection_criterion": "Highest validation F1, then ROC-AUC, then accuracy",
            "training_source": "Existing train feature cache only",
            "selection_source": "Existing validation feature cache only",
            "official_testing_set_used_for_selection": False,
            "seed": self.seed,
            "selected_model": selected["row"]["model"],
            "selected_artifact": str(selected_path),
            "results": rows,
            "limitation": "Validation-based model selection; official TestingSet remains held out from this comparison.",
        }
        output = self.reports_dir / "phase13_visual_model_comparison.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def evaluate_selected_on_test_once(self) -> Dict[str, Any]:
        selected_path = self.artifact_dir / "validation_selected_classifier.pkl"
        if not selected_path.exists():
            raise FileNotFoundError("Run validation model selection before official test evaluation.")
        test_cache_path = Path("reports/phase12b_official_test_results.features.pt")
        test = self._load_cache(test_cache_path)
        classifier = pickle.loads(selected_path.read_bytes())
        comparison = json.loads((self.reports_dir / "phase13_visual_model_comparison.json").read_text(encoding="utf-8"))
        x_test = test["features"].numpy()
        y_test = np.asarray(test["labels"], dtype=int)
        prediction = classifier.predict(x_test)
        probability = classifier.predict_proba(x_test)[:, 1]
        report = {
            "dataset": "AIForge-Doc-v1",
            "split": "official TestingSet",
            "selected_model": comparison["selected_model"],
            "classifier_artifact": str(selected_path),
            "test_feature_cache": str(test_cache_path),
            "sample_count": len(y_test),
            "metrics": self._metrics(y_test, prediction, probability),
            "confusion_matrix": {
                "tn": int(((y_test == 0) & (prediction == 0)).sum()),
                "fp": int(((y_test == 0) & (prediction == 1)).sum()),
                "fn": int(((y_test == 1) & (prediction == 0)).sum()),
                "tp": int(((y_test == 1) & (prediction == 1)).sum()),
            },
            "warning": "One-time held-out evaluation after model selection was locked using validation F1. These are tampering-model results, not overall fraud-detection performance.",
        }
        output = self.reports_dir / "phase13_visual_model_official_test.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
