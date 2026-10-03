from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from backend.services.tampering_ml_baseline_service import LABEL_MAP, TamperingBaselineService


@dataclass
class FeatureBaselineConfig:
    manifest_path: str = "reports/phase12b1_train_validation_split.json"
    artifact_dir: str = "models/tampering_feature_baseline"
    reports_dir: str = "reports"
    seed: int = 42
    image_size: int = 64
    batch_size: int = 64
    max_iter: int = 200
    pretrained: bool = True
    backbone: str = "mobilenet_v3_small"


class TamperingFeatureBaselineService:
    """Single-pass frozen MobileNet feature extraction plus a linear classifier."""

    def __init__(self, config: FeatureBaselineConfig | None = None):
        self.config = config or FeatureBaselineConfig()

    @staticmethod
    def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
        return TamperingBaselineService.load_manifest(manifest_path)

    @staticmethod
    def resolve_records(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        return TamperingBaselineService.resolve_records(manifest)

    @staticmethod
    def build_feature_extractor(pretrained: bool = True):
        try:
            import torch.nn as nn
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError("Torch and torchvision are required for feature extraction.") from exc
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        for parameter in extractor.parameters():
            parameter.requires_grad = False
        extractor.eval()
        return extractor

    @staticmethod
    def build_dataset(records: Sequence[Dict[str, Any]], split: str, image_size: int = 64):
        return TamperingBaselineService.build_dataset(records, split, image_size=image_size)

    @staticmethod
    def _save_feature_cache(path: Path, features, labels: Sequence[int], image_paths: Sequence[str]) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features.cpu(), "labels": list(labels), "image_paths": list(image_paths)}, path)

    @staticmethod
    def load_feature_cache(path: str | Path) -> Dict[str, Any]:
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)

    def extract_features(self, extractor, dataset, cache_path: str | Path) -> Dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        feature_batches = []
        labels: List[int] = []
        image_paths: List[str] = []
        extractor.eval()
        with torch.inference_mode():
            for images, batch_labels, paths in loader:
                feature_batches.append(extractor(images))
                labels.extend(int(label) for label in batch_labels.tolist())
                image_paths.extend(str(path) for path in paths)
        features = torch.cat(feature_batches, dim=0)
        path = Path(cache_path)
        self._save_feature_cache(path, features, labels, image_paths)
        return {"features": features, "labels": labels, "image_paths": image_paths, "cache_path": str(path)}

    @staticmethod
    def train_classifier(train_features, train_labels: Sequence[int], seed: int = 42, max_iter: int = 200):
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the feature classifier.") from exc
        classifier = LogisticRegression(random_state=seed, max_iter=max_iter, class_weight="balanced", solver="liblinear")
        classifier.fit(train_features.numpy(), list(train_labels))
        return classifier

    @staticmethod
    def calculate_metrics(labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
        return TamperingBaselineService.calculate_metrics(labels, predictions, probabilities)

    def predict_image(self, image_path: str | Path, classifier_path: str | Path) -> Dict[str, Any]:
        """Run the saved classifier on one image without fitting any model."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The active Python environment lacks torch/torchvision.") from exc
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        extractor = self.build_feature_extractor(pretrained=True)
        with Path(classifier_path).open("rb") as handle:
            classifier = pickle.load(handle)
        with torch.inference_mode():
            features = extractor(tensor).numpy()
        forged_probability = float(classifier.predict_proba(features)[0][1])
        prediction = "forged" if int(classifier.predict(features)[0]) == 1 else "authentic"
        return {
            "prediction": prediction,
            "forged_probability": forged_probability,
            "confidence": max(forged_probability, 1.0 - forged_probability),
            "model_name": "AIForge-Doc-v1 MobileNetV3-Small feature baseline",
            "model_version": {
                "backbone": self.config.backbone,
                "pretrained": True,
                "image_size": self.config.image_size,
                "normalization": "ImageNet mean/std",
                "classifier": "sklearn LogisticRegression",
                "classifier_artifact": str(classifier_path),
            },
            "evidence_source": "trained_visual_model",
            "calibration_note": "Forged probability is a model output, not a calibrated fraud probability or proof of fraud.",
        }

    @staticmethod
    def _load_evaluation_manifest(manifest_path: str | Path) -> Dict[str, Any]:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != "AIForge-Doc-v1":
            raise ValueError("Manifest is not for AIForge-Doc-v1.")
        if manifest.get("source_split") != "TestingSet":
            raise ValueError("Evaluation manifest must identify TestingSet as its source split.")
        if not manifest.get("records"):
            raise ValueError("Evaluation manifest contains no records.")
        return manifest

    def evaluate_manifest(
        self,
        manifest_path: str | Path,
        classifier_path: str | Path,
        split: str = "test",
        report_path: str | Path = "reports/phase12b_testing_evaluation.json",
    ) -> Dict[str, Any]:
        """Evaluate the saved classifier on a separate, already prepared manifest."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The active Python environment lacks torch/torchvision.") from exc
        started = time.perf_counter()
        base = TamperingBaselineService()
        base.seed_everything(self.config.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        manifest = self._load_evaluation_manifest(manifest_path)
        records = base.resolve_records(manifest)
        dataset = self.build_dataset(records, split, self.config.image_size)
        extractor = self.build_feature_extractor(pretrained=True)
        feature_data = self.extract_features(extractor, dataset, Path(report_path).with_suffix(".features.pt"))
        with Path(classifier_path).open("rb") as handle:
            classifier = pickle.load(handle)
        labels = [int(label) for label in feature_data["labels"]]
        probabilities = classifier.predict_proba(feature_data["features"].numpy())[:, 1].tolist()
        predictions = classifier.predict(feature_data["features"].numpy()).tolist()
        metrics = self.calculate_metrics(labels, predictions, probabilities)
        prediction_rows = [
            {
                "image": path,
                "label": int(label),
                "prediction": int(prediction),
                "probability": float(probability),
            }
            for path, label, prediction, probability in zip(
                feature_data["image_paths"], labels, predictions, probabilities
            )
        ]
        predictions_path = Path(report_path).with_name("phase12b_official_test_predictions.json")
        predictions_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
        report = {
            "status": "evaluated",
            "dataset": "AIForge-Doc-v1",
            "split": "TestingSet",
            "manifest": str(manifest_path),
            "classifier_artifact": str(classifier_path),
            "backbone": self.config.backbone,
            "pretrained_backbone": True,
            "backbone_frozen": True,
            "preprocessing": {
                "color_mode": "RGB",
                "image_size": [self.config.image_size, self.config.image_size],
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "batch_size": self.config.batch_size,
                "feature_extraction": "torch.inference_mode",
            },
            "sample_count": len(labels),
            "class_counts": {
                "authentic": sum(label == 0 for label in labels),
                "forged": sum(label == 1 for label in labels),
            },
            "metrics": metrics,
            "predictions_path": str(predictions_path),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "seed": self.config.seed,
            "classifier_fitted": False,
            "testing_set_used": True,
        }
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def run(self) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The active Python environment lacks torch/torchvision; use the verified system Python CPU environment.") from exc
        started = time.perf_counter()
        base = TamperingBaselineService()
        base.seed_everything(self.config.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        manifest = base.load_manifest(self.config.manifest_path)
        manifest_counts = base.validate_manifest_records(manifest)
        records = base.resolve_records(manifest)
        train_dataset = self.build_dataset(records, "train", self.config.image_size)
        validation_dataset = self.build_dataset(records, "validation", self.config.image_size)
        extractor = self.build_feature_extractor(pretrained=self.config.pretrained)
        artifact_dir = Path(self.config.artifact_dir)
        reports_dir = Path(self.config.reports_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        train_data = self.extract_features(extractor, train_dataset, artifact_dir / "train_features.pt")
        validation_data = self.extract_features(extractor, validation_dataset, artifact_dir / "validation_features.pt")
        classifier = self.train_classifier(train_data["features"], train_data["labels"], self.config.seed, self.config.max_iter)
        probabilities = classifier.predict_proba(validation_data["features"].numpy())[:, 1].tolist()
        predictions = classifier.predict(validation_data["features"].numpy()).tolist()
        metrics = self.calculate_metrics(validation_data["labels"], predictions, probabilities)
        classifier_path = artifact_dir / "logistic_regression.pkl"
        classifier_path.write_bytes(pickle.dumps(classifier))
        config_path = reports_dir / "phase12b_feature_baseline_config.json"
        predictions_path = reports_dir / "phase12b_feature_baseline_validation_predictions.json"
        metrics_path = reports_dir / "phase12b_feature_baseline.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        prediction_rows = [
            {"image": path, "label": int(label), "prediction": int(prediction), "probability": float(probability)}
            for path, label, prediction, probability in zip(validation_data["image_paths"], validation_data["labels"], predictions, probabilities)
        ]
        predictions_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
        result = {
            "status": "evaluated",
            "dataset": manifest["dataset"],
            "manifest": self.config.manifest_path,
            "split": "validation",
            "seed": self.config.seed,
            "device": "cpu",
            "backbone": self.config.backbone,
            "pretrained_backbone": self.config.pretrained,
            "backbone_frozen": True,
            "classifier": "sklearn LogisticRegression",
            "manifest_counts": manifest_counts,
            "metrics": metrics,
            "artifacts": {
                "train_features": str(artifact_dir / "train_features.pt"),
                "validation_features": str(artifact_dir / "validation_features.pt"),
                "classifier": str(classifier_path),
                "config": str(config_path),
                "predictions": str(predictions_path),
            },
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "warning": "Validation metrics use only the AIForge-Doc-v1 TrainingSet manifest split; TestingSet was not used.",
        }
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
