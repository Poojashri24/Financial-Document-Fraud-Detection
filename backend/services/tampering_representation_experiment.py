from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from backend.services.tampering_ml_baseline_service import TamperingBaselineService


@dataclass
class RepresentationExperimentConfig:
    manifest_path: str = "reports/phase12b1_train_validation_split.json"
    image_size: int = 96
    batch_size: int = 16
    max_epochs: int = 2
    patience: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.001
    seed: int = 42
    checkpoint_path: str = "models/tampering_feature_baseline/mobilenet_v3_small_finetuned_96.pt"
    report_path: str = "reports/phase13_representation_experiment.json"
    pretrained: bool = True
    unfreeze_last_feature_blocks: int = 3


class TamperingRepresentationExperiment:
    """Bounded MobileNet representation fine-tune selected only on validation."""

    def __init__(self, config: RepresentationExperimentConfig | None = None):
        self.config = config or RepresentationExperimentConfig()

    def build_model(self):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
        model = mobilenet_v3_small(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        blocks = list(model.features.children())
        unfreeze_from = max(0, len(blocks) - self.config.unfreeze_last_feature_blocks)
        for block in blocks[unfreeze_from:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return model

    def _metrics(self, labels: List[int], probabilities: List[float]) -> Dict[str, Any]:
        predictions = [int(value >= 0.5) for value in probabilities]
        return {
            "accuracy": round(float(accuracy_score(labels, predictions)), 6),
            "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
            "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
            "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6) if len(set(labels)) == 2 else None,
            "confusion_matrix": {
                "tn": sum(y == 0 and p == 0 for y, p in zip(labels, predictions)),
                "fp": sum(y == 0 and p == 1 for y, p in zip(labels, predictions)),
                "fn": sum(y == 1 and p == 0 for y, p in zip(labels, predictions)),
                "tp": sum(y == 1 and p == 1 for y, p in zip(labels, predictions)),
            },
            "sample_count": len(labels),
        }

    def _evaluate(self, model, loader, device):
        model.eval()
        labels: List[int] = []
        probabilities: List[float] = []
        with torch.inference_mode():
            for images, batch_labels, _paths in loader:
                logits = model(images.to(device))
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                labels.extend(int(label) for label in batch_labels.tolist())
        return self._metrics(labels, probabilities), labels, probabilities

    def _cache_prefix(self, prefix, loader):
        activations, labels = [], []
        prefix.eval()
        with torch.inference_mode():
            for images, batch_labels, _paths in loader:
                activations.append(prefix(images).cpu())
                labels.append(batch_labels.cpu())
        return torch.cat(activations), torch.cat(labels)

    def _evaluate_cached(self, tail, model, features, labels):
        model.eval()
        with torch.inference_mode():
            pooled = model.avgpool(tail(features)).flatten(1)
            logits = model.classifier(pooled)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        actual = labels.tolist()
        return self._metrics(actual, probabilities), actual, probabilities

    def run(self) -> Dict[str, Any]:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        base = TamperingBaselineService()
        manifest = base.load_manifest(self.config.manifest_path)
        counts = base.validate_manifest_records(manifest)
        records = base.resolve_records(manifest)
        train_ids = {item["image"] for item in records if item["split"] == "train"}
        validation_ids = {item["image"] for item in records if item["split"] == "validation"}
        if train_ids & validation_ids:
            raise ValueError("Train/validation image leakage detected.")
        if any("TestingSet" in item["image"] for item in records):
            raise ValueError("Official TestingSet records may not enter representation training or selection.")
        train_ds = base.build_dataset(records, "train", image_size=self.config.image_size)
        validation_ds = base.build_dataset(records, "validation", image_size=self.config.image_size)
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0, generator=generator)
        validation_loader = DataLoader(validation_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        model = self.build_model().to("cpu")
        blocks = list(model.features.children())
        cut = len(blocks) - self.config.unfreeze_last_feature_blocks
        prefix = nn.Sequential(*blocks[:cut])
        tail = nn.Sequential(*blocks[cut:])
        prefix_features_path = Path(self.config.checkpoint_path).with_name("train_frozen_prefix_features.pt")
        validation_features_path = Path(self.config.checkpoint_path).with_name("validation_frozen_prefix_features.pt")
        started = time.perf_counter()
        train_features, train_labels_tensor = self._cache_prefix(prefix, train_loader)
        extraction_seconds = time.perf_counter() - started
        validation_features, validation_labels_tensor = self._cache_prefix(prefix, validation_loader)
        prefix_features_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": train_features, "labels": train_labels_tensor, "split": "train"}, prefix_features_path)
        torch.save({"features": validation_features, "labels": validation_labels_tensor, "split": "validation"}, validation_features_path)
        train_labels = [item["label_id"] for item in records if item["split"] == "train"]
        class_counts = [max(train_labels.count(label), 1) for label in (0, 1)]
        class_weights = torch.tensor([len(train_labels) / (2 * count) for count in class_counts], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        history: List[Dict[str, Any]] = []
        best_f1 = -1.0
        best_state = None
        stale_epochs = 0
        training_started = time.perf_counter()
        for epoch in range(self.config.max_epochs):
            model.eval()
            tail.train()
            model.classifier.train()
            running_loss = 0.0
            order = torch.randperm(len(train_labels_tensor), generator=torch.Generator().manual_seed(self.config.seed + epoch))
            for batch_start in range(0, len(order), self.config.batch_size):
                batch_indices = order[batch_start:batch_start + self.config.batch_size]
                features = train_features[batch_indices]
                labels = train_labels_tensor[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                pooled = model.avgpool(tail(features)).flatten(1)
                logits = model.classifier(pooled)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(labels)
            validation_metrics, _labels, _probabilities = self._evaluate_cached(tail, model, validation_features, validation_labels_tensor)
            history.append({"epoch": epoch + 1, "train_loss": running_loss / len(train_ds), "validation": validation_metrics})
            if validation_metrics["f1"] > best_f1:
                best_f1 = validation_metrics["f1"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected.")
        model.load_state_dict(best_state)
        metrics, labels, probabilities = self._evaluate_cached(tail, model, validation_features, validation_labels_tensor)
        checkpoint = Path(self.config.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": asdict(self.config), "label_map": {"authentic": 0, "forged": 1}, "selected_validation_f1": best_f1}, checkpoint)
        result = {
            "model": "MobileNetV3-Small partial fine-tune",
            "features": "cached activations from frozen early blocks; final 3 MobileNetV3-Small feature blocks plus classifier fine-tuned",
            "preprocessing": {"rgb": True, "image_size": self.config.image_size, "normalization": "ImageNet mean/std", "train_augmentation": "small affine, crop, brightness/contrast from existing dataset builder"},
            "split_counts": counts,
            "validation_metrics": metrics,
            "history": history,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "feature_extraction_seconds": round(extraction_seconds, 3),
            "classifier_and_finetune_seconds": round(time.perf_counter() - training_started, 3),
            "seed": self.config.seed,
            "official_testing_set_used": False,
            "checkpoint": str(checkpoint),
            "cached_features": {"train": str(prefix_features_path), "validation": str(validation_features_path)},
            "selection_rule": "Best validation F1; early stopping on validation F1 only.",
            "pretrained_weights_downloaded": False,
        }
        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def evaluate_locked_model_on_official_test_once(self) -> Dict[str, Any]:
        """Evaluate the validation-locked checkpoint once; refuses to overwrite results."""
        test_report_path = Path("reports/phase13_representation_official_test.json")
        prediction_path = Path("reports/phase13_representation_official_test_predictions.json")
        if test_report_path.exists() or prediction_path.exists():
            raise FileExistsError("Official test evaluation artifacts already exist; refusing to evaluate again.")
        test_manifest_path = Path("reports/phase12b_official_testing_manifest.json")
        manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != "AIForge-Doc-v1" or manifest.get("source_split") != "TestingSet":
            raise ValueError("Official test manifest must identify AIForge-Doc-v1 TestingSet.")
        if not manifest.get("records"):
            raise ValueError("Official test manifest has no records.")
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("Torch is required for official test evaluation.") from exc
        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = self.build_model().cpu()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        records = TamperingBaselineService.resolve_records(manifest)
        test_records = []
        for record in records:
            record["split"] = "test"
            test_records.append(record)
        dataset = TamperingBaselineService.build_dataset(test_records, "test", image_size=self.config.image_size)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        labels: List[int] = []
        probabilities: List[float] = []
        paths: List[str] = []
        started = time.perf_counter()
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        with torch.inference_mode():
            for images, batch_labels, batch_paths in loader:
                logits = model(images.cpu())
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                labels.extend(int(label) for label in batch_labels.tolist())
                paths.extend(str(path) for path in batch_paths)
        metrics = self._metrics(labels, probabilities)
        predictions = [int(value >= 0.5) for value in probabilities]
        rows = [
            {"image": path, "label": label, "prediction": prediction, "forged_probability": probability}
            for path, label, prediction, probability in zip(paths, labels, predictions, probabilities)
        ]
        prediction_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        result = {
            "dataset": "AIForge-Doc-v1",
            "split": "official TestingSet",
            "model": "MobileNetV3-Small partial fine-tune at 96x96",
            "checkpoint": str(checkpoint_path),
            "manifest": str(test_manifest_path),
            "sample_count": len(labels),
            "class_counts": {"authentic": sum(label == 0 for label in labels), "forged": sum(label == 1 for label in labels)},
            "metrics": metrics,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "seed": self.config.seed,
            "selected_using": "validation F1 only",
            "classifier_fitted_on_test": False,
            "threshold_changed_on_test": False,
            "predictions_path": str(prediction_path),
            "warning": "Tampering-model results only; not overall fraud-detection performance.",
        }
        test_report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
