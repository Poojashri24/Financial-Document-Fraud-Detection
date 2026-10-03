from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


LABEL_MAP = {"authentic": 0, "forged": 1}


@dataclass
class TamperingTrainingConfig:
    manifest_path: str = "reports/phase12b1_train_validation_split.json"
    model_dir: str = "models/tampering_baseline"
    reports_dir: str = "reports"
    seed: int = 42
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 0.001
    num_workers: int = 0
    pretrained: bool = True
    backbone: str = "mobilenet_v3_small"


class TamperingBaselineService:
    """CPU-friendly transfer-learning baseline isolated from the inference pipeline."""

    def __init__(self, config: TamperingTrainingConfig | None = None):
        self.config = config or TamperingTrainingConfig()

    @staticmethod
    def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "AIForge-Doc-v1":
            raise ValueError("Manifest is not for AIForge-Doc-v1.")
        if payload.get("source_split") != "TrainingSet only":
            raise ValueError("Manifest must use TrainingSet only.")
        if not payload.get("records"):
            raise ValueError("Manifest contains no records.")
        return payload

    @staticmethod
    def label_for_record(record: Dict[str, Any]) -> int:
        label = record.get("label")
        if label not in LABEL_MAP:
            raise ValueError(f"Unsupported tampering label: {label}")
        return LABEL_MAP[label]

    @staticmethod
    def resolve_records(manifest: Dict[str, Any], source_root: str | Path | None = None) -> List[Dict[str, Any]]:
        root = Path(source_root or manifest.get("source_root", "."))
        records: List[Dict[str, Any]] = []
        for record in manifest.get("records", []):
            item = dict(record)
            item["label_id"] = TamperingBaselineService.label_for_record(item)
            item["image_path"] = str(root / item["image"])
            if item["mask"] is not None:
                item["mask_path"] = str(root / item["mask"])
            records.append(item)
        return records

    @staticmethod
    def validate_manifest_records(manifest: Dict[str, Any]) -> Dict[str, Any]:
        records = manifest.get("records", [])
        train = [item for item in records if item.get("split") == "train"]
        validation = [item for item in records if item.get("split") == "validation"]
        train_ids = {item.get("image") for item in train}
        validation_ids = {item.get("image") for item in validation}
        if train_ids & validation_ids:
            raise ValueError("Manifest has train/validation image overlap.")
        for item in records:
            label = TamperingBaselineService.label_for_record(item)
            if label == 1 and not item.get("mask"):
                raise ValueError(f"Forged record has no mask: {item.get('image')}")
            if label == 0 and item.get("mask") is not None:
                raise ValueError(f"Authentic record unexpectedly has a mask: {item.get('image')}")
        return {
            "total": len(records),
            "train": len(train),
            "validation": len(validation),
            "train_authentic": sum(1 for item in train if item.get("label") == "authentic"),
            "train_forged": sum(1 for item in train if item.get("label") == "forged"),
            "validation_authentic": sum(1 for item in validation if item.get("label") == "authentic"),
            "validation_forged": sum(1 for item in validation if item.get("label") == "forged"),
        }

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    @staticmethod
    def build_dataset(records: Sequence[Dict[str, Any]], split: str, image_size: int = 128):
        try:
            from PIL import Image
            from torch.utils.data import Dataset
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("Torch, torchvision, and Pillow are required for the ML baseline.") from exc

        selected = [item for item in records if item.get("split") == split]
        if not selected:
            raise ValueError(f"No records found for split: {split}")
        if split == "train":
            transform = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        class ManifestDataset(Dataset):
            def __len__(self):
                return len(selected)

            def __getitem__(self, index):
                item = selected[index]
                with Image.open(item["image_path"]) as image:
                    image = image.convert("RGB")
                return transform(image), item["label_id"], item["image"]

        return ManifestDataset()

    @staticmethod
    def build_model(pretrained: bool = True):
        try:
            import torch.nn as nn
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError("Torch and torchvision are required for the ML baseline.") from exc
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, 2)
        return model

    @staticmethod
    def calculate_metrics(labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
        if not labels or len(labels) != len(predictions) or len(labels) != len(probabilities):
            raise ValueError("Labels, predictions, and probabilities must be non-empty and aligned.")
        tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
        tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
        fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
        fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        roc_auc = None
        if len(set(labels)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = float(roc_auc_score(labels, probabilities))
            except ImportError:
                roc_auc = None
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "sample_count": len(labels),
        }

    def train(self) -> Dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("The active Python environment lacks torch/torchvision; use the verified system Python CPU environment.") from exc

        self.seed_everything(self.config.seed)
        manifest = self.load_manifest(self.config.manifest_path)
        manifest_counts = self.validate_manifest_records(manifest)
        records = self.resolve_records(manifest)
        train_dataset = self.build_dataset(records, "train", self.config.image_size)
        validation_dataset = self.build_dataset(records, "validation", self.config.image_size)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        model = self.build_model(pretrained=self.config.pretrained)
        device = torch.device("cpu")
        model.to(device)
        train_labels = [item["label_id"] for item in records if item["split"] == "train"]
        class_counts = [max(train_labels.count(index), 1) for index in (0, 1)]
        class_weights = torch.tensor([len(train_labels) / (2 * count) for count in class_counts], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=self.config.learning_rate)
        history: List[Dict[str, Any]] = []
        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0
            for images, labels, _ in train_loader:
                optimizer.zero_grad()
                logits = model(images.to(device))
                loss = criterion(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(labels)
            history.append({"epoch": epoch + 1, "train_loss": running_loss / len(train_dataset)})
        model.eval()
        labels: List[int] = []
        predictions: List[int] = []
        probabilities: List[float] = []
        prediction_rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for images, batch_labels, paths in validation_loader:
                logits = model(images.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long()
                for path, label, prediction, probability in zip(paths, batch_labels.tolist(), preds.tolist(), probs.tolist()):
                    labels.append(int(label))
                    predictions.append(int(prediction))
                    probabilities.append(float(probability))
                    prediction_rows.append({"image": path, "label": int(label), "prediction": int(prediction), "probability": float(probability)})
        metrics = self.calculate_metrics(labels, predictions, probabilities)
        model_dir = Path(self.config.model_dir)
        reports_dir = Path(self.config.reports_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "aiForge_doc_v1_mobilenet_v3_small.pt"
        torch.save({"state_dict": model.state_dict(), "config": asdict(self.config), "label_map": LABEL_MAP}, checkpoint_path)
        (reports_dir / "phase12b_tampering_training_config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        (reports_dir / "phase12b_tampering_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (reports_dir / "phase12b_tampering_validation_predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
        result = {
            "status": "evaluated",
            "dataset": "AIForge-Doc-v1",
            "manifest": self.config.manifest_path,
            "split": "validation",
            "device": str(device),
            "backbone": self.config.backbone,
            "pretrained_backbone": self.config.pretrained,
            "frozen_backbone": True,
            "manifest_counts": manifest_counts,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "history_path": str(reports_dir / "phase12b_tampering_training_history.json"),
            "predictions_path": str(reports_dir / "phase12b_tampering_validation_predictions.json"),
            "warning": "Validation metrics are from the AIForge-Doc-v1 training subset split only; TestingSet was not used.",
        }
        (reports_dir / "phase12b_tampering_baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
