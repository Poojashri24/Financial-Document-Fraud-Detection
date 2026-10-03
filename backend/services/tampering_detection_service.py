from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


class BaselineTamperingDetector:
    @staticmethod
    def _safe_open(image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert("RGB")
            return np.array(image)
        except Exception:
            return None

    @staticmethod
    def _compute_local_stats(image: np.ndarray) -> Tuple[float, float, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        variance = float(np.var(gray))
        return mean, std, variance

    @staticmethod
    def detect_noise_inconsistency(image: np.ndarray) -> Optional[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        mean_diff = float(diff.mean())
        if mean_diff > 18:
            return {
                "detector": "baseline_noise",
                "finding": "local noise inconsistency",
                "severity": "MEDIUM",
                "region": None,
                "explanation": "The image shows unusually strong local noise or blur mismatch, which can be consistent with tampering or post-processing.",
            }
        return None

    @staticmethod
    def detect_texture_change(image: np.ndarray) -> Optional[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        energy = float(np.var(laplacian))
        if energy > 850:
            return {
                "detector": "baseline_texture",
                "finding": "suspicious texture change",
                "severity": "MEDIUM",
                "region": None,
                "explanation": "The local texture energy is unusually high relative to a normal document scan, suggesting possible copy-and-paste or region editing.",
            }
        return None

    @staticmethod
    def detect_edge_anomaly(image: np.ndarray) -> Optional[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)
        if edge_ratio > 0.38:
            return {
                "detector": "baseline_edges",
                "finding": "edge-region inconsistency",
                "severity": "LOW",
                "region": None,
                "explanation": "The edge structure is unusually dense for a standard document-like image, which may indicate localized editing or synthetic content insertion.",
            }
        return None

    @staticmethod
    def detect_metadata_anomaly(image_path: str) -> Optional[Dict[str, Any]]:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif:
                    return None
                if len(exif) > 40:
                    return {
                        "detector": "baseline_metadata",
                        "finding": "metadata anomaly",
                        "severity": "LOW",
                        "region": None,
                        "explanation": "The image contains unusually large metadata content for a simple scanned or generated document image, which may be suspicious when combined with other evidence.",
                    }
        except Exception:
            return None
        return None

    @staticmethod
    def detect_image_statistic_shift(image: np.ndarray) -> Optional[Dict[str, Any]]:
        height, width = image.shape[:2]
        if height < 20 or width < 20:
            return None
        tile = image[height // 4 : (3 * height) // 4, width // 4 : (3 * width) // 4]
        mean, std, variance = BaselineTamperingDetector._compute_local_stats(tile)
        if std > 90 and variance > 7000:
            return {
                "detector": "baseline_statistics",
                "finding": "suspicious local intensity shift",
                "severity": "MEDIUM",
                "region": [width // 4, height // 4, (3 * width) // 4, (3 * height) // 4],
                "explanation": "A strongly shifted local intensity profile appears in the central document area, which can indicate a modified region or foreign content overlay.",
            }
        return None

    @staticmethod
    def detect_document(image_path: str) -> List[Dict[str, Any]]:
        image = BaselineTamperingDetector._safe_open(image_path)
        if image is None:
            return []

        findings: List[Dict[str, Any]] = []
        for detector in (
            BaselineTamperingDetector.detect_noise_inconsistency,
            BaselineTamperingDetector.detect_texture_change,
            BaselineTamperingDetector.detect_edge_anomaly,
            BaselineTamperingDetector.detect_image_statistic_shift,
            BaselineTamperingDetector.detect_metadata_anomaly,
        ):
            if detector == BaselineTamperingDetector.detect_metadata_anomaly:
                result = detector(image_path)
            else:
                result = detector(image)
            if result is not None:
                findings.append(result)
        return findings


class MLTamperingDetector:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path

    def load(self) -> bool:
        return bool(self.checkpoint_path) and Path(self.checkpoint_path).exists()

    def predict(self, image_path: str) -> Dict[str, Any]:
        if not self.load():
            return {
                "status": "not_loaded",
                "tampering_probability": 0.0,
                "findings": [],
                "regions": [],
                "details": "No trained ML checkpoint is available yet. This detector is reserved for a future trained model.",
            }
        return {
            "status": "loaded",
            "tampering_probability": 0.0,
            "findings": [],
            "regions": [],
            "details": "This placeholder ML detector is not trained in this phase. The checkpoint must be trained and evaluated before being used in production.",
        }


class DocTamperDatasetAdapter:
    def __init__(self, root: Optional[str] = None):
        self.root = root or str(Path("data/raw").resolve())

    def detect_dataset(self) -> bool:
        possible = [
            Path(self.root) / "DocTamper",
            Path(self.root) / "doctamper",
            Path(self.root) / "doc_tamper",
            Path(self.root) / "DocTamperDataset",
        ]
        return any(path.exists() for path in possible)

    def get_dataset_root(self) -> Optional[str]:
        candidates = [
            Path(self.root) / "DocTamper",
            Path(self.root) / "doctamper",
            Path(self.root) / "doc_tamper",
            Path(self.root) / "DocTamperDataset",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def available(self) -> bool:
        return self.get_dataset_root() is not None

    def describe(self) -> Dict[str, Any]:
        root = self.get_dataset_root()
        if root is None:
            return {
                "available": False,
                "root": None,
                "structure": "DocTamper not found locally. The baseline forensic detector remains active.",
            }
        return {
            "available": True,
            "root": root,
            "structure": "DocTamper directory detected; adapter is prepared for future labeling and split access when the dataset is actually present.",
        }

    def iter_images(self) -> List[str]:
        root = self.get_dataset_root()
        if root is None:
            return []
        images: List[str] = []
        for directory, _, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append(str(Path(directory) / name))
        return sorted(images)

    def train_validation_test_splits(self) -> Dict[str, Any]:
        images = self.iter_images()
        if not images:
            return {
                "available": False,
                "train": [],
                "validation": [],
                "test": [],
                "message": "DocTamper is not available locally, so no dataset splits are generated in this phase.",
            }
        return {
            "available": True,
            "train": images[: int(len(images) * 0.7)],
            "validation": images[int(len(images) * 0.7): int(len(images) * 0.85)],
            "test": images[int(len(images) * 0.85):],
            "message": "Dataset adapter is ready; training and validation logic can be activated once a trained model is explicitly requested.",
        }


class TamperingDetectionService:
    def __init__(self, detector_type: str = "baseline_forensic"):
        self.detector_type = detector_type

    def analyze_document(self, document_id: str, image_path: Optional[str] = None, doc_type: str = "document") -> Dict[str, Any]:
        if image_path is None:
            image_path = str(Path("data") / "uploads" / document_id / "document.png")

        normalized_path = str(Path(image_path))
        if not Path(normalized_path).exists():
            raise FileNotFoundError(f"Document image not found for {document_id}.")

        findings = BaselineTamperingDetector.detect_document(normalized_path)
        suspicious_regions = []
        for item in findings:
            region = item.get("region")
            if isinstance(region, (list, tuple)) and len(region) == 4:
                suspicious_regions.append(list(region))

        risk_score = 0
        severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        for finding in findings:
            risk_score += severity_order.get(str(finding.get("severity", "LOW")).upper(), 1)

        return {
            "document_id": document_id,
            "analysis_id": str(uuid.uuid4()),
            "detector_type": self.detector_type,
            "tampering_detected": bool(findings),
            "risk_score": min(risk_score, 100),
            "findings": findings,
            "suspicious_regions": suspicious_regions,
            "limitations": [
                "This detector is a baseline forensic heuristic and does not prove fraud.",
                "Evidence indicates suspicious characteristics only when the detector sees them in the image.",
                "A trained model and labeled dataset are required before claiming learned tampering performance.",
            ],
        }


class TamperingVisualizationUtility:
    @staticmethod
    def generate_annotated_image(source_image_path: str, findings: Sequence[Dict[str, Any]], output_path: str) -> str:
        try:
            image = Image.open(source_image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to open source image for annotation: {exc}") from exc

        annotated = image.copy()
        draw = None
        try:
            import PIL.ImageDraw as ImageDraw
            draw = ImageDraw.Draw(annotated)
        except Exception:
            draw = None

        regions = []
        for finding in findings:
            region = finding.get("region")
            if isinstance(region, (list, tuple)) and len(region) == 4:
                regions.append((region, finding.get("severity", "LOW"), finding.get("finding", "suspicious region")))

        if draw is not None:
            for region, severity, label in regions:
                x1, y1, x2, y2 = [int(v) for v in region]
                color = {
                    "LOW": "yellow",
                    "MEDIUM": "orange",
                    "HIGH": "red",
                }.get(str(severity).upper(), "yellow")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, max(0, y1 - 16)), label, fill=color)

        if not regions:
            if draw is not None:
                draw.text((10, 10), "No suspicious regions detected by the baseline detector.", fill="green")

        annotated.save(output_path)
        return output_path
