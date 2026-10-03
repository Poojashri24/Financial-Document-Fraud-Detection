from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_VALUES = {0, 1, 255}


@dataclass
class DatasetSpec:
    name: str
    dataset_type: str
    local_path: str | None = None
    available: bool = False
    image_directory: str | None = None
    mask_directory: str | None = None
    metadata_path: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    number_of_samples: int | None = None
    label_format: str = "unknown"
    license_access_notes: str = ""
    source_provenance: str = ""
    suitable_for_training: bool = False
    suitable_for_evaluation: bool = False
    expects_masks: bool = False
    purpose: str = ""
    official_splits: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TamperingDatasetService:
    """Local-only registry, validation, statistics, and deterministic splits."""

    def __init__(self, specs: Sequence[DatasetSpec] | None = None):
        self._specs: Dict[str, DatasetSpec] = {
            spec.name: spec for spec in (specs or self.default_specs())
        }

    @staticmethod
    def default_specs() -> List[DatasetSpec]:
        return [
            DatasetSpec(
                name="DocTamper",
                dataset_type="document_tampering",
                local_path=str(Path("data/raw/DocTamper")),
                image_directory="images",
                mask_directory="masks",
                label_format="authentic/tampered images with tampered-region masks (dataset-dependent)",
                license_access_notes="Requires authorized manual access; this service never downloads it automatically.",
                source_provenance="Configured local DocTamper directory only.",
                suitable_for_training=True,
                suitable_for_evaluation=True,
                expects_masks=True,
                purpose="document tampering research",
            ),
            DatasetSpec(
                name="AIForge-Doc",
                dataset_type="document_tampering",
                local_path=str(Path("data/raw/AIForge-Doc")),
                image_directory="images",
                mask_directory="masks",
                label_format="paired authentic/forged images with tampered-region masks",
                license_access_notes="Optional secondary dataset; configure or download manually when authorized.",
                source_provenance="Configured local AIForge-Doc directory only.",
                suitable_for_training=True,
                suitable_for_evaluation=True,
                expects_masks=True,
                purpose="paired document tampering research",
            ),
            DatasetSpec(
                name="SROIE",
                dataset_type="document_understanding",
                local_path=str(Path("data/raw/sroie")),
                label_format="receipt entities and OCR words; no tampering labels",
                license_access_notes="Available through the existing local Hugging Face cache when present.",
                source_provenance="SROIE / ICDAR2019-SROIE; local cache or configured local path.",
                suitable_for_training=False,
                suitable_for_evaluation=False,
                expects_masks=False,
                purpose="document understanding and field extraction; not a fraud/tampering benchmark",
            ),
        ]

    def register(self, spec: DatasetSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> DatasetSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {name}") from exc

    @staticmethod
    def _resolve_path(value: str | None) -> Path | None:
        return Path(value).expanduser() if value else None

    def status(self, name: str) -> Dict[str, Any]:
        spec = self.get(name)
        local_root = self._resolve_path(spec.local_path)
        local_available = bool(local_root and local_root.is_dir() and any(local_root.iterdir()))
        cache_available = False
        cache_path: Path | None = None
        if spec.name == "SROIE":
            cache_path = Path.home() / ".cache" / "huggingface" / "datasets" / "jsdnrs___icdar2019-sroie"
            cache_available = cache_path.is_dir()

        available = local_available or cache_available
        if available:
            reason = "configured local dataset path is available" if local_available else "local Hugging Face cache is available"
        elif not spec.local_path:
            reason = "local dataset path not configured"
        else:
            reason = "local dataset path is not configured or does not exist"

        payload = spec.to_dict()
        payload.update({
            "status": "AVAILABLE" if available else "NOT_AVAILABLE",
            "available": available,
            "reason": reason,
            "resolved_local_path": str(local_root) if local_root else None,
            "cache_path": str(cache_path) if cache_path else None,
        })
        if spec.name == "SROIE":
            payload["purpose"] = "document understanding / field extraction"
            payload["tampering_dataset"] = False
            payload["fraud_labels"] = False
        return payload

    def statuses(self) -> List[Dict[str, Any]]:
        return [self.status(name) for name in self._specs]

    @staticmethod
    def _image_files(directory: Path) -> List[Path]:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.rglob("*") if path.is_file())

    @staticmethod
    def _record_id(path: Path) -> str:
        return path.stem

    def discover_records(self, name: str) -> List[Dict[str, Any]]:
        spec = self.get(name)
        root = self._resolve_path(spec.local_path)
        image_dir = root / spec.image_directory if root and spec.image_directory else root
        if image_dir is None or not image_dir.is_dir():
            return []
        records = []
        for path in self._image_files(image_dir):
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            records.append({
                "document_id": self._record_id(path),
                "path": str(path),
                "label": None,
                "mask_path": None,
                "source": name,
            })
        return records

    def validate(self, name: str) -> Dict[str, Any]:
        spec = self.get(name)
        status = self.status(name)
        errors: List[str] = []
        warnings: List[str] = []
        records: List[Dict[str, Any]] = []
        root = self._resolve_path(spec.local_path)
        image_dir = root / spec.image_directory if root and spec.image_directory else root
        mask_dir = root / spec.mask_directory if root and spec.mask_directory else None

        if not status["available"] or not root or not root.is_dir():
            return {"dataset": name, "valid": False, "status": status["status"], "errors": [status["reason"]], "warnings": [], "records": []}
        if image_dir is None or not image_dir.is_dir():
            errors.append("image directory is missing")
        else:
            all_files = self._image_files(image_dir)
            unsupported = [str(path) for path in all_files if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS]
            if unsupported:
                errors.append(f"unsupported image formats found: {unsupported}")
            seen_ids: set[str] = set()
            for path in all_files:
                if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                record_id = self._record_id(path)
                if record_id in seen_ids:
                    errors.append(f"duplicate identifier: {record_id}")
                    continue
                seen_ids.add(record_id)
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    errors.append(f"unreadable image {path}: {exc}")
                    continue
                record = {"document_id": record_id, "path": str(path), "label": None, "mask_path": None, "source": name}
                if spec.expects_masks:
                    if mask_dir is None:
                        errors.append(f"mask directory is missing for {record_id}")
                    else:
                        mask_matches = [candidate for candidate in self._image_files(mask_dir) if candidate.stem == record_id]
                        if len(mask_matches) != 1:
                            errors.append(f"image/mask mismatch for {record_id}")
                        else:
                            mask_path = mask_matches[0]
                            try:
                                with Image.open(mask_path) as mask:
                                    mask.verify()
                                with Image.open(mask_path).convert("L") as mask:
                                    values = set(mask.getdata())
                                if not values or not values.issubset(MASK_VALUES):
                                    errors.append(f"mask contains unexpected values for {record_id}")
                                record["mask_path"] = str(mask_path)
                            except Exception as exc:
                                errors.append(f"unreadable mask {mask_path}: {exc}")
                records.append(record)

        metadata = self._load_metadata(spec)
        if metadata.get("error"):
            errors.append(metadata["error"])
        if metadata.get("records"):
            metadata_ids = {str(item.get("document_id")) for item in metadata["records"] if item.get("document_id")}
            record_ids = {item["document_id"] for item in records}
            if metadata_ids != record_ids:
                errors.append("metadata identifiers do not match discovered image identifiers")

        split_errors = self._validate_official_split_overlap(root, spec)
        errors.extend(split_errors)
        if not records and not errors:
            warnings.append("no supported images were discovered")
        return {
            "dataset": name,
            "valid": not errors,
            "status": status["status"],
            "errors": errors,
            "warnings": warnings,
            "records": records,
            "metadata": metadata,
        }

    def _load_metadata(self, spec: DatasetSpec) -> Dict[str, Any]:
        path = self._resolve_path(spec.metadata_path)
        if path is None and spec.local_path:
            candidate = Path(spec.local_path) / "metadata.json"
            path = candidate if candidate.exists() else None
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"path": str(path), "records": data if isinstance(data, list) else data.get("records", [])}
        except Exception as exc:
            return {"path": str(path), "error": f"metadata could not be read: {exc}"}

    @staticmethod
    def _validate_official_split_overlap(root: Path | None, spec: DatasetSpec) -> List[str]:
        if not root or not spec.official_splits:
            return []
        split_ids: Dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            split_root = root / split
            split_ids[split] = {path.stem for path in split_root.rglob("*") if path.is_file()} if split_root.is_dir() else set()
        errors = []
        names = list(split_ids)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                overlap = split_ids[first] & split_ids[second]
                if overlap:
                    errors.append(f"train/test split overlap: {sorted(overlap)}")
        return errors

    def statistics(self, name: str, records: Sequence[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        validation = self.validate(name) if records is None else {"records": list(records), "valid": True, "errors": []}
        items = list(records if records is not None else validation.get("records", []))
        authentic = sum(1 for item in items if item.get("label") in {0, "authentic", "real"})
        tampered = sum(1 for item in items if item.get("label") in {1, "tampered", "forged"})
        masks = [item.get("mask_path") for item in items if item.get("mask_path")]
        tampered_pixels = []
        for mask_path in masks:
            try:
                with Image.open(mask_path).convert("L") as mask:
                    pixels = list(mask.getdata())
                    tampered_pixels.append(sum(1 for value in pixels if value not in {0}) / len(pixels) if pixels else 0.0)
            except Exception:
                continue
        return {
            "dataset": name,
            "valid": validation.get("valid", True),
            "errors": validation.get("errors", []),
            "total_images": len(items),
            "authentic_count": authentic,
            "tampered_count": tampered,
            "train_count": sum(1 for item in items if item.get("split") == "train"),
            "validation_count": sum(1 for item in items if item.get("split") == "validation"),
            "test_count": sum(1 for item in items if item.get("split") == "test"),
            "mask_availability": len(masks),
            "tampered_pixel_fraction_mean": sum(tampered_pixels) / len(tampered_pixels) if tampered_pixels else None,
        }

    def prepare_splits(
        self,
        name: str,
        records: Sequence[Dict[str, Any]] | None = None,
        random_seed: int = 42,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
        output_path: str | None = None,
    ) -> Dict[str, Any]:
        items = [dict(item) for item in (records if records is not None else self.validate(name).get("records", []))]
        if not items:
            raise ValueError("No dataset records are available for splitting.")
        if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
            raise ValueError("Validation and test ratios must be non-negative and sum to less than one.")
        identifiers = [str(item.get("document_id")) for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate document identifiers cannot be split safely.")
        rng = random.Random(random_seed)
        shuffled = items[:]
        rng.shuffle(shuffled)
        test_count = int(len(shuffled) * test_ratio)
        validation_count = int(len(shuffled) * validation_ratio)
        if len(shuffled) > 1 and test_count == 0:
            test_count = 1
        if len(shuffled) > 2 and validation_count == 0:
            validation_count = 1
        train_count = len(shuffled) - validation_count - test_count
        if train_count < 1:
            train_count = 1
            test_count = max(0, len(shuffled) - validation_count - train_count)
        groups = {
            "train": shuffled[:train_count],
            "validation": shuffled[train_count:train_count + validation_count],
            "test": shuffled[train_count + validation_count:train_count + validation_count + test_count],
        }
        for split, group in groups.items():
            for item in group:
                item["split"] = split
        split_ids = {split: [item["document_id"] for item in group] for split, group in groups.items()}
        prepared = {
            "dataset": name,
            "random_seed": random_seed,
            "ratios": {"validation": validation_ratio, "test": test_ratio},
            "splits": split_ids,
            "records": [item for group in groups.values() for item in group],
            "no_document_leakage": len(set(split_ids["train"]) & set(split_ids["validation"])) == 0 and len(set(split_ids["train"]) & set(split_ids["test"])) == 0 and len(set(split_ids["validation"]) & set(split_ids["test"])) == 0,
        }
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
            prepared["persisted_to"] = str(path)
        return prepared
