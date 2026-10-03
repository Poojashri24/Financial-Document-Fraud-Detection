from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


class SROIEDatasetAdapter:
    def __init__(self, root: str | None = None):
        self.root = root or str((Path("data") / "raw" / "sroie").resolve())

    def describe(self) -> Dict[str, Any]:
        root = Path(self.root)
        exists = root.exists()
        available = exists and any(root.iterdir()) if exists else False
        return {
            "dataset_name": "SROIE",
            "status": "available" if available else "referenced_but_unavailable",
            "root": str(root),
            "available": available,
            "structure": "SROIE directory present and available for dataset preparation." if available else "SROIE is referenced but not populated locally.",
        }

    def list_samples(self) -> List[str]:
        root = Path(self.root)
        if not root.exists():
            return []
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".pdf"}:
                files.append(str(path))
        return files

    def load_samples(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for index, file_path in enumerate(self.list_samples(), start=1):
            document_id = Path(file_path).stem
            records.append({
                "document_id": document_id,
                "label": None,
                "source": "sroie",
                "path": file_path,
                "metadata": {"page_count": 1, "file_name": Path(file_path).name},
                "split": None,
            })
        return records


class DatasetPreparationService:
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        if not record.get("document_id"):
            return False
        if not record.get("path"):
            return False
        if "label" not in record:
            return False
        return True

    @staticmethod
    def prepare_records(
        records: Sequence[Dict[str, Any]],
        dataset_name: str = "dataset",
        random_seed: int = 42,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        if records is None:
            raise ValueError("Dataset records must not be None.")

        normalized = []
        for index, record in enumerate(records):
            if not DatasetPreparationService.validate_record(record):
                raise ValueError(f"Record at index {index} is missing document_id, path, or label.")
            doc_id = str(record["document_id"])
            normalized.append({
                "document_id": doc_id,
                "label": record["label"],
                "source": record.get("source", dataset_name),
                "path": str(record["path"]),
                "metadata": record.get("metadata") or {},
                "ground_truth": record.get("ground_truth"),
            })

        if not normalized:
            raise ValueError("No valid dataset records were supplied.")

        rng = random.Random(random_seed)
        shuffled = normalized[:]
        rng.shuffle(shuffled)

        total = len(shuffled)
        validation_count = int(total * validation_ratio)
        test_count = int(total * test_ratio)
        train_count = max(total - validation_count - test_count, 0)

        if total > 0 and train_count == 0:
            train_count = max(1, total - validation_count - test_count)

        train = shuffled[:train_count]
        validation = shuffled[train_count:train_count + validation_count]
        test = shuffled[train_count + validation_count:train_count + validation_count + test_count]

        if len(test) < 1 and total > 1:
            test = [shuffled[-1]]
            validation = [shuffled[-2]] if len(shuffled) > 2 else []
            reserved_ids = {item["document_id"] for item in test + validation}
            train = [sample for sample in shuffled if sample["document_id"] not in reserved_ids]

        split_map = {
            "train": [item["document_id"] for item in train],
            "validation": [item["document_id"] for item in validation],
            "test": [item["document_id"] for item in test],
        }
        overlaps = set(split_map["train"]) & set(split_map["validation"]) | set(split_map["train"]) & set(split_map["test"]) | set(split_map["validation"]) & set(split_map["test"])
        if overlaps:
            raise ValueError("Data leakage detected: document IDs overlap between train, validation, and test splits.")

        prepared = {
            "sample_count": len(normalized),
            "dataset_name": dataset_name,
            "splits": split_map,
            "split_sizes": {name: len(ids) for name, ids in split_map.items()},
            "records": normalized,
            "reproducibility": {
                "dataset_name": dataset_name,
                "dataset_version": "unknown",
                "source": "synthetic_or_reference_dataset",
                "random_seed": random_seed,
                "preprocessing": {
                    "shuffle": True,
                    "validation_ratio": validation_ratio,
                    "test_ratio": test_ratio,
                },
                "evaluation_configuration": {
                    "strategy": "deterministic_split",
                    "no_data_leakage": True,
                },
            },
        }
        return prepared

    @staticmethod
    def dataset_status(root: str | None = None) -> Dict[str, Any]:
        root_path = Path(root or str(Path("data") / "raw"))
        sroie = SROIEDatasetAdapter(root=str(root_path / "sroie")).describe()
        doctamper = __import__("backend.services.tampering_detection_service", fromlist=["DocTamperDatasetAdapter"]).DocTamperDatasetAdapter(root=str(root_path)).describe()
        return {
            "status": "available" if sroie["available"] or doctamper["available"] else "referenced_but_unavailable",
            "dataset_name": "SROIE",
            "sroie": sroie,
            "doctamper": doctamper,
            "synthetic_or_test_data": "available when sample fixtures are used in tests",
        }
