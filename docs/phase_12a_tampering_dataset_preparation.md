# Phase 12A - Tampering Dataset Preparation

## Scope

Phase 12A adds a local-only dataset registry and preparation layer for future tampering research. It does not train a model and does not change the existing upload, OCR, classification, extraction, verification, fusion, reasoning, or heuristic risk pipeline.

## Dataset Registry

The registry currently describes:

- **DocTamper**: real document-tampering dataset when authorized and configured locally. It is not downloaded automatically. It is `NOT_AVAILABLE` when its local path is absent.
- **AIForge-Doc**: optional real paired authentic/forged document dataset with masks when configured locally. It is not downloaded automatically.
- **SROIE**: real document-understanding and receipt-field dataset. It is not a fraud or tampering benchmark and its normal receipt images are not assigned tampering labels.

Each registry entry records dataset type, local path, image and mask directories, optional metadata, split information, label format, provenance/access notes, training/evaluation suitability, and purpose.

## Local Validation

For configured tampering datasets, validation checks:

- supported image extensions and readable images;
- image/mask filename correspondence when masks are expected;
- readable masks with binary-compatible values (`0`, `1`, or `255`);
- duplicate document identifiers;
- metadata identifier consistency when metadata exists;
- overlap between official train, validation, and test directories.

An unavailable dataset returns a clear `NOT_AVAILABLE` result and no invented sample count. The service never downloads data.

## Splits and Statistics

Datasets without official splits can be deterministically split into train, validation, and test sets with a fixed seed. Split records can be persisted as JSON. Document identifiers remain intact so the same document cannot leak across splits.

Statistics include total images, authentic/tampered counts when labels are supplied, split counts, mask availability, and mean tampered-pixel fraction when readable masks are available.

## Data Categories

### Real dataset

- DocTamper when authorized and locally configured.
- AIForge-Doc when locally configured.

### Document understanding

- SROIE, used for receipt understanding and field extraction only.

### Synthetic test data

- Existing project-generated PDFs, used only for integration testing. They are not research datasets and do not provide fraud labels.

## Reproducibility and Constraints

- CPU-only compatible.
- No GPU requirement.
- No large model installation.
- No automatic dataset download.
- No model training in Phase 12A.
- No fraud-detection accuracy or tampering-detection performance is reported by this phase.
