# Methodology

## Prototype Scope

This project follows a phased academic prototype approach. The current implementation focuses on robust infrastructure, OCR, and transparent rules-based classification and extraction without claiming trained AI performance.

## Current Methodology

1. Validate uploaded file types and document structure.
2. Assign a unique document ID and store the file safely.
3. Convert PDF pages to images when needed for OCR.
4. Use Tesseract OCR to extract text from the document.
5. Classify the document type with a rule-based heuristic that uses OCR phrases and labels.
6. Extract structured fields by document type and normalize values where possible.
7. Mask sensitive identifiers such as account numbers and document numbers.
8. Keep all logic CPU-friendly and synthetic-data-driven.

## Final Multimodal Architecture

The application combines real PDF/image processing, Tesseract OCR, rule-based document classification and structured extraction, rule-based cross-document verification, two separate tampering evidence sources, multimodal fusion, deterministic evidence-grounded reasoning, and heuristic risk scoring.

The tampering stage preserves the baseline forensic heuristic and, when its artifact and ML runtime are available, adds the frozen MobileNetV3-Small plus LogisticRegression baseline as `trained_visual_model` evidence. The model prediction and forged probability are visual-model outputs, not proof of fraud and not calibrated fraud probabilities.

The risk score remains a heuristic evidence score, not a calibrated fraud probability. Explicit cross-document FAIL/WARNING results contribute once. Provenance-tagged heuristic findings are not also scored as generic severity findings; their aggregate contribution is capped at 12 points. A trained visual prediction contributes 4 points only for a `forged` class prediction and never uses its probability as a fraud probability; `authentic` is informational. GNN evidence is advisory and contributes zero direct points. Matching documents can still have visual observations without any cross-document inconsistency.

AIForge-Doc-v1 is used for tampering-model training/validation and official held-out testing. SROIE is used for document understanding and receipt-field context, not as a fraud benchmark. Synthetic salary/bank PDFs are software integration fixtures only.

## Phase 13 Representation Learning and Graph Evidence

The original frozen MobileNetV3-Small + LogisticRegression baseline is preserved. A separate 96x96 MobileNetV3-Small partial fine-tuning experiment caches activations from frozen early blocks and fine-tunes the last three feature blocks and classification head. The fixed seed is 42; class-weighted loss is used; the existing training augmentation is applied only to training samples; validation has deterministic resize/tensor/ImageNet normalization. The selected checkpoint is selected using validation F1 only.

The validation-selected fine-tune did not improve the official held-out test result, so it is not promoted over the original active LogisticRegression baseline. Its checkpoint and metrics are retained as an experimental result only.

The cross-document GNN is a small two-layer PyG GCN with an edge classifier. Its supervision labels come from outcomes of the existing deterministic verification rules on a very small, explicitly synthetic scenario set. The train/validation/test scenario IDs are disjoint, but this is rule distillation, not independent graph ground truth, and its metrics must not be described as fraud accuracy or independent consistency generalization. The model is advisory; explicit rule outcomes remain authoritative.

## Phase 12A - Tampering Dataset Preparation

Phase 12A adds a local-only registry and validation layer for DocTamper and AIForge-Doc, while retaining SROIE strictly for document understanding and field extraction. The layer does not download datasets, assign tampering labels to SROIE, train models, or alter the existing document-analysis pipeline. See [phase_12a_tampering_dataset_preparation.md](phase_12a_tampering_dataset_preparation.md) for validation, split, and provenance rules.

## Design Principles

- modular code
- synthetic datasets for demonstration
- CPU compatibility
- explainability before black-box claims
- no fabricated metrics
- no unnecessary dependencies or model claims
