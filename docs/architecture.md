# Architecture

## Overview

This project is structured as an academic multimodal financial-document verification prototype. It combines real document ingestion/OCR with rule-based extraction and cross-document checks, separate tampering evidence sources, evidence fusion, deterministic reasoning, and heuristic risk scoring.

## Layered Design

### 1. Presentation Layer

- Streamlit dashboard for upload, analysis summaries, rule relationships, separate heuristic/model evidence, and explainable limitations

### 2. API Layer

- FastAPI application exposes upload, processing, OCR, classification/extraction, tampering, verification, fusion, reasoning, and risk endpoints
- API schemas are kept modular and explicit via Pydantic models

### 3. Service Layer

Current services include:

- document validation and storage
- PDF page extraction and image conversion
- OCR execution using Tesseract
- rule-based document classification and structured field extraction
- cross-document rule verification
- baseline forensic tampering heuristics and a trained AIForge-Doc-v1 visual tampering baseline
- multimodal evidence fusion, evidence-grounded reasoning, and heuristic risk assessment
- advisory cross-document GNN trained only to distill outcomes from synthetic rule scenarios

The trained visual baseline and heuristic detector are separate evidence sources. Neither independently confirms fraud. The rule engine remains authoritative for explicit field comparisons; GNN output is advisory and cannot override those results.

### 4. Data Layer

- local data directory for raw/processed/samples
- AIForge-Doc-v1 for tampering baseline research; SROIE for document understanding, not fraud labels
- synthetic fixtures for software integration tests only

### 5. Model Layer

- optional CPU-compatible model artifacts in models/
- checkpoints and trained outputs are stored separately from source files

## Document Ingestion Stage

The ingestion stage validates supported PDF/image files, assigns unique IDs, stores them safely, and prepares normalized document metadata for downstream OCR and extraction stages. PDFs are page-counted and page-converted to images before OCR processing when necessary.

## Cross-Document Graph Evidence

Verification results can be represented as two document nodes and a comparison edge with document types, normalized numeric/text indicators, rule type, and comparison features. A small PyTorch Geometric GCN plus edge classifier is trained on a tiny set of synthetic scenarios labeled by the existing deterministic verifier (`PASS`, `FAIL`, `INSUFFICIENT_DATA`). It is explicitly a rule-distillation prototype, not an independently labeled graph dataset or fraud model. Its output is added as `learned_graph_based_evidence`; it does not alter rule results or directly add risk score.

## Constraints

- CPU-friendly design
- transparent rules and explicit provenance for learned evidence
- optional LLM integration only
- no hard-coded secrets
- synthetic data generation used for demonstrations and tests
