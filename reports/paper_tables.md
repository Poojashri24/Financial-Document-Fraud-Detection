# Publication Tables

All visual model metrics are tampering-task metrics, not overall fraud detection. Synthetic integration examples are clearly separated from research evaluation.

## Table 1 — Dataset / Experimental Configuration

| Item | Verified configuration | Use |
|---|---|---|
| AIForge-Doc-v1 training subset | 1,624 authentic + 1,624 forged; 1,624 paired masks | Visual tampering training/validation |
| Train split | 1,299 authentic + 1,299 forged | Model fitting |
| Validation split | 325 authentic + 325 forged; seed 42 | Model selection / early stopping |
| Official TestingSet | 812 authentic + 812 forged; 1,624 total | Locked final visual-model evaluation only |
| Input/preprocessing | Original baseline 64x64 RGB; fine-tune 96x96 RGB; ImageNet normalization | Tampering model |
| SROIE | Document-understanding / receipt field context; no tampering labels | Not a fraud benchmark |
| Synthetic PDFs | Matching and mismatch salary/bank fixtures | Software integration tests only |

## Table 2 — Visual Model Comparison

| Model | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Validation ROC-AUC | Official Test Accuracy | Official Test Precision | Official Test Recall | Official Test F1 | Official Test ROC-AUC | Role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small + Logistic Regression | 0.607692 | 0.626812 | 0.532308 | 0.575707 | 0.68267 | 0.520936 | 0.526074 | 0.422414 | 0.468579 | 0.517846 | Active |
| MobileNetV3-Small + HistGradientBoosting | 0.650769 | 0.667808 | 0.6 | 0.632091 | 0.701387 | 0.493227 | 0.491876 | 0.410099 | 0.44728 | 0.502733 | Experimental / not active |
| MobileNetV3-Small partial fine-tune, 96x96 | 0.675385 | 0.69 | 0.636923 | 0.6624 | 0.762149 | 0.501847 | 0.502098 | 0.442118 | 0.470203 | 0.510476 | Experimental / not active |

## Table 3 — Active Model Confusion Matrix (Official Test)

| Actual / Predicted | Authentic | Forged |
|---|---|---|
| Authentic | 503 | 309 |
| Forged | 469 | 343 |

## Table 4 — Cross-Document Examples (Synthetic Integration Only)

| Scenario | Salary | Bank credit | Verification | Risk score/level |
|---|---|---|---|---|
| Synthetic matching integration | ₹35,000 | ₹35,000 | PASS | 16 / LOW |
| Synthetic mismatch integration | ₹35,000 | ₹25,000 | FAIL | 45 / MODERATE |

## Table 5 — GNN Prototype (Rule-Distillation Only)

| Architecture | Train scenarios | Validation scenarios | Test scenarios | Held-out agreement | Purpose | Limitations |
|---|---|---|---|---|---|---|
| 2-layer PyTorch Geometric GCNConv plus edge MLP | 7 | 2 | 2 | 0.000 on n=2 | Rule-distillation agreement | Synthetic rules; not independent fraud labels |

## Table 6 — System Components

| Component | Purpose | Implementation | Output |
|---|---|---|---|
| OCR | Extract text | Tesseract on PDF-rendered/image inputs | OCR text and status |
| Classification | Identify document type | Keyword/rule heuristic | Document type and evidence |
| Extraction | Read structured fields | Type-specific regex/normalization | Names, amounts, dates, identifiers |
| Cross-document verification | Check field consistency | Deterministic explicit rules | PASS/FAIL/WARNING/INSUFFICIENT_DATA |
| Tampering | Visual evidence | Forensic heuristic + AIForge visual baseline | Separate signals; neither proves fraud |
| Fusion | Combine evidence | Modality/source aggregation | Traceable evidence items |
| GNN | Relational advisory evidence | 2-layer PyG GCN + edge classifier | Rule-distillation prediction |
| Reasoning | Explain evidence | Deterministic evidence-grounded fallback | Findings and limitations |
| Risk scoring | Summarize evidence | Heuristic contributions; not calibrated | Score, level, breakdown |
| Explainability | Present comparisons | Consistency graph/dashboard | Rule status and explanation |
