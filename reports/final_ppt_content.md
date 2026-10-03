# Final Project PPT Content

## Slide 1: Title

Intelligent Financial Document Fraud Detection Using Multi-Modal AI
Team members: [Insert names] · Project guide: [Insert guide]

## Slide 2: Problem Statement

Financial documents can contain conflicting values or suspicious visual characteristics. This prototype combines document understanding and cross-document checks to surface evidence for human review.

## Slide 3: Motivation / Existing Limitations

Manual comparison is slow and error-prone. OCR can be noisy, heuristic visual detectors can produce false positives, and the trained visual baseline has weak held-out performance.

## Slide 4: Objectives

Build an upload-to-report pipeline; extract and compare financial fields; preserve visual heuristic and learned evidence separately; provide traceable explanations without claiming confirmed fraud.

## Slide 5: Literature Survey

The workspace contains no bibliographic/reference records. Insert supervisor-approved, independently verified papers here; no citations are fabricated in this draft.

## Slide 6: Proposed System

Use Fig. 1: document upload → preprocessing/OCR → classification/extraction → parallel visual evidence + deterministic cross-document verification + advisory GNN → fusion → reasoning → heuristic risk assessment.

## Slide 7: Methodology

Tesseract OCR; rule-based classification and extraction; deterministic cross-document rules; baseline forensic heuristic plus AIForge-Doc visual baseline; evidence fusion and deterministic fallback reasoning.

## Slide 8: Technologies and Dataset

FastAPI, Streamlit, Tesseract, PyMuPDF, PyTorch/torchvision, scikit-learn, PyTorch Geometric. AIForge-Doc-v1 is used for visual tampering; SROIE is document-understanding context only; synthetic PDFs are integration fixtures.

## Slide 9: Core Implementation

Upload and page rendering; OCR; document-type classification; structured fields; field-level PASS/FAIL/WARNING/INSUFFICIENT_DATA rules; independent heuristic and visual-model results.

## Slide 10: Multimodal Evidence + GNN

Fusion retains OCR, fields, metadata, rules, heuristic tampering, trained visual output, and graph output. The GNN is a small rule-distillation prototype and cannot override explicit rules.

## Slide 11: Experimental Results

Use Table 2 and Figs. 3–9. Active LogisticRegression visual baseline: official test accuracy 52.09%, F1 46.86%, ROC-AUC 51.78%. These are tampering metrics, not overall fraud accuracy.

## Slide 12: Graphs and Confusion Matrix

Show Figs. 7 and 8 for the active model. Show Fig. 12 for the rule-distillation GNN architecture; its 0/2 synthetic test agreement is not independent model performance.

## Slide 13: Demo Results

Use Fig. 10 and Fig. 14. Matching synthetic fixture: ₹35,000/₹35,000 → PASS → heuristic risk 16/LOW. Mismatch fixture: ₹35,000/₹25,000 → FAIL → 45/MODERATE. Software integration examples only.

## Slide 14: Limitations

Weak visual generalization; fine-tuned candidate did not improve official test; no independent end-to-end fraud labels; risk is not calibrated probability; heuristic signals are not proof; GNN supervision is tiny and rule-derived.

## Slide 15: Conclusion and Future Work

The project delivers an integrated evidence-based verification prototype, not a validated fraud classifier. Future work: independent labeled financial fraud and graph datasets, better visual representations, calibration, external validation, and broader user evaluation.
