# Final Results Summary

## Active visual tampering baseline

- Model: frozen MobileNetV3-Small features + LogisticRegression (`authentic=0`, `forged=1`).
- Validation: accuracy 0.607692; precision 0.626812; recall 0.532308; F1 0.575707; ROC-AUC 0.682670.
- Official TestingSet (n=1624): accuracy 0.520936; precision 0.526074; recall 0.422414; F1 0.468579; ROC-AUC 0.517846.
- Confusion matrix: TN 503, FP 309, FN 469, TP 343.

## Comparison and representation learning

The HistGradientBoosting alternative had validation F1 0.632091 and test F1 0.447280. The 96x96 partial fine-tune had the best validation F1 (0.662400), but its official test accuracy was 0.501847 and F1 0.470203, below the active model's official test accuracy. It is not the active model.

## GNN

Two-layer PyG GCNConv plus edge classifier. Targets are deterministic verifier outputs on 7 train, 2 validation, and 2 test synthetic cases. Test rule-distillation agreement was 0/2; ROC-AUC is undefined. This is not independent graph accuracy or fraud accuracy.

## Cross-document and risk examples

Synthetic integration fixtures: matching salary ₹35,000 / credit ₹35,000 → PASS, 0 explicit inconsistencies, heuristic risk 16/LOW. Mismatch salary ₹35,000 / credit ₹25,000 → FAIL, 1 explicit inconsistency, heuristic risk 45/MODERATE. These are software behavior examples, not model benchmark results.

## Limitations

The visual baseline is weak (official test ROC-AUC approximately 0.518). No independently labeled end-to-end financial fraud benchmark is present. The risk score is not calibrated probability. GNN supervision is a tiny rule-distillation set. SROIE is document-understanding context, not a fraud benchmark. Synthetic documents are integration fixtures only.

## Claims to Avoid

- Do not claim overall fraud accuracy of 52.09%.
- Do not claim 67.54% validation accuracy as final test accuracy.
- Do not claim GNN performance from its 0/2 synthetic test scenarios as independent accuracy.
- Do not call the risk score a fraud probability.
- Do not claim the system proves fraud.
- Do not claim greater than 80% accuracy.
