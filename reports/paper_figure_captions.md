# Paper Figure Captions

**Fig. 1 (fig01_system_architecture).** System architecture for the multimodal financial-document verification prototype, from document upload through explainable heuristic assessment.

**Fig. 2 (fig02_multimodal_fusion).** Evidence sources retained by multimodal fusion, including OCR, structured fields, rule verification, separate visual signals, metadata, and advisory GNN output.

**Fig. 3 (fig03_visual_accuracy).** Validation and official TestingSet accuracy for three visual tampering model configurations. Metrics are tampering-task results only.

**Fig. 4 (fig04_visual_f1).** Validation and official TestingSet F1 for the active LogisticRegression baseline, HistGradientBoosting alternative, and partial fine-tune.

**Fig. 5 (fig05_visual_roc_auc).** Validation and official TestingSet ROC-AUC for the evaluated visual tampering models.

**Fig. 6 (fig06_precision_recall_f1).** Precision, recall, and F1 across validation and official test splits for the evaluated visual tampering models.

**Fig. 7 (fig07_active_confusion_matrix).** Confusion matrix for the active MobileNetV3-Small frozen-feature LogisticRegression tampering baseline on the 1,624-image official AIForge-Doc-v1 TestingSet.

**Fig. 8 (fig08_active_roc_curve).** ROC curve computed from saved per-image probabilities for the active visual tampering baseline on the official TestingSet; AUC matches the recorded result.

**Fig. 9 (fig09_model_comparison_summary).** Summary of validation-based model comparison and official held-out visual tampering performance; the active model remains LogisticRegression because the fine-tune did not improve test accuracy.

**Fig. 10 (fig10_cross_document_examples).** Synthetic software integration examples of matching and mismatching salary/bank values; these are not research benchmark measurements.

**Fig. 11 (fig11_consistency_graph).** Conceptual rule-based salary-to-bank-credit comparison graph illustrating an explicit mismatch; not a GNN output.

**Fig. 12 (fig12_gnn_architecture).** Lightweight PyTorch Geometric GCN and edge classifier using deterministic verifier outputs as rule-distillation targets; not independent fraud ground truth.

**Fig. 13 (fig13_risk_assessment_flow).** Evidence contribution flow for heuristic risk assessment, with explicit cross-document failures distinguished from visual observations and advisory model outputs.

**Fig. 14 (fig14_synthetic_risk_cases).** Heuristic risk scores from matching and mismatching synthetic integration scenarios; these are not calibrated probabilities or research metrics.
