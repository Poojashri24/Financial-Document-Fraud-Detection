from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "paper_figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))

baseline_val = load("phase12b_feature_baseline.json")
active_test = load("phase12b_official_test_results.json")
comparison = load("phase13_visual_model_comparison.json")
hgb_test = load("phase13_visual_model_official_test.json")
finetune_val = load("phase13_representation_experiment.json")
finetune_test = load("phase13_representation_official_test.json")
gnn = load("phase13_gnn_rule_distillation.json")
test_predictions = load("phase12b_official_test_predictions.json")

models = [
    {
        "name": "MobileNetV3-Small + Logistic Regression",
        "short": "MobileNet-Small\n+ Logistic Reg.",
        "validation": {"accuracy": baseline_val["metrics"]["accuracy"], "precision": baseline_val["metrics"]["precision"], "recall": baseline_val["metrics"]["recall"], "f1": baseline_val["metrics"]["f1_score"], "roc_auc": baseline_val["metrics"]["roc_auc"]},
        "test": {"accuracy": active_test["metrics"]["accuracy"], "precision": active_test["metrics"]["precision"], "recall": active_test["metrics"]["recall"], "f1": active_test["metrics"]["f1_score"], "roc_auc": active_test["metrics"]["roc_auc"]},
        "active": True,
    },
    {
        "name": "MobileNetV3-Small + HistGradientBoosting",
        "short": "MobileNet-Small\n+ HistGradient",
        "validation": next(row for row in comparison["results"] if row["model"] == "HistGradientBoosting"),
        "test": hgb_test["metrics"],
        "active": False,
    },
    {
        "name": "MobileNetV3-Small partial fine-tune, 96x96",
        "short": "MobileNet-Small\npartial FT 96",
        "validation": finetune_val["validation_metrics"],
        "test": finetune_test["metrics"],
        "active": False,
    },
]


def metric_value(model, split, metric):
    values = model[split]
    if metric == "f1":
        return values.get("f1", values.get("f1_score"))
    return values[metric]


# Verify all metric rows against their saved JSON source objects before plotting/exporting.
assert models[0]["test"]["accuracy"] == active_test["metrics"]["accuracy"]
assert abs(models[0]["test"]["roc_auc"] - roc_auc_score([row["label"] for row in test_predictions], [row["probability"] for row in test_predictions])) < 1e-6
assert models[1]["validation"]["f1"] == next(row["f1"] for row in comparison["results"] if row["model"] == "HistGradientBoosting")
assert models[2]["validation"]["f1"] == finetune_val["validation_metrics"]["f1"]
assert models[2]["test"]["accuracy"] == finetune_test["metrics"]["accuracy"]

# This is a previously verified synthetic software-integration scenario, not research evaluation.
risk_examples = [
    {"name": "Matching synthetic documents", "salary": 35000, "credit": 35000, "verification": "PASS", "risk_score": 16, "risk_level": "LOW"},
    {"name": "Mismatching synthetic documents", "salary": 35000, "credit": 25000, "verification": "FAIL", "risk_score": 45, "risk_level": "MODERATE"},
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
NAVY = "#173042"
TEAL = "#168C88"
GREEN = "#27855B"
AMBER = "#D49B29"
RED = "#B64646"
BLUE = "#477B9E"
GREY = "#657987"
PALE = "#EEF3F5"


def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc=PALE, ec=NAVY, color=NAVY, fontsize=10, lw=1.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fontsize, wrap=True)


def arrow(ax, start, end, color=GREY, style="-|>", lw=1.5, connection="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, linewidth=lw, color=color, connectionstyle=connection))


def diagram_base(title, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, color=NAVY, pad=14, fontweight="bold")
    return fig, ax


# 1. System architecture
fig, ax = diagram_base("System Architecture: Multimodal Financial-Document Verification", (13, 8))
flow = [
    "Financial documents", "Upload + PDF/image processing", "OCR (Tesseract)",
    "Rule-based classification", "Structured field extraction", "Visual evidence: heuristic + trained baseline",
    "Rule-based cross-document verification", "Advisory GNN relational evidence", "Multimodal evidence fusion",
    "Evidence-grounded reasoning", "Heuristic risk assessment", "Explainable Streamlit assessment",
]
positions = [(0.28, 0.86 - i * 0.072) for i in range(len(flow))]
for i, (text, (x, y)) in enumerate(zip(flow, positions)):
    color = "#E5F0F4" if i < 5 else "#E8F4EF" if i in (5, 6, 7) else "#FFF4D7" if i == 8 else "#F0EDF7"
    box(ax, (x, y), (0.44, 0.05), text, fc=color, fontsize=9)
    if i:
        arrow(ax, (0.5, y + 0.072), (0.5, y + 0.05), color=TEAL if i in (6, 8) else GREY)
ax.text(.5, .015, "Rule outputs remain authoritative; visual and GNN signals are evidence, not proof of fraud.", ha="center", color=RED, fontsize=8.5)
save_figure(fig, "fig01_system_architecture")

# 2. Fusion diagram
fig, ax = diagram_base("Multimodal Evidence Fusion", (12, 7))
sources = ["OCR / text", "Structured fields", "Visual tampering\nheuristic", "Trained visual\nmodel", "Cross-document\nrules", "Metadata", "GNN advisory\nevidence"]
ys = np.linspace(.83, .17, len(sources))
for text, y in zip(sources, ys):
    box(ax, (.06, y - .035), (.26, .07), text, fc="#EAF1F4", fontsize=9)
    arrow(ax, (.33, y), (.47, .5), color=GREY, lw=1.0, connection="arc3,rad=0.05")
box(ax, (.47, .40), (.22, .20), "Multimodal\nEvidence Fusion", fc="#DDF1EE", ec=TEAL, fontsize=12, lw=2)
arrow(ax, (.70, .5), (.79, .5), color=TEAL, lw=2)
box(ax, (.79, .62), (.17, .11), "Reasoning", fc="#EEF0F8", fontsize=10)
box(ax, (.79, .35), (.17, .11), "Heuristic Risk\nAssessment", fc="#FFF1D3", ec=AMBER, fontsize=9)
arrow(ax, (.875, .61), (.875, .47), color=GREY)
ax.text(.875, .22, "Risk is not a calibrated\nfraud probability", ha="center", color=RED, fontsize=9)
save_figure(fig, "fig02_multimodal_fusion")

# Bar chart helper

def grouped_metric(metric, title, stem, ylabel="Score"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    width = .34
    vals_v = [metric_value(m, "validation", metric) for m in models]
    vals_t = [metric_value(m, "test", metric) for m in models]
    ax.bar(x - width/2, vals_v, width, label="Validation (n=650)", color=TEAL)
    ax.bar(x + width/2, vals_t, width, label="Official TestingSet (n=1,624)", color=BLUE)
    ax.set_xticks(x, [m["short"] for m in models])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="upper right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.text(.5, -.22, "Tampering-model metrics only; not overall fraud-detection performance.", transform=ax.transAxes, ha="center", color=GREY, fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)


grouped_metric("accuracy", "Visual Tampering Model Accuracy", "fig03_visual_accuracy")
grouped_metric("f1" if "f1" in models[0]["validation"] else "f1_score", "Visual Tampering Model F1", "fig04_visual_f1")
grouped_metric("roc_auc", "Visual Tampering Model ROC-AUC", "fig05_visual_roc_auc")

# 6. Precision, recall, F1
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
model_labels = ["Logistic\nRegression", "HistGradient\nBoosting", "Partial fine-tune\n96x96"]
width = .34
for ax, metric, title, color in zip(axes, ["precision", "recall", "f1"], ["Precision", "Recall", "F1"], [TEAL, BLUE, AMBER]):
    x = np.arange(len(models))
    values_v = [metric_value(model, "validation", metric) for model in models]
    values_t = [metric_value(model, "test", metric) for model in models]
    bars_v = ax.bar(x - width/2, values_v, width, color=color, label="Validation")
    bars_t = ax.bar(x + width/2, values_t, width, color=color, alpha=.45, hatch="//", label="Official test")
    ax.set_xticks(x, model_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, color=NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=.22)
    ax.bar_label(bars_v, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(bars_t, fmt="%.2f", padding=2, fontsize=7)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Score")
fig.suptitle("Precision, Recall, and F1 by Model and Split", color=NAVY, fontweight="bold")
fig.text(.5, .01, "Tampering-task metrics only; not overall fraud-detection performance.", ha="center", color=GREY, fontsize=9)
fig.tight_layout(rect=(0, .05, 1, .92))
save_figure(fig, "fig06_precision_recall_f1")

# 7. Active confusion matrix
cm = active_test["metrics"]["confusion_matrix"]
cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
fig, ax = plt.subplots(figsize=(6, 5.5))
image = ax.imshow(cm_array, cmap="Blues")
for (i, j), value in np.ndenumerate(cm_array):
    ax.text(j, i, str(value), ha="center", va="center", fontsize=16, color=NAVY, fontweight="bold")
ax.set_xticks([0, 1], ["Predicted authentic", "Predicted forged"])
ax.set_yticks([0, 1], ["Authentic", "Forged"])
ax.set_title("Active LogisticRegression Baseline: Official Test Confusion Matrix", color=NAVY, fontweight="bold")
fig.colorbar(image, ax=ax, fraction=.045)
fig.tight_layout()
save_figure(fig, "fig07_active_confusion_matrix")

# 8. ROC from actual saved per-image scores
labels = np.asarray([int(row["label"]) for row in test_predictions])
scores = np.asarray([float(row["probability"]) for row in test_predictions])
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc_saved = float(auc(fpr, tpr))
assert len(test_predictions) == active_test["sample_count"]
assert abs(roc_auc_saved - active_test["metrics"]["roc_auc"]) < 1e-5
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"Active model (AUC={roc_auc_saved:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, lw=1.2, label="Chance reference")
ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
ax.set_title("Official TestingSet ROC — Active Tampering Baseline", color=NAVY, fontweight="bold")
ax.legend(frameon=False)
ax.grid(alpha=.2)
ax.text(.5, -.19, "Computed from saved per-image official-test probabilities; tampering task only.", transform=ax.transAxes, ha="center", color=GREY, fontsize=8.5)
fig.tight_layout()
save_figure(fig, "fig08_active_roc_curve")

# 9. Summary matrix
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axis("off")
columns = ["Model", "Validation Acc.", "Validation F1", "Validation AUC", "Test Acc.", "Test F1", "Test AUC", "Role"]
rows = []
for model in models:
    rows.append([model["name"], f"{model['validation']['accuracy']:.3f}", f"{model['validation'].get('f1',model['validation'].get('f1_score')):.3f}", f"{model['validation']['roc_auc']:.3f}", f"{model['test']['accuracy']:.3f}", f"{model['test'].get('f1',model['test'].get('f1_score')):.3f}", f"{model['test']['roc_auc']:.3f}", "Active" if model["active"] else "Experiment"])
table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center", colWidths=[.28,.1,.1,.1,.1,.1,.1,.12])
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    elif row == 1:
        cell.set_facecolor("#E8F4EF")
    else:
        cell.set_facecolor("#F4F7F8" if row % 2 else "white")
ax.set_title("Validation Selection vs Official Held-Out Testing", color=NAVY, fontweight="bold", pad=15)
fig.tight_layout()
save_figure(fig, "fig09_model_comparison_summary")

# 10. Synthetic comparison examples
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, example in zip(axes, risk_examples):
    ax.axis("off")
    color = GREEN if example["verification"] == "PASS" else RED
    ax.set_title(example["name"], color=NAVY, fontweight="bold")
    box(ax, (.03, .58), (.35, .20), f"Salary Slip\n₹{example['salary']:,}", fc="#E9F1F4", fontsize=12)
    box(ax, (.62, .58), (.35, .20), f"Bank Statement\n₹{example['credit']:,}", fc="#E9F1F4", fontsize=12)
    arrow(ax, (.4, .68), (.60, .68), color=color, lw=2)
    box(ax, (.25, .25), (.50, .17), f"{example['verification']}\nRisk {example['risk_score']} · {example['risk_level']}", fc="#E8F4EF" if color==GREEN else "#FBEBEA", ec=color, color=color, fontsize=12, lw=2)
    ax.text(.5, .08, "Synthetic integration test only — not research performance", ha="center", color=GREY, fontsize=8.5, transform=ax.transAxes)
fig.suptitle("Cross-Document Salary/Credit Examples", color=NAVY, fontweight="bold", y=1.02)
fig.tight_layout()
save_figure(fig, "fig10_cross_document_examples")

# 11. Consistency graph conceptual
fig, ax = diagram_base("Explainable Cross-Document Consistency Graph", (10, 4.8))
box(ax, (.08, .55), (.28, .22), "Salary Slip\nnet_salary = ₹35,000", fc="#EAF1F4", fontsize=11)
box(ax, (.64, .55), (.28, .22), "Bank Statement\ncredits = ₹25,000", fc="#EAF1F4", fontsize=11)
arrow(ax, (.37, .66), (.63, .66), color=RED, lw=2)
box(ax, (.37, .23), (.26, .16), "salary_vs_bank_credit\nFAIL", fc="#FBEBEA", ec=RED, color=RED, fontsize=10)
arrow(ax, (.50, .54), (.50, .40), color=RED, lw=1.5)
ax.text(.5, .08, "Rule-based explanation graph; not a GNN prediction", ha="center", color=GREY, fontsize=9)
save_figure(fig, "fig11_consistency_graph")

# 12. GNN architecture
fig, ax = diagram_base("Advisory Cross-Document GNN: Rule-Distillation Prototype", (9, 8))
gnn_steps = [
    "Document nodes\n(type + extracted fields)",
    "Relationship features\n(rule type + value differences)",
    "GCNConv layer 1",
    "GCNConv layer 2",
    "Edge classifier\nPASS / FAIL / INSUFFICIENT_DATA",
    "Advisory graph evidence\n(rule output remains authoritative)",
]
gnn_y = .84
for index, label in enumerate(gnn_steps):
    color = "#E9E6F3" if "GCNConv" in label else "#EAF1F4" if index < 2 else "#FFF4D7"
    box(ax, (.28, gnn_y), (.44, .09), label, fc=color, ec="#71659A" if "GCNConv" in label else NAVY, fontsize=9)
    if index:
        arrow(ax, (.5, gnn_y + .12), (.5, gnn_y + .09), color=TEAL)
    gnn_y -= .14
ax.text(.5, .035, "Targets are deterministic verifier outcomes on 11 synthetic scenarios.\nRule-distillation agreement only; not independent fraud ground truth.", ha="center", va="center", color=RED, fontsize=9)
save_figure(fig, "fig12_gnn_architecture")

# 13. Risk evidence flow
fig, ax = diagram_base("Explainable Heuristic Risk Assessment", (12, 6))
inputs = [("Explicit cross-document\nFAIL/WARNING", .84, RED), ("Heuristic tampering\n(observation; capped)", .68, AMBER), ("Trained visual model\n(informational; uncalibrated)", .52, BLUE), ("GNN\n(advisory; 0 direct points)", .36, "#7666A6"), ("Missing / insufficient\nevidence", .20, GREY)]
for label, y, color in inputs:
    box(ax, (.06, y-.045), (.34, .09), label, fc="white", ec=color, color=NAVY, fontsize=9)
    arrow(ax, (.41, y), (.55, .5), color=color, lw=1.2)
box(ax, (.55, .39), (.22, .22), "Heuristic evidence\nscoring", fc="#FFF4D7", ec=AMBER, fontsize=11, lw=2)
arrow(ax, (.78, .5), (.84, .5), color=TEAL, lw=2)
box(ax, (.84, .41), (.14, .18), "Risk score\n+ level", fc="#E8F4EF", ec=GREEN, fontsize=10)
ax.text(.5, .06, "Not a calibrated fraud probability · Evidence signals do not prove fraud", ha="center", color=RED, fontsize=9)
save_figure(fig, "fig13_risk_assessment_flow")

# 14. Demo risk comparison
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [GREEN, AMBER]
values = [example["risk_score"] for example in risk_examples]
bars = ax.bar(["Matching · PASS", "Mismatch · FAIL"], values, color=bar_colors, width=.5)
ax.set_ylim(0, 60); ax.set_ylabel("Heuristic risk score"); ax.set_title("Synthetic Integration Scenarios", color=NAVY, fontweight="bold"); ax.grid(axis="y", alpha=.2)
for bar, example in zip(bars, risk_examples):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{example['risk_score']} · {example['risk_level']}", ha="center", fontweight="bold", color=NAVY)
ax.text(.5, -.18, "Software integration examples only; not research metrics or calibrated probabilities.", transform=ax.transAxes, ha="center", color=RED, fontsize=9)
fig.tight_layout()
save_figure(fig, "fig14_synthetic_risk_cases")

# Tables
visual_table = []
for model in models:
    visual_table.append({
        "Model": model["name"],
        "Validation Accuracy": model["validation"]["accuracy"],
        "Validation Precision": model["validation"]["precision"],
        "Validation Recall": model["validation"]["recall"],
        "Validation F1": model["validation"].get("f1", model["validation"].get("f1_score")),
        "Validation ROC-AUC": model["validation"]["roc_auc"],
        "Official Test Accuracy": model["test"]["accuracy"],
        "Official Test Precision": model["test"]["precision"],
        "Official Test Recall": model["test"]["recall"],
        "Official Test F1": model["test"].get("f1", model["test"].get("f1_score")),
        "Official Test ROC-AUC": model["test"]["roc_auc"],
        "Role": "Active" if model["active"] else "Experimental / not active",
    })

# Patch in GNN rule status sample information; no synthetic performance table is mixed with visual results.
dataset_rows = [
    {"Item": "AIForge-Doc-v1 training subset", "Verified configuration": "1,624 authentic + 1,624 forged; 1,624 paired masks", "Use": "Visual tampering training/validation"},
    {"Item": "Train split", "Verified configuration": "1,299 authentic + 1,299 forged", "Use": "Model fitting"},
    {"Item": "Validation split", "Verified configuration": "325 authentic + 325 forged; seed 42", "Use": "Model selection / early stopping"},
    {"Item": "Official TestingSet", "Verified configuration": "812 authentic + 812 forged; 1,624 total", "Use": "Locked final visual-model evaluation only"},
    {"Item": "Input/preprocessing", "Verified configuration": "Original baseline 64x64 RGB; fine-tune 96x96 RGB; ImageNet normalization", "Use": "Tampering model"},
    {"Item": "SROIE", "Verified configuration": "Document-understanding / receipt field context; no tampering labels", "Use": "Not a fraud benchmark"},
    {"Item": "Synthetic PDFs", "Verified configuration": "Matching and mismatch salary/bank fixtures", "Use": "Software integration tests only"},
]

test_cm = active_test["metrics"]["confusion_matrix"]
confusion_rows = [{"Actual / Predicted": "Authentic", "Authentic": test_cm["tn"], "Forged": test_cm["fp"]}, {"Actual / Predicted": "Forged", "Authentic": test_cm["fn"], "Forged": test_cm["tp"]}]
verification_rows = [
    {"Scenario": "Synthetic matching integration", "Salary": "₹35,000", "Bank credit": "₹35,000", "Verification": "PASS", "Risk score/level": "16 / LOW"},
    {"Scenario": "Synthetic mismatch integration", "Salary": "₹35,000", "Bank credit": "₹25,000", "Verification": "FAIL", "Risk score/level": "45 / MODERATE"},
]
gnn_rows = [{"Architecture": gnn["architecture"], "Train scenarios": gnn["counts"]["train"], "Validation scenarios": gnn["counts"]["validation"], "Test scenarios": gnn["counts"]["test"], "Held-out agreement": f"{gnn['metrics']['test']['accuracy']:.3f} on n={gnn['metrics']['test']['sample_count']}", "Purpose": "Rule-distillation agreement", "Limitations": "Synthetic rules; not independent fraud labels"}]
component_rows = [
    {"Component": "OCR", "Purpose": "Extract text", "Implementation": "Tesseract on PDF-rendered/image inputs", "Output": "OCR text and status"},
    {"Component": "Classification", "Purpose": "Identify document type", "Implementation": "Keyword/rule heuristic", "Output": "Document type and evidence"},
    {"Component": "Extraction", "Purpose": "Read structured fields", "Implementation": "Type-specific regex/normalization", "Output": "Names, amounts, dates, identifiers"},
    {"Component": "Cross-document verification", "Purpose": "Check field consistency", "Implementation": "Deterministic explicit rules", "Output": "PASS/FAIL/WARNING/INSUFFICIENT_DATA"},
    {"Component": "Tampering", "Purpose": "Visual evidence", "Implementation": "Forensic heuristic + AIForge visual baseline", "Output": "Separate signals; neither proves fraud"},
    {"Component": "Fusion", "Purpose": "Combine evidence", "Implementation": "Modality/source aggregation", "Output": "Traceable evidence items"},
    {"Component": "GNN", "Purpose": "Relational advisory evidence", "Implementation": "2-layer PyG GCN + edge classifier", "Output": "Rule-distillation prediction"},
    {"Component": "Reasoning", "Purpose": "Explain evidence", "Implementation": "Deterministic evidence-grounded fallback", "Output": "Findings and limitations"},
    {"Component": "Risk scoring", "Purpose": "Summarize evidence", "Implementation": "Heuristic contributions; not calibrated", "Output": "Score, level, breakdown"},
    {"Component": "Explainability", "Purpose": "Present comparisons", "Implementation": "Consistency graph/dashboard", "Output": "Rule status and explanation"},
]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def md_table(rows):
    headers = list(rows[0])
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    for row in rows:
        vals = [str(row[h]).replace("|", "\\|").replace("\n", " ") for h in headers]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

write_csv(REPORTS/"table1_dataset_experiment_config.csv", dataset_rows)
write_csv(REPORTS/"table2_visual_model_comparison.csv", visual_table)
write_csv(REPORTS/"table3_active_model_confusion_matrix.csv", confusion_rows)
write_csv(REPORTS/"table4_cross_document_integration_examples.csv", verification_rows)
write_csv(REPORTS/"table5_gnn_prototype.csv", gnn_rows)
write_csv(REPORTS/"table6_system_components.csv", component_rows)

(REPORTS/"paper_tables.md").write_text(
    "# Publication Tables\n\nAll visual model metrics are tampering-task metrics, not overall fraud detection. Synthetic integration examples are clearly separated from research evaluation.\n\n## Table 1 — Dataset / Experimental Configuration\n\n" + md_table(dataset_rows) +
    "\n\n## Table 2 — Visual Model Comparison\n\n" + md_table(visual_table) +
    "\n\n## Table 3 — Active Model Confusion Matrix (Official Test)\n\n" + md_table(confusion_rows) +
    "\n\n## Table 4 — Cross-Document Examples (Synthetic Integration Only)\n\n" + md_table(verification_rows) +
    "\n\n## Table 5 — GNN Prototype (Rule-Distillation Only)\n\n" + md_table(gnn_rows) +
    "\n\n## Table 6 — System Components\n\n" + md_table(component_rows) + "\n", encoding="utf-8")

# Paper Results section
paper = f'''# Results and Analysis

## 5.1 Visual Tampering Detection

The active visual tampering baseline uses frozen MobileNetV3-Small features (576 dimensions) and Logistic Regression, with authentic=0 and forged=1. On the AIForge-Doc-v1 validation split (n=650), it achieved accuracy {baseline_val['metrics']['accuracy']:.6f}, precision {baseline_val['metrics']['precision']:.6f}, recall {baseline_val['metrics']['recall']:.6f}, F1 {baseline_val['metrics']['f1_score']:.6f}, and ROC-AUC {baseline_val['metrics']['roc_auc']:.6f}. On the official held-out TestingSet (n={active_test['sample_count']}), it achieved accuracy {active_test['metrics']['accuracy']:.6f}, precision {active_test['metrics']['precision']:.6f}, recall {active_test['metrics']['recall']:.6f}, F1 {active_test['metrics']['f1_score']:.6f}, and ROC-AUC {active_test['metrics']['roc_auc']:.6f}.

These figures evaluate visual tampering classification only. They are not financial fraud-detection or complete-system accuracy.

## 5.2 Model Comparison

The validation comparison considered the existing Logistic Regression, standardized Logistic Regression, linear and RBF SVM, Random Forest, HistGradientBoosting, MLP, and a controlled MobileNetV3-Small partial fine-tune at 96x96. Selection priority was validation F1, then ROC-AUC, then accuracy. The fine-tune had the best validation F1 ({finetune_val['validation_metrics']['f1']:.6f}) and ROC-AUC ({finetune_val['validation_metrics']['roc_auc']:.6f}). Its official test accuracy was {finetune_test['metrics']['accuracy']:.6f} and F1 was {finetune_test['metrics']['f1']:.6f}, below the original LogisticRegression test accuracy {active_test['metrics']['accuracy']:.6f}. This validation/test gap indicates limited generalization; the fine-tune was not promoted, and the original classifier remains active. HistGradientBoosting also had better validation than test results and was not promoted.

## 5.3 Cross-Document Verification

The deterministic verifier explicitly compares extracted fields and returns PASS, FAIL, WARNING, or INSUFFICIENT_DATA. In software integration fixtures, salary net ₹35,000 matched bank credit ₹35,000 (PASS); salary ₹35,000 versus credit ₹25,000 produced FAIL. These are synthetic software-flow examples, not research performance data.

## 5.4 Multimodal Evidence Fusion

Fusion retains OCR, structured fields, cross-document rules, heuristic tampering, trained visual-model output, GNN advisory evidence, and metadata as separate traceable sources. Heuristic tampering observations are not counted as cross-document inconsistencies.

## 5.5 GNN-Based Relational Evidence

The lightweight two-layer PyTorch Geometric GCN with edge classifier was trained on 7 synthetic rule-distillation scenarios and evaluated on 2 validation and 2 held-out synthetic scenarios. Targets are generated by the existing deterministic verifier, not independent ground truth. Held-out agreement was {gnn['metrics']['test']['accuracy']:.3f} on n={gnn['metrics']['test']['sample_count']}; ROC-AUC is undefined for this split because all classes are not represented. This does not demonstrate independent graph generalization or fraud accuracy. The GNN is advisory and cannot override deterministic rules.

## 5.6 Explainable Risk Assessment

Risk is a heuristic evidence score, not a calibrated probability. Explicit cross-document failures contribute materially. Weak heuristic tampering observations are retained but their aggregate contribution is capped; trained visual probabilities are not treated as calibrated fraud probabilities; GNN output contributes zero direct risk points. Synthetic integration risk examples yielded 16/LOW for the matching fixture and 45/MODERATE for the mismatch fixture; these are software behavior examples only.

## 5.7 Limitations

The active visual baseline is modest (official test ROC-AUC {active_test['metrics']['roc_auc']:.6f}). The 96x96 fine-tune did not improve official test accuracy. There is no independent end-to-end financial fraud ground-truth benchmark, no calibrated fraud probability, and no independent graph-labelled corpus. Heuristic tampering can produce false positives. The GNN metrics reflect rule distillation over a tiny synthetic set.

## 5.8 Discussion

The system demonstrates an integrated, evidence-traceable verification workflow rather than a single validated fraud classifier. The official test results show weak visual tampering generalization; higher validation performance did not transfer to improved held-out test performance. Results should therefore be presented as a baseline study and software prototype. No overall fraud-detection accuracy is claimed.
'''
(REPORTS/"paper_results.md").write_text(paper, encoding="utf-8")

captions = [
("fig01_system_architecture", "System architecture for the multimodal financial-document verification prototype, from document upload through explainable heuristic assessment."),
("fig02_multimodal_fusion", "Evidence sources retained by multimodal fusion, including OCR, structured fields, rule verification, separate visual signals, metadata, and advisory GNN output."),
("fig03_visual_accuracy", "Validation and official TestingSet accuracy for three visual tampering model configurations. Metrics are tampering-task results only."),
("fig04_visual_f1", "Validation and official TestingSet F1 for the active LogisticRegression baseline, HistGradientBoosting alternative, and partial fine-tune."),
("fig05_visual_roc_auc", "Validation and official TestingSet ROC-AUC for the evaluated visual tampering models."),
("fig06_precision_recall_f1", "Precision, recall, and F1 across validation and official test splits for the evaluated visual tampering models."),
("fig07_active_confusion_matrix", "Confusion matrix for the active MobileNetV3-Small frozen-feature LogisticRegression tampering baseline on the 1,624-image official AIForge-Doc-v1 TestingSet."),
("fig08_active_roc_curve", "ROC curve computed from saved per-image probabilities for the active visual tampering baseline on the official TestingSet; AUC matches the recorded result."),
("fig09_model_comparison_summary", "Summary of validation-based model comparison and official held-out visual tampering performance; the active model remains LogisticRegression because the fine-tune did not improve test accuracy."),
("fig10_cross_document_examples", "Synthetic software integration examples of matching and mismatching salary/bank values; these are not research benchmark measurements."),
("fig11_consistency_graph", "Conceptual rule-based salary-to-bank-credit comparison graph illustrating an explicit mismatch; not a GNN output."),
("fig12_gnn_architecture", "Lightweight PyTorch Geometric GCN and edge classifier using deterministic verifier outputs as rule-distillation targets; not independent fraud ground truth."),
("fig13_risk_assessment_flow", "Evidence contribution flow for heuristic risk assessment, with explicit cross-document failures distinguished from visual observations and advisory model outputs."),
("fig14_synthetic_risk_cases", "Heuristic risk scores from matching and mismatching synthetic integration scenarios; these are not calibrated probabilities or research metrics."),
]
(REPORTS/"paper_figure_captions.md").write_text("# Paper Figure Captions\n\n" + "\n\n".join(f"**Fig. {i+1} ({stem}).** {caption}" for i,(stem,caption) in enumerate(captions)) + "\n", encoding="utf-8")

# PPT content
slides = [
("Title", "Intelligent Financial Document Fraud Detection Using Multi-Modal AI\nTeam members: [Insert names] · Project guide: [Insert guide]"),
("Problem Statement", "Financial documents can contain conflicting values or suspicious visual characteristics. This prototype combines document understanding and cross-document checks to surface evidence for human review."),
("Motivation / Existing Limitations", "Manual comparison is slow and error-prone. OCR can be noisy, heuristic visual detectors can produce false positives, and the trained visual baseline has weak held-out performance."),
("Objectives", "Build an upload-to-report pipeline; extract and compare financial fields; preserve visual heuristic and learned evidence separately; provide traceable explanations without claiming confirmed fraud."),
("Literature Survey", "The workspace contains no bibliographic/reference records. Insert supervisor-approved, independently verified papers here; no citations are fabricated in this draft."),
("Proposed System", "Use Fig. 1: document upload → preprocessing/OCR → classification/extraction → parallel visual evidence + deterministic cross-document verification + advisory GNN → fusion → reasoning → heuristic risk assessment."),
("Methodology", "Tesseract OCR; rule-based classification and extraction; deterministic cross-document rules; baseline forensic heuristic plus AIForge-Doc visual baseline; evidence fusion and deterministic fallback reasoning."),
("Technologies and Dataset", "FastAPI, Streamlit, Tesseract, PyMuPDF, PyTorch/torchvision, scikit-learn, PyTorch Geometric. AIForge-Doc-v1 is used for visual tampering; SROIE is document-understanding context only; synthetic PDFs are integration fixtures."),
("Core Implementation", "Upload and page rendering; OCR; document-type classification; structured fields; field-level PASS/FAIL/WARNING/INSUFFICIENT_DATA rules; independent heuristic and visual-model results."),
("Multimodal Evidence + GNN", "Fusion retains OCR, fields, metadata, rules, heuristic tampering, trained visual output, and graph output. The GNN is a small rule-distillation prototype and cannot override explicit rules."),
("Experimental Results", "Use Table 2 and Figs. 3–9. Active LogisticRegression visual baseline: official test accuracy 52.09%, F1 46.86%, ROC-AUC 51.78%. These are tampering metrics, not overall fraud accuracy."),
("Graphs and Confusion Matrix", "Show Figs. 7 and 8 for the active model. Show Fig. 12 for the rule-distillation GNN architecture; its 0/2 synthetic test agreement is not independent model performance."),
("Demo Results", "Use Fig. 10 and Fig. 14. Matching synthetic fixture: ₹35,000/₹35,000 → PASS → heuristic risk 16/LOW. Mismatch fixture: ₹35,000/₹25,000 → FAIL → 45/MODERATE. Software integration examples only."),
("Limitations", "Weak visual generalization; fine-tuned candidate did not improve official test; no independent end-to-end fraud labels; risk is not calibrated probability; heuristic signals are not proof; GNN supervision is tiny and rule-derived."),
("Conclusion and Future Work", "The project delivers an integrated evidence-based verification prototype, not a validated fraud classifier. Future work: independent labeled financial fraud and graph datasets, better visual representations, calibration, external validation, and broader user evaluation."),
]
(REPORTS/"final_ppt_content.md").write_text("# Final Project PPT Content\n\n" + "\n\n".join(f"## Slide {i+1}: {title}\n\n{body}" for i,(title,body) in enumerate(slides)) + "\n", encoding="utf-8")

# Demo script
(REPORTS/"demo_script.md").write_text('''# Final Demonstration Script (5–7 minutes)

## 0:00–0:35 — Introduction
“This is an intelligent financial-document verification prototype. It combines OCR, structured extraction, explicit cross-document rules, heuristic visual observations, a trained visual tampering baseline, advisory graph evidence, and a heuristic risk assessment. None of these outputs alone proves fraud.”

## 0:35–1:30 — Matching documents
Upload the existing synthetic salary slip and matching bank statement. State clearly that these are software integration fixtures, not research data. Show OCR status and extracted employee/account names and amounts.

## 1:30–2:15 — Cross-document PASS
Show Salary Slip net salary ₹35,000 and Bank Statement credit ₹35,000. Point to `salary_vs_bank_credit = PASS` and the zero explicit inconsistency count.

## 2:15–3:00 — Separate visual evidence
Show the forensic heuristic signal separately from the trained visual model prediction. Explain that the heuristic may flag texture on clean synthetic images; the trained model’s AUTHENTIC output is also only evidence and its probability is not calibrated.

## 3:00–3:40 — GNN evidence
Show the graph-based section beside explicit verification. Explain the GNN is a small rule-distillation prototype trained on synthetic scenarios; it is advisory and cannot change PASS/FAIL.

## 3:40–4:20 — Risk breakdown
Show 16/LOW for the matching integration fixture. Explain this is a heuristic score, not a fraud probability; weak texture signals are visible and capped.

## 4:20–5:30 — Mismatch documents
Upload the same salary slip and the existing mismatch bank statement. Show ₹35,000 versus ₹25,000, `salary_vs_bank_credit = FAIL`, and the preserved explicit explanation.

## 5:30–6:15 — Mismatch risk and close
Show 45/MODERATE and the score breakdown: explicit cross-document mismatch contributes materially; heuristic observations remain bounded. Close by distinguishing software integration examples from AIForge-Doc visual tampering research metrics.

## Presenter guardrails
- Do not call the visual tampering metrics overall fraud accuracy.
- Do not describe model predictions or heuristic signals as proof of fraud.
- Do not call the risk score a probability.
- Do not present the tiny GNN rule-distillation agreement as independent GNN accuracy.
''', encoding="utf-8")

# Results summary
summary = f'''# Final Results Summary

## Active visual tampering baseline

- Model: frozen MobileNetV3-Small features + LogisticRegression (`authentic=0`, `forged=1`).
- Validation: accuracy {baseline_val['metrics']['accuracy']:.6f}; precision {baseline_val['metrics']['precision']:.6f}; recall {baseline_val['metrics']['recall']:.6f}; F1 {baseline_val['metrics']['f1_score']:.6f}; ROC-AUC {baseline_val['metrics']['roc_auc']:.6f}.
- Official TestingSet (n={active_test['sample_count']}): accuracy {active_test['metrics']['accuracy']:.6f}; precision {active_test['metrics']['precision']:.6f}; recall {active_test['metrics']['recall']:.6f}; F1 {active_test['metrics']['f1_score']:.6f}; ROC-AUC {active_test['metrics']['roc_auc']:.6f}.
- Confusion matrix: TN {active_test['metrics']['confusion_matrix']['tn']}, FP {active_test['metrics']['confusion_matrix']['fp']}, FN {active_test['metrics']['confusion_matrix']['fn']}, TP {active_test['metrics']['confusion_matrix']['tp']}.

## Comparison and representation learning

The HistGradientBoosting alternative had validation F1 0.632091 and test F1 0.447280. The 96x96 partial fine-tune had the best validation F1 (0.662400), but its official test accuracy was {finetune_test['metrics']['accuracy']:.6f} and F1 {finetune_test['metrics']['f1']:.6f}, below the active model's official test accuracy. It is not the active model.

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
'''
(REPORTS/"final_results_summary.md").write_text(summary, encoding="utf-8")

print(json.dumps({"figure_dir": str(FIGURES), "figures": len(list(FIGURES.glob('*.png'))), "tables": ["table1_dataset_experiment_config.csv", "table2_visual_model_comparison.csv", "table3_active_model_confusion_matrix.csv", "table4_cross_document_integration_examples.csv", "table5_gnn_prototype.csv", "table6_system_components.csv"], "roc_auc_from_saved_scores": round(roc_auc_saved,6), "roc_auc_report": active_test["metrics"]["roc_auc"], "citation_records_found": 0}, indent=2))
