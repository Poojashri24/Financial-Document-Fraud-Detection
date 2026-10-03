# Final Demonstration Script (5–7 minutes)

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
