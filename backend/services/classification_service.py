from __future__ import annotations

import re
from typing import Dict, List


class ClassificationService:
    @staticmethod
    def classify_text(ocr_text: str) -> Dict[str, object]:
        text = (ocr_text or "").strip()
        lowered = text.lower()

        score_map = {
            "Invoice": 0,
            "Bank Statement": 0,
            "Salary Slip": 0,
            "Identity Document": 0,
        }

        if re.search(r"\binvoice\b|invoice number|subtotal|tax|total", lowered):
            score_map["Invoice"] += 2
        if re.search(r"\bvendor\b|company|currency|due date", lowered):
            score_map["Invoice"] += 1
        if re.search(r"\bbank statement\b|account holder|account number|closing balance", lowered):
            score_map["Bank Statement"] += 3
        if re.search(r"\bstatement period\b|credits|debits|bank\b", lowered):
            score_map["Bank Statement"] += 2
        if re.search(r"\bsalary slip\b|basic salary|net salary|employee id|pay period", lowered):
            score_map["Salary Slip"] += 3
        if re.search(r"\bemployer\b|allowances|deductions", lowered):
            score_map["Salary Slip"] += 1
        if re.search(r"\bidentity document\b|date of birth|document number|expiry date", lowered):
            score_map["Identity Document"] += 3
        if re.search(r"\bname\b|document number|expiry date", lowered):
            score_map["Identity Document"] += 1

        best_type, best_score = "Unknown", 0
        for doc_type, score in score_map.items():
            if score > best_score:
                best_type = doc_type
                best_score = score

        if best_score == 0:
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "classification_method": "heuristic_ocr_rules",
                "evidence": [],
            }

        confidence = min(0.99, 0.55 + (best_score * 0.12))
        evidence: List[str] = []
        if best_type == "Invoice":
            evidence = ["invoice markers such as invoice number, subtotal, tax, or total were detected"]
        elif best_type == "Bank Statement":
            evidence = ["banking markers such as account number, statement period, credits, debits, or closing balance were detected"]
        elif best_type == "Salary Slip":
            evidence = ["salary markers such as employee ID, basic salary, net salary, or pay period were detected"]
        elif best_type == "Identity Document":
            evidence = ["identity markers such as date of birth, expiry date, or document number were detected"]

        return {
            "document_type": best_type,
            "confidence": round(confidence, 3),
            "classification_method": "heuristic_ocr_rules",
            "evidence": evidence,
        }
