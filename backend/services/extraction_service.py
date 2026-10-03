from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class ExtractionService:
    @staticmethod
    def _line_match(label: str, text: str) -> Optional[str]:
        for line in text.splitlines():
            match = re.match(rf"^\s*{re.escape(label)}\s*[:\-]?\s*(.+?)\s*$", line, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip()
            if not value:
                continue
            if value.upper() in {"STATEMENT", "SLIP", "DOCUMENT", "INVOICE"}:
                continue
            return value
        return None

    @staticmethod
    def _last_match(pattern: str, text: str) -> Optional[str]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not matches:
            return None
        return matches[-1].strip()

    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
            return cleaned
        return cleaned if cleaned else None

    @staticmethod
    def _mask_identifier(value: Optional[str], keep_last: int = 4) -> Optional[str]:
        if value is None:
            return None
        raw = re.sub(r"\s+", "", value)
        if not raw:
            return None
        if any(ch.isalpha() for ch in raw):
            prefix = raw[:2]
            if len(raw) <= 2:
                return "*" * max(1, len(raw))
            return prefix + ("*" * 8)
        if len(raw) <= keep_last:
            return "*" * max(1, len(raw))
        return "*" * (len(raw) - keep_last) + raw[-keep_last:]

    @staticmethod
    def extract_fields(document_type: str, ocr_text: str) -> Dict[str, Any]:
        text = ocr_text or ""
        doc_type = (document_type or "").strip()

        if doc_type == "Invoice":
            invoice_number = ExtractionService._line_match("Invoice Number", text)
            vendor = ExtractionService._line_match("Vendor", text)
            invoice_date = ExtractionService._line_match("Date", text)
            due_date = ExtractionService._line_match("Due Date", text)
            subtotal = ExtractionService._last_match(r"subtotal\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            tax = ExtractionService._last_match(r"tax\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            total = ExtractionService._last_match(r"total\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            currency = ExtractionService._line_match("Currency", text)

            return {
                "invoice_number": invoice_number,
                "vendor": vendor,
                "invoice_date": ExtractionService._parse_date(invoice_date),
                "due_date": ExtractionService._parse_date(due_date),
                "subtotal": ExtractionService._parse_float(subtotal),
                "tax": ExtractionService._parse_float(tax),
                "total": ExtractionService._parse_float(total),
                "currency": currency,
            }

        if doc_type == "Bank Statement":
            account_holder = ExtractionService._line_match("Account Holder", text)
            account_number = ExtractionService._line_match("Account Number", text)
            bank = ExtractionService._line_match("Bank", text)
            statement_period = ExtractionService._line_match("Statement Period", text)
            credits = ExtractionService._last_match(r"(?:salary\s+)?credits?\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            debits = ExtractionService._last_match(r"debits\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            closing_balance = ExtractionService._last_match(r"closing\s*balance\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)

            return {
                "account_holder": account_holder,
                "account_number": ExtractionService._mask_identifier(account_number),
                "bank": bank,
                "statement_period": statement_period,
                "transactions": [],
                "credits": ExtractionService._parse_float(credits),
                "debits": ExtractionService._parse_float(debits),
                "closing_balance": ExtractionService._parse_float(closing_balance),
            }

        if doc_type == "Salary Slip":
            employee_name = ExtractionService._line_match("Employee Name", text)
            employer = ExtractionService._line_match("Employer", text)
            employee_id = ExtractionService._line_match("Employee ID", text)
            pay_period = ExtractionService._line_match("Pay Period", text)
            basic_salary = ExtractionService._last_match(r"basic\s*salary\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            allowances = ExtractionService._last_match(r"allowances\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            deductions = ExtractionService._last_match(r"deductions\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)
            net_salary = ExtractionService._last_match(r"net\s*salary\s*[:\-]?\s*\$?([0-9,]+(?:\.\d{1,2})?)", text)

            return {
                "employee_name": employee_name,
                "employer": employer,
                "employee_id": employee_id,
                "pay_period": pay_period,
                "basic_salary": ExtractionService._parse_float(basic_salary),
                "allowances": ExtractionService._parse_float(allowances),
                "deductions": ExtractionService._parse_float(deductions),
                "net_salary": ExtractionService._parse_float(net_salary),
            }

        if doc_type == "Identity Document":
            name = ExtractionService._line_match("Name", text)
            dob = ExtractionService._line_match("Date of Birth", text)
            document_number = ExtractionService._line_match("Document Number", text)
            expiry_date = ExtractionService._line_match("Expiry Date", text)

            return {
                "name": name,
                "date_of_birth": ExtractionService._parse_date(dob),
                "document_number": ExtractionService._mask_identifier(document_number),
                "expiry_date": ExtractionService._parse_date(expiry_date),
            }

        return {
            "document_type": doc_type or "Unknown",
            "raw_text": text,
        }
