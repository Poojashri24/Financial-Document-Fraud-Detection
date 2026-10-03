from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class CrossDocumentVerificationService:
    _field_registry: Dict[str, Dict[str, Any]] = {}
    _analysis_store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_document_fields(cls, document_id: str, document_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "document_id": document_id,
            "document_type": document_type,
            "fields": dict(fields or {}),
        }
        cls._field_registry[document_id] = payload
        return payload

    @classmethod
    def get_analysis(cls, analysis_id: str) -> Optional[Dict[str, Any]]:
        return cls._analysis_store.get(analysis_id)

    @classmethod
    def fetch_documents_by_ids(cls, document_ids: Sequence[str]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for document_id in document_ids:
            record = cls._field_registry.get(document_id)
            if record is not None:
                docs.append(record)
        return docs

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "").replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _as_date(value: Any) -> Optional[datetime.date]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None

    @classmethod
    def _combined_values(cls, doc: Dict[str, Any], keys: Sequence[str]) -> List[str]:
        result: List[str] = []
        for key in keys:
            value = doc.get("fields", {}).get(key)
            if value is not None and str(value).strip():
                result.append(str(value).strip())
        return result

    @classmethod
    def _rule_result(cls, rule_id: str, rule_name: str, status: str, documents: Sequence[str], fields_compared: Sequence[str], values: Dict[str, Any], explanation: str, evidence: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "status": status,
            "documents": list(documents),
            "fields_compared": list(fields_compared),
            "values": values,
            "explanation": explanation,
            "evidence": list(evidence or []),
        }

    @classmethod
    def _extract_name_documents(cls, docs: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
        name_docs: List[Tuple[str, str]] = []
        for doc in docs:
            fields = doc.get("fields", {})
            name_candidates = [
                fields.get("name"),
                fields.get("employee_name"),
                fields.get("account_holder"),
            ]
            for value in name_candidates:
                if value is not None and str(value).strip():
                    name_docs.append((doc["document_id"], str(value).strip()))
                    break
        return name_docs

    @classmethod
    def _rule_name_consistency(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        name_docs = cls._extract_name_documents(docs)
        if len(name_docs) < 2:
            return cls._rule_result(
                "name_consistency",
                "Name consistency",
                "INSUFFICIENT_DATA",
                [],
                ["name", "employee_name", "account_holder"],
                {},
                "Name comparison is not meaningful because there are fewer than two comparable identity names in the uploaded documents.",
                [],
            )

        unique_names = {cls._normalize_text(value) for _, value in name_docs}
        if len(unique_names) == 1:
            return cls._rule_result(
                "name_consistency",
                "Name consistency",
                "PASS",
                [document_id for document_id, _ in name_docs],
                ["name", "employee_name", "account_holder"],
                {"names": [value for _, value in name_docs]},
                "The comparable names across the uploaded documents are consistent.",
                [f"{document_id}: {value}" for document_id, value in name_docs],
            )

        doc_ids = [document_id for document_id, _ in name_docs]
        values = {document_id: value for document_id, value in name_docs}
        mismatch_text = ", ".join(f"{document_id}={value}" for document_id, value in name_docs)
        return cls._rule_result(
            "name_consistency",
            "Name consistency",
            "FAIL",
            doc_ids,
            ["name", "employee_name", "account_holder"],
            values,
            f"The comparable names across the uploaded documents do not align: {mismatch_text}.",
            [f"{document_id}: {value}" for document_id, value in name_docs],
        )

    @classmethod
    def _rule_employer_company_consistency(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        company_values: List[Tuple[str, str]] = []
        for doc in docs:
            fields = doc.get("fields", {})
            for key in ("employer", "vendor", "company"):
                value = fields.get(key)
                if value is not None and str(value).strip():
                    company_values.append((doc["document_id"], str(value).strip()))
                    break

        if len(company_values) < 2:
            return cls._rule_result(
                "employer_company_consistency",
                "Employer/company consistency",
                "INSUFFICIENT_DATA",
                [],
                ["employer", "vendor", "company"],
                {},
                "No meaningful employer or company values were available for comparison across the uploaded documents.",
                [],
            )

        unique_values = {cls._normalize_text(value) for _, value in company_values}
        if len(unique_values) == 1:
            return cls._rule_result(
                "employer_company_consistency",
                "Employer/company consistency",
                "PASS",
                [document_id for document_id, _ in company_values],
                ["employer", "vendor", "company"],
                {"companies": [value for _, value in company_values]},
                "The employer or company values present in the related documents are consistent.",
                [f"{document_id}: {value}" for document_id, value in company_values],
            )

        return cls._rule_result(
            "employer_company_consistency",
            "Employer/company consistency",
            "FAIL",
            [document_id for document_id, _ in company_values],
            ["employer", "vendor", "company"],
            {document_id: value for document_id, value in company_values},
            "The employer or company values differ across the related documents and do not align.",
            [f"{document_id}: {value}" for document_id, value in company_values],
        )

    @classmethod
    def _rule_employee_id_consistency(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        employee_ids: List[Tuple[str, str]] = []
        for doc in docs:
            value = doc.get("fields", {}).get("employee_id")
            if value is not None and str(value).strip():
                employee_ids.append((doc["document_id"], str(value).strip()))

        if len(employee_ids) < 2:
            return cls._rule_result(
                "employee_id_consistency",
                "Employee ID consistency",
                "INSUFFICIENT_DATA",
                [],
                ["employee_id"],
                {},
                "There are not enough employee identifiers available to compare across documents.",
                [],
            )

        unique_ids = {cls._normalize_text(value) for _, value in employee_ids}
        if len(unique_ids) == 1:
            return cls._rule_result(
                "employee_id_consistency",
                "Employee ID consistency",
                "PASS",
                [document_id for document_id, _ in employee_ids],
                ["employee_id"],
                {"employee_ids": [value for _, value in employee_ids]},
                "The employee identifiers available across the uploaded documents are consistent.",
                [f"{document_id}: {value}" for document_id, value in employee_ids],
            )

        return cls._rule_result(
            "employee_id_consistency",
            "Employee ID consistency",
            "FAIL",
            [document_id for document_id, _ in employee_ids],
            ["employee_id"],
            {document_id: value for document_id, value in employee_ids},
            "The employee identifiers in the related documents conflict with each other.",
            [f"{document_id}: {value}" for document_id, value in employee_ids],
        )

    @classmethod
    def _rule_salary_vs_bank_credit(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        salary_docs = []
        bank_docs = []
        for doc in docs:
            fields = doc.get("fields", {})
            if doc.get("document_type") == "Salary Slip" and fields.get("net_salary") is not None:
                salary_docs.append((doc["document_id"], cls._as_float(fields.get("net_salary"))))
            if doc.get("document_type") == "Bank Statement" and fields.get("credits") is not None:
                bank_docs.append((doc["document_id"], cls._as_float(fields.get("credits"))))

        if not salary_docs or not bank_docs:
            return cls._rule_result(
                "salary_vs_bank_credit",
                "Salary vs bank credit",
                "INSUFFICIENT_DATA",
                [],
                ["net_salary", "credits"],
                {},
                "Salary and bank credit values are unavailable for a valid salary vs bank comparison.",
                [],
            )

        salary_value = salary_docs[0][1]
        bank_value = bank_docs[0][1]
        if salary_value is None or bank_value is None:
            return cls._rule_result(
                "salary_vs_bank_credit",
                "Salary vs bank credit",
                "INSUFFICIENT_DATA",
                [salary_docs[0][0], bank_docs[0][0]],
                ["net_salary", "credits"],
                {"net_salary": salary_value, "bank_credit": bank_value},
                "One of the required salary or bank credit values is missing or not numeric.",
                [],
            )

        if abs(salary_value - bank_value) <= 0.01:
            return cls._rule_result(
                "salary_vs_bank_credit",
                "Salary vs bank credit",
                "PASS",
                [salary_docs[0][0], bank_docs[0][0]],
                ["net_salary", "credits"],
                {"net_salary": salary_value, "bank_credit": bank_value},
                f"The salary-slip net salary ({salary_value}) matches the bank credit amount ({bank_value}).",
                [f"{salary_docs[0][0]}: net_salary={salary_value}", f"{bank_docs[0][0]}: credits={bank_value}"],
            )

        return cls._rule_result(
            "salary_vs_bank_credit",
            "Salary vs bank credit",
            "FAIL",
            [salary_docs[0][0], bank_docs[0][0]],
            ["net_salary", "credits"],
            {"net_salary": salary_value, "bank_credit": bank_value},
            f"The salary-slip net salary ({salary_value}) differs from the bank credit amount ({bank_value}), which indicates a salary mismatch.",
            [f"{salary_docs[0][0]}: net_salary={salary_value}", f"{bank_docs[0][0]}: credits={bank_value}"],
        )

    @classmethod
    def _rule_invoice_total_vs_payment_amount(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        invoice_docs = []
        payment_docs = []
        for doc in docs:
            fields = doc.get("fields", {})
            if doc.get("document_type") == "Invoice" and fields.get("total") is not None:
                invoice_docs.append((doc["document_id"], cls._as_float(fields.get("total"))))
            if doc.get("document_type") == "Bank Statement" and (fields.get("transaction_amount") is not None or fields.get("credits") is not None):
                payment_value = fields.get("transaction_amount")
                if payment_value is None:
                    payment_value = fields.get("credits")
                payment_docs.append((doc["document_id"], cls._as_float(payment_value)))

        if not invoice_docs or not payment_docs:
            return cls._rule_result(
                "invoice_total_vs_payment_amount",
                "Invoice total vs payment amount",
                "INSUFFICIENT_DATA",
                [],
                ["total", "transaction_amount", "credits"],
                {},
                "Invoice totals and related payment amounts are not both available for a meaningful comparison.",
                [],
            )

        invoice_value = invoice_docs[0][1]
        payment_value = payment_docs[0][1]
        if invoice_value is None or payment_value is None:
            return cls._rule_result(
                "invoice_total_vs_payment_amount",
                "Invoice total vs payment amount",
                "INSUFFICIENT_DATA",
                [invoice_docs[0][0], payment_docs[0][0]],
                ["total", "transaction_amount", "credits"],
                {"invoice_total": invoice_value, "payment_amount": payment_value},
                "One or both comparison values is missing or not numeric.",
                [],
            )

        if abs(invoice_value - payment_value) <= 0.01:
            return cls._rule_result(
                "invoice_total_vs_payment_amount",
                "Invoice total vs payment amount",
                "PASS",
                [invoice_docs[0][0], payment_docs[0][0]],
                ["total", "transaction_amount", "credits"],
                {"invoice_total": invoice_value, "payment_amount": payment_value},
                f"The invoice total ({invoice_value}) matches the related payment amount ({payment_value}).",
                [f"{invoice_docs[0][0]}: total={invoice_value}", f"{payment_docs[0][0]}: amount={payment_value}"],
            )

        return cls._rule_result(
            "invoice_total_vs_payment_amount",
            "Invoice total vs payment amount",
            "FAIL",
            [invoice_docs[0][0], payment_docs[0][0]],
            ["total", "transaction_amount", "credits"],
            {"invoice_total": invoice_value, "payment_amount": payment_value},
            f"The invoice total ({invoice_value}) does not match the related payment amount ({payment_value}).",
            [f"{invoice_docs[0][0]}: total={invoice_value}", f"{payment_docs[0][0]}: amount={payment_value}"],
        )

    @classmethod
    def _rule_invoice_date_vs_payment_date(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        invoice_docs = []
        payment_docs = []
        for doc in docs:
            fields = doc.get("fields", {})
            if doc.get("document_type") == "Invoice" and fields.get("invoice_date") is not None:
                invoice_docs.append((doc["document_id"], str(fields.get("invoice_date"))))
            if doc.get("document_type") == "Bank Statement" and (fields.get("payment_date") is not None or fields.get("date") is not None):
                payment_value = fields.get("payment_date") or fields.get("date")
                payment_docs.append((doc["document_id"], str(payment_value)))

        if not invoice_docs or not payment_docs:
            return cls._rule_result(
                "invoice_date_vs_payment_date",
                "Invoice date vs payment date",
                "INSUFFICIENT_DATA",
                [],
                ["invoice_date", "payment_date"],
                {},
                "Invoice date and bank payment date values are not both available for a meaningful comparison.",
                [],
            )

        invoice_date = cls._as_date(invoice_docs[0][1])
        payment_date = cls._as_date(payment_docs[0][1])
        if invoice_date is None or payment_date is None:
            return cls._rule_result(
                "invoice_date_vs_payment_date",
                "Invoice date vs payment date",
                "INSUFFICIENT_DATA",
                [invoice_docs[0][0], payment_docs[0][0]],
                ["invoice_date", "payment_date"],
                {"invoice_date": invoice_docs[0][1], "payment_date": payment_docs[0][1]},
                "The invoice date or payment date cannot be parsed into a valid date value.",
                [],
            )

        if payment_date < invoice_date:
            return cls._rule_result(
                "invoice_date_vs_payment_date",
                "Invoice date vs payment date",
                "FAIL",
                [invoice_docs[0][0], payment_docs[0][0]],
                ["invoice_date", "payment_date"],
                {"invoice_date": invoice_docs[0][1], "payment_date": payment_docs[0][1]},
                f"The payment date ({payment_docs[0][1]}) occurs before the invoice date ({invoice_docs[0][1]}), which is inconsistent.",
                [f"{invoice_docs[0][0]}: {invoice_docs[0][1]}", f"{payment_docs[0][0]}: {payment_docs[0][1]}"],
            )

        return cls._rule_result(
            "invoice_date_vs_payment_date",
            "Invoice date vs payment date",
            "PASS",
            [invoice_docs[0][0], payment_docs[0][0]],
            ["invoice_date", "payment_date"],
            {"invoice_date": invoice_docs[0][1], "payment_date": payment_docs[0][1]},
            f"The payment date ({payment_docs[0][1]}) is not earlier than the invoice date ({invoice_docs[0][1]}).",
            [f"{invoice_docs[0][0]}: {invoice_docs[0][1]}", f"{payment_docs[0][0]}: {payment_docs[0][1]}"],
        )

    @classmethod
    def _rule_account_holder_consistency(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        account_names: List[Tuple[str, str]] = []
        for doc in docs:
            fields = doc.get("fields", {})
            name_values = [fields.get("account_holder"), fields.get("employee_name"), fields.get("name")]
            for value in name_values:
                if value is not None and str(value).strip():
                    account_names.append((doc["document_id"], str(value).strip()))
                    break

        if len(account_names) < 2:
            return cls._rule_result(
                "account_holder_consistency",
                "Account-holder consistency",
                "INSUFFICIENT_DATA",
                [],
                ["account_holder", "employee_name", "name"],
                {},
                "There are not enough account-holder or person-name values to compare across the related documents.",
                [],
            )

        unique_names = {cls._normalize_text(value) for _, value in account_names}
        if len(unique_names) == 1:
            return cls._rule_result(
                "account_holder_consistency",
                "Account-holder consistency",
                "PASS",
                [document_id for document_id, _ in account_names],
                ["account_holder", "employee_name", "name"],
                {"names": [value for _, value in account_names]},
                "The account-holder and person-name values are consistent across the relevant documents.",
                [f"{document_id}: {value}" for document_id, value in account_names],
            )

        return cls._rule_result(
            "account_holder_consistency",
            "Account-holder consistency",
            "FAIL",
            [document_id for document_id, _ in account_names],
            ["account_holder", "employee_name", "name"],
            {document_id: value for document_id, value in account_names},
            "The account-holder and person-name values do not align across the related documents.",
            [f"{document_id}: {value}" for document_id, value in account_names],
        )

    @classmethod
    def _rule_currency_consistency(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        currency_values: List[Tuple[str, str]] = []
        for doc in docs:
            value = doc.get("fields", {}).get("currency")
            if value is not None and str(value).strip():
                currency_values.append((doc["document_id"], str(value).strip().upper()))

        if len(currency_values) < 2:
            return cls._rule_result(
                "currency_consistency",
                "Currency consistency",
                "INSUFFICIENT_DATA",
                [],
                ["currency"],
                {},
                "No meaningful multi-document currency comparison is possible because the comparison data is incomplete.",
                [],
            )

        unique_currencies = {value for _, value in currency_values}
        if len(unique_currencies) == 1:
            return cls._rule_result(
                "currency_consistency",
                "Currency consistency",
                "PASS",
                [document_id for document_id, _ in currency_values],
                ["currency"],
                {"currencies": [value for _, value in currency_values]},
                "All available related currency values are consistent.",
                [f"{document_id}: {value}" for document_id, value in currency_values],
            )

        return cls._rule_result(
            "currency_consistency",
            "Currency consistency",
            "FAIL",
            [document_id for document_id, _ in currency_values],
            ["currency"],
            {document_id: value for document_id, value in currency_values},
            "The related documents use conflicting currencies and therefore do not agree on the transaction denomination.",
            [f"{document_id}: {value}" for document_id, value in currency_values],
        )

    @classmethod
    def _rule_duplicate_document_identifiers(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        seen: Dict[str, List[str]] = {}
        for doc in docs:
            for key in ("invoice_number", "document_number", "employee_id", "account_number"):
                value = doc.get("fields", {}).get(key)
                if value is not None and str(value).strip():
                    seen.setdefault(str(value).strip(), []).append(doc["document_id"])

        duplicates = {value: ids for value, ids in seen.items() if len(ids) > 1}
        if not duplicates:
            if not seen:
                return cls._rule_result(
                    "duplicate_document_identifiers",
                    "Duplicate document identifiers",
                    "INSUFFICIENT_DATA",
                    [],
                    ["invoice_number", "document_number", "employee_id", "account_number"],
                    {},
                    "No duplicate document identifiers were detected and no identifying values were present for comparison.",
                    [],
                )
            return cls._rule_result(
                "duplicate_document_identifiers",
                "Duplicate document identifiers",
                "PASS",
                list({doc_id for ids in seen.values() for doc_id in ids}),
                ["invoice_number", "document_number", "employee_id", "account_number"],
                {value: ids for value, ids in seen.items()},
                "No duplicate invoice or document identifiers were found across the uploaded documents.",
                [f"{document_id}" for ids in seen.values() for document_id in ids],
            )

        return cls._rule_result(
            "duplicate_document_identifiers",
            "Duplicate document identifiers",
            "FAIL",
            list({doc_id for ids in duplicates.values() for doc_id in ids}),
            ["invoice_number", "document_number", "employee_id", "account_number"],
            {value: ids for value, ids in duplicates.items()},
            "The same document identifier appears more than once across the uploaded documents, which indicates a duplicate reference.",
            [f"{value}: {', '.join(ids)}" for value, ids in duplicates.items()],
        )

    @classmethod
    def _rule_date_inconsistencies(cls, docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        contradictions: List[str] = []
        for doc in docs:
            fields = doc.get("fields", {})
            invoice_date = fields.get("invoice_date")
            due_date = fields.get("due_date")
            if invoice_date and due_date:
                invoice_dt = cls._as_date(invoice_date)
                due_dt = cls._as_date(due_date)
                if invoice_dt and due_dt and due_dt < invoice_dt:
                    contradictions.append(f"{doc['document_id']}: due date {due_date} is earlier than invoice date {invoice_date}")

            dob = fields.get("date_of_birth")
            expiry = fields.get("expiry_date")
            if dob and expiry:
                dob_dt = cls._as_date(dob)
                expiry_dt = cls._as_date(expiry)
                if dob_dt and expiry_dt and expiry_dt < dob_dt:
                    contradictions.append(f"{doc['document_id']}: expiry date {expiry} is earlier than DOB {dob}")

            period = fields.get("statement_period")
            if period and " to " in str(period):
                start_text, end_text = str(period).split(" to ", 1)
                start_dt = cls._as_date(start_text.strip())
                end_dt = cls._as_date(end_text.strip())
                if start_dt and end_dt and end_dt < start_dt:
                    contradictions.append(f"{doc['document_id']}: statement period ends before it begins")

        has_date_data = any(
            doc.get("fields", {}).get(key) is not None
            for doc in docs
            for key in ("invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period")
        )

        if contradictions:
            return cls._rule_result(
                "date_inconsistencies",
                "Date inconsistencies",
                "FAIL",
                [doc["document_id"] for doc in docs if any(doc.get("fields", {}).get(key) for key in ("invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"))],
                ["invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"],
                {"contradictions": contradictions},
                "The uploaded documents include clearly inconsistent date relationships that cannot be reconciled logically.",
                contradictions,
            )

        if not has_date_data:
            return cls._rule_result(
                "date_inconsistencies",
                "Date inconsistencies",
                "INSUFFICIENT_DATA",
                [],
                ["invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"],
                {},
                "No date fields were available for cross-document date validation.",
                [],
            )

        return cls._rule_result(
            "date_inconsistencies",
            "Date inconsistencies",
            "PASS",
            [doc["document_id"] for doc in docs if any(doc.get("fields", {}).get(key) for key in ("invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"))],
            ["invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"],
            {"checked_dates": [doc.get("document_id") for doc in docs]},
            "No impossible or contradictory date relationships were detected in the available date fields.",
            [f"{doc['document_id']}: date fields examined" for doc in docs if any(doc.get("fields", {}).get(key) for key in ("invoice_date", "due_date", "date_of_birth", "expiry_date", "statement_period"))],
        )

    @classmethod
    def verify_documents(cls, documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        normalized_docs = []
        for document in documents:
            if not isinstance(document, dict):
                continue
            normalized = {
                "document_id": document.get("document_id", f"doc-{len(normalized_docs) + 1}"),
                "document_type": document.get("document_type", "Unknown"),
                "fields": dict(document.get("fields", {}) or {}),
            }
            normalized_docs.append(normalized)

        rule_functions = [
            cls._rule_name_consistency,
            cls._rule_employer_company_consistency,
            cls._rule_employee_id_consistency,
            cls._rule_salary_vs_bank_credit,
            cls._rule_invoice_total_vs_payment_amount,
            cls._rule_invoice_date_vs_payment_date,
            cls._rule_account_holder_consistency,
            cls._rule_currency_consistency,
            cls._rule_duplicate_document_identifiers,
            cls._rule_date_inconsistencies,
        ]

        results = [func(normalized_docs) for func in rule_functions]
        summary = {
            "number_of_rules_checked": len(results),
            "passed_rules": sum(1 for item in results if item["status"] == "PASS"),
            "failed_rules": sum(1 for item in results if item["status"] == "FAIL"),
            "warnings": sum(1 for item in results if item["status"] == "WARNING"),
            "insufficient_data_checks": sum(1 for item in results if item["status"] == "INSUFFICIENT_DATA"),
            "overall_inconsistency_count": sum(1 for item in results if item["status"] in {"FAIL", "WARNING"}),
        }

        analysis_id = str(uuid.uuid4())
        analysis = {
            "analysis_id": analysis_id,
            "document_count": len(normalized_docs),
            "results": results,
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        cls._analysis_store[analysis_id] = analysis
        return analysis
