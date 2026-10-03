from backend.services.cross_document_verification_service import CrossDocumentVerificationService


def _docs_from_fields(*doc_fields):
    docs = []
    for item in doc_fields:
        docs.append({
            "document_id": item["document_id"],
            "document_type": item["document_type"],
            "fields": item["fields"],
        })
    return docs


def test_consistent_salary_and_bank_statement():
    docs = _docs_from_fields(
        {
            "document_id": "doc-101",
            "document_type": "Salary Slip",
            "fields": {
                "employee_name": "John Smith",
                "employer": "Acme Ltd",
                "employee_id": "EMP-204",
                "net_salary": 4300.0,
                "currency": "USD",
            },
        },
        {
            "document_id": "doc-102",
            "document_type": "Bank Statement",
            "fields": {
                "account_holder": "John Smith",
                "bank": "Metro Finance",
                "credits": 4300.0,
                "currency": "USD",
            },
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    result_by_rule = {item["rule_id"]: item for item in analysis["results"]}

    assert result_by_rule["name_consistency"]["status"] == "PASS"
    assert result_by_rule["salary_vs_bank_credit"]["status"] == "PASS"
    assert analysis["summary"]["failed_rules"] == 0


def test_salary_mismatch():
    docs = _docs_from_fields(
        {
            "document_id": "doc-201",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith", "net_salary": 4300.0, "currency": "USD"},
        },
        {
            "document_id": "doc-202",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "John Smith", "credits": 3500.0, "currency": "USD"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "salary_vs_bank_credit")

    assert rule["status"] == "FAIL"
    assert "4300.0" in rule["explanation"] or "3500.0" in rule["explanation"]
    assert rule["evidence"]


def test_name_mismatch():
    docs = _docs_from_fields(
        {
            "document_id": "doc-301",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith"},
        },
        {
            "document_id": "doc-302",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "Jane Smith"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "name_consistency")

    assert rule["status"] == "FAIL"
    assert "John Smith" in rule["explanation"] or "Jane Smith" in rule["explanation"]


def test_employer_mismatch():
    docs = _docs_from_fields(
        {
            "document_id": "doc-401",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith", "employer": "Acme Ltd"},
        },
        {
            "document_id": "doc-402",
            "document_type": "Invoice",
            "fields": {"vendor": "Beta Corp", "invoice_number": "INV-55"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "employer_company_consistency")

    assert rule["status"] == "FAIL"
    assert "Acme Ltd" in str(rule["values"]) or "Beta Corp" in str(rule["values"]) 


def test_invoice_total_mismatch():
    docs = _docs_from_fields(
        {
            "document_id": "doc-501",
            "document_type": "Invoice",
            "fields": {"invoice_number": "INV-77", "total": 1650.0, "currency": "USD"},
        },
        {
            "document_id": "doc-502",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "John Doe", "transaction_amount": 1500.0, "currency": "USD"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "invoice_total_vs_payment_amount")

    assert rule["status"] == "FAIL"
    assert "1650.0" in rule["explanation"] or "1500.0" in rule["explanation"]


def test_currency_mismatch():
    docs = _docs_from_fields(
        {
            "document_id": "doc-601",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith", "currency": "USD", "net_salary": 4300.0},
        },
        {
            "document_id": "doc-602",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "John Smith", "currency": "EUR", "credits": 4300.0},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "currency_consistency")

    assert rule["status"] == "FAIL"
    assert "USD" in str(rule["values"]) or "EUR" in str(rule["values"]) 


def test_duplicate_invoice_number():
    docs = _docs_from_fields(
        {
            "document_id": "doc-701",
            "document_type": "Invoice",
            "fields": {"invoice_number": "INV-1042", "total": 100.0},
        },
        {
            "document_id": "doc-702",
            "document_type": "Invoice",
            "fields": {"invoice_number": "INV-1042", "total": 120.0},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "duplicate_document_identifiers")

    assert rule["status"] == "FAIL"
    assert "INV-1042" in str(rule["values"]) or "INV-1042" in rule["explanation"]


def test_missing_fields_return_insufficient_data():
    docs = _docs_from_fields(
        {
            "document_id": "doc-801",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith"},
        },
        {
            "document_id": "doc-802",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "John Smith"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "salary_vs_bank_credit")

    assert rule["status"] == "INSUFFICIENT_DATA"
    assert rule["evidence"] == []


def test_unrelated_documents_are_not_assumed_related():
    docs = _docs_from_fields(
        {
            "document_id": "doc-901",
            "document_type": "Invoice",
            "fields": {"vendor": "Acme Ltd", "total": 100.0},
        },
        {
            "document_id": "doc-902",
            "document_type": "Identity Document",
            "fields": {"name": "Mary Jane Watson", "document_number": "AB1234567"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    rule = next(item for item in analysis["results"] if item["rule_id"] == "name_consistency")

    assert rule["status"] == "INSUFFICIENT_DATA"
    assert rule["documents"] == []


def test_multiple_simultaneous_inconsistencies():
    docs = _docs_from_fields(
        {
            "document_id": "doc-1001",
            "document_type": "Salary Slip",
            "fields": {"employee_name": "John Smith", "employer": "Acme Ltd", "net_salary": 4300.0, "currency": "USD"},
        },
        {
            "document_id": "doc-1002",
            "document_type": "Bank Statement",
            "fields": {"account_holder": "Jane Smith", "bank": "Metro Finance", "credits": 3000.0, "currency": "EUR"},
        },
        {
            "document_id": "doc-1003",
            "document_type": "Invoice",
            "fields": {"invoice_number": "INV-1042", "total": 1650.0, "currency": "USD"},
        },
        {
            "document_id": "doc-1004",
            "document_type": "Invoice",
            "fields": {"invoice_number": "INV-1042", "total": 1500.0, "currency": "USD"},
        },
    )

    analysis = CrossDocumentVerificationService.verify_documents(docs)
    statuses = {item["rule_id"]: item["status"] for item in analysis["results"]}

    assert statuses["name_consistency"] == "FAIL"
    assert statuses["salary_vs_bank_credit"] == "FAIL"
    assert statuses["currency_consistency"] == "FAIL"
    assert statuses["duplicate_document_identifiers"] == "FAIL"
    assert analysis["summary"]["overall_inconsistency_count"] >= 4
