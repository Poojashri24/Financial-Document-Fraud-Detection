from fastapi.testclient import TestClient

from backend.main import app
from backend.services.classification_service import ClassificationService
from backend.services.extraction_service import ExtractionService


SAMPLE_INVOICE_TEXT = """
INVOICE
Invoice Number INV-1042
Vendor ABC Traders
Date 2024-03-15
Due Date 2024-04-15
Subtotal 1500.00
Tax 150.00
Total 1650.00
Currency USD
"""

SAMPLE_BANK_TEXT = """
BANK STATEMENT
Account Holder Jane Doe
Account Number 123456789012
Bank Metro Finance
Statement Period 2024-01-01 to 2024-01-31
Credits 3200.00
Debits 1200.00
Closing Balance 2000.00
"""

SAMPLE_SALARY_TEXT = """
SALARY SLIP
Employee Name John Smith
Employer Acme Ltd
Employee ID EMP-204
Pay Period 2024-02
Basic Salary 4000.00
Allowances 600.00
Deductions 300.00
Net Salary 4300.00
"""

SAMPLE_ID_TEXT = """
IDENTITY DOCUMENT
Name Mary Jane Watson
Date of Birth 1990-05-12
Document Number AB1234567
Expiry Date 2032-05-12
"""


def test_invoice_classification_heuristic():
    result = ClassificationService.classify_text(SAMPLE_INVOICE_TEXT)
    assert result["document_type"] == "Invoice"
    assert result["confidence"] >= 0.0
    assert result["classification_method"] == "heuristic_ocr_rules"


def test_bank_statement_classification_heuristic():
    result = ClassificationService.classify_text(SAMPLE_BANK_TEXT)
    assert result["document_type"] == "Bank Statement"
    assert result["classification_method"] == "heuristic_ocr_rules"


def test_salary_slip_classification_heuristic():
    result = ClassificationService.classify_text(SAMPLE_SALARY_TEXT)
    assert result["document_type"] == "Salary Slip"


def test_identity_document_classification_heuristic():
    result = ClassificationService.classify_text(SAMPLE_ID_TEXT)
    assert result["document_type"] == "Identity Document"


def test_unknown_classification():
    result = ClassificationService.classify_text("some generic note with no financial indicators")
    assert result["document_type"] == "Unknown"


def test_invoice_field_extraction():
    extracted = ExtractionService.extract_fields("Invoice", SAMPLE_INVOICE_TEXT)
    assert extracted["invoice_number"] == "INV-1042"
    assert extracted["vendor"] == "ABC Traders"
    assert extracted["currency"] == "USD"
    assert extracted["total"] == 1650.0


def test_bank_statement_field_extraction():
    extracted = ExtractionService.extract_fields("Bank Statement", SAMPLE_BANK_TEXT)
    assert extracted["account_holder"] == "Jane Doe"
    assert extracted["bank"] == "Metro Finance"
    assert extracted["closing_balance"] == 2000.0
    assert extracted["account_number"] == "********9012"


def test_salary_slip_field_extraction():
    extracted = ExtractionService.extract_fields("Salary Slip", SAMPLE_SALARY_TEXT)
    assert extracted["employee_name"] == "John Smith"
    assert extracted["net_salary"] == 4300.0
    assert extracted["employee_id"] == "EMP-204"


def test_identity_document_field_extraction():
    extracted = ExtractionService.extract_fields("Identity Document", SAMPLE_ID_TEXT)
    assert extracted["name"] == "Mary Jane Watson"
    assert extracted["document_number"] == "AB********"
    assert extracted["expiry_date"] == "2032-05-12"


def test_missing_fields_are_safe():
    fields = ExtractionService.extract_fields("Invoice", "INVOICE\nTotal 500.00")
    assert fields.get("vendor") is None
    assert fields.get("invoice_number") is None


def test_numeric_and_date_parsing():
    parsed = ExtractionService.extract_fields("Invoice", "Invoice Number INV-5\nTotal 1234.56\nDate 2024-07-01")
    assert parsed["total"] == 1234.56
    assert parsed["invoice_date"] == "2024-07-01"


def test_classify_api_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/documents/placeholder/classify",
        json={"ocr_text": SAMPLE_INVOICE_TEXT},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "Invoice"


def test_extraction_api_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/documents/placeholder/extract",
        json={"document_type": "Salary Slip", "ocr_text": SAMPLE_SALARY_TEXT},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["employee_name"] == "John Smith"


def test_fields_endpoint_for_document_id_requires_document():
    client = TestClient(app)
    response = client.get("/api/documents/does-not-exist/fields")
    assert response.status_code == 404
