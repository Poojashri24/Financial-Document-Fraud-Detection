# Intelligent Financial Document Fraud Detection

This project is an academic prototype for detecting suspicious financial documents and cross-document inconsistencies using OCR, document preprocessing, structured extraction, and rule-based classification.

## Current Phases

- Phase 1: Project foundation and startup scaffolding
- Phase 2: Document upload and file processing
- Phase 3: OCR pipeline
- Phase 4: Rule-based document classification and field extraction

## Project Structure

- backend/: API application and business logic
- frontend/: Streamlit application
- data/: raw, processed, and sample data
- models/: checkpoints and trained model artifacts
- notebooks/: experimentation notebooks
- docs/: architecture and methodology documents
- tests/: automated tests
- reports/: generated analysis artifacts

## Run Backend

```bash
cd "C:\Users\pooja\Downloads\FYP-FIN-DOC-FRAUD-DETECT"
C:/Users/pooja/AppData/Local/Programs/Python/Python311/python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Frontend

```bash
cd "C:\Users\pooja\Downloads\FYP-FIN-DOC-FRAUD-DETECT"
C:/Users/pooja/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run frontend/app.py --server.port 8501 --server.headless true
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Upload API Example

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "files=@C:\path\to\invoice.pdf" \
  -F "files=@C:\path\to\statement.png"
```

## Supported Upload Formats

- PDF
- PNG
- JPG
- JPEG

## File Size Configuration

The default upload size limit is configured in the backend settings as `max_upload_size_mb` and is defined in [backend/core/config.py](backend/core/config.py).

## Notes

- CPU-compatible baseline only
- OCR and classification use transparent rule-based logic, not claimed ML training
- LLM integration remains optional and disabled by default
- Real datasets should remain outside the repository and should be added to .gitignore
- Synthetic data is used for local demonstration and tests
