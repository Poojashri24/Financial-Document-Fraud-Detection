from fastapi import FastAPI

from backend.api.routes.analysis import router as analysis_router
from backend.api.routes.classification import router as classification_router
from backend.api.routes.documents import router as document_router
from backend.api.routes.reasoning import router as reasoning_router
from backend.api.routes.risk import router as risk_router
from backend.api.routes.tampering import router as tampering_router
from backend.api.routes.verification import router as verification_router
from backend.core.config import settings
from backend.core.logging_config import setup_logging

logger = setup_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(document_router)
app.include_router(classification_router)
app.include_router(verification_router)
app.include_router(tampering_router)
app.include_router(analysis_router)
app.include_router(reasoning_router)
app.include_router(risk_router)


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }


@app.get("/")
def root():
    return {
        "message": "Financial document fraud detection prototype is running.",
        "service": settings.app_name,
    }
