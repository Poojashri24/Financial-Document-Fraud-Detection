from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.fusion import MultimodalAnalysisRequest
from backend.services.multimodal_fusion_service import MultimodalFusionEngine

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/multimodal")
async def analyze_multimodal(payload: MultimodalAnalysisRequest) -> dict:
    try:
        analysis = MultimodalFusionEngine().fuse_documents([doc.model_dump() for doc in payload.documents])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis


@router.get("/multimodal/{analysis_id}")
async def get_multimodal_analysis(analysis_id: str) -> dict:
    analysis = MultimodalFusionEngine.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Multimodal analysis not found.")
    return analysis
