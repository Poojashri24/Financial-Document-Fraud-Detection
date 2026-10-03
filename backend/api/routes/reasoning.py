from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.llm_reasoning_service import LLMReasoningService
from backend.services.multimodal_fusion_service import MultimodalFusionEngine
from backend.schemas.reasoning import ReasoningRequest

router = APIRouter(prefix="/api/analysis", tags=["reasoning"])


@router.post("/reason")
async def generate_reasoning(payload: ReasoningRequest) -> dict:
    if not payload.analysis_id and payload.multimodal_analysis is None:
        raise HTTPException(status_code=400, detail="Provide either an analysis_id or a multimodal_analysis payload.")

    analysis = payload.multimodal_analysis
    if payload.analysis_id:
        analysis = MultimodalFusionEngine.get_analysis(payload.analysis_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="Multimodal analysis not found.")

    if not isinstance(analysis, dict):
        raise HTTPException(status_code=400, detail="The supplied multimodal analysis is not valid.")

    service = LLMReasoningService(provider=payload.provider, api_key=payload.api_key)
    result = service.reason_about_analysis(analysis)
    return result


@router.get("/reason/{reasoning_id}")
async def get_reasoning(reasoning_id: str) -> dict:
    store = getattr(LLMReasoningService, "_reasoning_store", None)
    if store is None:
        LLMReasoningService._reasoning_store = {}
        store = LLMReasoningService._reasoning_store

    if reasoning_id not in store:
        raise HTTPException(status_code=404, detail="Reasoning result not found.")
    return store[reasoning_id]
