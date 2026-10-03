from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.risk import RiskAssessmentRequest
from backend.services.llm_reasoning_service import LLMReasoningService
from backend.services.risk_scoring_service import RiskScoringService

router = APIRouter(prefix="/api/analysis", tags=["risk"])


@router.post("/risk")
async def assess_risk(payload: RiskAssessmentRequest) -> dict:
    if not payload.reasoning_id and payload.reasoning is None:
        raise HTTPException(status_code=400, detail="Provide either reasoning_id or a direct reasoning payload.")

    reasoning = payload.reasoning
    if payload.reasoning_id:
        reasoning = None
        for record in LLMReasoningService._reasoning_store.values():
            if record.get("reasoning_id") == payload.reasoning_id:
                reasoning = record
                break
        if reasoning is None:
            raise HTTPException(status_code=404, detail="Reasoning result not found.")

    if not isinstance(reasoning, dict):
        raise HTTPException(status_code=400, detail="The supplied reasoning payload is not valid.")

    assessment = RiskScoringService().score_reasoning(reasoning)
    return assessment


@router.get("/risk/{analysis_id}")
async def get_risk_assessment(analysis_id: str) -> dict:
    store = RiskScoringService._risk_store
    if analysis_id not in store:
        raise HTTPException(status_code=404, detail="Risk assessment not found.")
    return store[analysis_id]
