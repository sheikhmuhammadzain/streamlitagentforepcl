from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException

from ..models.schemas import AgentRunResponse
from ..services.agent import generate_agent_response

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/run", response_model=AgentRunResponse)
async def run_agent(
    question: str = Query(..., description="User question to analyze using pandas code + LLM"),
    dataset: str = Query("incident", description="Which dataset to use: incident|hazard|audit|inspection|all"),
    model: str = Query("gpt-4o", description="LLM model for code-gen and summary"),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Parameter 'question' is required")
    resp = generate_agent_response(question, dataset=dataset, model=model)
    return AgentRunResponse(
        code=resp.code,
        stdout=resp.stdout,
        error=resp.error,
        result_preview=resp.result_preview,
        figure=resp.figure,
        mpl_png_base64=resp.mpl_png_base64,
        analysis=resp.analysis,
    )
