from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import json

from ..services.agent import generate_agent_response
from ..services.intelligent_agent import run_intelligent_analyst_stream
from ..models.schemas import AgentRunResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/run", response_model=AgentRunResponse)
async def run_agent(
    question: str = Query(..., description="User question to analyze using pandas code + LLM"),
    dataset: str = Query("all", description="Which dataset to use: all (default - loads ALL sheets)|incident|hazard|audit|inspection"),
    model: str = Query("z-ai/glm-4.6", description="LLM model for code-gen and summary"),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Parameter 'question' is required")
    resp = generate_agent_response(question, dataset=dataset, model=model)
    return resp


@router.get("/stream")
async def run_agent_stream(
    question: str = Query(..., description="User question to analyze using pandas code + LLM"),
    dataset: str = Query("all", description="Which dataset to use: all (default - loads ALL sheets)|incident|hazard|audit|inspection"),
    model: str = Query("z-ai/glm-4.6", description="LLM model for streaming (default: FREE Grok via OpenRouter)"),
):
    """
    Streaming version of the agent that shows real-time progress.
    Returns Server-Sent Events (SSE) stream.
    
    Event types:
    - progress: Stage updates (generating_code, executing, verifying, etc.)
    - code_chunk: Real-time code generation
    - analysis_chunk: Real-time analysis generation
    - error: Execution errors
    - verification: Result verification status
    - complete: Final result with all data
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Parameter 'question' is required")
    
    async def event_generator():
        try:
            # Use intelligent analyst with reflection and verification
            async for event in run_intelligent_analyst_stream(question, dataset=dataset, model=model):
                # Format as Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Stream error: {error_details}")  # Log to console
            error_event = {
                "type": "error",
                "message": f"Stream error: {str(e)}",
                "details": error_details
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
