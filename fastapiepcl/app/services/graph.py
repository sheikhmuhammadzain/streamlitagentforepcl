"""
LangGraph-based workflow for Safety Copilot Agent
Built by Qbit - Using Grok's 7-step analytical approach
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
import operator
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .agent import (
    _is_conversational_query,
    _generate_conversational_response,
    load_default_sheets,
    _select_primary_df,
    build_workbook_context,
    ask_openai,
    extract_python_code,
    run_user_code,
    verify_result_quality,
    _result_preview,
    _fig_to_dict,
    _mpl_to_png_b64,
    _to_native_jsonable,
    _normalize_or_recompute,
)


# ==================== State Schema ====================

class SafetyCopilotState(TypedDict):
    """State schema for Safety Copilot LangGraph workflow"""
    
    # User interaction
    query: str
    query_type: Literal["conversational", "analytical"]
    dataset: str
    model: str
    
    # Data context
    workbook: Optional[Dict[str, pd.DataFrame]]
    primary_df: Optional[pd.DataFrame]
    dfs: Optional[Dict[str, pd.DataFrame]]
    context: str
    
    # Code generation & execution
    code: str
    execution_result: Optional[Any]
    execution_env: Dict
    stdout: str
    stderr: str
    
    # Verification & iteration
    verification: Dict
    past_attempts: Annotated[List[Dict], operator.add]
    attempts: int
    max_retries: int
    best_result: Optional[Any]
    best_verification: Dict
    best_code: str
    
    # Final output
    result_preview: List[Dict]
    figure: Optional[Dict]
    mpl_png_base64: Optional[str]
    analysis: str
    
    # Progress tracking
    current_stage: str
    error_message: Optional[str]


# ==================== Node Functions ====================

def query_router_node(state: SafetyCopilotState) -> Command:
    """Route queries to conversational or analytical path"""
    if _is_conversational_query(state["query"]):
        return Command(
            goto="conversational_handler",
            update={
                "query_type": "conversational",
                "current_stage": "routing_conversational"
            }
        )
    
    return Command(
        goto="data_loader",
        update={
            "query_type": "analytical",
            "current_stage": "routing_analytical"
        }
    )


def conversational_handler_node(state: SafetyCopilotState) -> Dict:
    """Handle conversational queries instantly"""
    response = _generate_conversational_response(
        state["query"],
        model=state.get("model", "z-ai/glm-4.6")
    )
    
    return {
        "analysis": response,
        "current_stage": "conversational_complete",
        "code": "# Conversational query - no code generated",
        "attempts": 1
    }


def data_loader_node(state: SafetyCopilotState) -> Dict:
    """Load all data sheets and build context"""
    dataset_l = (state.get("dataset") or "incident").lower()
    workbook = load_default_sheets()
    dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
    
    preferred = dataset_l if dataset_l in ("incident", "hazard", "audit", "inspection") else None
    primary_df = _select_primary_df(dfs, state["query"], preferred=preferred)
    context = build_workbook_context(workbook, query=state["query"])
    
    return {
        "workbook": workbook,
        "dfs": dfs,
        "primary_df": primary_df,
        "context": context,
        "current_stage": "data_loaded"
    }


def code_generator_node(state: SafetyCopilotState) -> Dict:
    """Generate pandas/plotly code using Grok's 7-step approach"""
    # Enhance context with past failures for self-correction
    enhanced_context = state["context"]
    
    if state.get("past_attempts"):
        failures_context = "\n\nPREVIOUS FAILED ATTEMPTS:\n"
        for i, attempt in enumerate(state["past_attempts"], 1):
            failures_context += f"\nAttempt {i}:\n"
            failures_context += f"Code:\n{attempt.get('code', 'N/A')}\n"
            failures_context += f"Issue: {attempt.get('error', 'N/A')}\n"
            if attempt.get('verification', {}).get('suggestions'):
                failures_context += f"Suggestions: {attempt['verification']['suggestions']}\n"
        
        failures_context += "\n\nGENERATE CORRECTED CODE that fixes the issues above.\n"
        enhanced_context += failures_context
    
    # Generate code
    code_response = ask_openai(
        question=state["query"],
        context=enhanced_context,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=True,
        multi_df=True
    )
    
    code_block = extract_python_code(code_response)
    
    # Fallback if no code generated
    if not code_block:
        code_block = """# Fallback code
_df = df
if _df is None and isinstance(dfs, dict) and dfs:
    _df = list(dfs.values())[0]
if _df is not None:
    result = _df.head(10)
"""
    
    return {
        "code": code_block,
        "current_stage": "code_generated"
    }


def code_executor_node(state: SafetyCopilotState) -> Dict:
    """Execute generated code"""
    env, stdout, stderr = run_user_code(
        state["code"],
        df=state.get("primary_df"),
        dfs=state.get("dfs")
    )
    
    has_error = bool(stderr and stderr.strip())
    
    return {
        "execution_result": env.get("result"),
        "execution_env": env,
        "stdout": stdout,
        "stderr": stderr,
        "current_stage": "code_executed_error" if has_error else "code_executed_success"
    }


def result_verifier_node(state: SafetyCopilotState) -> Dict:
    """Verify result quality using LLM"""
    verification = verify_result_quality(
        query=state["query"],
        code=state["code"],
        result=state.get("execution_result"),
        error=state.get("stderr", ""),
        model=state.get("model", "z-ai/glm-4.6")
    )
    
    # Record attempt
    attempt = {
        "attempt": state["attempts"] + 1,
        "code": state["code"],
        "success": not bool(state.get("stderr")),
        "error": state.get("stderr"),
        "verification": verification
    }
    
    # Track best result
    confidence = verification.get("confidence", 0.0)
    is_valid = verification.get("is_valid", False)
    best_confidence = state.get("best_verification", {}).get("confidence", 0.0)
    
    updates = {
        "verification": verification,
        "past_attempts": [attempt],
        "attempts": state["attempts"] + 1,
        "current_stage": "result_verified"
    }
    
    # Update best result if this is better
    if confidence > best_confidence or (is_valid and not state.get("best_verification", {}).get("is_valid")):
        updates.update({
            "best_result": state.get("execution_result"),
            "best_verification": verification,
            "best_code": state["code"]
        })
    
    return updates


def result_finalizer_node(state: SafetyCopilotState) -> Dict:
    """Finalize results and generate analysis"""
    # Use best result
    result = state.get("best_result") or state.get("execution_result")
    env = state.get("execution_env", {})
    
    # Normalize results
    normalized_result, normalized_fig = _normalize_or_recompute(
        state["query"],
        result,
        env.get("fig"),
        state.get("primary_df"),
        state.get("dfs")
    )
    
    result_preview = _result_preview(normalized_result)
    result_preview = _to_native_jsonable(result_preview)
    
    fig_dict = _fig_to_dict(normalized_fig)
    fig_dict = _to_native_jsonable(fig_dict) if fig_dict is not None else None
    
    mpl_png = _mpl_to_png_b64(env.get("mpl_fig")) if env.get("mpl_fig") is not None else None
    
    # Generate analysis
    summary_ctx_parts = []
    if result_preview:
        summary_ctx_parts.append(f"Result preview:\n{pd.DataFrame(result_preview).to_string(index=False)}")
    summary_ctx = "\n\n".join(summary_ctx_parts) or "No outputs."
    
    analysis = ask_openai(
        question="Provide a concise analysis based on the outputs.",
        context=summary_ctx,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=False
    )
    
    # Add verification details to analysis
    if state.get("best_verification", {}).get("explanation"):
        analysis += f"\n\n**Verification:** {state['best_verification']['explanation']}"
    
    if state["attempts"] > 1:
        analysis += f"\n\n**Self-Correction:** Completed in {state['attempts']} attempt(s). Confidence: {state.get('best_verification', {}).get('confidence', 0.0):.2f}"
    
    return {
        "result_preview": result_preview,
        "figure": fig_dict,
        "mpl_png_base64": mpl_png,
        "analysis": analysis,
        "current_stage": "finalized"
    }


# ==================== Conditional Logic ====================

def should_retry(state: SafetyCopilotState) -> Literal["retry", "finalize"]:
    """Determine if we should retry code generation"""
    is_valid = state.get("verification", {}).get("is_valid", False)
    confidence = state.get("verification", {}).get("confidence", 0.0)
    attempts = state.get("attempts", 0)
    max_retries = state.get("max_retries", 3)
    
    # Stop if good enough
    if is_valid and confidence >= 0.8:
        return "finalize"
    
    # Retry if attempts left and not valid
    if attempts < max_retries and not is_valid:
        return "retry"
    
    # Otherwise finalize with best attempt
    return "finalize"


# ==================== Graph Builder ====================

def create_safety_copilot_graph():
    """Create the complete Safety Copilot workflow graph with LangGraph"""
    
    # Initialize state graph
    builder = StateGraph(SafetyCopilotState)
    
    # Add all nodes
    builder.add_node("router", query_router_node)
    builder.add_node("conversational_handler", conversational_handler_node)
    builder.add_node("data_loader", data_loader_node)
    builder.add_node("code_generator", code_generator_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("result_verifier", result_verifier_node)
    builder.add_node("result_finalizer", result_finalizer_node)
    
    # Define graph flow
    builder.add_edge(START, "router")
    
    # Conversational path (Command handles routing from router)
    builder.add_edge("conversational_handler", END)
    
    # Analytical path (Command handles routing from router)
    builder.add_edge("data_loader", "code_generator")
    builder.add_edge("code_generator", "code_executor")
    builder.add_edge("code_executor", "result_verifier")
    
    # Conditional retry or finalize
    builder.add_conditional_edges(
        "result_verifier",
        should_retry,
        {
            "retry": "code_generator",  # Loop back for retry
            "finalize": "result_finalizer"
        }
    )
    
    builder.add_edge("result_finalizer", END)
    
    # Add memory saver for conversation history
    memory = MemorySaver()
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    return graph


# ==================== Helper Function for Streaming ====================

async def run_safety_copilot_graph_stream(
    query: str,
    dataset: str = "all",
    model: str = "z-ai/glm-4.6",
    max_retries: int = 3
):
    """Run Safety Copilot graph with streaming updates
    
    Args:
        query: User question to analyze
        dataset: Which dataset to use - "all" loads ALL sheets (default), or specify: incident|hazard|audit|inspection
        model: LLM model to use (default: z-ai/glm-4.6)
        max_retries: Maximum retry attempts for self-correction
    """
    
    graph = create_safety_copilot_graph()
    
    # Initial state - dataset defaults to "all" for full access
    initial_state = {
        "query": query,
        "dataset": dataset or "all",  # Use "all" to access all sheets
        "model": model,
        "max_retries": max_retries,
        "attempts": 0,
        "past_attempts": [],
        "verification": {},
        "best_verification": {},
        "context": "",
        "code": "",
        "stdout": "",
        "stderr": "",
        "analysis": "",
        "current_stage": "starting"
    }
    
    # Stream events
    config = {"configurable": {"thread_id": "safety-copilot-session"}}
    
    async for event in graph.astream(initial_state, config=config):
        # Extract node name and state
        node_name = list(event.keys())[0]
        node_state = event[node_name]
        
        # Clean state for JSON serialization (remove non-serializable objects)
        clean_state = {}
        for key, value in node_state.items():
            if key in ["workbook", "dfs", "primary_df", "execution_result", "execution_env", "best_result"]:
                # Skip DataFrames and complex objects
                continue
            clean_state[key] = value
        
        # Yield progress update
        yield {
            "type": "progress",
            "node": node_name,
            "stage": node_state.get("current_stage", node_name),
            "message": f"📍 {node_name}: {node_state.get('current_stage', 'processing')}"
        }
        
        # Yield specific event types
        if node_name == "code_generator":
            yield {
                "type": "code_generated",
                "code": node_state.get("code", "")
            }
        
        elif node_name == "result_verifier":
            yield {
                "type": "verification",
                "verification": node_state.get("verification", {}),
                "attempts": node_state.get("attempts", 0)
            }
        
        elif node_name == "result_finalizer":
            yield {
                "type": "complete",
                "data": {
                    "code": node_state.get("best_code", node_state.get("code", "")),
                    "result_preview": node_state.get("result_preview", []),
                    "figure": node_state.get("figure"),
                    "mpl_png_base64": node_state.get("mpl_png_base64"),
                    "analysis": node_state.get("analysis", ""),
                    "attempts": node_state.get("attempts", 0),
                    "verification_score": node_state.get("best_verification", {}).get("confidence", 0.0)
                }
            }
