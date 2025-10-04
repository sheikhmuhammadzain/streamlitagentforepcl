"""
Intelligent Data Analyst Agent using LangGraph Best Practices
Built by Qbit - Powered by Grok's 7-step methodology

Based on latest LangGraph patterns:
- Reflection and self-correction
- Code verification with execution checks
- Multi-sheet intelligent analysis
- Structured reasoning with state management
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
import operator
import pandas as pd
import json

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .agent import (
    _is_conversational_query,
    _generate_conversational_response,
    load_default_sheets,
    build_workbook_context,
    ask_openai,
    ask_openai_stream,
    extract_python_code,
    run_user_code,
    verify_result_quality,
    _result_preview,
    _fig_to_dict,
    _mpl_to_png_b64,
    _to_native_jsonable,
    _normalize_or_recompute,
)


# ==================== Enhanced State Schema ====================

class IntelligentAnalystState(TypedDict):
    """Enhanced state schema with reflection and verification"""
    
    # User interaction
    query: str
    query_type: Literal["conversational", "analytical"]
    dataset: str
    model: str
    
    # Data context (ALL sheets)
    workbook: Optional[Dict[str, pd.DataFrame]]
    dfs: Optional[Dict[str, pd.DataFrame]]
    context: str
    
    # Code generation with reflection
    code: str
    code_prefix: str  # Explanation of approach
    execution_result: Optional[Any]
    execution_env: Dict
    stdout: str
    stderr: str
    
    # Reflection & self-correction (LangGraph best practice)
    reflection: str  # LLM reflection on errors
    past_attempts: Annotated[List[Dict], operator.add]
    iterations: int
    max_iterations: int
    error_flag: Literal["yes", "no"]
    
    # Verification with scoring
    verification: Dict
    confidence_score: float
    best_result: Optional[Any]
    best_code: str
    best_confidence: float
    
    # Final output
    result_preview: List[Dict]
    figure: Optional[Dict]
    mpl_png_base64: Optional[str]
    analysis: str
    insights: List[str]  # Key insights extracted
    
    # Progress tracking
    current_stage: str


# ==================== Node Functions with Best Practices ====================

def query_router_node(state: IntelligentAnalystState) -> Command:
    """Smart query routing - instant conversational vs deep analytical"""
    if _is_conversational_query(state["query"]):
        return Command(
            goto="conversational_handler",
            update={"query_type": "conversational", "current_stage": "conversational_routing"}
        )
    
    return Command(
        goto="data_loader",
        update={"query_type": "analytical", "current_stage": "analytical_routing"}
    )


def conversational_handler_node(state: IntelligentAnalystState) -> Dict:
    """Handle conversational queries with context"""
    response = _generate_conversational_response(
        state["query"],
        model=state.get("model", "z-ai/glm-4.6")
    )
    
    return {
        "analysis": response,
        "current_stage": "conversational_complete",
        "code": "# Conversational query",
        "iterations": 1,
        "confidence_score": 1.0
    }


def data_loader_node(state: IntelligentAnalystState) -> Dict:
    """Load data intelligently - fast for simple queries, comprehensive for complex"""
    query_lower = state["query"].lower()
    
    # Detect if query needs multiple sheets
    needs_multi_sheet = any(keyword in query_lower for keyword in [
        'compare', 'correlation', 'relationship', 'across', 'between',
        'all datasets', 'all sheets', 'combined', 'merge', 'join'
    ])
    
    # Fast path: Load only requested dataset for simple queries
    if not needs_multi_sheet:
        dataset = state.get("dataset", "incident").lower()
        workbook = load_default_sheets()  # Still loads all (cached)
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        # Build lightweight context (faster)
        primary_sheet = dataset if dataset in dfs else "incident"
        context = f"# PRIMARY DATASET: {primary_sheet}\n"
        context += f"Shape: {dfs[primary_sheet].shape}\n"
        context += f"Columns: {', '.join(dfs[primary_sheet].columns[:10])}\n"
        context += f"\nOther sheets available in dfs: {', '.join(dfs.keys())}\n"
    else:
        # Comprehensive path: Full context for complex queries
        workbook = load_default_sheets()
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        context = build_workbook_context(workbook, query=state["query"])
        
        # Add cross-sheet hints
        context += "\n\n# CROSS-SHEET ANALYSIS:\n"
        for sheet_name in dfs.keys():
            context += f"- dfs['{sheet_name}']: {len(dfs[sheet_name])} rows\n"
    
    return {
        "workbook": workbook,
        "dfs": dfs,
        "context": context,
        "current_stage": "data_loaded"
    }


def intelligent_code_generator_node(state: IntelligentAnalystState) -> Dict:
    """Generate code with reflection on past failures (LangGraph best practice)"""
    
    # Build enhanced context with reflection
    enhanced_context = state["context"]
    
    # Add reflection from past attempts (LangGraph pattern)
    if state.get("past_attempts"):
        reflection_context = "\n\n# REFLECTION ON PREVIOUS ATTEMPTS:\n"
        
        for i, attempt in enumerate(state["past_attempts"], 1):
            reflection_context += f"\n## Attempt {i} Analysis:\n"
            reflection_context += f"Code tried:\n```python\n{attempt.get('code', 'N/A')}\n```\n"
            
            if attempt.get('error'):
                reflection_context += f"Error encountered: {attempt['error']}\n"
            
            if attempt.get('reflection'):
                reflection_context += f"Reflection: {attempt['reflection']}\n"
            
            if attempt.get('verification', {}).get('issues'):
                reflection_context += f"Issues identified: {', '.join(attempt['verification']['issues'])}\n"
            
            if attempt.get('verification', {}).get('suggestions'):
                reflection_context += f"Suggestions: {attempt['verification']['suggestions']}\n"
        
        reflection_context += "\n# YOUR TASK:\n"
        reflection_context += "Learn from the above reflections and generate CORRECTED code.\n"
        reflection_context += "Apply the suggestions and avoid the previous errors.\n"
        
        enhanced_context += reflection_context
    
    # Generate code with Grok's 7-step approach
    code_response = ask_openai(
        question=state["query"],
        context=enhanced_context,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=True,
        multi_df=True
    )
    
    code_block = extract_python_code(code_response)
    
    # Validate code completeness (check for truncation)
    if code_block:
        # Check for incomplete syntax
        open_parens = code_block.count('(') - code_block.count(')')
        open_brackets = code_block.count('[') - code_block.count(']')
        open_braces = code_block.count('{') - code_block.count('}')
        
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            # Code is truncated - add closing syntax
            code_block += ')' * open_parens
            code_block += ']' * open_brackets
            code_block += '}' * open_braces
            code_block += "\n# Note: Code was auto-completed due to truncation"
        
        # Ensure result variable is assigned
        if 'result =' not in code_block and 'result=' not in code_block:
            # Try to find the last meaningful variable assignment
            lines = code_block.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    if var_name and var_name.isidentifier():
                        code_block += f"\nresult = {var_name}  # Auto-assigned for output"
                        break
    
    # Extract code prefix (explanation) if present
    code_prefix = ""
    if "```" in code_response:
        code_prefix = code_response.split("```")[0].strip()
    
    return {
        "code": code_block or "# No code generated",
        "code_prefix": code_prefix,
        "current_stage": "code_generated_with_reasoning"
    }


def code_executor_node(state: IntelligentAnalystState) -> Dict:
    """Execute code with comprehensive error capture"""
    env, stdout, stderr = run_user_code(
        state["code"],
        df=state.get("dfs", {}).get(state.get("dataset", "incident")),  # Smart df selection
        dfs=state.get("dfs")
    )
    
    has_error = bool(stderr and stderr.strip())
    
    return {
        "execution_result": env.get("result"),
        "execution_env": env,
        "stdout": stdout,
        "stderr": stderr,
        "error_flag": "yes" if has_error else "no",
        "current_stage": "executed_with_error" if has_error else "executed_successfully"
    }


def reflection_node(state: IntelligentAnalystState) -> Dict:
    """Generate reflection on errors (LangGraph best practice for self-correction)"""
    
    if state.get("error_flag") == "no":
        # No error, no reflection needed
        return {
            "reflection": "Code executed successfully without errors.",
            "current_stage": "reflection_positive"
        }
    
    # Generate reflection on the error
    reflection_prompt = f"""
You are a code review expert. Analyze this error and provide reflection.

Original Query: {state['query']}

Code Attempted:
```python
{state['code']}
```

Error Encountered:
{state['stderr']}

Provide a structured reflection:
1. What went wrong (root cause)
2. Why it happened (technical explanation)
3. How to fix it (specific solution)
4. What to avoid next time (learning)

Be specific and actionable.
"""
    
    reflection = ask_openai(
        question="Provide reflection on this error",
        context=reflection_prompt,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=False
    )
    
    return {
        "reflection": reflection,
        "current_stage": "reflection_generated"
    }


def verification_node(state: IntelligentAnalystState) -> Dict:
    """Verify result quality with confidence scoring"""
    
    # Build verification prompt
    result_str = ""
    if isinstance(state.get("execution_result"), pd.DataFrame):
        df_result = state["execution_result"]
        result_str = df_result.head(20).to_string() if not df_result.empty else "Empty DataFrame"
    else:
        result_str = str(state.get("execution_result", "No result"))[:1000]
    
    verification_prompt = f"""
Verify if this analysis correctly answers the user's query.

Query: {state['query']}

Code:
```python
{state['code']}
```

Result:
{result_str}

Error: {state.get('stderr', 'None')}

Provide verification in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues if any"],
    "suggestions": "specific improvements if needed",
    "insights": ["key insights from the data"],
    "explanation": "brief verification summary"
}}
"""
    
    verification_response = ask_openai(
        question="Verify this analysis",
        context=verification_prompt,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=False
    )
    
    # Parse verification JSON
    try:
        verification = json.loads(verification_response)
    except:
        # Fallback parsing
        verification = {
            "is_valid": state.get("error_flag") == "no",
            "confidence": 0.5 if state.get("error_flag") == "no" else 0.0,
            "issues": [state.get("stderr", "")] if state.get("stderr") else [],
            "suggestions": "Review the code and error message",
            "insights": [],
            "explanation": "Verification parsing failed"
        }
    
    # Record attempt with reflection
    attempt = {
        "iteration": state["iterations"] + 1,
        "code": state["code"],
        "error": state.get("stderr"),
        "reflection": state.get("reflection", ""),
        "verification": verification
    }
    
    # Track best result
    confidence = verification.get("confidence", 0.0)
    best_confidence = state.get("best_confidence", 0.0)
    
    updates = {
        "verification": verification,
        "confidence_score": confidence,
        "past_attempts": [attempt],
        "iterations": state["iterations"] + 1,
        "current_stage": "verified"
    }
    
    # Update best if this is better
    if confidence > best_confidence:
        updates.update({
            "best_result": state.get("execution_result"),
            "best_code": state["code"],
            "best_confidence": confidence
        })
    
    return updates


def intelligent_finalizer_node(state: IntelligentAnalystState) -> Dict:
    """Generate comprehensive analysis with insights"""
    
    # Use best result (handle DataFrame ambiguity)
    result = state.get("best_result")
    if result is None:
        result = state.get("execution_result")
    env = state.get("execution_env", {})
    
    # Normalize results with better fallback
    result_preview = _result_preview(result)
    result_preview = _to_native_jsonable(result_preview)
    
    # If no result preview, try to extract from environment variables
    if not result_preview or len(result_preview) == 0:
        # Check if there's a DataFrame in env
        for key, value in env.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                result_preview = _result_preview(value)
                result_preview = _to_native_jsonable(result_preview)
                break
        
        # If still empty, provide a helpful message
        if not result_preview or len(result_preview) == 0:
            result_preview = [{
                "info": "Analysis completed successfully",
                "note": "Results are visualized in the chart above" if env.get("fig") or env.get("mpl_fig") else "Check the analysis below for insights"
            }]
    
    fig_dict = _fig_to_dict(env.get("fig"))
    fig_dict = _to_native_jsonable(fig_dict) if fig_dict is not None else None
    
    mpl_png = _mpl_to_png_b64(env.get("mpl_fig")) if env.get("mpl_fig") is not None else None
    
    # Generate comprehensive analysis
    analysis_context = f"""
# ANALYSIS TASK

Query: {state['query']}

Code Solution:
```python
{state.get('best_code', state['code'])}
```

Results:
{pd.DataFrame(result_preview).to_string(index=False) if result_preview else 'No results'}

Verification:
- Confidence: {state.get('best_confidence', 0.0):.2f}
- Attempts: {state['iterations']}
- Issues: {state.get('verification', {}).get('issues', [])}

# YOUR TASK:
Provide a comprehensive analysis following this structure:

## 📊 KEY FINDINGS
- What the data shows (specific metrics, numbers, trends)
- Most important observations

## 💡 INSIGHTS
- Why these findings matter
- Patterns and correlations discovered
- Root causes or contributing factors

## 🎯 RECOMMENDATIONS
- Actionable next steps (prioritized)
- Strategic suggestions based on data
- Areas requiring attention

## ⚠️ LIMITATIONS
- Data gaps or assumptions
- Caveats to consider
- Confidence level notes

Be specific, data-driven, and actionable. Use bullet points.
"""
    
    analysis = ask_openai(
        question="Generate comprehensive analysis",
        context=analysis_context,
        model=state.get("model", "z-ai/glm-4.6"),
        code_mode=False
    )
    
    # Extract insights from verification
    insights = state.get("verification", {}).get("insights", [])
    
    # Add self-correction summary
    if state["iterations"] > 1:
        analysis += f"\n\n---\n**Self-Correction Summary:** Completed in {state['iterations']} iteration(s). "
        analysis += f"Final confidence: {state.get('best_confidence', 0.0):.2f}"
    
    return {
        "result_preview": result_preview,
        "figure": fig_dict,
        "mpl_png_base64": mpl_png,
        "analysis": analysis,
        "insights": insights,
        "current_stage": "analysis_complete"
    }


# ==================== Conditional Logic (LangGraph Pattern) ====================

def should_continue_or_reflect(state: IntelligentAnalystState) -> Literal["reflect", "verify"]:
    """Decide if we need reflection on errors or can proceed to verification"""
    if state.get("error_flag") == "yes":
        return "reflect"  # Error occurred, need reflection
    return "verify"  # No error, proceed to verification


def should_retry_or_finalize(state: IntelligentAnalystState) -> Literal["retry", "finalize"]:
    """Determine if we should retry with reflection or finalize (LangGraph pattern)"""
    
    error_flag = state.get("error_flag", "no")
    is_valid = state.get("verification", {}).get("is_valid", False)
    confidence = state.get("confidence_score", 0.0)
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # Finalize if good enough
    if is_valid and confidence >= 0.8:
        return "finalize"
    
    # Finalize if max iterations reached
    if iterations >= max_iterations:
        return "finalize"
    
    # Retry if error or low confidence
    if error_flag == "yes" or (not is_valid and confidence < 0.8):
        return "retry"
    
    # Default: finalize
    return "finalize"


# ==================== Graph Builder ====================

def create_intelligent_analyst_graph():
    """
    Create intelligent data analyst graph with LangGraph best practices:
    - Reflection on errors
    - Self-correction loops
    - Verification with confidence scoring
    - Multi-sheet intelligent analysis
    """
    
    builder = StateGraph(IntelligentAnalystState)
    
    # Add nodes
    builder.add_node("router", query_router_node)
    builder.add_node("conversational_handler", conversational_handler_node)
    builder.add_node("data_loader", data_loader_node)
    builder.add_node("code_generator", intelligent_code_generator_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("reflect", reflection_node)  # Reflection node (LangGraph best practice)
    builder.add_node("verify", verification_node)
    builder.add_node("finalize", intelligent_finalizer_node)
    
    # Graph flow
    builder.add_edge(START, "router")
    
    # Conversational path (handled by Command in router)
    builder.add_edge("conversational_handler", END)
    
    # Analytical path (handled by Command in router)
    builder.add_edge("data_loader", "code_generator")
    builder.add_edge("code_generator", "code_executor")
    
    # Conditional: error → reflect, no error → verify
    builder.add_conditional_edges(
        "code_executor",
        should_continue_or_reflect,
        {
            "reflect": "reflect",
            "verify": "verify"
        }
    )
    
    # After reflection, go to verify
    builder.add_edge("reflect", "verify")
    
    # Conditional: retry with reflection or finalize
    builder.add_conditional_edges(
        "verify",
        should_retry_or_finalize,
        {
            "retry": "code_generator",  # Loop back with reflection
            "finalize": "finalize"
        }
    )
    
    builder.add_edge("finalize", END)
    
    # Compile WITHOUT checkpointer to avoid DataFrame serialization issues
    # Note: DataFrames are not msgpack serializable
    # For conversation history, we'd need custom serializer or store only metadata
    graph = builder.compile()
    
    return graph


# ==================== Streaming Interface ====================

async def run_intelligent_analyst_stream(
    query: str,
    dataset: str = "all",
    model: str = "z-ai/glm-4.6",
    max_iterations: int = 3
):
    """
    Run intelligent data analyst with streaming updates
    
    Features:
    - ALL sheets loaded by default
    - Reflection on errors (LangGraph best practice)
    - Self-correction with learning
    - Confidence scoring
    - Comprehensive insights
    """
    
    graph = create_intelligent_analyst_graph()
    
    # Initial state
    initial_state = {
        "query": query,
        "dataset": dataset or "all",
        "model": model,
        "max_iterations": max_iterations,
        "iterations": 0,
        "past_attempts": [],
        "verification": {},
        "best_confidence": 0.0,
        "error_flag": "no",
        "context": "",
        "code": "",
        "stdout": "",
        "stderr": "",
        "reflection": "",
        "analysis": "",
        "insights": [],
        "current_stage": "initializing"
    }
    
    # Send instant start signal
    yield {
        "type": "start",
        "message": "🚀 Starting analysis..."
    }
    
    # Stream events (no config needed since we're not using checkpointer)
    try:
        async for event in graph.astream(initial_state):
            # Extract node name and state
            node_name = list(event.keys())[0]
            node_state = event[node_name]
            
            # Map stages to user-friendly messages (shorter for speed)
            stage_messages = {
                "router": "🔀 Routing...",
                "conversational_handler": "💬 Responding...",
                "data_loader": "📊 Loading data...",
                "code_generator": "🧠 Generating code...",
                "code_executor": "⚙️ Executing...",
                "reflect": "🤔 Analyzing error...",
                "verify": "✅ Verifying...",
                "finalize": "📝 Finalizing..."
            }
            
            # Yield progress immediately
            yield {
                "type": "progress",
                "stage": node_name,
                "message": stage_messages.get(node_name, f"Processing...")
            }
            
            # Yield code when generated with chain-of-thought streaming
            if node_name == "code_generator" and node_state.get("code"):
                code = node_state["code"]
                
                # Extract and stream chain of thought (7-step comments)
                if "# GROK'S 7-STEP" in code or "# 1. UNDERSTAND" in code:
                    lines = code.split('\n')
                    thought_lines = []
                    code_lines = []
                    
                    # Separate thoughts from code
                    in_thoughts = False
                    for line in lines:
                        if line.strip().startswith('#') and any(step in line for step in ['STEP', 'UNDERSTAND', 'EXPLORE', 'CLEAN', 'ANALYZE', 'VISUALIZE', 'VALIDATE', 'COMMUNICATION']):
                            in_thoughts = True
                            thought_lines.append(line)
                        elif line.strip().startswith('#') and in_thoughts:
                            thought_lines.append(line)
                        elif line.strip() and not line.strip().startswith('#'):
                            in_thoughts = False
                            code_lines.append(line)
                        else:
                            if in_thoughts:
                                thought_lines.append(line)
                            else:
                                code_lines.append(line)
                    
                    # Stream chain of thought
                    if thought_lines:
                        thought_text = '\n'.join(thought_lines)
                        # Stream in chunks for readability
                        thought_chunks = thought_text.split('\n\n')  # Split by double newline (sections)
                        for chunk in thought_chunks:
                            if chunk.strip():
                                yield {
                                    "type": "chain_of_thought",
                                    "content": chunk.strip()
                                }
                
                # Yield full code
                yield {
                    "type": "code_chunk",
                    "chunk": code
                }
                
                # Yield reasoning prefix if available
                if node_state.get("code_prefix"):
                    yield {
                        "type": "reasoning",
                        "content": node_state["code_prefix"]
                    }
            
            # Yield reflection in chunks (chain of thought for error analysis)
            if node_name == "reflect" and node_state.get("reflection"):
                reflection_text = node_state["reflection"]
                
                # Split reflection into sections for streaming
                sections = reflection_text.split('\n\n')
                for section in sections:
                    if section.strip():
                        yield {
                            "type": "reflection_chunk",
                            "content": section.strip()
                        }
                
                # Also yield complete reflection
                yield {
                    "type": "reflection",
                    "content": reflection_text
                }
            
            # Yield verification
            if node_name == "verify":
                yield {
                    "type": "verification",
                    "is_valid": node_state.get("verification", {}).get("is_valid", False),
                    "confidence": node_state.get("confidence_score", 0.0),
                    "attempts": node_state.get("iterations", 0)
                }
            
            # Yield final result with streaming analysis
            if node_name == "finalize":
                # First yield the data without analysis
                yield {
                    "type": "data_ready",
                    "data": {
                        "code": node_state.get("best_code", node_state.get("code", "")),
                        "stdout": node_state.get("stdout", ""),
                        "error": node_state.get("stderr", ""),
                        "result_preview": node_state.get("result_preview", []),
                        "figure": node_state.get("figure"),
                        "mpl_png_base64": node_state.get("mpl_png_base64"),
                        "attempts": node_state.get("iterations", 0),
                        "verification_score": node_state.get("best_confidence", 0.0),
                        "insights": node_state.get("insights", [])
                    }
                }
                
                # Stream the analysis preserving markdown formatting
                analysis_text = node_state.get("analysis", "")
                if analysis_text:
                    # Split by lines to preserve markdown structure
                    lines = analysis_text.split('\n')
                    current_chunk = ""
                    
                    for line in lines:
                        # Add line to current chunk
                        current_chunk += line + '\n'
                        
                        # Yield chunk when we hit a section break or accumulate enough
                        if line.strip() == "" or line.startswith('#') or len(current_chunk) > 200:
                            if current_chunk.strip():
                                yield {
                                    "type": "analysis_chunk",
                                    "chunk": current_chunk
                                }
                            current_chunk = ""
                    
                    # Yield any remaining content
                    if current_chunk.strip():
                        yield {
                            "type": "analysis_chunk",
                            "chunk": current_chunk
                        }
                
                # Final complete signal
                yield {
                    "type": "complete",
                    "message": "Analysis complete"
                }
    
    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "message": f"Graph execution error: {str(e)}",
            "details": traceback.format_exc()
        }
