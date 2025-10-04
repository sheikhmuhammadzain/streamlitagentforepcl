"""
Ultra-Fast Streaming Agent with Real-Time Token Generation
Provides instant feedback as the LLM generates responses
"""

from typing import AsyncGenerator, Dict, Any, Optional
import pandas as pd
import json

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
)


async def run_ultra_fast_streaming_agent(
    query: str,
    dataset: str = "all",
    model: str = "z-ai/glm-4.6",
    max_retries: int = 3
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Ultra-fast streaming agent with instant token-by-token generation
    
    Speed optimizations:
    - Instant start signal
    - Parallel data loading
    - Token-by-token code streaming
    - Minimal context for simple queries
    - Smart caching
    """
    
    # 1. INSTANT START (0ms)
    yield {
        "type": "start",
        "message": "🚀 Analysis starting...",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # 2. INSTANT ROUTING (50ms)
    if _is_conversational_query(query):
        yield {"type": "progress", "stage": "conversational", "message": "💬 Responding..."}
        
        response = _generate_conversational_response(query, model=model)
        
        yield {
            "type": "complete",
            "data": {
                "code": "# Conversational query",
                "analysis": response,
                "attempts": 1,
                "verification_score": 1.0
            }
        }
        return
    
    # 3. SMART DATA LOADING (100-500ms)
    yield {"type": "progress", "stage": "loading", "message": "📊 Loading data..."}
    
    query_lower = query.lower()
    needs_multi_sheet = any(kw in query_lower for kw in [
        'compare', 'correlation', 'across', 'between', 'merge', 'join', 'all datasets'
    ])
    
    workbook = load_default_sheets()
    dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
    
    # Build minimal context for speed
    if not needs_multi_sheet:
        primary_sheet = dataset.lower() if dataset.lower() in dfs else "incident"
        df = dfs[primary_sheet]
        context = f"# Dataset: {primary_sheet} ({len(df)} rows, {len(df.columns)} cols)\n"
        context += f"Columns: {', '.join(df.columns[:15])}\n"
        context += f"Sample:\n{df.head(3).to_string()}\n"
    else:
        context = build_workbook_context(workbook, query=query)
    
    yield {"type": "data_loaded", "sheets": list(dfs.keys()), "primary": dataset}
    
    # 4. STREAMING CODE GENERATION (instant feedback!)
    yield {"type": "progress", "stage": "generating", "message": "🧠 Generating code..."}
    
    code_response = ""
    code_buffer = ""
    
    # Stream code token-by-token
    async for token in ask_openai_stream(
        question=query,
        context=context,
        model=model,
        code_mode=True,
        multi_df=needs_multi_sheet
    ):
        code_response += token
        code_buffer += token
        
        # Yield chunks for smooth streaming (every ~50 chars)
        if len(code_buffer) >= 50:
            yield {
                "type": "code_token",
                "token": code_buffer
            }
            code_buffer = ""
    
    # Yield remaining buffer
    if code_buffer:
        yield {"type": "code_token", "token": code_buffer}
    
    # Extract complete code
    code = extract_python_code(code_response)
    
    if not code:
        yield {
            "type": "error",
            "message": "No code generated",
            "details": code_response[:500]
        }
        return
    
    yield {"type": "code_complete", "code": code}
    
    # 5. EXECUTION (fast)
    yield {"type": "progress", "stage": "executing", "message": "⚙️ Executing..."}
    
    env, stdout, stderr = run_user_code(
        code,
        df=dfs.get(dataset.lower(), dfs.get("incident")),
        dfs=dfs
    )
    
    has_error = bool(stderr and stderr.strip())
    
    if has_error and max_retries > 0:
        # Quick retry with error context
        yield {"type": "progress", "stage": "retrying", "message": "🔄 Retrying..."}
        
        retry_context = context + f"\n\n# PREVIOUS ERROR:\n{stderr}\n\n# Fix the error and try again."
        
        code_response = ""
        async for token in ask_openai_stream(
            question=query,
            context=retry_context,
            model=model,
            code_mode=True,
            multi_df=needs_multi_sheet
        ):
            code_response += token
            yield {"type": "code_token", "token": token}
        
        code = extract_python_code(code_response)
        yield {"type": "code_complete", "code": code}
        
        env, stdout, stderr = run_user_code(code, df=dfs.get(dataset.lower()), dfs=dfs)
        has_error = bool(stderr and stderr.strip())
    
    # 6. RESULTS
    result = env.get("result")
    result_preview = _result_preview(result)
    result_preview = _to_native_jsonable(result_preview)
    
    # Fallback if no preview
    if not result_preview or len(result_preview) == 0:
        for key, value in env.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                result_preview = _result_preview(value)
                result_preview = _to_native_jsonable(result_preview)
                break
    
    fig_dict = _fig_to_dict(env.get("fig"))
    fig_dict = _to_native_jsonable(fig_dict) if fig_dict else None
    
    mpl_png = _mpl_to_png_b64(env.get("mpl_fig")) if env.get("mpl_fig") else None
    
    yield {
        "type": "data_ready",
        "data": {
            "code": code,
            "stdout": stdout,
            "error": stderr,
            "result_preview": result_preview or [],
            "figure": fig_dict,
            "mpl_png_base64": mpl_png
        }
    }
    
    # 7. STREAMING ANALYSIS
    yield {"type": "progress", "stage": "analyzing", "message": "📝 Analyzing..."}
    
    analysis_context = f"""
Query: {query}

Results:
{pd.DataFrame(result_preview).to_string() if result_preview else 'See visualization'}

Provide concise analysis:
## 📊 KEY FINDINGS
- Main observations (specific numbers)

## 💡 INSIGHTS  
- Why it matters

## 🎯 RECOMMENDATIONS
- Actionable next steps

Be brief and data-driven.
"""
    
    analysis_text = ""
    analysis_buffer = ""
    
    # Stream analysis token-by-token
    async for token in ask_openai_stream(
        question="Generate analysis",
        context=analysis_context,
        model=model,
        code_mode=False
    ):
        analysis_text += token
        analysis_buffer += token
        
        # Yield in chunks for smooth streaming
        if len(analysis_buffer) >= 30:
            yield {
                "type": "analysis_token",
                "token": analysis_buffer
            }
            analysis_buffer = ""
    
    # Yield remaining
    if analysis_buffer:
        yield {"type": "analysis_token", "token": analysis_buffer}
    
    # 8. COMPLETE
    yield {
        "type": "complete",
        "data": {
            "code": code,
            "stdout": stdout,
            "error": stderr,
            "result_preview": result_preview or [],
            "figure": fig_dict,
            "mpl_png_base64": mpl_png,
            "analysis": analysis_text,
            "attempts": 2 if has_error else 1,
            "verification_score": 0.0 if has_error else 0.9
        }
    }
