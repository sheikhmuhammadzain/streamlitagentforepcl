from __future__ import annotations

import base64
import io
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
import re
import json
import asyncio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .excel import (
    get_incident_df,
    get_hazard_df,
    get_audit_df,
    get_inspection_df,
    load_default_sheets,
)

try:
    from openai import OpenAI, AsyncOpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ------------------ OpenRouter model normalization helpers ------------------
def _normalize_openrouter_model(model: str) -> str:
    """Normalize user-provided model aliases to known OpenRouter model IDs.
    Returns a safe default if input is empty or deprecated.

    Known Grok models on OpenRouter include:
    - z-ai/glm-4.6 (FREE - recommended for development)
    - x-ai/grok-2-latest
    - x-ai/grok-2-1212
    - x-ai/grok-2-vision-1212

    Deprecated/invalid examples that we remap:
    - x-ai/grok-beta -> z-ai/glm-4.6
    """
    if not model:
        return "z-ai/glm-4.6"
    m = model.strip().lower()
    aliases = {
        "grok": "z-ai/glm-4.6",
        "x-ai/grok": "z-ai/glm-4.6",
        "x-ai/grok-beta": "z-ai/glm-4.6",
        "x-ai/grok-latest": "z-ai/glm-4.6",
    }
    return aliases.get(m, model)


def _openrouter_fallback_models(primary: str) -> list[str]:
    """Order of fallback models to try on OpenRouter if a model is invalid/unavailable.
    
    OPTIMIZED: If primary is z-ai/glm-4.6, return ONLY that model for speed.
    """
    normalized = _normalize_openrouter_model(primary)
    
    # Fast path: if requesting the working model, use ONLY it
    if normalized == "z-ai/glm-4.6":
        return ["z-ai/glm-4.6"]
    
    # Fallback path: try multiple models
    fallbacks = [
        normalized,
        "z-ai/glm-4.6",  # Primary free Grok
        "x-ai/grok-2-latest",
        "x-ai/grok-2-1212",
        # Other free fallbacks
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for f in fallbacks:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _head_records(df: Optional[pd.DataFrame], n: int = 20) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    safe = df.head(n).copy()
    # stringify datetimes to avoid serialization issues
    for c in safe.columns:
        if np.issubdtype(safe[c].dtype, np.datetime64):
            safe[c] = pd.to_datetime(safe[c], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return safe.to_dict(orient="records")


def build_df_context(df: Optional[pd.DataFrame], *, sample_rows: int = 5, max_numeric: int = 8, max_cat: int = 8, query: Optional[str] = None) -> str:
    if df is None or df.empty:
        return "DataFrame is empty or None."
    lines: List[str] = []
    # shape
    lines.append(f"Shape: rows={len(df):,}, cols={len(df.columns):,}")
    # dtypes and non-null counts
    nn = df.notna().sum()
    dtypes = [f"{c}:{str(df[c].dtype)} nn={int(nn.get(c, 0)):,}" for c in df.columns]
    lines.append("Columns: " + ", ".join(dtypes))
    # numeric summary (limit)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_numeric]
    if num_cols:
        try:
            desc = df[num_cols].describe().to_string()
            lines.append("Numeric summary (subset):\n" + desc)
        except Exception:
            pass
    # categorical sample (limit)
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])][:max_cat]
    for c in cat_cols:
        try:
            vc = df[c].astype(str).value_counts().head(10)
            if not vc.empty:
                lines.append(f"Top values for {c}:\n" + vc.to_string())
        except Exception:
            pass
    # query-aware hints
    if query:
        lines.append(_build_query_hints(query, df))
    # sample rows (first 5)
    try:
        lines.append("Sample rows (head):\n" + df.head(sample_rows).to_string())
    except Exception:
        pass
    return "\n\n".join([s for s in lines if s])


def _is_conversational_query(query: str) -> bool:
    """Detect if query is conversational/meta rather than data analytical."""
    query_lower = query.lower().strip()
    
    # Conversational patterns
    conversational_patterns = [
        "who are you", "what are you", "tell me about yourself",
        "what can you do", "what's your name", "your name",
        "how do you work", "what is your purpose", "introduce yourself",
        "hello", "hi ", "hey ", "greetings",
        "help", "what can i ask", "how to use",
        "are you ai", "are you a bot", "are you human",
        "who made you", "who created you", "who built you",
        "what is safety copilot", "what is qbit"
    ]
    
    # Check if query matches conversational patterns
    for pattern in conversational_patterns:
        if pattern in query_lower:
            return True
    
    # Very short queries without data keywords
    data_keywords = ["show", "find", "analyze", "calculate", "count", "sum", "average", 
                     "trend", "compare", "list", "top", "bottom", "filter", "incident", 
                     "hazard", "audit", "inspection", "department", "severity", "data"]
    
    if len(query_lower.split()) <= 5:
        has_data_keyword = any(keyword in query_lower for keyword in data_keywords)
        if not has_data_keyword:
            return True
    
    return False


def _generate_conversational_response(query: str, model: str = "z-ai/glm-4.6") -> str:
    """Generate a conversational response for meta/greeting queries."""
    conversational_context = """
You are Safety Copilot, an intelligent data analysis agent built by Qbit.

ABOUT YOU:
- Name: Safety Copilot
- Built by: Qbit
- Purpose: Analyze safety data (incidents, hazards, audits, inspections)
- Capabilities: Advanced data analysis, visualization, insights generation
- Methodology: Grok's 7-step analytical approach
- Model: Powered by x-ai/grok-4-fast (free)

YOUR CAPABILITIES:
- Analyze incidents, hazards, audits, and inspections
- Generate charts and visualizations
- Identify trends and patterns
- Provide actionable insights and recommendations
- Multi-sheet data analysis
- Self-correcting with verification

EXAMPLE QUERIES YOU CAN HANDLE:
- "Show top 10 incidents by severity"
- "Analyze incident trends over last 6 months"
- "Compare hazards vs incidents by department"
- "What are the main risk factors?"
- "Forecast incident rates for next quarter"

Be friendly, concise, and helpful. Encourage users to ask data analysis questions.
"""
    
    try:
        response = ask_openai(
            question=query,
            context=conversational_context,
            model=model,
            code_mode=False,
            multi_df=False
        )
        return response
    except Exception as e:
        # Fallback response
        return (
            "👋 Hi! I'm **Safety Copilot**, an intelligent data analysis agent built by Qbit.\n\n"
            "I can help you analyze safety data including:\n"
            "- 📊 Incidents\n"
            "- ⚠️ Hazards\n"
            "- ✅ Audits\n"
            "- 🔍 Inspections\n\n"
            "**Ask me questions like:**\n"
            "- Show top 10 incidents by severity\n"
            "- Analyze incident trends over last 6 months\n"
            "- Compare hazards by department\n"
            "- What are the main risk factors?\n\n"
            "How can I help you today?"
        )


def build_workbook_context(workbook: Dict[str, pd.DataFrame], *, sample_rows: int = 5, query: Optional[str] = None) -> str:
    if not workbook:
        return "Workbook is empty."
    chunks: List[str] = []
    for name, df in workbook.items():
        chunks.append(f"# Sheet: {name}\n" + build_df_context(df, sample_rows=sample_rows, query=query))
    return "\n\n".join(chunks)


def _build_query_hints(question: str, df: pd.DataFrame) -> str:
    """Extract tokens from the question and surface likely relevant columns and sample distributions."""
    try:
        q = (question or "").lower()
        tokens = set([t.strip(" ,.?;:!()[]{}\"'\n\r\t") for t in q.split() if len(t) >= 3])
        matched_cols = [c for c in df.columns if any(tok in str(c).lower() for tok in tokens)]
        if not matched_cols:
            return ""
        lines = ["Query hints:", f"Matched columns: {', '.join(map(str, matched_cols))}"]
        for c in matched_cols[:6]:
            try:
                vc = df[c].astype(str).value_counts().head(5)
                if not vc.empty:
                    lines.append(f"Top values for {c}:\n" + vc.to_string())
            except Exception:
                pass
        return "\n".join(lines)
    except Exception:
        return ""


def ask_openai(question: str, context: str, *, model: str = "gpt-4o", code_mode: bool = False, multi_df: bool = False) -> str:
    if not _OPENAI_AVAILABLE:
        return "OpenAI Python package is not installed. Please run: pip install openai"
    
    # Check if using OpenRouter
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    
    if use_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Missing OPENROUTER_API_KEY. Set it in your environment when USE_OPENROUTER=true."
        base_url = "https://openrouter.ai/api/v1"
        site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "Safety Copilot")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Missing OPENAI_API_KEY. Set it in your environment."
        base_url = None
        site_url = None
        site_name = None
    
    try:
        # Create client with appropriate base_url
        if use_openrouter:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        if code_mode:
            if multi_df:
                system_prompt = (
                    "You are Safety Copilot, an expert data scientist built by Qbit. Use your structured 7-step analytical approach.\n"
                    "You have access to multiple DataFrames in `dfs` dict (sheet names as keys) and a primary `df`.\n\n"
                    
                    "GROK'S 7-STEP DATA ANALYSIS APPROACH:\n\n"
                    
                    "1. UNDERSTAND THE PROBLEM & DATA CONTEXT\n"
                    "   - Clarify the question's intent and objectives\n"
                    "   - Review data structure: columns, types, relationships\n"
                    "   - Identify relevant metrics and KPIs for safety data\n\n"
                    
                    "2. DATA EXPLORATION (EDA)\n"
                    "   - Perform initial summaries: df.describe(), df.info()\n"
                    "   - Check distributions, correlations, patterns\n"
                    "   - Identify anomalies, outliers, missing values\n\n"
                    
                    "3. DATA CLEANING & PREPARATION\n"
                    "   - Handle missing values intelligently (fill/drop/flag)\n"
                    "   - Remove duplicates, fix data types\n"
                    "   - Transform as needed (normalize, encode, aggregate)\n\n"
                    
                    "4. CORE ANALYSIS (choose appropriate type)\n"
                    "   - Descriptive: Summarize what happened (aggregates, trends)\n"
                    "   - Diagnostic: Explain why (correlations, segmentation)\n"
                    "   - Predictive: Forecast trends (time series, regression)\n"
                    "   - Prescriptive: Recommend actions (optimization, insights)\n\n"
                    
                    "5. VISUALIZATION & INTERPRETATION\n"
                    "   - Create intuitive charts (bar, line, scatter, heatmap)\n"
                    "   - Highlight key insights visually\n"
                    "   - Use plotly for interactive or matplotlib for static\n\n"
                    
                    "6. VALIDATION & ITERATION\n"
                    "   - Cross-check results for accuracy\n"
                    "   - Handle edge cases and biases\n"
                    "   - Validate assumptions\n\n"
                    
                    "7. COMMUNICATION (in code comments)\n"
                    "   - Add clear comments explaining your reasoning\n"
                    "   - Note key findings and limitations\n\n"
                    
                    "TECHNICAL CAPABILITIES:\n"
                    "- Advanced pandas: merge, groupby, pivot, rolling, resample\n"
                    "- Statistical methods: correlations, distributions, tests\n"
                    "- Multi-sheet analysis: combine datasets intelligently\n"
                    "- Libraries available: pd, np, px, go, plt (no imports needed)\n\n"
                    
                    "OUTPUT FORMAT:\n"
                    "- Return ONE fenced Python code block\n"
                    "- Set `result` to your analytical findings (DataFrame/scalar)\n"
                    "- Set `fig` (plotly) or `mpl_fig` (matplotlib) for visualizations\n"
                    "- Add comments following your 7-step reasoning\n"
                    "- Do NOT read files, access network, or call .show()"
                )
            else:
                system_prompt = (
                    "You are an expert data scientist with analytical intelligence. "
                    "You have a DataFrame `df`. Use the provided context.\n\n"
                    
                    "APPROACH:\n"
                    "- Understand query intent and find the best solution\n"
                    "- Apply statistical and analytical thinking\n"
                    "- Create insightful visualizations\n"
                    "- Calculate derived metrics that add value\n\n"
                    
                    "OUTPUT:\n"
                    "- Return ONE fenced Python code block\n"
                    "- Set `result` to your analysis\n"
                    "- Optionally set `fig` or `mpl_fig`\n"
                    "- Libraries: pd, np, px, go, plt\n"
                    "- Do NOT read files, access network, or call .show()"
                )
            user_prompt = f"Context about the data:\n\n{context}\n\nQuery: {question}\n\nProvide your best intelligent solution."
        else:
            system_prompt = (
                "You are Grok, an expert data analyst built by xAI. Analyze data using structured reasoning.\n\n"
                
                "ANALYSIS FRAMEWORK:\n"
                "1. FINDINGS: What the data shows (key metrics, patterns, trends)\n"
                "2. INSIGHTS: Why it matters (correlations, root causes, implications)\n"
                "3. RECOMMENDATIONS: Actionable next steps (prioritized by impact)\n"
                "4. LIMITATIONS: Data gaps, assumptions, caveats\n\n"
                
                "Be clear, specific, and actionable. Use bullet points and structured format."
            )
            user_prompt = f"Context:\n\n{context}\n\nQuery: {question}\n\nProvide structured analysis following your framework."
        
        # Add extra headers for OpenRouter
        extra_kwargs = {}
        if use_openrouter and site_url and site_name:
            extra_kwargs["extra_headers"] = {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        
        # Try model with OpenRouter normalization/fallback when enabled
        models_to_try = [model]
        if use_openrouter:
            models_to_try = _openrouter_fallback_models(model)
        
        last_err = None
        for m in models_to_try:
            try:
                # Increase max_tokens for code generation to avoid truncation
                # Code mode needs more tokens for 7-step comments + code
                token_limit = 4000 if code_mode else 2000
                
                resp = client.chat.completions.create(
                    model=m,
                    temperature=0.1,  # Lower for faster, more deterministic responses
                    max_tokens=token_limit,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **extra_kwargs
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = str(e)
                # Continue on invalid model errors
                if any(tok in last_err.lower() for tok in ["invalid model", "model_not_found", "invalid_request_error"]):
                    continue
                # Non-model error -> break early
                break
        return f"OpenAI request failed: {last_err or 'All models unavailable'}"
    except Exception as e:
        return f"OpenAI request failed: {e}"


def _select_primary_df(dfs: Dict[str, pd.DataFrame] | None, question: str, preferred: str | None = None) -> Optional[pd.DataFrame]:
    """Pick the most relevant sheet as primary df based on query tokens and optional preferred dataset token.
    Scoring:
      +3 if preferred token appears in sheet key
      +2 per matched query token in sheet key
      +1 per matched query token in column names (capped per sheet)
    Fallback: largest sheet.
    """
    if not dfs:
        return None
    q = (question or "").lower()
    qtokens = [t for t in [tok.strip(" ,.?;:!()[]{}\"'\n\r\t") for tok in q.split()] if len(t) >= 3]
    best_key = None
    best_score = -1
    cap_col_matches = 20
    for key, df in dfs.items():
        score = 0
        k = str(key).lower()
        if preferred and preferred in k:
            score += 3
        # tokens in key
        for t in qtokens:
            if t in k:
                score += 2
        # tokens in columns
        col_hits = 0
        cols_l = [str(c).lower() for c in getattr(df, 'columns', [])]
        for t in qtokens:
            for c in cols_l:
                if t in c:
                    col_hits += 1
                    if col_hits >= cap_col_matches:
                        break
            if col_hits >= cap_col_matches:
                break
        score += min(col_hits, cap_col_matches)  # +1 per hit up to cap
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is not None:
        return dfs.get(best_key)
    # fallback: largest sheet by rows
    sizes = sorted(((k, len(v) if isinstance(v, pd.DataFrame) else 0) for k, v in dfs.items()), key=lambda x: x[1], reverse=True)
    return dfs.get(sizes[0][0]) if sizes else None


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    fences = ["```python", "``` py", "```py", "```"]
    start = -1
    for f in fences:
        start = text.find(f)
        if start != -1:
            start += len(f)
            break
    if start == -1:
        return ""
    end = text.find("```", start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def _sanitize_user_code(code: str) -> str:
    """Remove import statements for libraries we already provide in the sandbox,
    preventing usage of __import__ which is blocked for safety.
    """
    try:
        patterns = [
            r"^\s*import\s+pandas\s+as\s+pd\s*$",
            r"^\s*import\s+numpy\s+as\s+np\s*$",
            r"^\s*import\s+plotly\.express\s+as\s+px\s*$",
            r"^\s*import\s+plotly\.graph_objects\s+as\s+go\s*$",
            r"^\s*import\s+matplotlib\.pyplot\s+as\s+plt\s*$",
            r"^\s*from\s+pandas\b.*$",
            r"^\s*from\s+numpy\b.*$",
            r"^\s*from\s+plotly\b.*$",
            r"^\s*from\s+matplotlib\b.*$",
        ]
        sanitized = code
        for pat in patterns:
            sanitized = re.sub(pat, "# sanitized: removed import", sanitized, flags=re.MULTILINE)
        return sanitized
    except Exception:
        return code


# ---------- Normalization helpers for common queries (e.g., Top-N departments) ----------
def _recompute_department_topN(base_df: Optional[pd.DataFrame], top_n: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[go.Figure]]:
    """From a base dataframe, compute Top-N departments and an optional bar chart.
    Returns (df, fig) where df has columns [department, count].
    """
    if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return None, None
    candidate_cols = [
        "department",
        "dept",
        "department_name",
        "sub_department",
        "Department",
    ]
    dept_col = next((c for c in candidate_cols if c in base_df.columns), None)
    if dept_col is None:
        return None, None
    s = base_df[dept_col].astype(str).str.strip()
    s = s.mask(s.eq(""), "Unknown")
    try:
        top_n = int(top_n)
    except Exception:
        top_n = 10
    top_n = max(1, min(top_n, 100))
    vc = (
        s.value_counts(dropna=False)
         .head(top_n)
         .rename_axis("department")
         .reset_index(name="count")
    )
    fig = None
    try:
        fig = go.Figure(data=[go.Bar(x=vc["count"], y=vc["department"], orientation="h")])
        fig.update_layout(title=f"Top {top_n} Departments by Incident Count", yaxis=dict(autorange="reversed"))
    except Exception:
        fig = None
    return vc, fig


def _extract_top_n_from_query(question: str, default: int = 10) -> int:
    try:
        q = (question or "").lower()
        # Look for patterns like 'top 5', 'top10', 'top-7'
        m = re.search(r"top\s*-?\s*(\d{1,3})", q)
        if m:
            n = int(m.group(1))
            return max(1, min(n, 100))
    except Exception:
        pass
    return default


def _normalize_or_recompute(
    question: str,
    result_obj: Any,
    fig_obj: Any,
    df: Optional[pd.DataFrame],
    dfs: Optional[Dict[str, pd.DataFrame]],
) -> Tuple[Any, Any]:
    """If the query asks about departments, ensure a consistent (department,count) schema
    and create a bar chart if missing. Recompute from base data if needed.
    """
    try:
        q = (question or "").lower()
        if "department" not in q:
            return result_obj, fig_obj
        top_n = _extract_top_n_from_query(q, default=10)

        # Prefer recompute from primary df
        res_df, res_fig = _recompute_department_topN(df, top_n=top_n)
        if res_df is None and isinstance(dfs, dict) and dfs:
            # Try dynamic sheet selection similar to primary selection
            chosen = _select_primary_df(dfs, question, preferred="incident")
            res_df, res_fig = _recompute_department_topN(chosen, top_n=top_n)
        if res_df is not None:
            return res_df, (fig_obj or res_fig)

        # Otherwise, attempt to normalize existing result dataframe
        if isinstance(result_obj, pd.DataFrame) and not result_obj.empty:
            cand = result_obj.copy()
            # Find department-like column (stringy) and count-like column (numeric or named like count)
            dept_candidates = [
                c for c in cand.columns
                if (cand[c].dtype == object) or ("dept" in str(c).lower()) or ("department" in str(c).lower())
            ]
            count_candidates = [
                c for c in cand.columns
                if pd.api.types.is_numeric_dtype(cand[c]) or ("count" in str(c).lower())
            ]
            if dept_candidates and count_candidates:
                dcol = dept_candidates[0]
                ccol = count_candidates[0]
                out = cand[[dcol, ccol]].copy()
                out.columns = ["department", "count"]
                # Make a simple figure if missing
                if fig_obj is None:
                    try:
                        fig_obj = go.Figure(data=[go.Bar(x=out["count"], y=out["department"], orientation="h")])
                        fig_obj.update_layout(title=f"Top {top_n} Departments by Count", yaxis=dict(autorange="reversed"))
                    except Exception:
                        pass
                return out, fig_obj
    except Exception:
        pass
    return result_obj, fig_obj

def run_user_code(code: str, df: Optional[pd.DataFrame], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Dict[str, Any], str, str]:
    """Execute user code in a restricted environment. Returns (env, stdout, stderr).
    Provides a minimal set of safe builtins similar to the Streamlit implementation.
    """
    # Build execution environment
    env: Dict[str, Any] = {}
    safe_builtins = {
        'len': len, 'range': range, 'min': min, 'max': max, 'sum': sum,
        'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'abs': abs, 'round': round,
        'print': print, 'any': any, 'all': all, 'map': map, 'filter': filter,
        # common types/constructors
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple
    }
    allowed_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        # Libraries
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    try:
        import matplotlib.pyplot as plt  # import lazily
        allowed_globals["plt"] = plt
    except Exception:
        pass

    # Locals include df / dfs and placeholders for results
    local_env: Dict[str, Any] = {
        "df": df,
        "dfs": dfs or {},
        "result": None,
        "fig": None,
        "mpl_fig": None,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        import contextlib
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            sanitized_code = _sanitize_user_code(code)
            exec(sanitized_code, allowed_globals, local_env)
    except Exception:
        traceback.print_exc(file=stderr_buf)
    return local_env, stdout_buf.getvalue(), stderr_buf.getvalue()


@dataclass
class AgentResponse:
    code: str
    stdout: str
    error: str
    result_preview: List[Dict[str, Any]]
    figure: Optional[Dict[str, Any]]
    mpl_png_base64: Optional[str]
    analysis: str
    attempts: int = 1
    verification_score: float = 0.0
    correction_log: List[str] = None

    def __post_init__(self):
        if self.correction_log is None:
            self.correction_log = []


def _result_preview(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, pd.DataFrame):
        return _head_records(obj)
    # scalar or list-like
    try:
        return pd.DataFrame({"result": [obj]}).to_dict(orient="records")
    except Exception:
        return [{"result": str(obj)}]


def _fig_to_dict(fig_obj: Any) -> Optional[Dict[str, Any]]:
    """Safely convert a Plotly figure to dict. Avoids local redefinition of `go`.
    Accepts either a plotly.graph_objects.Figure or an already-serialized dict.
    """
    try:
        if fig_obj is None:
            return None
        if isinstance(fig_obj, dict):
            return fig_obj
        if isinstance(fig_obj, go.Figure):
            return fig_obj.to_plotly_json()
    except Exception:
        return None
    return None


def _mpl_to_png_b64(mpl_fig: Any) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        buf = io.BytesIO()
        mpl_fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def _to_native_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to native JSON-safe Python types."""
    try:
        # Scalars
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, np.generic):
            return obj.item()
        # Pandas timestamps
        try:
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
        except Exception:
            pass
        # Sequences
        if isinstance(obj, (list, tuple, set)):
            return [ _to_native_jsonable(v) for v in obj ]
        # Numpy arrays
        if isinstance(obj, np.ndarray):
            return _to_native_jsonable(obj.tolist())
        # Dicts
        if isinstance(obj, dict):
            return { str(k): _to_native_jsonable(v) for k, v in obj.items() }
        # DataFrames / Series
        if isinstance(obj, pd.DataFrame):
            return _to_native_jsonable(obj.to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            return _to_native_jsonable(obj.to_dict())
        # Fallback for NaN/NaT and others
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return str(obj)
    except Exception:
        return None

def verify_result_quality(query: str, code: str, result: Any, error: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Use LLM to verify if the result correctly answers the query."""
    # Helper to check if result exists
    def has_result(r):
        if r is None:
            return False
        if isinstance(r, pd.DataFrame):
            return not r.empty
        if isinstance(r, pd.Series):
            return len(r) > 0
        return True
    
    if not _OPENAI_AVAILABLE:
        return {"is_valid": has_result(result) and not error, "confidence": 0.5, "issues": [], "suggestions": ""}
    
    # Check if using OpenRouter
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    
    if use_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"is_valid": has_result(result) and not error, "confidence": 0.5, "issues": [], "suggestions": ""}
        base_url = "https://openrouter.ai/api/v1"
        site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "Safety Copilot")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"is_valid": has_result(result) and not error, "confidence": 0.5, "issues": [], "suggestions": ""}
        base_url = None
        site_url = None
        site_name = None
    
    try:
        # Create client with appropriate base_url
        if use_openrouter:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        
        if error:
            verification_prompt = f"""The pandas code execution failed.
            
Original query: {query}
Code attempted: {code}
Error: {error}
            
Analyze the error and provide:
1. Root cause of the error
2. Corrected pandas code that will work
3. What was missing or wrong
            
Respond in JSON format:
{{
    "is_valid": false,
    "confidence": 0.0,
    "issues": ["list of specific issues"],
    "corrected_code": "fixed pandas code",
    "explanation": "why it failed and how to fix it"
}}"""
        else:
            # Convert result to string for verification (safely handle DataFrames)
            if isinstance(result, pd.DataFrame):
                result_str = result.head(10).to_string() if not result.empty else "Empty DataFrame"
            elif isinstance(result, pd.Series):
                result_str = result.head(10).to_string() if len(result) > 0 else "Empty Series"
            else:
                result_str = str(result)[:500] if result is not None else "No result"
            
            verification_prompt = f"""User asked: {query}
            
Pandas code executed:
{code}
            
Result obtained:
{result_str}
            
Verify if this result FULLY and CORRECTLY answers the user's question.
Check for:
1. Correct filtering/conditions applied
2. Proper sorting/ordering
3. Right aggregations (count, sum, avg, etc.)
4. Complete data (not truncated incorrectly)
5. Appropriate columns selected
6. Accurate calculations
            
Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of any issues found"],
    "suggestions": "if invalid, suggest specific code correction",
    "explanation": "brief explanation of verification result"
}}"""
        
        # Add extra headers for OpenRouter
        extra_kwargs = {}
        if use_openrouter and site_url and site_name:
            extra_kwargs["extra_headers"] = {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        
        # Try model with OpenRouter normalization/fallback when enabled
        models_to_try = [model]
        if use_openrouter:
            models_to_try = _openrouter_fallback_models(model)
        
        last_err = None
        for m in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=m,
                    temperature=0.1,
                    max_tokens=800,
                    messages=[
                        {"role": "system", "content": "You are a code verification expert. Analyze results and provide JSON responses only."},
                        {"role": "user", "content": verification_prompt}
                    ],
                    **extra_kwargs
                )
                break  # Success, exit loop
            except Exception as e:
                last_err = str(e)
                # Continue on invalid model errors
                if any(tok in last_err.lower() for tok in ["invalid model", "model_not_found", "invalid_request_error"]):
                    continue
                # Non-model error -> break early
                raise
        else:
            # All models failed
            raise Exception(f"All verification models failed: {last_err}")
        
        content = response.choices[0].message.content or "{}"
        # Extract JSON from response
        import json
        try:
            # Try direct JSON parse
            return json.loads(content)
        except:
            # Try extracting JSON from markdown code block
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            return json.loads(content)
    except Exception as e:
        # Fallback: simple heuristic with safe result check
        def has_result_safe(r):
            if r is None:
                return False
            if isinstance(r, pd.DataFrame):
                return not r.empty
            if isinstance(r, pd.Series):
                return len(r) > 0
            return True
        
        has_valid_result = has_result_safe(result) and not error
        return {
            "is_valid": has_valid_result,
            "confidence": 0.6 if has_valid_result else 0.0,
            "issues": [str(e)] if error else [],
            "suggestions": "Retry with corrected logic",
            "explanation": f"Verification failed: {str(e)}"
        }


def generate_corrected_code(query: str, context: str, previous_attempts: List[Dict], model: str = "gpt-4o") -> str:
    """Generate corrected code based on previous failed attempts."""
    if not previous_attempts:
        return ask_openai(query, context, model=model, code_mode=True, multi_df=True)
    
    # Build context with previous failures
    failures_context = "\n\nPREVIOUS FAILED ATTEMPTS:\n"
    for i, attempt in enumerate(previous_attempts, 1):
        failures_context += f"\nAttempt {i}:\n"
        failures_context += f"Code:\n{attempt.get('code', 'N/A')}\n"
        failures_context += f"Issue: {attempt.get('issue', 'N/A')}\n"
        if attempt.get('suggestions'):
            failures_context += f"Suggestion: {attempt['suggestions']}\n"
    
    failures_context += "\n\nGENERATE CORRECTED CODE that fixes the issues above.\n"
    failures_context += "IMPORTANT: Learn from the failures and implement the suggestions.\n"
    
    enhanced_context = context + failures_context
    return ask_openai(query, enhanced_context, model=model, code_mode=True, multi_df=True)


async def ask_openai_stream(question: str, context: str, *, model: str = "z-ai/glm-4.6", code_mode: bool = False, multi_df: bool = False) -> AsyncGenerator[str, None]:
    """Streaming version of ask_openai for real-time updates."""
    if not _OPENAI_AVAILABLE:
        yield "OpenAI Python package is not installed."
        return
    
    # Check if using OpenRouter
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    
    if use_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            yield "Missing OPENROUTER_API_KEY."
            return
        base_url = "https://openrouter.ai/api/v1"
        site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
        site_name = os.getenv("OPENROUTER_SITE_NAME", "Safety Copilot")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            yield "Missing OPENAI_API_KEY."
            return
        base_url = None
        site_url = None
        site_name = None
    
    try:
        # Create async client
        if use_openrouter:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client = AsyncOpenAI(api_key=api_key)
        
        if code_mode:
            if multi_df:
                system_prompt = (
                    "You are a senior data analyst writing Python for pandas and plotly to analyze multiple DataFrames in a dict named `dfs` (keys are lowercased sheet names). "
                    "A primary DataFrame `df` may also be provided. Use ONLY the provided context (schema, summaries, head rows). "
                    "Rules: \n"
                    "- Return ONE fenced Python code block ONLY. No prose.\n"
                    "- Choose the correct DataFrame by matching query terms to `dfs` keys and column names. If ambiguous, default to `df` or the largest table.\n"
                    "- VERIFY columns exist before using; handle missing columns gracefully.\n"
                    "- Prefer vectorized pandas ops; avoid loops.\n"
                    "- Set `result` to a DataFrame or scalar. Optionally set `fig` (plotly) or `mpl_fig` (matplotlib).\n"
                    "- Do NOT read files, do NOT access network, do NOT call .show()."
                )
            else:
                system_prompt = (
                    "You are a senior data analyst writing Python for pandas and plotly to analyze a DataFrame named `df`. "
                    "Use ONLY the provided context (schema, summaries, head rows). \n"
                    "Rules: \n"
                    "- Return ONE fenced Python code block ONLY. No prose.\n"
                    "- VERIFY columns exist before using; handle missing columns gracefully.\n"
                    "- Prefer vectorized pandas ops; avoid loops.\n"
                    "- Set `result` to a DataFrame or scalar. Optionally set `fig` (plotly) or `mpl_fig` (matplotlib).\n"
                    "- Do NOT read files, do NOT access network, do NOT call .show().\n"
                    "- Do NOT write import statements for pandas/numpy/plotly/matplotlib; use pd, np, px, go, plt provided."
                )
            user_prompt = f"Context about the data:\n\n{context}\n\nTask: Write Python code to answer: {question}"
        else:
            system_prompt = (
                "You are a helpful data analyst. Use ONLY the provided context to produce a concise, prescriptive analysis (Findings, Recommendations, Next steps)."
            )
            user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
        
        # Add extra headers for OpenRouter
        extra_kwargs = {}
        if use_openrouter and site_url and site_name:
            extra_kwargs["extra_headers"] = {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        
        # Try model with OpenRouter normalization/fallback when enabled
        models_to_try = [model]
        if use_openrouter:
            models_to_try = _openrouter_fallback_models(model)
        
        last_err = None
        for m in models_to_try:
            try:
                # Use same token limits as non-streaming for consistency
                token_limit = 4000 if code_mode else 2000
                
                stream = await client.chat.completions.create(
                    model=m,
                    temperature=0.1,  # Lower for faster, more deterministic
                    max_tokens=token_limit,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                    **extra_kwargs
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, exit function
            except Exception as e:
                last_err = str(e)
                # Continue on invalid model errors
                if any(tok in last_err.lower() for tok in ["invalid model", "model_not_found", "invalid_request_error"]):
                    continue
                # Non-model error -> yield error and return
                yield f"\n\nError: {str(e)}"
                return
        
        # All models failed
        yield f"\n\nError: All models failed - {last_err}"
                
    except Exception as e:
        yield f"\n\nError: {str(e)}"


async def generate_agent_response_stream(query: str, *, dataset: str = "incident", model: str = "z-ai/glm-4.6", max_retries: int = 3) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming version of generate_agent_response.
    Yields progress updates in real-time.
    """
    # Check if conversational query - handle instantly
    if _is_conversational_query(query):
        yield {
            "type": "progress",
            "stage": "conversational",
            "message": "💬 Detected conversational query..."
        }
        
        conversational_response = _generate_conversational_response(query, model=model)
        
        yield {
            "type": "complete",
            "data": {
                "code": "# Conversational query - no code generated",
                "stdout": "",
                "error": "",
                "result_preview": [],
                "figure": None,
                "mpl_png_base64": None,
                "analysis": conversational_response,
                "attempts": 1,
                "verification_score": 1.0,
            }
        }
        return
    
    try:
        # Setup
        yield {
            "type": "progress",
            "stage": "initializing",
            "message": "🔧 Loading data..."
        }
        
        dataset_l = (dataset or "incident").lower()
        workbook = load_default_sheets()
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        preferred = dataset_l if dataset_l in ("incident", "hazard", "audit", "inspection") else None
        primary_df = _select_primary_df(dfs, query, preferred=preferred)
        df = primary_df
        context = build_workbook_context(workbook, query=query)
        multi_df = True
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Initialization error: {str(e)}"
        }
        return
    
    attempts = []
    best_result = None
    best_verification = {"is_valid": False, "confidence": 0.0}
    final_code = ""
    final_stdout = ""
    final_stderr = ""
    final_env = {}
    
    try:
        for attempt_num in range(1, max_retries + 1):
            # Send progress update
            yield {
                "type": "progress",
                "stage": "generating_code",
                "attempt": attempt_num,
                "max_attempts": max_retries,
                "message": f"🔄 Attempt {attempt_num}/{max_retries}: Generating code..."
            }
            
            # Generate code with streaming
            code_chunks = []
            try:
                async for chunk in ask_openai_stream(query, context, model=model, code_mode=True, multi_df=multi_df):
                    code_chunks.append(chunk)
                    # Stream code generation progress
                    yield {
                        "type": "code_chunk",
                        "chunk": chunk,
                        "message": "📝 Generating code..."
                    }
            except Exception as e:
                yield {
                    "type": "error",
                    "message": f"Code generation error: {str(e)}"
                }
                # Use fallback code
                code_chunks = ["# Fallback due to error\nresult = df.head(10)"]
            
            code_resp = "".join(code_chunks)
            code_block = extract_python_code(code_resp)
            
            if not code_block:
                yield {
                    "type": "progress",
                    "stage": "fallback",
                    "message": "⚠️  No code generated, using fallback..."
                }
                query_lower = (query or '').lower()
                code_block = f"""# Fallback code
_df = df
if _df is None and isinstance(dfs, dict) and dfs:
    _best = list(dfs.keys())[0]
    _df = dfs[_best]
if _df is not None:
    result = _df.head(20)
"""
            
            yield {
                "type": "progress",
                "stage": "executing",
                "message": "⚙️  Executing code..."
            }
            
            # Execute
            env, stdout, stderr = run_user_code(code_block, df=df, dfs=dfs)
            
            has_error = bool(stderr and stderr.strip())
            
            if has_error:
                yield {
                    "type": "error",
                    "error": stderr[:200],
                    "message": f"❌ Execution error"
                }
            else:
                yield {
                    "type": "progress",
                    "stage": "success",
                    "message": f"✅ Execution successful"
                }
            
            # Verify
            yield {
                "type": "progress",
                "stage": "verifying",
                "message": "🔍 Verifying result quality..."
            }
            
            verification = verify_result_quality(query, code_block, env.get("result"), stderr, model=model)
            confidence = verification.get("confidence", 0.0)
            is_valid = verification.get("is_valid", False)
            
            yield {
                "type": "verification",
                "is_valid": is_valid,
                "confidence": confidence,
                "message": f"🔍 Verification: valid={is_valid}, confidence={confidence:.2f}"
            }
            
            # Record attempt
            attempts.append({
                "attempt": attempt_num,
                "code": code_block,
                "success": not has_error,
                "error": stderr if has_error else None,
                "verification": verification,
            })
            
            # Keep best result
            if confidence > best_verification.get("confidence", 0.0) or (is_valid and not best_verification.get("is_valid")):
                best_verification = verification
                best_result = env.get("result")
                final_code = code_block
                final_stdout = stdout
                final_stderr = stderr
                final_env = env
                
                yield {
                    "type": "progress",
                    "stage": "best_result",
                    "message": f"⭐ New best result (confidence: {confidence:.2f})"
                }
            
            # Stop if good enough
            if is_valid and confidence >= 0.8:
                yield {
                    "type": "progress",
                    "stage": "complete",
                    "message": f"✨ SUCCESS! Found valid answer in attempt {attempt_num}"
                }
                break
            
            if attempt_num < max_retries and not is_valid:
                yield {
                    "type": "progress",
                    "stage": "retrying",
                    "message": "🔄 Retrying with corrections..."
                }
        
        # Final processing
        yield {
            "type": "progress",
            "stage": "finalizing",
            "message": "📊 Preparing final results..."
        }
        
        env = final_env
        normalized_result, normalized_fig = _normalize_or_recompute(query, env.get("result"), env.get("fig"), df, dfs)
        result_preview = _result_preview(normalized_result)
        result_preview = _to_native_jsonable(result_preview)
        fig_dict = _fig_to_dict(normalized_fig)
        fig_dict = _to_native_jsonable(fig_dict) if fig_dict is not None else None
        mpl_png = _mpl_to_png_b64(env.get("mpl_fig")) if env.get("mpl_fig") is not None else None
        
        # Generate analysis
        yield {
            "type": "progress",
            "stage": "analyzing",
            "message": "🤔 Generating analysis..."
        }
        
        summary_ctx_parts: List[str] = []
        if result_preview:
            summary_ctx_parts.append(f"Result preview:\n{pd.DataFrame(result_preview).to_string(index=False)}")
        summary_ctx = "\n\n".join(summary_ctx_parts) or "No outputs."
        
        analysis_chunks = []
        async for chunk in ask_openai_stream(
            "Provide a concise analysis based on the outputs.",
            context=summary_ctx,
            model=model,
            code_mode=False
        ):
            analysis_chunks.append(chunk)
            yield {
                "type": "analysis_chunk",
                "chunk": chunk
            }
        
        analysis = "".join(analysis_chunks)
        
        if best_verification.get("explanation"):
            analysis += f"\n\n**Verification:** {best_verification['explanation']}"
        if len(attempts) > 1:
            analysis += f"\n\n**Self-Correction:** Completed in {len(attempts)} attempt(s). Confidence: {best_verification.get('confidence', 0.0):.2f}"
    
        # Send final result
        yield {
            "type": "complete",
            "data": {
                "code": final_code,
                "stdout": final_stdout,
                "error": final_stderr,
                "result_preview": result_preview,
                "figure": fig_dict,
                "mpl_png_base64": mpl_png,
                "analysis": analysis,
                "attempts": len(attempts),
                "verification_score": best_verification.get("confidence", 0.0),
            }
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        yield {
            "type": "error",
            "message": f"Fatal error in streaming: {str(e)}",
            "details": error_details
        }


def generate_agent_response(query: str, *, dataset: str = "incident", model: str = "gpt-4o", max_retries: int = 3) -> AgentResponse:
    # Check if conversational query - handle instantly
    if _is_conversational_query(query):
        conversational_response = _generate_conversational_response(query, model=model)
        return AgentResponse(
            code="# Conversational query - no code generated",
            stdout="",
            error="",
            result_preview=[],
            figure=None,
            mpl_png_base64=None,
            analysis=conversational_response,
            attempts=1,
            verification_score=1.0,
            correction_log=["✅ Conversational query handled instantly"]
        )
    
    # Always gather all sheets and build multi-DF context; also pick a primary df smartly
    dataset_l = (dataset or "incident").lower()
    workbook = load_default_sheets()
    dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
    # preferred token from dataset param if provided
    preferred = dataset_l if dataset_l in ("incident", "hazard", "audit", "inspection") else None
    primary_df = _select_primary_df(dfs, query, preferred=preferred)
    df = primary_df
    context = build_workbook_context(workbook, query=query)
    multi_df = True

    # Self-correction loop
    attempts = []
    correction_log = []
    best_result = None
    best_verification = {"is_valid": False, "confidence": 0.0}
    final_code = ""
    final_stdout = ""
    final_stderr = ""
    final_env = {}
    
    for attempt_num in range(1, max_retries + 1):
        correction_log.append(f"🔄 Attempt {attempt_num}/{max_retries}")
        
        # 1) Generate code (corrected if not first attempt)
        if attempt_num == 1:
            code_resp = ask_openai(query, context, model=model, code_mode=True, multi_df=multi_df)
            code_block = extract_python_code(code_resp)
        else:
            code_resp = generate_corrected_code(query, context, attempts, model=model)
            code_block = extract_python_code(code_resp)
        
        correction_log.append(f"📝 Generated code ({len(code_block)} chars)")
        
        if not code_block:
            correction_log.append("⚠️  No code generated, using fallback")
            # Heuristic fallback: choose a working DataFrame and compute something meaningful
            query_lower = (query or '').lower()
            fallback_code = f"""# Heuristic fallback because the LLM did not return executable code
_df = df
if _df is None and isinstance(dfs, dict) and dfs:
    # Dynamic sheet selection by query tokens
    _q = '{query_lower}'
    _q_toks = [t for t in _q.split() if len(t) >= 3]
    _best = None
    _best_score = -1
    for _k, _v in dfs.items():
        _s = 0
        _kl = str(_k).lower()
        for _t in _q_toks:
            if _t in _kl: _s += 2
        _cols = [str(c).lower() for c in getattr(_v, 'columns', [])]
        _hits = 0
        for _t in _q_toks:
            for _c in _cols:
                if _t in _c: _hits += 1
        _s += min(_hits, 20)
        if _s > _best_score: _best_score, _best = _s, _k
    if _best is None:
        _best = list(dfs.keys())[0]
    _df = dfs[_best]
# Guard if still None
if _df is None:
    result = None
else:
    _cp = _df.copy()
    if 'department' in _cp.columns:
        _vc = _cp['department'].astype(str).value_counts().head(10).rename_axis('department').reset_index(name='count')
        result = _vc
        try:
            fig = go.Figure(data=[go.Bar(x=_vc['count'], y=_vc['department'], orientation='h')])
            fig.update_layout(title='Top Departments by Count')
        except Exception:
            pass
    else:
        result = _cp.head(20)
"""
            code_block = fallback_code

        # 2) Execute
        env, stdout, stderr = run_user_code(code_block, df=df, dfs=dfs)
        
        has_error = bool(stderr and stderr.strip())
        has_result = env.get("result") is not None
        
        if has_error:
            correction_log.append(f"❌ Execution error: {stderr[:200]}...")
        else:
            correction_log.append(f"✅ Execution successful, result type: {type(env.get('result')).__name__}")
        
        # 3) Verify result quality
        verification = verify_result_quality(query, code_block, env.get("result"), stderr, model=model)
        confidence = verification.get("confidence", 0.0)
        is_valid = verification.get("is_valid", False)
        
        correction_log.append(f"🔍 Verification: valid={is_valid}, confidence={confidence:.2f}")
        
        # Record attempt
        attempts.append({
            "attempt": attempt_num,
            "code": code_block,
            "success": not has_error,
            "error": stderr if has_error else None,
            "result": str(env.get("result", ""))[:200],
            "verification": verification,
            "issue": "; ".join(verification.get("issues", [])) if not is_valid else None,
            "suggestions": verification.get("suggestions", "")
        })
        
        # Keep best result
        if confidence > best_verification.get("confidence", 0.0) or (is_valid and not best_verification.get("is_valid")):
            best_verification = verification
            best_result = env.get("result")
            final_code = code_block
            final_stdout = stdout
            final_stderr = stderr
            final_env = env
            correction_log.append(f"⭐ New best result (confidence: {confidence:.2f})")
        
        # If result is valid with high confidence, stop early
        if is_valid and confidence >= 0.8:
            correction_log.append(f"✨ SUCCESS! Found valid answer in attempt {attempt_num}")
            break
        
        # If we have more retries, continue with corrections
        if attempt_num < max_retries:
            if not is_valid:
                correction_log.append(f"🔄 Retrying with corrections...")
                if verification.get("suggestions"):
                    correction_log.append(f"💡 Suggestion: {verification['suggestions'][:150]}...")
    
    # Use best result found
    env = final_env
    stdout = final_stdout
    stderr = final_stderr
    code_block = final_code
    # 3) Collect outputs (normalize schema when appropriate)
    normalized_result, normalized_fig = _normalize_or_recompute(query, env.get("result"), env.get("fig"), df, dfs)
    result_preview = _result_preview(normalized_result)
    # Sanitize for JSON safety
    result_preview = _to_native_jsonable(result_preview)
    fig_dict = _fig_to_dict(normalized_fig)
    fig_dict = _to_native_jsonable(fig_dict) if fig_dict is not None else None
    mpl_png = _mpl_to_png_b64(env.get("mpl_fig")) if env.get("mpl_fig") is not None else None

    # 4) Prescriptive analysis
    # Build a compact run context for the summary LLM call
    summary_ctx_parts: List[str] = []
    if result_preview:
        summary_ctx_parts.append(f"Result preview (first {len(result_preview)} rows):\n{pd.DataFrame(result_preview).to_string(index=False)}")
    if fig_dict:
        summary_ctx_parts.append("A Plotly figure was generated (structure summarized).")
    if stdout:
        summary_ctx_parts.append("Stdout:\n" + stdout[:1000])
    if stderr:
        summary_ctx_parts.append("Errors:\n" + stderr[:1000])
    summary_ctx = "\n\n".join(summary_ctx_parts) or "No outputs."

    analysis = ask_openai(
        "Provide a concise, prescriptive analysis and recommendations based on the outputs above. Structure as: Findings, Recommendations, Potential next steps.",
        context=summary_ctx,
        model=model,
        code_mode=False,
    )

    # Add verification info to analysis
    if best_verification.get("explanation"):
        analysis += f"\n\n**Verification:** {best_verification['explanation']}"
    if len(attempts) > 1:
        analysis += f"\n\n**Self-Correction:** Completed in {len(attempts)} attempt(s). Confidence: {best_verification.get('confidence', 0.0):.2f}"
    
    return AgentResponse(
        code=code_block,
        stdout=stdout,
        error=stderr,
        result_preview=result_preview,
        figure=fig_dict,
        mpl_png_base64=mpl_png,
        analysis=analysis,
        attempts=len(attempts),
        verification_score=best_verification.get("confidence", 0.0),
        correction_log=correction_log
    )
