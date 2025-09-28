from __future__ import annotations

import base64
import io
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

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
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Missing OPENAI_API_KEY. Set it in your environment."
    try:
        client = OpenAI(api_key=api_key)
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
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
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

def generate_agent_response(query: str, *, dataset: str = "incident", model: str = "gpt-4o") -> AgentResponse:
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

    # 1) Ask LLM to produce code
    code_resp = ask_openai(query, context, model=model, code_mode=True, multi_df=multi_df)
    code_block = extract_python_code(code_resp)
    if not code_block:
        # Heuristic fallback: choose a working DataFrame and compute something meaningful
        fallback_code = "\n".join([
            "# Heuristic fallback because the LLM did not return executable code",
            "_df = df",
            "if _df is None and isinstance(dfs, dict) and dfs:",
            "    # Dynamic sheet selection by query tokens",
            "    _q = '''""""""""'''.lower()",
            "    _q_toks = [t for t in _q.split() if len(t) >= 3]",
            "    _best = None",
            "    _best_score = -1",
            "    for _k, _v in dfs.items():",
            "        _s = 0",
            "        _kl = str(_k).lower()",
            "        for _t in _q_toks:",
            "            if _t in _kl: _s += 2",
            "        _cols = [str(c).lower() for c in getattr(_v, 'columns', [])]",
            "        _hits = 0",
            "        for _t in _q_toks:",
            "            for _c in _cols:",
            "                if _t in _c: _hits += 1",
            "        _s += min(_hits, 20)",
            "        if _s > _best_score: _best_score, _best = _s, _k",
            "    if _best is None:",
            "        _best = list(dfs.keys())[0]",
            "    _df = dfs[_best]",
            "# Guard if still None",
            "if _df is None:",
            "    result = None",
            "else:",
            "    _cp = _df.copy()",
            "    if 'department' in _cp.columns:",
            "        _vc = _cp['department'].astype(str).value_counts().head(10).rename_axis('department').reset_index(name='count')",
            "        result = _vc",
            "        try:",
            "            fig = go.Figure(data=[go.Bar(x=_vc['count'], y=_vc['department'], orientation='h')])",
            "            fig.update_layout(title='Top Departments by Count')",
            "        except Exception:",
            "            pass",
            "    else:",
            "        result = _cp.head(20)",
        ]).replace("'''\"\"\"\"\"\"\"\"\"\"'''.lower()", f"'{(query or '').lower()}'.lower()")
        code_block = fallback_code

    # 2) Execute
    env, stdout, stderr = run_user_code(code_block, df=df, dfs=dfs)
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

    return AgentResponse(
        code=code_block,
        stdout=stdout,
        error=stderr,
        result_preview=result_preview,
        figure=fig_dict,
        mpl_png_base64=mpl_png,
        analysis=analysis,
    )
