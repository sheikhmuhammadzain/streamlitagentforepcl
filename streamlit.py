import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
import plotly.io as pio
import textwrap
import os
import io
import contextlib
import traceback
import matplotlib.pyplot as plt
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False
warnings.filterwarnings('ignore')

# Global theming for Plotly: white background with green accents
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#16A34A",  # Green 600
    "#22C55E",  # Green 500
    "#059669",  # Emerald 600
    "#10B981",  # Emerald 500
    "#065F46",  # Green 800
    "#6EE7B7",  # Emerald 300
    "#34D399"   # Emerald 400
]

# Use a green continuous scale for heatmaps/gradients
px.defaults.color_continuous_scale = px.colors.sequential.Greens

# Page configuration
st.set_page_config(
    page_title="HSE Analytics Dashboard",
    page_icon="🧯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #FFFFFF;
    }
    /* Metric cards */
    .stMetric {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E5E7EB; /* gray-200 */
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        border-left: 4px solid #16A34A; /* primary green accent */
    }
    /* Equalize metric card heights */
    .stMetric, div[data-testid="stMetric"] {
        height: 130px;
        min-height: 130px;
    }
    div[data-testid="stMetric"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    /* Ensure column children stretch */
    div[data-testid="column"] > div {
        height: 100%;
    }
    /* Improve section dividers */
    hr, .stMarkdown hr { border-color: #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🧯 EPCL HSE Analytics Dashboard")
st.markdown("### Incidents, Hazards, Audits, Inspections – Interactive Analysis")

# Sidebar for file upload and filters
with st.sidebar:
    st.header("📁 Data Input")
    
    # File uploader for Excel workbook
    uploaded_file = st.file_uploader(
        "Upload EPCL processed Excel (.xlsx)",
        type=['xlsx'],
        help="Upload the EPCL_VEHS_Data_Processed.xlsx or a similar processed workbook"
    )
    
    # Use example data option
    use_example = st.checkbox("Use local example file (EPCL_VEHS_Data_Processed.xlsx)", value=True if not uploaded_file else False)
    
    # Divider
    st.markdown("---")

# Load workbook and helpers
@st.cache_data
def load_workbook(file):
    """Load an Excel workbook and parse all sheets into DataFrames with datetime coercion."""
    try:
        if file is not None:
            xls = pd.ExcelFile(file)
        else:
            # Local example file
            xls = pd.ExcelFile('EPCL_VEHS_Data_Processed.xlsx')
        sheets = {}
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
            except Exception:
                continue
            # Coerce potential datetime-like columns
            for col in df.columns:
                lc = str(col).lower()
                if ('date' in lc) or ('time' in lc) or lc.startswith('entered_') or lc in ['start_date', 'entered_closed']:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception:
                        pass
            sheets[sheet] = df
        return sheets
    except FileNotFoundError:
        st.error("Example file 'EPCL_VEHS_Data_Processed.xlsx' not found. Please upload the workbook.")
        return None
    except Exception as e:
        st.error(f"Failed to load workbook: {e}")
        return None

def _first_present(df: pd.DataFrame, candidates):
    """Return the first column name from candidates that exists in df, else None.
    Avoid advanced typing so this runs on older Python versions.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_schema(df: pd.DataFrame, sheet_name: str) -> dict:
    s = sheet_name.lower()
    # Date/time primary column candidates per sheet
    if 'incident' in s:
        date_candidates = ['occurrence_date', 'reported_date', 'entered_date', 'completion_date']
        status_candidates = ['status']
        title_candidates = ['title']
        category_candidates = ['category', 'incident_type']
        consequence_candidates = ['worst_case_consequence_incident', 'actual_consequence_incident', 'relevant_consequence_incident']
    elif 'hazard' in s:
        date_candidates = ['occurrence_date', 'reported_date', 'entered_date', 'entered_closed']
        status_candidates = ['status']
        title_candidates = ['title']
        category_candidates = ['category', 'violation_type_hazard_id']
        consequence_candidates = ['worst_case_consequence_potential_hazard_id', 'relevant_consequence_hazard_id']
    elif 'audit findings' in s:
        date_candidates = ['start_date', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category']
        consequence_candidates = ['worst_case_consequence']
    elif 'audit' in s:
        date_candidates = ['start_date', 'entered_in_progress', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category', 'auditing_body']
        consequence_candidates = ['worst_case_consequence']
    elif 'inspection' in s:
        date_candidates = ['start_date', 'entered_in_progress', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category']
        consequence_candidates = ['worst_case_consequence']
    else:
        # Fallback: choose any datetime col
        date_candidates = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        status_candidates = ['status', 'audit_status']
        title_candidates = ['title', 'audit_title']
        category_candidates = ['category', 'audit_category']
        consequence_candidates = []

    date_col = _first_present(df, date_candidates) if date_candidates else None
    status_col = _first_present(df, status_candidates)
    title_col = _first_present(df, title_candidates)
    category_col = _first_present(df, category_candidates)

    # Common dimensions
    dept_col = _first_present(df, ['department', 'sub_department'])
    loc_col = _first_present(df, ['location', 'sublocation', 'location.1'])
    id_col = _first_present(df, ['incident_id', 'audit_id'])
    consequence_col = _first_present(df, consequence_candidates) if consequence_candidates else None

    # Metrics
    severity_col = _first_present(df, ['severity_score'])
    risk_col = _first_present(df, ['risk_score', 'department_avg_risk'])
    cost_col = _first_present(df, ['estimated_cost_impact'])
    manhours_col = _first_present(df, ['estimated_manhours_impact'])
    reporting_delay_col = _first_present(df, ['reporting_delay_days'])
    resolution_time_col = _first_present(df, ['resolution_time_days'])

    flags = [c for c in ['root_cause_is_missing', 'corrective_actions_is_missing'] if c in df.columns]

    return {
        'date_col': date_col,
        'status_col': status_col,
        'title_col': title_col,
        'category_col': category_col,
        'dept_col': dept_col,
        'loc_col': loc_col,
        'id_col': id_col,
        'consequence_col': consequence_col,
        'severity_col': severity_col,
        'risk_col': risk_col,
        'cost_col': cost_col,
        'manhours_col': manhours_col,
        'reporting_delay_col': reporting_delay_col,
        'resolution_time_col': resolution_time_col,
        'flags': flags,
    }

def apply_filters(df: pd.DataFrame, schema: dict,
                  date_range=None, statuses=None, departments=None, locations=None, categories=None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    dc = schema.get('date_col')
    if dc and date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        m &= df[dc].between(start, end, inclusive='both')
    sc = schema.get('status_col')
    if sc and statuses:
        m &= df[sc].isin(statuses)
    dpc = schema.get('dept_col')
    if dpc and departments:
        m &= df[dpc].isin(departments)
    lc = schema.get('loc_col')
    if lc and locations:
        m &= df[lc].isin(locations)
    cc = schema.get('category_col')
    if cc and categories:
        m &= df[cc].isin(categories)
    return df[m]

def safe_value_counts(df: pd.DataFrame, col: str, top: int = 10):
    if col and col in df.columns:
        return df[col].value_counts(dropna=True).head(top)
    return pd.Series(dtype='int64')

def _fmt_num(v):
    try:
        if pd.isna(v):
            return "0"
        if abs(float(v)) >= 1000:
            return f"{float(v):,.0f}"
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

# ---------- AI helpers ----------
def build_ai_context(df: pd.DataFrame, max_numeric_cols: int = 6, max_cat_cols: int = 6, sample_rows: int = 5) -> str:
    """Build a concise text context describing the filtered dataframe.
    Limits numeric/categorical summaries and includes a few sample rows to keep token usage low.
    """
    if df is None or len(df) == 0:
        return "No data available."

    lines = []
    lines.append("DATAFRAME OVERVIEW")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")
    lines.append("COLUMNS (name: dtype, non-null count):")
    for col in list(df.columns)[:50]:
        dtype = str(df[col].dtype)
        nonnull = int(df[col].notna().sum())
        lines.append(f"- {col}: {dtype}, non-null={nonnull:,}")

    # Numeric summaries
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        lines.append("")
        lines.append("NUMERIC SUMMARIES (mean, std, min, max):")
        for col in num_cols[:max_numeric_cols]:
            desc = df[col].describe()
            try:
                mean = float(desc.get('mean', 0))
                std = float(desc.get('std', 0))
                min_v = float(desc.get('min', 0))
                max_v = float(desc.get('max', 0))
                lines.append(f"- {col}: mean={mean:.2f}, std={std:.2f}, min={min_v:.2f}, max={max_v:.2f}")
            except Exception:
                pass

    # Categorical top values
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        lines.append("")
        lines.append("CATEGORICAL TOP VALUES (top 5):")
        for col in cat_cols[:max_cat_cols]:
            vc = df[col].value_counts(dropna=True).head(5)
            compact = ", ".join([f"{idx} ({cnt})" for idx, cnt in vc.items()])
            lines.append(f"- {col}: {compact}")

    # Sample rows
    try:
        n = min(sample_rows, len(df))
        sample = df.head(n)
        lines.append("")
        lines.append(f"SAMPLE ROWS (first {n}):")
        # Keep sample compact by converting datetimes to string
        sample_cp = sample.copy()
        for c in sample_cp.columns:
            if np.issubdtype(sample_cp[c].dtype, np.datetime64):
                sample_cp[c] = sample_cp[c].dt.strftime('%Y-%m-%d %H:%M:%S')
        lines.append(sample_cp.to_csv(index=False))
    except Exception:
        pass

    return "\n".join(lines)


def ask_openai(question: str, context: str, model: str = "gpt-4o", code_mode: bool = False) -> str:
    """Ask OpenAI using a chat completion. Expects API key in env or session state.
    If code_mode is True, the assistant should return a single Python fenced code block
    that sets a variable named `result` (and optionally `fig` for Plotly).
    """
    if not _OPENAI_AVAILABLE:
        return "OpenAI Python package is not installed. Please run: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        return "Missing OPENAI_API_KEY. Add it to environment or enter it in the field above."

    try:
        client = OpenAI(api_key=api_key)
        if code_mode:
            system_prompt = (
                "You are a helpful data analyst writing Python to analyze a pandas DataFrame named df. "
                "Use ONLY the provided context to infer column names/types and avoid external I/O or network access. "
                "Return a single fenced Python code block only (no prose), which: "
                "1) computes the answer using the DataFrame df, "
                "2) assigns the main output to a variable named result (DataFrame/Series/scalar), and "
                "3) optionally assigns a Plotly figure to a variable named fig OR a Matplotlib figure to a variable named mpl_fig. "
                "Do not import modules. Use provided pd/np/px/plt if needed. "
                "Do NOT call fig.show() or plt.show(); Streamlit will render the figure. "
                "Use the variable name df exactly as provided. If using pyplot without explicit figure management, set mpl_fig = plt.gcf() at the end."
            )
            user_prompt = (
                f"Context about df:\n\n{context}\n\n"
                f"Task: Write Python code to answer the user's question on df. Question: {question}"
            )
        else:
            system_prompt = (
                "You are a helpful data analyst. Use ONLY the provided context to answer the user's question. "
                "If the answer is not derivable from the context, say 'I cannot determine from the provided data.' "
                "When relevant, include concise reasoning and, if helpful, short pandas code snippets to reproduce the answer."
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
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI request failed: {e}"


def extract_python_code(text: str) -> str:
    """Extract the first Python fenced code block from the LLM text."""
    if not text:
        return ""
    fences = [
        ("```python", "```"),
        ("```py", "```"),
        ("```", "```"),
    ]
    for start, end in fences:
        if start in text:
            start_idx = text.find(start) + len(start)
            end_idx = text.find(end, start_idx)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()
    return ""


def run_user_code(code: str, df: pd.DataFrame):
    """Execute user-provided code in a restricted environment.
    Exposes df, pd, np, px, go, plt. Requires code to optionally set `result` and/or
    `fig` (Plotly) or `mpl_fig` (Matplotlib).
    Returns (env, stdout_text, error_text).
    """
    safe_builtins = {
        'len': len, 'range': range, 'min': min, 'max': max, 'sum': sum,
        'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'abs': abs, 'round': round,
        'print': print, 'any': any, 'all': all, 'map': map, 'filter': filter
    }
    g = {
        '__builtins__': safe_builtins,
        'pd': pd, 'np': np, 'px': px, 'go': go, 'plt': plt,
    }
    l = {'df': df}
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, g, l)
        return l, stdout_buf.getvalue(), None
    except Exception:
        tb = traceback.format_exc(limit=2)
        return l, stdout_buf.getvalue(), tb

# Build a compact context from code execution outputs for the prescriptive summary step
def build_run_context(question: str, code: str, env: dict, stdout_text: str, err_text: str) -> str:
    lines = []
    lines.append("QUESTION")
    lines.append(question)
    lines.append("")
    lines.append("CODE USED")
    lines.append(code)
    lines.append("")
    if err_text:
        lines.append("EXECUTION ERROR")
        lines.append(err_text)
        return "\n".join(lines)

    # Summarize result
    res = env.get('result', None)
    fig_present = (env.get('fig', None) is not None) or (env.get('mpl_fig', None) is not None)
    lines.append("RESULT SUMMARY")
    if res is None:
        lines.append("No variable named 'result' was produced.")
    elif isinstance(res, pd.DataFrame):
        lines.append(f"DataFrame shape: {res.shape}")
        cols = list(res.columns)
        lines.append(f"Columns ({min(len(cols), 20)} shown): {cols[:20]}")
        # Include a small sample to ground the LLM
        sample_n = min(10, len(res))
        try:
            sample_csv = res.head(sample_n).to_csv(index=False)
            lines.append("Sample rows (CSV):")
            lines.append(sample_csv)
        except Exception:
            pass
    elif isinstance(res, pd.Series):
        lines.append(f"Series length: {len(res)}")
        try:
            lines.append("Head:")
            lines.append(res.head(10).to_string())
        except Exception:
            pass
    else:
        # scalar or other types
        try:
            lines.append(f"Value: {str(res)}")
        except Exception:
            pass

    # Stdout
    if stdout_text:
        lines.append("")
        lines.append("CONSOLE OUTPUT")
        # Truncate very long prints
        if len(stdout_text) > 4000:
            lines.append(stdout_text[:4000] + "\n...[truncated]...")
        else:
            lines.append(stdout_text)

    # Figure presence
    lines.append("")
    lines.append(f"FIGURE PRESENT: {fig_present}")
    return "\n".join(lines)

if uploaded_file is not None or use_example:
    # Load workbook (all sheets)
    workbook = load_workbook(uploaded_file if uploaded_file else None)
    if workbook:
        sheet_names = list(workbook.keys())
        # Sheet selector
        with st.sidebar:
            st.header("📄 Sheet")
            selected_sheet = st.selectbox("Select sheet", options=sheet_names, index=0)
            st.success(f"✅ Loaded {len(sheet_names)} sheets: {', '.join(sheet_names)}")

        df = workbook[selected_sheet].copy()
        schema = infer_schema(df, selected_sheet)

        # Sidebar filters
        with st.sidebar:
            st.header("🎛️ Filters")
            # Date filter
            date_col = schema.get('date_col')
            if date_col and date_col in df.columns:
                min_dt = pd.to_datetime(df[date_col]).min()
                max_dt = pd.to_datetime(df[date_col]).max()
                if pd.isna(min_dt) or pd.isna(max_dt):
                    date_range = ()
                else:
                    date_range = st.date_input(
                        f"Date range ({date_col})",
                        value=(min_dt.date(), max_dt.date()),
                        min_value=min_dt.date(),
                        max_value=max_dt.date()
                    )
            else:
                date_range = ()

            # Status
            statuses = None
            if schema.get('status_col') and schema['status_col'] in df.columns:
                options = sorted([x for x in df[schema['status_col']].dropna().unique()])
                statuses = st.multiselect("Status", options=options, default=[])

            # Department
            departments = None
            if schema.get('dept_col') and schema['dept_col'] in df.columns:
                options = sorted([x for x in df[schema['dept_col']].dropna().unique()])
                departments = st.multiselect("Department", options=options, default=[])

            # Location
            locations = None
            if schema.get('loc_col') and schema['loc_col'] in df.columns:
                options = sorted([x for x in df[schema['loc_col']].dropna().unique()])
                locations = st.multiselect("Location", options=options, default=[])

            # Category
            categories = None
            if schema.get('category_col') and schema['category_col'] in df.columns:
                options = sorted([x for x in df[schema['category_col']].dropna().unique()])
                categories = st.multiselect("Category", options=options, default=[])

        # Apply filters
        filtered_df = apply_filters(df, schema, date_range, statuses, departments, locations, categories)

        # Tabs (added Overall across-sheets view)
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌐 Overall",
            "📊 Overview",
            "🔎 Deep Dive",
            "📋 Data Table",
            "📑 Summary Report",
            "🧠 HSE Data Agent",
        ])

        # Tab 0: Overall (across all sheets)
        with tab0:
            st.subheader("Overall – Incidents vs Hazards vs Relationships")
            # Detect sheets by keyword
            hazard_sheet = next((s for s in sheet_names if 'hazard' in s.lower()), None)
            incident_sheet = next((s for s in sheet_names if 'incident' in s.lower()), None)
            rel_sheet = next((s for s in sheet_names if 'relationship' in s.lower()), None)

            df_haz = workbook.get(hazard_sheet) if hazard_sheet else None
            df_inc = workbook.get(incident_sheet) if incident_sheet else None
            df_rel = workbook.get(rel_sheet) if rel_sheet else None

            hazard_count = len(df_haz) if df_haz is not None else 0
            incident_count = len(df_inc) if df_inc is not None else 0
            relationships_count = len(df_rel) if df_rel is not None else 0

            # Try to infer columns in Relationships sheet that map Hazard and Incident
            rel_haz_nunique = 0
            rel_inc_nunique = 0
            if df_rel is not None and len(df_rel.columns) > 0:
                cols_lower = {c: str(c).lower() for c in df_rel.columns}
                haz_rel_cols = [c for c, lc in cols_lower.items() if 'hazard' in lc]
                inc_rel_cols = [c for c, lc in cols_lower.items() if 'incident' in lc]
                haz_rel_col = haz_rel_cols[0] if haz_rel_cols else None
                inc_rel_col = inc_rel_cols[0] if inc_rel_cols else None
                if haz_rel_col:
                    rel_haz_nunique = df_rel[haz_rel_col].nunique(dropna=True)
                if inc_rel_col:
                    rel_inc_nunique = df_rel[inc_rel_col].nunique(dropna=True)

            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Total Incidents", f"{incident_count:,}")
            with c2:
                st.metric("Total Hazards", f"{hazard_count:,}")
            with c3:
                st.metric("Relationships (rows)", f"{relationships_count:,}")
            with c4:
                st.metric("Incidents in Relationships", f"{rel_inc_nunique:,}")
            with c5:
                st.metric("Hazards in Relationships", f"{rel_haz_nunique:,}")

            st.markdown("---")

            # Bar chart summarizing counts
            summary_df = pd.DataFrame({
                'Type': ['Incidents', 'Hazards', 'Relationships'],
                'Count': [incident_count, hazard_count, relationships_count]
            })
            fig_overall = px.bar(summary_df, x='Type', y='Count', title='Overall Counts Across Workbook')
            fig_overall.update_layout(height=360)
            st.plotly_chart(fig_overall, width='stretch')

            # Linked vs Unlinked (stacked) using Relationships sheet
            st.subheader("Linked vs Unlinked")
            stacked_rows = []

            # Identify id columns in Hazard and Incident sheets
            hazard_id_col = None
            incident_id_col = None
            if df_haz is not None:
                for c in df_haz.columns:
                    lc = str(c).lower()
                    if ('hazard' in lc) and ('id' in lc):
                        hazard_id_col = c
                        break
            if df_inc is not None:
                for c in df_inc.columns:
                    lc = str(c).lower()
                    if ('incident' in lc) and ('id' in lc):
                        incident_id_col = c
                        break

            # Build linked/unlinked counts for Hazards
            if df_haz is not None and hazard_id_col and df_rel is not None:
                linked_haz_ids = set()
                if haz_rel_col:
                    linked_haz_ids = set(pd.Series(df_rel[haz_rel_col]).dropna().astype(str).unique())
                total_haz_ids = set(pd.Series(df_haz[hazard_id_col]).dropna().astype(str).unique())
                linked_haz_count = len(total_haz_ids.intersection(linked_haz_ids))
                unlinked_haz_count = max(0, len(total_haz_ids) - linked_haz_count)
                stacked_rows += [
                    {"Type": "Hazards", "Link": "Linked", "Count": linked_haz_count},
                    {"Type": "Hazards", "Link": "Unlinked", "Count": unlinked_haz_count},
                ]
            elif df_haz is not None:
                # Fallback: no IDs detected; show total as Unlinked-only
                stacked_rows.append({"Type": "Hazards", "Link": "Total", "Count": hazard_count})

            # Build linked/unlinked counts for Incidents
            if df_inc is not None and incident_id_col and df_rel is not None:
                linked_inc_ids = set()
                if inc_rel_col:
                    linked_inc_ids = set(pd.Series(df_rel[inc_rel_col]).dropna().astype(str).unique())
                total_inc_ids = set(pd.Series(df_inc[incident_id_col]).dropna().astype(str).unique())
                linked_inc_count = len(total_inc_ids.intersection(linked_inc_ids))
                unlinked_inc_count = max(0, len(total_inc_ids) - linked_inc_count)
                stacked_rows += [
                    {"Type": "Incidents", "Link": "Linked", "Count": linked_inc_count},
                    {"Type": "Incidents", "Link": "Unlinked", "Count": unlinked_inc_count},
                ]
            elif df_inc is not None:
                stacked_rows.append({"Type": "Incidents", "Link": "Total", "Count": incident_count})

            if stacked_rows:
                stacked_df = pd.DataFrame(stacked_rows)
                fig_stack = px.bar(
                    stacked_df, x="Type", y="Count", color="Link",
                    barmode="stack", title="Linked vs Unlinked by Type"
                )
                fig_stack.update_layout(height=360)
                st.plotly_chart(fig_stack, width='stretch')
            else:
                st.info("No sufficient ID columns detected to compute linked vs unlinked.")

        # Tab 1: Overview
        with tab1:
            st.subheader(f"Overview – {selected_sheet}")
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Records", f"{len(filtered_df):,}")
            with col2:
                sev = schema.get('severity_col')
                if sev and sev in filtered_df.columns and filtered_df[sev].notna().any():
                    st.metric("Avg Severity", _fmt_num(filtered_df[sev].mean()))
                else:
                    st.metric("Avg Severity", "–")
            with col3:
                rk = schema.get('risk_col')
                if rk and rk in filtered_df.columns and filtered_df[rk].notna().any():
                    st.metric("Avg Risk", _fmt_num(filtered_df[rk].mean()))
                else:
                    st.metric("Avg Risk", "–")
            with col4:
                costc = schema.get('cost_col')
                if costc and costc in filtered_df.columns:
                    st.metric("Est. Cost Impact", _fmt_num(filtered_df[costc].sum()))
                else:
                    st.metric("Est. Cost Impact", "–")
            with col5:
                mhc = schema.get('manhours_col')
                if mhc and mhc in filtered_df.columns:
                    st.metric("Est. Manhours", _fmt_num(filtered_df[mhc].sum()))
                else:
                    st.metric("Est. Manhours", "–")

            st.markdown("---")

            # Charts
            colA, colB = st.columns(2)
            with colA:
                # Time series
                dc = schema.get('date_col')
                if dc and dc in filtered_df.columns and filtered_df[dc].notna().any():
                    ts = filtered_df.copy()
                    ts['_dt'] = pd.to_datetime(ts[dc])
                    series = ts.groupby(ts['_dt'].dt.to_period('D')).size().reset_index(name='count')
                    series['_dt'] = series['_dt'].dt.to_timestamp()
                    fig_ts = px.line(series, x='_dt', y='count', title=f"Records over time ({dc})")
                    fig_ts.update_layout(height=300)
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No date column detected for time series.")

                # Top Departments
                depc = schema.get('dept_col')
                vc = safe_value_counts(filtered_df, depc)
                if not vc.empty:
                    fig_dep = px.bar(x=vc.values, y=vc.index, orientation='h', title="Top Departments")
                    fig_dep.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_dep, use_container_width=True)
                else:
                    st.info("No department column available.")

            with colB:
                # Status distribution
                sc = schema.get('status_col')
                vc = safe_value_counts(filtered_df, sc)
                if not vc.empty:
                    fig_status = px.pie(values=vc.values, names=vc.index, title="Status Distribution")
                    fig_status.update_layout(height=300)
                    st.plotly_chart(fig_status, use_container_width=True)
                else:
                    st.info("No status column available.")

                # Top Locations
                locc = schema.get('loc_col')
                vc = safe_value_counts(filtered_df, locc)
                if not vc.empty:
                    fig_loc = px.bar(x=vc.values, y=vc.index, orientation='h', title="Top Locations")
                    fig_loc.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_loc, use_container_width=True)
                else:
                    st.info("No location column available.")

        # Tab 2: Deep Dive
        with tab2:
            st.subheader("Deep Dive")
            col1, col2 = st.columns(2)
            with col1:
                # Category distribution
                cc = schema.get('category_col')
                vc = safe_value_counts(filtered_df, cc)
                if not vc.empty:
                    fig_cat = px.bar(x=vc.index, y=vc.values, title="Category Distribution")
                    fig_cat.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_cat, use_container_width=True)
                # Consequence
                cons = schema.get('consequence_col')
                vc = safe_value_counts(filtered_df, cons)
                if not vc.empty:
                    fig_cons = px.pie(values=vc.values, names=vc.index, title="Worst/Relevant Consequences")
                    st.plotly_chart(fig_cons, use_container_width=True)

            with col2:
                # Severity and Risk histograms if present
                sev = schema.get('severity_col')
                if sev and sev in filtered_df.columns and filtered_df[sev].notna().any():
                    fig_sev = px.histogram(filtered_df, x=sev, nbins=20, title="Severity Score Distribution")
                    st.plotly_chart(fig_sev, use_container_width=True)
                rk = schema.get('risk_col')
                if rk and rk in filtered_df.columns and filtered_df[rk].notna().any():
                    fig_risk = px.histogram(filtered_df, x=rk, nbins=20, title="Risk Score Distribution")
                    st.plotly_chart(fig_risk, use_container_width=True)

            # Timeliness metrics
            col3, col4 = st.columns(2)
            with col3:
                rd = schema.get('reporting_delay_col')
                if rd and rd in filtered_df.columns and filtered_df[rd].notna().any():
                    st.subheader("Reporting Delay (days)")
                    colA, colB, colC = st.columns(3)
                    with colA: st.metric("Avg", _fmt_num(filtered_df[rd].mean()))
                    with colB: st.metric("P90", _fmt_num(filtered_df[rd].quantile(0.9)))
                    with colC: st.metric("Max", _fmt_num(filtered_df[rd].max()))
                    st.plotly_chart(px.histogram(filtered_df, x=rd, nbins=30, title="Reporting Delay Histogram"), use_container_width=True)
            with col4:
                rt = schema.get('resolution_time_col')
                if rt and rt in filtered_df.columns and filtered_df[rt].notna().any():
                    st.subheader("Resolution Time (days)")
                    colA, colB, colC = st.columns(3)
                    with colA: st.metric("Avg", _fmt_num(filtered_df[rt].mean()))
                    with colB: st.metric("P90", _fmt_num(filtered_df[rt].quantile(0.9)))
                    with colC: st.metric("Max", _fmt_num(filtered_df[rt].max()))
                    st.plotly_chart(px.histogram(filtered_df, x=rt, nbins=30, title="Resolution Time Histogram"), use_container_width=True)

            # Data quality flags
            flags = schema.get('flags', [])
            if flags:
                st.subheader("Data Quality Flags")
                cols = st.columns(len(flags))
                for i, fl in enumerate(flags):
                    cnt = filtered_df[fl].sum() if fl in filtered_df.columns else 0
                    with cols[i]:
                        st.metric(fl.replace('_', ' ').title(), f"{int(cnt):,}")

        # Tab 3: Data Table
        with tab3:
            st.header("📋 Data Table")
            search = st.text_input("🔍 Search in data", "")
            if search:
                mask = filtered_df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
                display_df = filtered_df[mask]
            else:
                display_df = filtered_df
            st.info(f"Showing {len(display_df):,} rows")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_na = st.checkbox("Show rows with missing values only", False)
            with col2:
                show_recent = st.checkbox("Sort by latest date first", True)
            with col3:
                rows_to_show = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)
            if show_na:
                display_df = display_df[display_df.isna().any(axis=1)]
            dcol = schema.get('date_col')
            if show_recent and dcol and dcol in display_df.columns:
                display_df = display_df.sort_values(dcol, ascending=False, na_position='last')
            total_pages = len(display_df) // rows_to_show + (1 if len(display_df) % rows_to_show > 0 else 0)
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
            start_idx = (page - 1) * rows_to_show
            end_idx = min(start_idx + rows_to_show, len(display_df))
            st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True, height=600)
            # Download
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_{selected_sheet.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Tab 4: Summary Report
        with tab4:
            st.header("📑 Summary Report")
            dcol = schema.get('date_col')
            if dcol and dcol in filtered_df.columns and filtered_df[dcol].notna().any():
                start_time = pd.to_datetime(filtered_df[dcol]).min()
                end_time = pd.to_datetime(filtered_df[dcol]).max()
                duration_days = (end_time - start_time).days if (pd.notna(end_time) and pd.notna(start_time)) else 0
            else:
                start_time = end_time = None
                duration_days = 0

            sev = schema.get('severity_col')
            rk = schema.get('risk_col')
            costc = schema.get('cost_col')
            mhc = schema.get('manhours_col')

            report_lines = [
                f"# HSE DATA ANALYSIS REPORT – {selected_sheet}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## BASIC STATISTICS",
                f"- Total Records: {len(filtered_df):,}",
            ]
            if start_time is not None and end_time is not None:
                report_lines += [
                    f"- Time Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
                    f"- Duration: {duration_days} days",
                ]
            if sev and sev in filtered_df.columns and filtered_df[sev].notna().any():
                report_lines.append(f"- Avg Severity: {filtered_df[sev].mean():.2f}")
            if rk and rk in filtered_df.columns and filtered_df[rk].notna().any():
                report_lines.append(f"- Avg Risk: {filtered_df[rk].mean():.2f}")
            if costc and costc in filtered_df.columns:
                report_lines.append(f"- Est. Cost Impact (sum): {_fmt_num(filtered_df[costc].sum())}")
            if mhc and mhc in filtered_df.columns:
                report_lines.append(f"- Est. Manhours (sum): {_fmt_num(filtered_df[mhc].sum())}")

            # Top items
            for label, col in [("STATUS", schema.get('status_col')), ("CATEGORY", schema.get('category_col')), ("DEPARTMENT", schema.get('dept_col')), ("LOCATION", schema.get('loc_col'))]:
                if col and col in filtered_df.columns:
                    report_lines.append("")
                    report_lines.append(f"## TOP {label}")
                    for i, (val, count) in enumerate(filtered_df[col].value_counts().head(5).items(), 1):
                        report_lines.append(f"{i}. {val}: {count:,} ({count/len(filtered_df)*100:.1f}%)")

            report_text = "\n".join(report_lines)
            st.markdown(report_text)
            st.download_button(
                label="📥 Download Report as Markdown",
                data=report_text,
                file_name=f"hse_report_{selected_sheet.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        # Tab 5: AI Agent
        with tab5:
            st.header("🧠 HSE Data Agent")
            st.caption("Ask a question. The agent will generate Python, run it on the filtered data, then provide a prescriptive summary.")
            question = st.text_area("Your question about the filtered data", placeholder="e.g., Which departments have the highest risk and what should we focus on?")
            sample_rows = st.slider("Sample rows included in context", 3, 10, 5)
            show_ctx = st.checkbox("Show context sent to AI", value=False)
            show_code = st.checkbox("Show generated code", value=True)
            run_agent = st.button("Run HSE Data Agent", type="primary")

            if run_agent and question:
                with st.spinner("Generating analysis code from your question..."):
                    context_text = build_ai_context(filtered_df, sample_rows=sample_rows)
                    code_resp = ask_openai(question, context_text, code_mode=True)
                if show_ctx:
                    with st.expander("Context sent to AI"):
                        st.code(context_text)
                code_block = extract_python_code(code_resp)
                if not code_block:
                    st.error("The agent did not produce runnable Python. Showing raw response below.")
                    st.code(code_resp)
                    st.stop()
                if show_code:
                    st.markdown("### Generated code")
                    st.code(code_block, language="python")
                with st.spinner("Running code on filtered data..."):
                    env, stdout_text, err = run_user_code(code_block, filtered_df.copy())
                if err:
                    st.error("Execution failed. See traceback:")
                    st.code(err)
                else:
                    if 'fig' in env and env['fig'] is not None:
                        try:
                            st.plotly_chart(env['fig'], use_container_width=True)
                        except Exception:
                            st.warning("'fig' was not a valid Plotly figure.")
                    if 'mpl_fig' in env and env['mpl_fig'] is not None:
                        try:
                            st.pyplot(env['mpl_fig'])
                            try:
                                plt.close(env['mpl_fig'])
                            except Exception:
                                pass
                        except Exception:
                            st.warning("'mpl_fig' was not a valid Matplotlib figure.")
                    # Fallbacks
                    if (("fig" not in env) or (env.get('fig') is None)):
                        try:
                            for _k, _v in env.items():
                                if isinstance(_v, go.Figure):
                                    st.plotly_chart(_v, use_container_width=True)
                                    break
                        except Exception:
                            pass
                    if (("mpl_fig" not in env) or (env.get('mpl_fig') is None)):
                        try:
                            fignums = plt.get_fignums()
                            if fignums:
                                last_num = fignums[-1]
                                _fig = plt.figure(last_num)
                                st.pyplot(_fig)
                                try:
                                    plt.close(_fig)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    if 'result' in env:
                        res = env['result']
                        st.markdown("#### Result")
                        if isinstance(res, pd.DataFrame):
                            st.dataframe(res, use_container_width=True, height=500)
                        elif isinstance(res, pd.Series):
                            st.write(res)
                        else:
                            st.write(res)
                    if stdout_text:
                        with st.expander("Console output"):
                            st.code(stdout_text)
                # Prescriptive summary
                with st.spinner("Generating prescriptive summary and recommendations..."):
                    run_ctx = build_run_context(question, code_block, env, stdout_text, err)
                    summary_question = (
                        "Provide a concise, prescriptive analysis and recommendations based on the outputs above. "
                        "Structure as: Findings, Recommendations, Potential next steps. If a figure is present, reference it."
                    )
                    summary_resp = ask_openai(summary_question, run_ctx, code_mode=False)
                st.markdown("### Prescriptive summary")
                st.markdown(summary_resp)

    else:
        st.error("Failed to load workbook. Please upload a valid .xlsx or place 'EPCL_VEHS_Data_Processed.xlsx' alongside this app.")

else:
    # Welcome screen when no data is loaded
    st.info("👈 Please upload an Excel workbook (.xlsx) with EPCL HSE data using the sidebar. You can also tick 'Use local example file' if EPCL_VEHS_Data_Processed.xlsx is present next to this app.")

    # Expected workbook structure
    st.subheader("Expected Workbook Structure")
    st.markdown("""
    The workbook should contain some of the following sheets (names can vary slightly):
    - Incident
    - Hazard ID
    - Audit
    - Audit Findings
    - Inspection

    Key columns the dashboard looks for (case-insensitive):
    - Dates: occurrence_date, reported_date, entered_date, completion_date, start_date, entered_closed
    - Status: status, audit_status
    - Category: category, incident_type, violation_type_hazard_id, audit_category
    - Department: department, sub_department
    - Location: location, sublocation, location.1
    - Metrics: severity_score, risk_score, department_avg_risk, estimated_cost_impact, estimated_manhours_impact, reporting_delay_days, resolution_time_days

    The app automatically detects available columns per sheet and adapts filters and visuals.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>HSE Analytics Dashboard v1.0 | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)