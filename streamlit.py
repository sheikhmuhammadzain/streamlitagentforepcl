import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from analytics.hazard_incident import render_conversion_page, create_conversion_metrics_card
from analytics.wordclouds import (
    get_incident_hazard_department_words,
    create_python_wordcloud_image,
)
from analytics.maps import add_coordinates_to_df as maps_add_coords, build_combined_map_html
import streamlit.components.v1 as components
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen, MousePosition, MeasureControl
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

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
    page_title="Safety Co-pilot",
    page_icon="🧯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
      <style>
      /* Import Geist Sans font faces */
      @font-face {
          font-family: 'Geist Sans';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-sans/Geist-Regular.woff2') format('woff2');
          font-weight: 400;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Sans';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-sans/Geist-Medium.woff2') format('woff2');
          font-weight: 500;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Sans';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-sans/Geist-SemiBold.woff2') format('woff2');
          font-weight: 600;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Sans';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-sans/Geist-Bold.woff2') format('woff2');
          font-weight: 700;
          font-style: normal;
          font-display: swap;
      }

      /* Apply Geist Sans globally */
      html, body, .stApp, [class*="css"] {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          text-rendering: optimizeLegibility;
      }

      /* Headings */
      h1, h2, h3, h4, h5, h6 {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
          font-weight: 600;
          letter-spacing: -0.015em;
      }

      /* Markdown and text */
      .stMarkdown, .stText, p {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
          line-height: 1.6;
          letter-spacing: -0.011em;
      }

      /* Buttons */
      .stButton > button {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
          font-weight: 500;
      }

      /* Sidebar */
      [data-testid="stSidebar"], .css-1d391kg, [class*="sidebar"] {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
      }

      /* Inputs */
      .stTextInput input,
      .stSelectbox select,
      .stTextArea textarea,
      [data-baseweb="select"] div,
      [data-baseweb="input"] input {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
      }

      /* Tables and metrics */
      .stTable, .dataframe, [data-testid="stMetric"], .stMetric {
          font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
      }

      /* Layout spacing refinements */
      [data-testid="stAppViewContainer"] {
          padding-top: 0rem !important;
      }
      .block-container {
          padding-top: 0rem !important;
          padding-bottom: 0.75rem !important;
      }
      h1, h2, h3 {
          margin-top: 0.15rem !important;
          margin-bottom: 0.15rem !important;
      }
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
          margin-top: 0.15rem !important;
          margin-bottom: 0.15rem !important;
      }
      /* Collapse the default Streamlit header area */
      header[data-testid="stHeader"] {
          background: transparent;
          height: 0 !important;
          min-height: 0 !important;
          padding: 0 !important;
      }
      /* Ensure the very first element doesn't add extra top space */
      .block-container > :first-child { margin-top: 0 !important; }

      /* Tabs polish */
      .stTabs [data-baseweb="tab-list"] {
          margin-bottom: 0.25rem;
          border-bottom: 1px solid #E5E7EB;
          gap: 4px;
      }
      .stTabs [data-baseweb="tab"] {
          padding: 6px 10px;
      }
      .stTabs [aria-selected="true"] {
          border-bottom: 2px solid #16A34A !important;
          color: #111827 !important;
      }

      /* Code blocks - Geist Mono (preserve syntax colors) */
      @font-face {
          font-family: 'Geist Mono';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-mono/GeistMono-Regular.woff2') format('woff2');
          font-weight: 400;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Mono';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-mono/GeistMono-Medium.woff2') format('woff2');
          font-weight: 500;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Mono';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-mono/GeistMono-SemiBold.woff2') format('woff2');
          font-weight: 600;
          font-style: normal;
          font-display: swap;
      }
      @font-face {
          font-family: 'Geist Mono';
          src: url('https://cdn.jsdelivr.net/npm/geist@1.0.0/dist/fonts/geist-mono/GeistMono-Bold.woff2') format('woff2');
          font-weight: 700;
          font-style: normal;
          font-display: swap;
      }
      div[data-testid="stCodeBlock"] pre,
      pre,
      code,
      kbd,
      samp {
          font-family: 'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace !important;
          font-variant-ligatures: none;
      }

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
st.title("Safety Co-pilot")
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

# Coerce a pandas Series to numeric days if it's timedelta/datetime-like
def _to_days(series: pd.Series) -> pd.Series:
    try:
        if series is None:
            return series
        s = series
        if np.issubdtype(s.dtype, np.timedelta64):
            return s.dt.total_seconds() / 86400.0
        if np.issubdtype(s.dtype, np.datetime64):
            # Datetime values cannot be meaningfully compared to day counts; coerce to NaN
            return pd.to_numeric(s, errors='coerce')
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.to_numeric(series, errors='coerce')
# ---------- Cross-sheet helpers ----------
def get_sheet_df(workbook: dict, name_contains: str):
    """Return first sheet DataFrame whose name contains the given token (case-insensitive)."""
    if not workbook:
        return None
    for s in workbook.keys():
        if name_contains.lower() in s.lower():
            return workbook[s]
    return None

# ---------- Advanced analytics chart builders ----------
def create_unified_hse_scorecard(incident_df, hazard_df, audit_df, inspection_df):
    """Single-row KPI indicators only. Other charts are shown on subsequent rows."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Incidents', 'Hazards', 'Audits Completed', 'Inspections'],
        horizontal_spacing=0.08,
    )

    inc_count = len(incident_df) if incident_df is not None else 0
    haz_count = len(hazard_df) if hazard_df is not None else 0
    audits_completed = 0
    if audit_df is not None and 'audit_status' in audit_df.columns:
        audits_completed = (audit_df['audit_status'].astype(str).str.lower() == 'closed').sum()
    insp_count = len(inspection_df) if inspection_df is not None else 0

    # Indicators with large number display
    fig.add_trace(go.Indicator(mode="number", value=inc_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=haz_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=audits_completed, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=3)
    fig.add_trace(go.Indicator(mode="number", value=insp_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=4)

    fig.update_layout(
        title={"text": "Unified HSE Scorecard", "x": 0.01, "xanchor": "left"},
        height=260,
        margin=dict(t=50, l=30, r=20, b=20),
        showlegend=False,
    )
    return fig

def create_hse_performance_index(df):
    if df is None or len(df) == 0 or 'department' not in df.columns:
        return go.Figure()
    # Work on a copy and ensure numeric types
    cp = df.copy()
    for c in ['severity_score','risk_score']:
        if c not in cp.columns:
            cp[c] = np.nan
        cp[c] = pd.to_numeric(cp[c], errors='coerce')
    # Delay and resolution as numeric days
    if 'reporting_delay_days' not in cp.columns:
        cp['reporting_delay_days'] = np.nan
    cp['reporting_delay_days'] = _to_days(cp['reporting_delay_days'])
    if 'resolution_time_days' not in cp.columns:
        cp['resolution_time_days'] = np.nan
    cp['resolution_time_days'] = _to_days(cp['resolution_time_days'])
    # Flags: ensure numeric 0..1
    for c in ['root_cause_is_missing','corrective_actions_is_missing']:
        if c not in cp.columns:
            cp[c] = np.nan
        # Treat truthy as 1, falsy as 0
        cp[c] = pd.to_numeric(cp[c].astype(float), errors='coerce')

    dept_metrics = cp.groupby('department').agg({
        'severity_score': 'mean',
        'risk_score': 'mean',
        'reporting_delay_days': 'mean',
        'resolution_time_days': 'mean',
        'root_cause_is_missing': 'mean',
        'corrective_actions_is_missing': 'mean'
    }).fillna(0)

    # Normalize into 0-100 where higher is better
    sev = dept_metrics['severity_score'].clip(lower=0, upper=5)
    risk = dept_metrics['risk_score'].clip(lower=0, upper=5)
    rep = dept_metrics['reporting_delay_days'].clip(lower=0, upper=30)
    res = dept_metrics['resolution_time_days'].clip(lower=0, upper=60)
    rc_miss = dept_metrics['root_cause_is_missing'].clip(lower=0, upper=1)
    ca_miss = dept_metrics['corrective_actions_is_missing'].clip(lower=0, upper=1)

    idx = (
        (5 - sev)/5 * 0.25 +
        (5 - risk)/5 * 0.25 +
        (30 - rep)/30 * 0.2 +
        (60 - res)/60 * 0.2 +
        (1 - rc_miss) * 0.05 +
        (1 - ca_miss) * 0.05
    ) * 100

    dept_metrics = dept_metrics.assign(hse_index=idx)
    fig = px.bar(
        dept_metrics.reset_index(), x='hse_index', y='department', orientation='h',
        color='hse_index', color_continuous_scale=['red','yellow','green'],
        title='HSE Performance Index by Department (0-100)'
    )
    # Improve readability: larger height, left margin for long department names
    fig.update_layout(
        height=520,
        margin=dict(t=60, l=160, r=40, b=40),
        yaxis=dict(automargin=True)
    )
    return fig

# def create_incident_action_funnel(incident_df, relationships_df):
#     if incident_df is None:
#         return go.Figure()
#     inc_total = len(incident_df)
#     invest_started = incident_df['entered_investigation'].notna().sum() if 'entered_investigation' in incident_df.columns else 0
#     if 'root_cause_is_missing' in incident_df.columns:
#         rc = incident_df['root_cause_is_missing']
#         # Treat NaN as missing (1); identified when value is 0/False
#         rc_num = pd.to_numeric(rc, errors='coerce').fillna(1).astype(int)
#         root_identified = (rc_num == 0).sum()
#     else:
#         root_identified = 0
#     closed_inc = (incident_df['status'].astype(str).str.lower() == 'closed').sum() if 'status' in incident_df.columns else 0
#     with_actions = 0
#     if relationships_df is not None and {'source_type','source_id','target_type'}.issubset(relationships_df.columns):
#         with_actions = relationships_df[relationships_df['source_type'].astype(str).str.lower().eq('incident')]['source_id'].nunique()
#     fig = go.Figure(go.Funnel(
#         y=['Total Incidents','Investigation Started','Root Cause Identified','Corrective Actions Generated','Incidents Closed'],
#         x=[inc_total, invest_started, root_identified, with_actions, closed_inc],
#         textposition='inside', textinfo='value+percent initial',
#         marker={'color':['#16A34A','#34D399','#A7F3D0','#F59E0B','#065F46']}
#     ))
#     fig.update_layout(title='Incident Management Funnel')
#     return fig

def create_risk_calendar_heatmap(df):
    if df is None or 'occurrence_date' not in df.columns or 'department' not in df.columns or 'risk_score' not in df.columns:
        return go.Figure()
    cp = df.copy()
    # Create monthly period, then pivot; convert Periods to strings before plotting
    cp['month'] = pd.to_datetime(cp['occurrence_date'], errors='coerce').dt.to_period('M')
    risk_pivot = cp.pivot_table(values='risk_score', index='department', columns='month', aggfunc='mean')
    # Ensure JSON-serializable labels
    x_labels = risk_pivot.columns.astype(str).tolist()
    y_labels = risk_pivot.index.astype(str).tolist()
    z = risk_pivot.to_numpy()
    fig = px.imshow(
        z,
        labels=dict(x='Month', y='Department', color='Avg Risk Score'),
        x=x_labels,
        y=y_labels,
        color_continuous_scale='RdYlGn_r',
        title='Department Risk Score Evolution',
        aspect='auto',
        text_auto=True,
    )
    fig.update_xaxes(tickangle=-45)
    return fig

def create_psm_breakdown(incident_df):
    if incident_df is None:
        return go.Figure()
    psm_counts = incident_df['psm'].value_counts(dropna=True) if 'psm' in incident_df.columns else pd.Series(dtype=int)
    pse_counts = incident_df['pse_category'].value_counts(dropna=True) if 'pse_category' in incident_df.columns else pd.Series(dtype=int)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['PSM Elements','PSE Categories'], specs=[[{'type':'pie'},{'type':'bar'}]])
    if not psm_counts.empty:
        fig.add_trace(go.Pie(labels=psm_counts.index, values=psm_counts.values, hole=0.4), row=1, col=1)
    if not pse_counts.empty:
        fig.add_trace(go.Bar(x=pse_counts.values, y=pse_counts.index, orientation='h'), row=1, col=2)
    fig.update_layout(title='Process Safety Management Analysis')
    return fig

def create_consequence_matrix(df):
    if df is None or 'actual_consequence_incident' not in df.columns or 'worst_case_consequence_incident' not in df.columns:
        return go.Figure()
    ct = pd.crosstab(df['actual_consequence_incident'], df['worst_case_consequence_incident'])
    fig = px.imshow(ct, labels=dict(x='Worst Case', y='Actual', color='Count'), title='Actual vs Worst Case Consequence Matrix', color_continuous_scale='YlOrRd', text_auto=True)
    return fig

def create_data_quality_metrics(incident_df):
    if incident_df is None:
        return go.Figure()
    fig = make_subplots(rows=2, cols=3, subplot_titles=['Root Cause Missing','Corrective Actions Missing','Reporting Delays','Resolution Times by Status','', ''])
    if 'department' in incident_df.columns and 'root_cause_is_missing' in incident_df.columns:
        missing_rc = incident_df.groupby('department')['root_cause_is_missing'].sum()
        fig.add_trace(go.Bar(x=missing_rc.index, y=missing_rc.values, name='Root Cause Missing'), row=1, col=1)
    if 'department' in incident_df.columns and 'corrective_actions_is_missing' in incident_df.columns:
        missing_ca = incident_df.groupby('department')['corrective_actions_is_missing'].sum()
        fig.add_trace(go.Bar(x=missing_ca.index, y=missing_ca.values, name='Actions Missing'), row=1, col=2)
    if 'reporting_delay_days' in incident_df.columns:
        fig.add_trace(go.Histogram(x=_to_days(incident_df['reporting_delay_days']), nbinsx=30, name='Reporting Delay'), row=1, col=3)
    if {'resolution_time_days','status'}.issubset(incident_df.columns):
        fig.add_trace(go.Box(y=_to_days(incident_df['resolution_time_days']), x=incident_df['status'], name='Resolution by Status'), row=2, col=1)
    fig.update_layout(title='Data Quality Metrics')
    return fig

def create_comprehensive_timeline(df):
    if df is None or 'occurrence_date' not in df.columns:
        return go.Figure()
    cp = df.copy()
    cp['week'] = pd.to_datetime(cp['occurrence_date'], errors='coerce').dt.to_period('W')
    agg_dict = {}
    count_col = 'incident_id' if 'incident_id' in cp.columns else cp.columns[0]
    agg_dict[count_col] = 'count'
    if 'severity_score' in cp.columns:
        agg_dict['severity_score'] = 'mean'
    if 'risk_score' in cp.columns:
        agg_dict['risk_score'] = 'mean'
    if 'estimated_cost_impact' in cp.columns:
        agg_dict['estimated_cost_impact'] = 'sum'
    if 'estimated_manhours_impact' in cp.columns:
        agg_dict['estimated_manhours_impact'] = 'sum'
    agg = cp.groupby('week').agg(agg_dict).reset_index()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Incident Count','Risk & Severity Scores','Cost & Manhour Impact'], row_heights=[0.3,0.35,0.35])
    fig.add_trace(go.Bar(x=agg['week'].astype(str), y=agg[count_col], name='Count'), row=1, col=1)
    if 'severity_score' in agg.columns:
        fig.add_trace(go.Scatter(x=agg['week'].astype(str), y=agg['severity_score'], name='Severity', line=dict(color='red')), row=2, col=1)
    if 'risk_score' in agg.columns:
        fig.add_trace(go.Scatter(x=agg['week'].astype(str), y=agg['risk_score'], name='Risk', line=dict(color='orange')), row=2, col=1)
    if 'estimated_cost_impact' in agg.columns:
        fig.add_trace(go.Bar(x=agg['week'].astype(str), y=agg['estimated_cost_impact'], name='Cost ($)', marker_color='green'), row=3, col=1)
    if 'estimated_manhours_impact' in agg.columns:
        fig.add_trace(go.Bar(x=agg['week'].astype(str), y=agg['estimated_manhours_impact'], name='Manhours', marker_color='#34D399'), row=3, col=1)
    fig.update_layout(title='Comprehensive HSE Timeline')
    return fig

def create_audit_inspection_tracker(audit_df, inspection_df):
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Audit Status Over Time','Inspection Status Over Time'])
    if audit_df is not None and {'start_date','audit_status'}.issubset(audit_df.columns):
        aud = audit_df.copy()
        aud['_m'] = pd.to_datetime(aud['start_date'], errors='coerce').dt.to_period('M')
        timeline = aud.groupby([aud['_m'], 'audit_status']).size().unstack(fill_value=0)
        for status in timeline.columns:
            fig.add_trace(go.Bar(x=timeline.index.astype(str), y=timeline[status], name=str(status)), row=1, col=1)
    if inspection_df is not None and {'start_date','audit_status'}.issubset(inspection_df.columns):
        ins = inspection_df.copy()
        ins['_m'] = pd.to_datetime(ins['start_date'], errors='coerce').dt.to_period('M')
        timeline = ins.groupby([ins['_m'], 'audit_status']).size().unstack(fill_value=0)
        for status in timeline.columns:
            fig.add_trace(go.Bar(x=timeline.index.astype(str), y=timeline[status], name=str(status)), row=2, col=1)
    fig.update_layout(barmode='stack', title='Audit & Inspection Compliance Tracking')
    return fig

def create_location_risk_treemap(df):
    if df is None or not {'location','sublocation'}.issubset(df.columns):
        return go.Figure()
    cp = df.copy()
    # Provide defaults if metrics are missing
    if 'incident_id' not in cp.columns: cp['incident_id'] = 1
    for c in ['severity_score','risk_score','estimated_cost_impact']:
        if c not in cp.columns: cp[c] = np.nan
    location_data = cp.groupby(['location','sublocation']).agg({
        'incident_id': 'count', 'severity_score':'mean','risk_score':'mean','estimated_cost_impact':'sum'
    }).reset_index()
    location_data['size'] = location_data['incident_id']
    location_data['hover_text'] = (
        'Count: ' + location_data['incident_id'].astype(str) +
        '<br>Avg Severity: ' + location_data['severity_score'].round(2).astype(str) +
        '<br>Total Cost: ' + location_data['estimated_cost_impact'].round(0).astype(str)
    )
    fig = px.treemap(location_data, path=['location','sublocation'], values='size', color='risk_score', hover_data={'hover_text':True}, color_continuous_scale='RdYlGn_r', title='Location Risk Map (Size=Count, Color=Risk)')
    return fig

def create_department_spider(df):
    if df is None or 'department' not in df.columns:
        return go.Figure()
    cp = df.copy()
    for col in ['severity_score','risk_score','reporting_delay_days','resolution_time_days','root_cause_is_missing','corrective_actions_is_missing']:
        if col not in cp.columns: cp[col] = np.nan
    # Coerce delay/resolution to days
    cp['reporting_delay_days'] = _to_days(cp['reporting_delay_days'])
    cp['resolution_time_days'] = _to_days(cp['resolution_time_days'])
    dept_metrics = cp.groupby('department').agg({
        'severity_score': lambda x: 5 - np.nanmean(x),
        'risk_score': lambda x: 5 - np.nanmean(x),
        'reporting_delay_days': lambda x: max(0, 30 - np.nanmean(x)),
        'resolution_time_days': lambda x: max(0, 60 - np.nanmean(x)),
        'root_cause_is_missing': lambda x: 100 * (1 - np.nanmean(x)),
        'corrective_actions_is_missing': lambda x: 100 * (1 - np.nanmean(x)),
    }).fillna(0)
    # Normalize 0-100
    for col in dept_metrics.columns:
        m = dept_metrics[col].max()
        if m and m > 0:
            dept_metrics[col] = (dept_metrics[col] / m) * 100
    fig = go.Figure()
    labels = ['Low Severity','Low Risk','Fast Reporting','Quick Resolution','Root Cause ID','Actions Taken']
    for dept in dept_metrics.index[:5]:
        fig.add_trace(go.Scatterpolar(r=dept_metrics.loc[dept].values, theta=labels, fill='toself', name=str(dept)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), title='Department HSE Performance Radar', showlegend=True)
    return fig

def create_violation_analysis(hazard_df):
    if hazard_df is None:
        return go.Figure()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Violation Types','Consequences Distribution','Reporting Delays','Department Violations'],
        specs=[[{'type':'xy'}, {'type':'domain'}],
               [{'type':'xy'}, {'type':'xy'}]]  # heatmap also works in an 'xy' subplot
    )
    if 'violation_type_hazard_id' in hazard_df.columns:
        vc = hazard_df['violation_type_hazard_id'].value_counts()
        fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation='h'), row=1, col=1)
    if 'worst_case_consequence_potential_hazard_id' in hazard_df.columns:
        vc = hazard_df['worst_case_consequence_potential_hazard_id'].value_counts()
        if not vc.empty:
            try:
                fig.add_trace(go.Pie(labels=vc.index, values=vc.values), row=1, col=2)
            except ValueError:
                # Fallback: render as a bar if subplot type is not domain due to environment quirks
                fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation='h', name='Consequences'), row=1, col=2)
    if 'reporting_delay_days' in hazard_df.columns:
        fig.add_trace(go.Histogram(x=_to_days(hazard_df['reporting_delay_days']), nbinsx=20), row=2, col=1)
    if {'department','violation_type_hazard_id'}.issubset(hazard_df.columns):
        ctab = pd.crosstab(hazard_df['department'], hazard_df['violation_type_hazard_id'])
        if ctab.size > 0:
            fig.add_trace(go.Heatmap(z=ctab.values, x=ctab.columns, y=ctab.index, colorscale='YlOrRd'), row=2, col=2)
    fig.update_layout(title='Hazard Violation Analysis')
    return fig

def create_cost_prediction_analysis(df):
    if df is None or 'estimated_cost_impact' not in df.columns:
        return go.Figure()
    numeric_cols = [c for c in ['severity_score','risk_score','reporting_delay_days','resolution_time_days','estimated_manhours_impact'] if c in df.columns]
    sub = df[numeric_cols + ['estimated_cost_impact']].dropna()
    fig = make_subplots(rows=2, cols=2, subplot_titles=['Cost Correlations','Cost vs Severity','Cost vs Risk','Cost by Category'])
    if not sub.empty and len(numeric_cols) > 0:
        corrs = sub.corr(numeric_only=True)['estimated_cost_impact'].drop('estimated_cost_impact', errors='ignore')
        if not corrs.empty:
            fig.add_trace(go.Bar(x=corrs.values, y=corrs.index, orientation='h', marker_color=corrs.values, marker_colorscale='RdBu'), row=1, col=1)
    if {'severity_score','estimated_cost_impact'}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df['severity_score'], y=df['estimated_cost_impact'], mode='markers', marker=dict(size=5)), row=1, col=2)
    if {'risk_score','estimated_cost_impact'}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df['risk_score'], y=df['estimated_cost_impact'], mode='markers', marker=dict(size=5)), row=2, col=1)
    if {'category','estimated_cost_impact'}.issubset(df.columns):
        fig.add_trace(go.Box(x=df['category'], y=df['estimated_cost_impact']), row=2, col=2)
    fig.update_layout(title='Cost Impact Analysis')
    return fig

# ---------- Heatmap functions ----------
# Facility layout coordinates for heatmap visualization
FACILITY_ZONES = {
    'Admin Building': {'x': 1, 'y': 5, 'area': 'Administration'},
    'EVCM 200': {'x': 2, 'y': 4, 'area': 'EDC/VCM'},
    'EVCM 300': {'x': 3, 'y': 4, 'area': 'EDC/VCM'},
    'PVC I Front End': {'x': 4, 'y': 3, 'area': 'PVC'},
    'PVC III Feedstock': {'x': 5, 'y': 3, 'area': 'PVC'},
    'HPO': {'x': 2, 'y': 2, 'area': 'HPO'},
    'HPO Process Area': {'x': 3, 'y': 2, 'area': 'HPO'},
    'HTDC': {'x': 4, 'y': 1, 'area': 'HTDC'},
    'CA-1650 and HCL Loading': {'x': 5, 'y': 1, 'area': 'Chlor Alkali'},
    'Container Offices': {'x': 1, 'y': 4, 'area': 'Administration'},
    'Manufacturing Facility': {'x': 3, 'y': 3, 'area': 'Main'},
    'Karachi': {'x': 3, 'y': 3, 'area': 'Main'},  # Default location
}

# GPS coordinates for geographical mapping (example coordinates around Karachi)
LOCATION_COORDINATES = {
    'Karachi': {'lat': 24.8607, 'lon': 67.0011},
    'Manufacturing Facility': {'lat': 24.8607, 'lon': 67.0011},
    'EVCM 200': {'lat': 24.8610, 'lon': 67.0015},
    'EVCM 300': {'lat': 24.8612, 'lon': 67.0018},
    'PVC I Front End': {'lat': 24.8608, 'lon': 67.0020},
    'PVC III Feedstock': {'lat': 24.8605, 'lon': 67.0022},
    'HPO': {'lat': 24.8603, 'lon': 67.0025},
    'HPO Process Area': {'lat': 24.8602, 'lon': 67.0027},
    'HTDC': {'lat': 24.8600, 'lon': 67.0030},
    'CA-1650 and HCL Loading': {'lat': 24.8598, 'lon': 67.0032},
    'Admin Building': {'lat': 24.8615, 'lon': 67.0010},
    'Container Offices': {'lat': 24.8613, 'lon': 67.0008},
}

def add_coordinates_to_df(df):
    """Add lat/lon coordinates to dataframe based on location columns (vectorized)."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()

    # Build mapping functions
    def map_series(series, key):
        return series.astype(str).map(lambda v: LOCATION_COORDINATES.get(v, {}).get(key) if pd.notna(v) else None)

    lat = pd.Series(pd.NA, index=df.index)
    lon = pd.Series(pd.NA, index=df.index)

    if 'location.1' in df.columns:
        lat = map_series(df['location.1'], 'lat')
        lon = map_series(df['location.1'], 'lon')
    if 'sublocation' in df.columns:
        lat = lat.fillna(map_series(df['sublocation'], 'lat'))
        lon = lon.fillna(map_series(df['sublocation'], 'lon'))
    if 'location' in df.columns:
        lat = lat.fillna(map_series(df['location'], 'lat'))
        lon = lon.fillna(map_series(df['location'], 'lon'))

    df['latitude'] = lat
    df['longitude'] = lon

    # Deterministic jitter only where we have coordinates (prevents re-runs from changing data)
    coords_mask = df['latitude'].notna() & df['longitude'].notna()
    if coords_mask.any():
        # Build a stable key for hashing
        if 'incident_id' in df.columns:
            key_series = df['incident_id'].astype(str)
        elif 'title' in df.columns:
            key_series = df['title'].astype(str)
        elif 'location.1' in df.columns:
            key_series = df['location.1'].astype(str)
        elif 'location' in df.columns:
            key_series = df['location'].astype(str)
        else:
            key_series = pd.Series(df.index.astype(str), index=df.index)
        h = pd.util.hash_pandas_object(key_series, index=False).astype(np.uint32)
        lat_j = ((h % 1000) / 1000.0 - 0.5) * 0.00008
        lon_j = (((h // 1000) % 1000) / 1000.0 - 0.5) * 0.00008
        df.loc[coords_mask, 'latitude'] = df.loc[coords_mask, 'latitude'].astype(float) + lat_j[coords_mask].values
        df.loc[coords_mask, 'longitude'] = df.loc[coords_mask, 'longitude'].astype(float) + lon_j[coords_mask].values

    return df

def create_facility_layout_heatmap(incident_df, hazard_df):
    """Create an interactive facility layout heatmap using Plotly"""

    # Create heatmap data for incidents and hazards
    incident_heatmap = create_zone_heatmap_data(incident_df, FACILITY_ZONES, 'Incidents')
    hazard_heatmap = create_zone_heatmap_data(hazard_df, FACILITY_ZONES, 'Hazards')

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🔴 Incident Heat Map', '⚠️ Hazard Heat Map'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.15
    )

    # Add incident heatmap
    fig.add_trace(
        go.Scatter(
            x=incident_heatmap['x'],
            y=incident_heatmap['y'],
            mode='markers',
            marker=dict(
                size=incident_heatmap['size'],
                color=incident_heatmap['intensity'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(x=0.45, title='Incidents', len=0.8),
                line=dict(width=2, color='darkred'),
                sizemode='diameter',
                sizeref=2,
                sizemin=20
            ),
            hovertemplate='<b>%{hovertext}</b><br>Count: %{marker.color}<extra></extra>',
            hovertext=incident_heatmap['hover'],
            showlegend=False
        ),
        row=1, col=1
    )

    # Location labels for incidents (always visible, small)
    fig.add_trace(
        go.Scatter(
            x=incident_heatmap['x'],
            y=incident_heatmap['y'],
            mode='text',
            text=incident_heatmap['labels'],
            textposition='top center',
            textfont=dict(color='#374151', size=11),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add incident counts with dynamic text color for visibility
    inc_vals = incident_heatmap['intensity']
    inc_thr = (max(inc_vals) * 0.55) if inc_vals else 0
    inc_hi_idx = [i for i, v in enumerate(inc_vals) if v >= inc_thr]
    inc_lo_idx = [i for i, v in enumerate(inc_vals) if v < inc_thr]
    if inc_hi_idx:
        fig.add_trace(
            go.Scatter(
                x=[incident_heatmap['x'][i] for i in inc_hi_idx],
                y=[incident_heatmap['y'][i] for i in inc_hi_idx],
                mode='text',
                text=[incident_heatmap['text'][i] for i in inc_hi_idx],
                textposition='middle center',
                textfont=dict(color='white', size=12, family='Arial Black'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )
    if inc_lo_idx:
        fig.add_trace(
            go.Scatter(
                x=[incident_heatmap['x'][i] for i in inc_lo_idx],
                y=[incident_heatmap['y'][i] for i in inc_lo_idx],
                mode='text',
                text=[incident_heatmap['text'][i] for i in inc_lo_idx],
                textposition='middle center',
                textfont=dict(color='#111827', size=12, family='Arial Black'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )

    # Add hazard heatmap
    fig.add_trace(
        go.Scatter(
            x=hazard_heatmap['x'],
            y=hazard_heatmap['y'],
            mode='markers',
            marker=dict(
                size=hazard_heatmap['size'],
                color=hazard_heatmap['intensity'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(x=1.0, title='Hazards', len=0.8),
                line=dict(width=2, color='darkorange'),
                sizemode='diameter',
                sizeref=2,
                sizemin=20
            ),
            hovertemplate='<b>%{hovertext}</b><br>Count: %{marker.color}<extra></extra>',
            hovertext=hazard_heatmap['hover'],
            showlegend=False
        ),
        row=1, col=2
    )

    # Location labels for hazards
    fig.add_trace(
        go.Scatter(
            x=hazard_heatmap['x'],
            y=hazard_heatmap['y'],
            mode='text',
            text=hazard_heatmap['labels'],
            textposition='top center',
            textfont=dict(color='#374151', size=11),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )

    # Add hazard counts with dynamic text color
    haz_vals = hazard_heatmap['intensity']
    haz_thr = (max(haz_vals) * 0.55) if haz_vals else 0
    haz_hi_idx = [i for i, v in enumerate(haz_vals) if v >= haz_thr]
    haz_lo_idx = [i for i, v in enumerate(haz_vals) if v < haz_thr]
    if haz_hi_idx:
        fig.add_trace(
            go.Scatter(
                x=[hazard_heatmap['x'][i] for i in haz_hi_idx],
                y=[hazard_heatmap['y'][i] for i in haz_hi_idx],
                mode='text',
                text=[hazard_heatmap['text'][i] for i in haz_hi_idx],
                textposition='middle center',
                textfont=dict(color='white', size=12, family='Arial Black'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=2
        )
    if haz_lo_idx:
        fig.add_trace(
            go.Scatter(
                x=[hazard_heatmap['x'][i] for i in haz_lo_idx],
                y=[hazard_heatmap['y'][i] for i in haz_lo_idx],
                mode='text',
                text=[hazard_heatmap['text'][i] for i in haz_lo_idx],
                textposition='middle center',
                textfont=dict(color='#111827', size=12, family='Arial Black'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=2
        )

    # Update layout for facility appearance
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        zeroline=False, showticklabels=False,
        range=[0, 6], title='',
        row=1, col=1
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        zeroline=False, showticklabels=False,
        range=[0, 6], title='',
        row=1, col=2
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        zeroline=False, showticklabels=False,
        range=[0, 6], title='',
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='LightGray',
        zeroline=False, showticklabels=False,
        range=[0, 6], title='',
        row=1, col=2
    )

    fig.update_layout(
        height=500,
        title_text="🏭 Facility Risk Heat Map - Real-time HSE Status",
        title_font_size=16,
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(t=60, l=40, r=40, b=40)
    )

    return fig

def create_zone_heatmap_data(df, zones, data_type):
    """Process dataframe to create zone-based heatmap data"""
    zone_counts = {}

    # Initialize all zones
    for zone_name, zone_info in zones.items():
        zone_counts[zone_name] = {
            'count': 0,
            'severity_sum': 0,
            'risk_sum': 0,
            'x': zone_info['x'],
            'y': zone_info['y'],
            'area': zone_info['area']
        }

    # Count events per zone
    for zone_name in zones.keys():
        count = 0
        severity_sum = 0
        risk_sum = 0

        # Check all location columns
        for col in ['location.1', 'sublocation', 'location']:
            if col in df.columns:
                # Case-insensitive partial matching
                matches = df[df[col].astype(str).str.contains(zone_name, case=False, na=False)]
                count += len(matches)
                if 'severity_score' in df.columns:
                    severity_sum += matches['severity_score'].sum()
                if 'risk_score' in df.columns:
                    risk_sum += matches['risk_score'].sum()

        zone_counts[zone_name]['count'] = count
        zone_counts[zone_name]['severity_sum'] = severity_sum
        zone_counts[zone_name]['risk_sum'] = risk_sum

    # Prepare data for plotting
    x, y, intensity, size, text, hover, labels = [], [], [], [], [], [], []

    for zone_name, data in zone_counts.items():
        x.append(data['x'])
        y.append(data['y'])
        intensity.append(data['count'])
        # Size based on count (minimum size 30)
        size.append(max(30, min(100, data['count'] * 8 + 20)))
        text.append(f"{data['count']}" if data['count'] > 0 else "")
        avg_severity = data['severity_sum'] / data['count'] if data['count'] > 0 else 0
        avg_risk = data['risk_sum'] / data['count'] if data['count'] > 0 else 0
        hover.append(f"{zone_name}<br>Area: {data['area']}<br>{data_type}: {data['count']}<br>Avg Severity: {avg_severity:.1f}<br>Avg Risk: {avg_risk:.1f}")
        labels.append(zone_name)

    return {
        'x': x, 'y': y, 'intensity': intensity,
        'size': size, 'text': text, 'hover': hover,
        'labels': labels,
    }

def create_folium_heatmap(df, map_title, color_gradient, max_points: int = 5000):
    """Create a Folium heatmap with custom styling. Optionally subsample for speed."""

    # Filter out rows without coordinates
    df_coords = df.dropna(subset=['latitude', 'longitude'])
    # Subsample to avoid huge heatmaps
    if len(df_coords) > max_points:
        df_coords = df_coords.sample(max_points, random_state=42)
    if df_coords.empty:
        # Return empty map if no coordinates
        m = folium.Map(location=[24.8607, 67.0011], zoom_start=13, tiles='CartoDB dark_matter')
        folium.Marker([24.8607, 67.0011], popup="No location data available").add_to(m)
        return m

    # Get center coordinates
    center_lat = df_coords['latitude'].mean()
    center_lon = df_coords['longitude'].mean()

    # Create base map with dark theme (similar to Snapchat)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='CartoDB dark_matter',  # Dark theme like Snapchat
        control_scale=True
    )

    # Prepare data for heatmap
    # Vectorized weight computation
    weight = pd.Series(1.0, index=df_coords.index)
    if 'severity_score' in df_coords.columns:
        w = pd.to_numeric(df_coords['severity_score'], errors='coerce') / 5.0
        weight = w.fillna(weight)
    elif 'risk_score' in df_coords.columns:
        w = pd.to_numeric(df_coords['risk_score'], errors='coerce') / 5.0
        weight = w.fillna(weight)
    heat_data = list(zip(df_coords['latitude'].astype(float), df_coords['longitude'].astype(float), weight.astype(float)))

    # Add heatmap layer
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=18,
            radius=25,
            blur=20,
            gradient=color_gradient,
            overlay=True,
            control=True,
            show=True
        ).add_to(m)

    # Add marker clusters for detailed view
    marker_cluster = MarkerCluster(name='Event Details').add_to(m)

    for idx, row in df_coords.head(2000).iterrows():
        # Create popup text
        popup_text = f"""
        <div style='width: 200px'>
            <b>Location:</b> {row.get('location.1', row.get('sublocation', 'Unknown'))}<br>
            <b>Date:</b> {row.get('occurrence_date', 'N/A')}<br>
            <b>Status:</b> {row.get('status', 'N/A')}<br>
            <b>Severity:</b> {row.get('severity_score', 'N/A')}<br>
            <b>Risk:</b> {row.get('risk_score', 'N/A')}<br>
            <b>Title:</b> {str(row.get('title', 'N/A'))[:50]}...
        </div>
        """

        # Color based on severity
        severity = row.get('severity_score', 1)
        if severity >= 4:
            icon_color = 'red'
        elif severity >= 3:
            icon_color = 'orange'
        elif severity >= 2:
            icon_color = 'yellow'
        else:
            icon_color = 'green'

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=icon_color, icon='warning', prefix='fa'),
        ).add_to(marker_cluster)

    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:20px; color: white; background-color: rgba(0,0,0,0.6); padding: 10px; border-radius: 5px;">
        <b>{map_title} Heat Map</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add fullscreen button
    folium.plugins.Fullscreen().add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

def create_incident_hazard_heatmaps(incident_df, hazard_df):
    """Create a single combined Folium map with both incident and hazard heat layers."""

    # Add coordinates
    incident_df = add_coordinates_to_df(incident_df if incident_df is not None else pd.DataFrame())
    hazard_df = add_coordinates_to_df(hazard_df if hazard_df is not None else pd.DataFrame())

    # Filter coords
    inc_coords = incident_df.dropna(subset=['latitude', 'longitude']) if not incident_df.empty else pd.DataFrame()
    haz_coords = hazard_df.dropna(subset=['latitude', 'longitude']) if not hazard_df.empty else pd.DataFrame()

    # Determine center
    if not inc_coords.empty:
        center_lat = inc_coords['latitude'].astype(float).mean()
        center_lon = inc_coords['longitude'].astype(float).mean()
    elif not haz_coords.empty:
        center_lat = haz_coords['latitude'].astype(float).mean()
        center_lon = haz_coords['longitude'].astype(float).mean()
    else:
        center_lat, center_lon = 24.8607, 67.0011

    # Create base map
    if FOLIUM_AVAILABLE:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB dark_matter', control_scale=True, prefer_canvas=True)
    else:
        st.error("Folium not installed. Install with: pip install folium streamlit-folium")
        return

    # Heat layer for incidents
    if not inc_coords.empty:
        weight_inc = pd.Series(1.0, index=inc_coords.index)
        if 'severity_score' in inc_coords.columns:
            w = pd.to_numeric(inc_coords['severity_score'], errors='coerce') / 5.0
            weight_inc = w.fillna(weight_inc)
        elif 'risk_score' in inc_coords.columns:
            w = pd.to_numeric(inc_coords['risk_score'], errors='coerce') / 5.0
            weight_inc = w.fillna(weight_inc)
        if len(inc_coords) > 3000:
            inc_coords = inc_coords.sample(3000, random_state=42)
            weight_inc = weight_inc.loc[inc_coords.index]
        inc_heat = list(zip(inc_coords['latitude'].astype(float), inc_coords['longitude'].astype(float), weight_inc.astype(float)))
        HeatMap(inc_heat, name='Incidents Heat', min_opacity=0.2, max_zoom=18, radius=16, blur=10, gradient={0.0: 'blue', 0.5: 'yellow', 0.75: 'orange', 1.0: 'red'}).add_to(m)

    # Heat layer for hazards
    if not haz_coords.empty:
        weight_haz = pd.Series(1.0, index=haz_coords.index)
        if 'severity_score' in haz_coords.columns:
            w = pd.to_numeric(haz_coords['severity_score'], errors='coerce') / 5.0
            weight_haz = w.fillna(weight_haz)
        elif 'risk_score' in haz_coords.columns:
            w = pd.to_numeric(haz_coords['risk_score'], errors='coerce') / 5.0
            weight_haz = w.fillna(weight_haz)
        if len(haz_coords) > 3000:
            haz_coords = haz_coords.sample(3000, random_state=42)
            weight_haz = weight_haz.loc[haz_coords.index]
        haz_heat = list(zip(haz_coords['latitude'].astype(float), haz_coords['longitude'].astype(float), weight_haz.astype(float)))
        HeatMap(haz_heat, name='Hazards Heat', min_opacity=0.2, max_zoom=18, radius=16, blur=10, gradient={0.0: 'green', 0.5: 'yellow', 0.75: 'orange', 1.0: 'darkred'}).add_to(m)

    # Add count labels per place (top N) for clarity
    def _add_count_labels(df_coords: pd.DataFrame, label_name: str, color: str, top_n: int = 25):
        if df_coords.empty:
            return
        place_col = 'location.1' if 'location.1' in df_coords.columns else ('sublocation' if 'sublocation' in df_coords.columns else ('location' if 'location' in df_coords.columns else None))
        if not place_col:
            return
        records = []
        for place, grp in df_coords.groupby(df_coords[place_col].astype(str)):
            lat = grp['latitude'].astype(float).mean()
            lon = grp['longitude'].astype(float).mean()
            cnt = len(grp)
            sev_avg = pd.to_numeric(grp.get('severity_score', pd.Series(dtype=float)), errors='coerce').mean()
            records.append({'place': place, 'lat': lat, 'lon': lon, 'count': cnt, 'sev': sev_avg})
        if not records:
            return
        rec_df = pd.DataFrame(records).sort_values('count', ascending=False).head(top_n)
        fg = folium.FeatureGroup(name=label_name, show=True)
        for _, r in rec_df.iterrows():
            html = f"""
            <div style='background: rgba(255,255,255,0.85); border:1px solid #999; border-radius:4px; padding:2px 6px; font-size:12px; font-weight:700; color:{color}; box-shadow:0 1px 2px rgba(0,0,0,0.2)'>
                {r['place']}: {int(r['count'])}
            </div>
            """
            folium.Marker(
                location=[r['lat'], r['lon']],
                icon=folium.DivIcon(html=html)
            ).add_to(fg)
        fg.add_to(m)

    _add_count_labels(inc_coords, 'Incident Counts (Top)', '#d32f2f')
    _add_count_labels(haz_coords, 'Hazard Counts (Top)', '#c05621')

    # Add limited detail markers with popups for both layers
    if not inc_coords.empty:
        inc_cluster = MarkerCluster(name='Incident Details', show=False)
        for _, row in inc_coords.head(600).iterrows():
            popup = f"<b>Incident</b><br>Location: {row.get('location.1', row.get('sublocation', 'Unknown'))}<br>Date: {row.get('occurrence_date','N/A')}<br>Status: {row.get('status','N/A')}<br>Severity: {row.get('severity_score','N/A')}<br>Risk: {row.get('risk_score','N/A')}"
            folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=3, color='red', fill=True, fill_opacity=0.6, popup=popup).add_to(inc_cluster)
        inc_cluster.add_to(m)
    if not haz_coords.empty:
        haz_cluster = MarkerCluster(name='Hazard Details', show=False)
        for _, row in haz_coords.head(600).iterrows():
            popup = f"<b>Hazard</b><br>Location: {row.get('location.1', row.get('sublocation', 'Unknown'))}<br>Date: {row.get('occurrence_date','N/A')}<br>Status: {row.get('status','N/A')}<br>Severity: {row.get('severity_score','N/A')}<br>Risk: {row.get('risk_score','N/A')}"
            folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=3, color='orange', fill=True, fill_opacity=0.6, popup=popup).add_to(haz_cluster)
        haz_cluster.add_to(m)

    # Fit bounds to data if we have any coordinates
    if not inc_coords.empty or not haz_coords.empty:
        lat_concat = pd.concat([inc_coords['latitude'], haz_coords['latitude']]).astype(float)
        lon_concat = pd.concat([inc_coords['longitude'], haz_coords['longitude']]).astype(float)
        m.fit_bounds([[lat_concat.min(), lon_concat.min()], [lat_concat.max(), lon_concat.max()]])

    folium.LayerControl().add_to(m)
    st_folium(m, key="combined_heatmap", width=None, height=520)

    # Metrics row under the single map
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Incidents (mapped)", len(inc_coords))
    with col2:
        st.metric("Total Hazards (mapped)", len(haz_coords))

def create_3d_facility_heatmap(df, event_type='Incidents'):
    """Create a 3D surface heatmap of the facility"""

    # Create a grid representing the facility
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)

    # Initialize Z values (intensity)
    Z = np.zeros_like(X)

    # Define zone centers and add Gaussian peaks for each event
    zone_centers = {
        'EVCM': (3, 7),
        'PVC': (7, 7),
        'HPO': (3, 3),
        'HTDC': (7, 3),
        'Admin': (1, 8),
        'Default': (5, 5)
    }

    for idx, row in df.iterrows():
        # Get location coordinates
        loc = str(row.get('location.1', row.get('sublocation', '')))

        # Find matching zone center
        cx, cy = zone_centers['Default']
        for zone_key, coords in zone_centers.items():
            if zone_key.lower() in loc.lower():
                cx, cy = coords
                break

        # Add Gaussian peak for this event
        severity = row.get('severity_score', row.get('risk_score', 1))
        if pd.notna(severity):
            Z += severity * np.exp(-((X - cx)**2 + (Y - cy)**2) / 2)

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Hot',
        name=event_type,
        showscale=True,
        colorbar=dict(title=f"{event_type} Intensity"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
        )
    )])

    fig.update_layout(
        title=f'3D {event_type} Heat Map - Facility Risk Visualization',
        scene=dict(
            xaxis_title='Facility Width',
            yaxis_title='Facility Length',
            zaxis_title='Risk Intensity',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(t=40, l=20, r=20, b=20)
    )
    return fig

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


def build_multi_sheet_context(workbook: dict, sample_rows: int = 5, max_numeric_cols: int = 6, max_cat_cols: int = 6, max_sheets: int = 20) -> str:
    """Build compact context for ALL sheets, including sample rows per sheet.
    Limits number of sheets and per-sheet summaries to keep tokens bounded.
    """
    if not workbook:
        return "No workbook loaded."
    lines = []
    lines.append("MULTI-SHEET CONTEXT")
    for i, (name, df) in enumerate(workbook.items()):
        if i >= max_sheets:
            lines.append(f"... {len(workbook) - max_sheets} more sheets omitted ...")
            break
        lines.append("")
        lines.append(f"=== SHEET: {name} ===")
        lines.append(build_ai_context(df, max_numeric_cols=max_numeric_cols, max_cat_cols=max_cat_cols, sample_rows=sample_rows))
    return "\n".join(lines)

def ask_openai(question: str, context: str, model: str = "gpt-4o", code_mode: bool = False, multi_df: bool = False) -> str:
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
            if multi_df:
                system_prompt = (
                    "You are a helpful data analyst writing Python to analyze multiple pandas DataFrames provided in a dict named dfs. "
                    "Keys are sheet names (lowercased) and values are DataFrames. A single-sheet DataFrame named df is also provided. "
                    "Use ONLY the provided context to infer column names/types and avoid external I/O or network access. "
                    "Return a single fenced Python code block only (no prose), which: "
                    "1) computes the answer using one or more DataFrames from dfs (and df if helpful), "
                    "2) assigns the main output to a variable named result (DataFrame/Series/scalar), and "
                    "3) optionally assigns a Plotly figure to a variable named fig OR a Matplotlib figure to a variable named mpl_fig. "
                    "Do not import modules. Use provided pd/np/px/plt if needed. "
                    "Do NOT call fig.show() or plt.show(); Streamlit will render the figure. "
                    "If using pyplot without explicit figure management, set mpl_fig = plt.gcf() at the end."
                )
            else:
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


def run_user_code(code: str, df: pd.DataFrame, dfs: dict | None = None):
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
    l = {'df': df, 'dfs': dfs or {}}
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

# ---------- Agent fallbacks ----------
def _fallback_hazard_to_incident_count(workbook: dict) -> dict:
    try:
        dfs = {str(k).lower(): v.copy() for k, v in (workbook or {}).items()}
    except Exception:
        dfs = {}
    incidents_df = dfs.get('incident', pd.DataFrame())
    hazard_df = dfs.get('hazard id', dfs.get('hazard', pd.DataFrame()))
    rel_key = next((k for k in dfs.keys() if 'relationship' in k), None)
    rel_df = dfs.get(rel_key) if rel_key else None

    def _find_col(df: pd.DataFrame, tokens, fallback=None):
        if df is None or df.empty:
            return None
        toks = [t.lower() for t in tokens]
        for c in df.columns:
            lc = str(c).lower()
            if all(t in lc for t in toks):
                return c
        return fallback if (fallback and fallback in df.columns) else None

    inc_id = _find_col(incidents_df, ['incident','id'], 'incident_id')
    haz_id = (_find_col(hazard_df, ['hazard','id']) or _find_col(hazard_df, ['id']) or
              ('incident_id' if (hazard_df is not None and 'incident_id' in getattr(hazard_df, 'columns', [])) else None))

    # Prefer a relationships sheet
    if rel_df is not None:
        haz_rel = _find_col(rel_df, ['hazard'])
        inc_rel = _find_col(rel_df, ['incident'])
        if haz_rel is not None and inc_rel is not None:
            haz_ids_rel = pd.Series(rel_df[haz_rel]).dropna().astype(str).unique()
            if hazard_df is not None and haz_id and haz_id in hazard_df.columns:
                haz_ids = pd.Series(hazard_df[haz_id]).dropna().astype(str).unique()
                count = len(set(haz_ids_rel).intersection(haz_ids))
            else:
                count = len(haz_ids_rel)
            sample = rel_df[[haz_rel, inc_rel]].dropna().head(10)
            return {'count': int(count), 'method': 'relationships', 'sample': sample}

    # Fallback: intersect IDs directly
    if (incidents_df is not None and not incidents_df.empty and hazard_df is not None and not hazard_df.empty
            and inc_id and inc_id in incidents_df.columns and haz_id and haz_id in hazard_df.columns):
        inc_ids = pd.Series(incidents_df[inc_id]).dropna().astype(str).unique()
        haz_ids = pd.Series(hazard_df[haz_id]).dropna().astype(str).unique()
        count = len(set(haz_ids).intersection(inc_ids))
        return {'count': int(count), 'method': 'id_intersection'}

    return {'count': 0, 'method': 'unknown_columns'}

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

        # Tabs (added Overall across-sheets view + Advanced Analytics + Hazard-Incident Analysis)
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🌐 Overall",
            "📊 Overview",
            "🔎 Deep Dive",
            "📋 Data Table",
            "📑 Summary Report",
            "🧠 Data Agent",
            "🚀 Advanced Analytics",
            "🔄 Hazard-Incident Analysis",
        ])

        # Tab 0: Overall (across all sheets)
        with tab0:
            st.subheader("🌐 Overall – Incidents vs Hazards vs Relationships")

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

            # Metrics row FIRST
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

            # Heatmap visualization type selector (now after the cards)
            st.markdown("### 🗺️ Real-time Risk Heat Maps")
            heatmap_type = st.radio(
                "Select Heat Map Type:",
                ["Facility Layout (2D)", "Geographical Map", "3D Surface"],
                horizontal=True,
                label_visibility="collapsed"
            )

            # Display selected heatmap
            if heatmap_type == "Facility Layout (2D)":
                if df_inc is not None and df_haz is not None:
                    fig_heatmap = create_facility_layout_heatmap(df_inc, df_haz)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                elif df_inc is not None:
                    fig_heatmap = create_facility_layout_heatmap(df_inc, pd.DataFrame())
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                elif df_haz is not None:
                    fig_heatmap = create_facility_layout_heatmap(pd.DataFrame(), df_haz)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No incident or hazard data available for heatmap visualization.")

            elif heatmap_type == "Geographical Map":
                st.subheader("🗺️ Geographical Heatmaps with Details")
                try:
                    incident_df = get_sheet_df(workbook, 'incident')
                    hazard_df = get_sheet_df(workbook, 'hazard')
                    if incident_df is None and hazard_df is None:
                        st.info("No incident or hazard data available for geographical mapping.")
                    else:
                        # Build one combined static HTML map to avoid reruns during pan/zoom
                        inc_df = maps_add_coords(incident_df if incident_df is not None else pd.DataFrame(), LOCATION_COORDINATES)
                        haz_df = maps_add_coords(hazard_df if hazard_df is not None else pd.DataFrame(), LOCATION_COORDINATES)
                        html_map = build_combined_map_html(inc_df, haz_df)
                        components.html(html_map, height=540, scrolling=False)
                        # Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Incidents (mapped)", len(inc_df.dropna(subset=['latitude','longitude'])) if inc_df is not None and not inc_df.empty else 0)
                        with col2:
                            st.metric("Total Hazards (mapped)", len(haz_df.dropna(subset=['latitude','longitude'])) if haz_df is not None and not haz_df.empty else 0)
                except Exception as e:
                    st.error(f"Failed to render maps: {e}")

            elif heatmap_type == "3D Surface":
                col1, col2 = st.columns(2)
                with col1:
                    if df_inc is not None and len(df_inc) > 0:
                        fig_3d_inc = create_3d_facility_heatmap(df_inc, 'Incidents')
                        st.plotly_chart(fig_3d_inc, use_container_width=True)
                    else:
                        st.info("No incident data available for 3D visualization.")
                with col2:
                    if df_haz is not None and len(df_haz) > 0:
                        fig_3d_haz = create_3d_facility_heatmap(df_haz, 'Hazards')
                        st.plotly_chart(fig_3d_haz, use_container_width=True)
                    else:
                        st.info("No hazard data available for 3D visualization.")

            st.markdown("---")

            # Bar chart summarizing counts
            summary_df = pd.DataFrame({
                'Type': ['Incidents', 'Hazards', 'Relationships'],
                'Count': [incident_count, hazard_count, relationships_count]
            })
            fig_overall = px.bar(summary_df, x='Type', y='Count', title='Overall Counts Across Workbook')
            fig_overall.update_layout(height=360)
            st.plotly_chart(fig_overall, width='stretch')

            # Prevention gauge (modular metric)
            try:
                gauge_fig = create_conversion_metrics_card(workbook)
                st.plotly_chart(gauge_fig, use_container_width=True)
            except Exception:
                pass

            # Linked vs Unlinked (stacked) using Relationships sheet
            st.subheader("🔗 Linked vs Unlinked Analysis")
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

            # 🔥 Hottest Zones Summary
            if df_inc is not None or df_haz is not None:
                st.markdown("### 🔥 Hottest Risk Zones")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top Incident Areas:**")
                    if df_inc is not None and 'location.1' in df_inc.columns:
                        top_incident_areas = df_inc['location.1'].value_counts().head(5)
                        for area, count in top_incident_areas.items():
                            severity_avg = df_inc[df_inc['location.1'] == area]['severity_score'].mean()
                            color = "🔴" if severity_avg >= 3 else "🟡" if severity_avg >= 2 else "🟢"
                            st.write(f"{color} **{area}**: {count} incidents (Avg Severity: {severity_avg:.1f})")
                    else:
                        st.write("No location data available")

                with col2:
                    st.markdown("**Top Hazard Areas:**")
                    if df_haz is not None and 'location.1' in df_haz.columns:
                        top_hazard_areas = df_haz['location.1'].value_counts().head(5)
                        for area, count in top_hazard_areas.items():
                            risk_avg = df_haz[df_haz['location.1'] == area]['risk_score'].mean()
                            color = "🔴" if risk_avg >= 3 else "🟡" if risk_avg >= 2 else "🟢"
                            st.write(f"{color} **{area}**: {count} hazards (Avg Risk: {risk_avg:.1f})")
                    else:
                        st.write("No location data available")

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
                    st.plotly_chart(fig_ts, width='stretch')
                else:
                    st.info("No date column detected for time series.")

                # Top Departments
                depc = schema.get('dept_col')
                vc = safe_value_counts(filtered_df, depc)
                if not vc.empty:
                    fig_dep = px.bar(x=vc.values, y=vc.index, orientation='h', title="Top Departments")
                    fig_dep.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_dep, width='stretch')
                else:
                    st.info("No department column available.")

            with colB:
                # Status distribution
                sc = schema.get('status_col')
                vc = safe_value_counts(filtered_df, sc)
                if not vc.empty:
                    fig_status = px.pie(values=vc.values, names=vc.index, title="Status Distribution")
                    fig_status.update_layout(height=300)
                    st.plotly_chart(fig_status, width='stretch')
                else:
                    st.info("No status column available.")

                # Top Locations
                locc = schema.get('loc_col')
                vc = safe_value_counts(filtered_df, locc)
                if not vc.empty:
                    fig_loc = px.bar(x=vc.values, y=vc.index, orientation='h', title="Top Locations")
                    fig_loc.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_loc, width='stretch')
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
                    st.plotly_chart(fig_cat, width='stretch')
                # Consequence
                cons = schema.get('consequence_col')
                vc = safe_value_counts(filtered_df, cons)
                if not vc.empty:
                    fig_cons = px.pie(values=vc.values, names=vc.index, title="Worst/Relevant Consequences")
                    st.plotly_chart(fig_cons, width='stretch')

            with col2:
                # Severity and Risk histograms if present
                sev = schema.get('severity_col')
                if sev and sev in filtered_df.columns and filtered_df[sev].notna().any():
                    fig_sev = px.histogram(filtered_df, x=sev, nbins=20, title="Severity Score Distribution")
                    st.plotly_chart(fig_sev, width='stretch')
                rk = schema.get('risk_col')
                if rk and rk in filtered_df.columns and filtered_df[rk].notna().any():
                    fig_risk = px.histogram(filtered_df, x=rk, nbins=20, title="Risk Score Distribution")
                    st.plotly_chart(fig_risk, width='stretch')

            # Timeliness metrics
            col3, col4 = st.columns(2)
            with col3:
                rd = schema.get('reporting_delay_col')
                if rd and rd in filtered_df.columns and filtered_df[rd].notna().any():
                    st.subheader("Reporting Delay (days)")
                    colA, colB, colC = st.columns(3)
                    _rd = _to_days(filtered_df[rd])
                    with colA: st.metric("Avg", _fmt_num(_rd.mean()))
                    with colB: st.metric("P90", _fmt_num(_rd.quantile(0.9)))
                    with colC: st.metric("Max", _fmt_num(_rd.max()))
                    st.plotly_chart(px.histogram(_rd.dropna(), nbins=30, title="Reporting Delay Histogram"), width='stretch')
            with col4:
                rt = schema.get('resolution_time_col')
                if rt and rt in filtered_df.columns and filtered_df[rt].notna().any():
                    st.subheader("Resolution Time (days)")
                    colA, colB, colC = st.columns(3)
                    _rt = _to_days(filtered_df[rt])
                    with colA: st.metric("Avg", _fmt_num(_rt.mean()))
                    with colB: st.metric("P90", _fmt_num(_rt.quantile(0.9)))
                    with colC: st.metric("Max", _fmt_num(_rt.max()))
                    st.plotly_chart(px.histogram(_rt.dropna(), nbins=30, title="Resolution Time Histogram"), width='stretch')

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
            st.header("🧠 Data Agent")
            st.caption("Ask a question. The agent will generate Python, run it on the filtered data, then provide a prescriptive summary.")
            question = st.text_area("Your question about the filtered data", placeholder="e.g., Which departments have the highest risk and what should we focus on?")
            with st.expander("⚙️ Agent settings", expanded=False):
                sample_rows = st.slider("Sample rows included in context", 3, 10, 5)
                show_ctx = st.checkbox("Show context sent to AI", value=False)
                show_code = st.checkbox("Show generated code", value=True)
                multi_sheets = st.checkbox("Enable multi-sheet reasoning (use all sheets)", value=True)
            _c1, _c2, _c3, _c4, _c5 = st.columns([1, 1, 1, 1, 1])
            with _c5:
                run_agent = st.button("Run EPCL Data Agent", type="primary")

            if run_agent and question:
                with st.spinner("Generating analysis code from your question..."):
                    if multi_sheets:
                        context_text = build_multi_sheet_context(workbook, sample_rows=sample_rows)
                    else:
                        context_text = build_ai_context(filtered_df, sample_rows=sample_rows)
                    code_resp = ask_openai(question, context_text, code_mode=True, multi_df=multi_sheets)
                if show_ctx:
                    with st.expander("Context sent to AI"):
                        st.code(context_text)
                code_block = extract_python_code(code_resp)
                if not code_block:
                    st.error("The agent did not produce runnable Python. Showing raw response below.")
                    st.code(code_resp)
                    st.stop()
                if show_code:
                    with st.expander("Generated code", expanded=False):
                        st.code(code_block, language="python")
                with st.spinner("Running code on filtered data..."):
                    dfs_payload = None
                    if multi_sheets:
                        try:
                            # Provide all sheets as lowercase-name dict for the agent
                            dfs_payload = {str(k).lower(): v.copy() for k, v in workbook.items()}
                        except Exception:
                            dfs_payload = None
                    env, stdout_text, err = run_user_code(code_block, filtered_df.copy(), dfs=dfs_payload)
                # If execution failed for a known question, use a guarded fallback
                if err and ('hazard' in question.lower() and 'incident' in question.lower() and 'how many' in question.lower()):
                    safe = _fallback_hazard_to_incident_count(workbook)
                    env['result'] = safe
                    err = None
                if err:
                    st.error("Execution failed. See traceback:")
                    st.code(err)
                else:
                    if 'fig' in env and env['fig'] is not None:
                        try:
                            st.plotly_chart(env['fig'], width='stretch')
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
                                    st.plotly_chart(_v, width='stretch')
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

        # Tab 6: Advanced Analytics
        with tab6:
            st.header("Advanced  Analytics")
            # Gather common DataFrames
            incident_df = get_sheet_df(workbook, 'incident')
            hazard_df = get_sheet_df(workbook, 'hazard')
            audit_df = get_sheet_df(workbook, 'audit')
            inspection_df = get_sheet_df(workbook, 'inspection')
            relationships_df = get_sheet_df(workbook, 'relationship')

            analysis_category = st.selectbox(
                "Select Analysis Category",
                [
                    "Executive Overview",
                    "Risk Analysis",
                    "Time Trends",
                    "Department Performance",
                    "Word Clouds",
                    "Data Quality",
                    "Predictive Analytics",
                    "PSM Analysis",
                    "Hazard Violation Analysis",
                ]
            )

            if analysis_category == "Executive Overview":
                # Row 1: Full-width unified scorecard
                fig1 = create_unified_hse_scorecard(incident_df, hazard_df, audit_df, inspection_df)
                st.plotly_chart(fig1, width='stretch')

                # Row 2: Full-width HSE Performance Index
                st.markdown("---")
                base_df = incident_df if incident_df is not None else filtered_df
                fig2 = create_hse_performance_index(base_df)
                st.plotly_chart(fig2, width='stretch')

                                                                                                                                                                            # # Row 3: Full-width Incident management funnel
                                                                                                                                                                            # st.markdown("---")
                                                                                                                                                                            # fig3 = create_incident_action_funnel(incident_df, relationships_df)
                                                                                                                                                                            # st.plotly_chart(fig3, width='stretch')

            elif analysis_category == "Risk Analysis":
                # Consequence matrix (incident)
                fig_cm = create_consequence_matrix(incident_df)
                st.plotly_chart(fig_cm, width='stretch')
                # Risk calendar heatmap (incident)
                fig_heat = create_risk_calendar_heatmap(incident_df)
                st.plotly_chart(fig_heat, width='stretch')

            elif analysis_category == "Time Trends":
                fig_tl = create_comprehensive_timeline(incident_df)
                st.plotly_chart(fig_tl, width='stretch')
                fig_tracker = create_audit_inspection_tracker(audit_df, inspection_df)
                st.plotly_chart(fig_tracker, width='stretch')

            elif analysis_category == "Department Performance":
                fig_spider = create_department_spider(incident_df if incident_df is not None else filtered_df)
                st.plotly_chart(fig_spider, width='stretch')
                fig_loc = create_location_risk_treemap(incident_df if incident_df is not None else filtered_df)
                st.plotly_chart(fig_loc, width='stretch')

            elif analysis_category == "Word Clouds":
                st.subheader("Top Departments by Mentions")
                top_n = st.slider("Max words", 10, 150, 75)
                min_count = st.slider("Min department frequency", 1, 10, 1)
                extra_stop = st.text_input("Extra stopwords (comma-separated)", value="")
                extra_set = {s.strip().lower() for s in extra_stop.split(',') if s.strip()} if extra_stop else set()
                words = get_incident_hazard_department_words(incident_df, hazard_df, top_n=top_n, min_count=min_count, extra_stopwords=extra_set)

                # Python WordCloud (default and only)
                st.markdown("#### Incident Departments")
                img_inc = create_python_wordcloud_image(words.get("incident", []), width=1600, height=520, colormap="coolwarm")
                if img_inc:
                    st.image(img_inc, caption="Incident Departments", use_container_width=True)
                else:
                    st.info("Python wordcloud not available or no data to display.")

                st.markdown("#### Hazard Departments")
                img_haz = create_python_wordcloud_image(words.get("hazard", []), width=1600, height=520, colormap="coolwarm")
                if img_haz:
                    st.image(img_haz, caption="Hazard Departments", use_container_width=True)
                else:
                    st.info("Python wordcloud not available or no data to display.")

            elif analysis_category == "Data Quality":
                fig_dq = create_data_quality_metrics(incident_df)
                st.plotly_chart(fig_dq, width='stretch')

            elif analysis_category == "Predictive Analytics":
                fig_cost = create_cost_prediction_analysis(incident_df if incident_df is not None else filtered_df)
                st.plotly_chart(fig_cost, width='stretch')

            elif analysis_category == "PSM Analysis":
                fig_psm = create_psm_breakdown(incident_df)
                st.plotly_chart(fig_psm, width='stretch')

            elif analysis_category == "Hazard Violation Analysis":
                fig_vi = create_violation_analysis(hazard_df)
                st.plotly_chart(fig_vi, width='stretch')

        # Tab 7: Hazard-Incident Analysis (modular)
        with tab7:
            st.header("🔄 Hazard-Incident Analysis")
            render_conversion_page(workbook)

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
        <p>Safety Co-pilot v1.0 </p>
    </div>
    """, unsafe_allow_html=True)