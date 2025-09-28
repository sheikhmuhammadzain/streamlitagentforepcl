from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_days(series: pd.Series) -> pd.Series:
    try:
        if series is None:
            return series
        s = series
        if np.issubdtype(s.dtype, np.timedelta64):
            return s.dt.total_seconds() / 86400.0
        if np.issubdtype(s.dtype, np.datetime64):
            return pd.to_numeric(s, errors='coerce')
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.to_numeric(series, errors='coerce')


def create_unified_hse_scorecard(incident_df, hazard_df, audit_df, inspection_df):
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Incidents', 'Hazards', 'Audits Completed', 'Inspections'],
        horizontal_spacing=0.08,
    )
    inc_count = len(incident_df) if isinstance(incident_df, pd.DataFrame) else 0
    haz_count = len(hazard_df) if isinstance(hazard_df, pd.DataFrame) else 0
    audits_completed = 0
    if isinstance(audit_df, pd.DataFrame) and 'audit_status' in audit_df.columns:
        audits_completed = (audit_df['audit_status'].astype(str).str.lower() == 'closed').sum()
    insp_count = len(inspection_df) if isinstance(inspection_df, pd.DataFrame) else 0
    fig.add_trace(go.Indicator(mode="number", value=inc_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=haz_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=audits_completed, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=3)
    fig.add_trace(go.Indicator(mode="number", value=insp_count, number={'font': {'size': 48}, 'valueformat': ",d"}), row=1, col=4)
    fig.update_layout(title={"text": "Unified HSE Scorecard", "x": 0.01, "xanchor": "left"}, height=260, margin=dict(t=50, l=30, r=20, b=20), showlegend=False)
    return fig


def create_hse_performance_index(df: Optional[pd.DataFrame]):
    if df is None or len(df) == 0 or 'department' not in df.columns:
        return go.Figure()
    cp = df.copy()
    for c in ['severity_score', 'risk_score']:
        if c not in cp.columns:
            cp[c] = np.nan
        cp[c] = pd.to_numeric(cp[c], errors='coerce')
    if 'reporting_delay_days' not in cp.columns:
        cp['reporting_delay_days'] = np.nan
    cp['reporting_delay_days'] = _to_days(cp['reporting_delay_days'])
    if 'resolution_time_days' not in cp.columns:
        cp['resolution_time_days'] = np.nan
    cp['resolution_time_days'] = _to_days(cp['resolution_time_days'])
    for c in ['root_cause_is_missing', 'corrective_actions_is_missing']:
        if c not in cp.columns:
            cp[c] = np.nan
        cp[c] = pd.to_numeric(cp[c].astype(float), errors='coerce')
    dept_metrics = cp.groupby('department').agg({
        'severity_score': 'mean',
        'risk_score': 'mean',
        'reporting_delay_days': 'mean',
        'resolution_time_days': 'mean',
        'root_cause_is_missing': 'mean',
        'corrective_actions_is_missing': 'mean'
    }).fillna(0)
    sev = dept_metrics['severity_score'].clip(lower=0, upper=5)
    risk = dept_metrics['risk_score'].clip(lower=0, upper=5)
    rep = dept_metrics['reporting_delay_days'].clip(lower=0, upper=30)
    res = dept_metrics['resolution_time_days'].clip(lower=0, upper=60)
    rc_miss = dept_metrics['root_cause_is_missing'].clip(lower=0, upper=1)
    ca_miss = dept_metrics['corrective_actions_is_missing'].clip(lower=0, upper=1)
    idx = ((5 - sev)/5 * 0.25 + (5 - risk)/5 * 0.25 + (30 - rep)/30 * 0.2 + (60 - res)/60 * 0.2 + (1 - rc_miss) * 0.05 + (1 - ca_miss) * 0.05) * 100
    dept_metrics = dept_metrics.assign(hse_index=idx)
    fig = px.bar(dept_metrics.reset_index(), x='hse_index', y='department', orientation='h', color='hse_index', color_continuous_scale=['red', 'yellow', 'green'], title='HSE Performance Index by Department (0-100)')
    fig.update_layout(height=520, margin=dict(t=60, l=160, r=40, b=40), yaxis=dict(automargin=True))
    return fig


def create_risk_calendar_heatmap(df: Optional[pd.DataFrame]):
    if df is None or 'occurrence_date' not in df.columns or 'department' not in df.columns or 'risk_score' not in df.columns:
        return go.Figure()
    cp = df.copy()
    # Coerce risk_score to numeric to ensure mean aggregation works
    cp['risk_score'] = pd.to_numeric(cp['risk_score'], errors='coerce')
    cp['month'] = pd.to_datetime(cp['occurrence_date'], errors='coerce').dt.to_period('M')
    risk_pivot = cp.pivot_table(values='risk_score', index='department', columns='month', aggfunc='mean')
    z = risk_pivot.to_numpy()
    # If no finite values in mean risk matrix, fallback to counts
    if not (isinstance(z, np.ndarray) and np.isfinite(z).any()):
        count_pivot = cp.pivot_table(values='risk_score', index='department', columns='month', aggfunc='count')
        z = count_pivot.to_numpy()
        x_labels = count_pivot.columns.astype(str).tolist()
        y_labels = count_pivot.index.astype(str).tolist()
        fig = px.imshow(z, labels=dict(x='Month', y='Department', color='Count'), x=x_labels, y=y_labels, color_continuous_scale='YlOrRd', title='Department Events Count (Fallback)', aspect='auto', text_auto=True)
        fig.update_xaxes(tickangle=-45)
        return fig
    x_labels = risk_pivot.columns.astype(str).tolist()
    y_labels = risk_pivot.index.astype(str).tolist()
    fig = px.imshow(z, labels=dict(x='Month', y='Department', color='Avg Risk Score'), x=x_labels, y=y_labels, color_continuous_scale='RdYlGn_r', title='Department Risk Score Evolution', aspect='auto', text_auto=True)
    fig.update_xaxes(tickangle=-45)
    return fig


def create_psm_breakdown(incident_df: Optional[pd.DataFrame]):
    if incident_df is None:
        return go.Figure()
    psm_counts = incident_df['psm'].value_counts(dropna=True) if 'psm' in incident_df.columns else pd.Series(dtype=int)
    pse_counts = incident_df['pse_category'].value_counts(dropna=True) if 'pse_category' in incident_df.columns else pd.Series(dtype=int)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['PSM Elements', 'PSE Categories'], specs=[[{'type': 'pie'}, {'type': 'bar'}]])
    if not psm_counts.empty:
        fig.add_trace(go.Pie(labels=psm_counts.index, values=psm_counts.values, hole=0.4), row=1, col=1)
    if not pse_counts.empty:
        fig.add_trace(go.Bar(x=pse_counts.values, y=pse_counts.index, orientation='h'), row=1, col=2)
    fig.update_layout(title='Process Safety Management Analysis')
    return fig


def create_consequence_matrix(df: Optional[pd.DataFrame]):
    if df is None or 'actual_consequence_incident' not in df.columns or 'worst_case_consequence_incident' not in df.columns:
        return go.Figure()
    ct = pd.crosstab(df['actual_consequence_incident'], df['worst_case_consequence_incident'])
    fig = px.imshow(ct, labels=dict(x='Worst Case', y='Actual', color='Count'), title='Actual vs Worst Case Consequence Matrix', color_continuous_scale='YlOrRd', text_auto=True)
    return fig


def create_data_quality_metrics(incident_df: Optional[pd.DataFrame]):
    if incident_df is None:
        return go.Figure()
    fig = make_subplots(rows=2, cols=3, subplot_titles=['Root Cause Missing', 'Corrective Actions Missing', 'Reporting Delays', 'Resolution Times by Status', '', ''])
    if 'department' in incident_df.columns and 'root_cause_is_missing' in incident_df.columns:
        missing_rc = incident_df.groupby('department')['root_cause_is_missing'].sum()
        fig.add_trace(go.Bar(x=missing_rc.index, y=missing_rc.values, name='Root Cause Missing'), row=1, col=1)
    if 'department' in incident_df.columns and 'corrective_actions_is_missing' in incident_df.columns:
        missing_ca = incident_df.groupby('department')['corrective_actions_is_missing'].sum()
        fig.add_trace(go.Bar(x=missing_ca.index, y=missing_ca.values, name='Actions Missing'), row=1, col=2)
    if 'reporting_delay_days' in incident_df.columns:
        fig.add_trace(go.Histogram(x=_to_days(incident_df['reporting_delay_days']), nbinsx=30, name='Reporting Delay'), row=1, col=3)
    if {'resolution_time_days', 'status'}.issubset(incident_df.columns):
        fig.add_trace(go.Box(y=_to_days(incident_df['resolution_time_days']), x=incident_df['status'], name='Resolution by Status'), row=2, col=1)
    fig.update_layout(title='Data Quality Metrics')
    return fig


def create_comprehensive_timeline(df: Optional[pd.DataFrame]):
    if df is None or 'occurrence_date' not in df.columns:
        return go.Figure()
    cp = df.copy()
    # Use monthly aggregation to align with KPI cards summing first trace
    cp['_m'] = pd.to_datetime(cp['occurrence_date'], errors='coerce').dt.to_period('M')
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
    agg = cp.groupby('_m').agg(agg_dict).reset_index()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['Incident Count', 'Risk & Severity Scores', 'Cost & Manhour Impact'], row_heights=[0.3, 0.35, 0.35])
    # First trace: total monthly counts (frontend will sum this series for KPI)
    fig.add_trace(go.Bar(x=agg['_m'].astype(str), y=agg[count_col], name='Count'), row=1, col=1)
    if 'severity_score' in agg.columns:
        fig.add_trace(go.Scatter(x=agg['_m'].astype(str), y=agg['severity_score'], name='Severity', line=dict(color='red')), row=2, col=1)
    if 'risk_score' in agg.columns:
        fig.add_trace(go.Scatter(x=agg['_m'].astype(str), y=agg['risk_score'], name='Risk', line=dict(color='orange')), row=2, col=1)
    if 'estimated_cost_impact' in agg.columns:
        fig.add_trace(go.Bar(x=agg['_m'].astype(str), y=agg['estimated_cost_impact'], name='Cost ($)', marker_color='green'), row=3, col=1)
    if 'estimated_manhours_impact' in agg.columns:
        fig.add_trace(go.Bar(x=agg['_m'].astype(str), y=agg['estimated_manhours_impact'], name='Manhours', marker_color='#34D399'), row=3, col=1)
    fig.update_layout(title='Comprehensive HSE Timeline')
    return fig


def create_audit_inspection_tracker(audit_df: Optional[pd.DataFrame], inspection_df: Optional[pd.DataFrame]):
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Audit Status Over Time', 'Inspection Status Over Time'])
    if isinstance(audit_df, pd.DataFrame) and {'start_date', 'audit_status'}.issubset(audit_df.columns):
        aud = audit_df.copy()
        aud['_m'] = pd.to_datetime(aud['start_date'], errors='coerce').dt.to_period('M')
        # First trace: total audits per month
        totals = aud.groupby(aud['_m']).size()
        fig.add_trace(go.Bar(x=totals.index.astype(str), y=totals.values, name='Audits Total'), row=1, col=1)
        # Additional per-status traces
        timeline = aud.groupby([aud['_m'], 'audit_status']).size().unstack(fill_value=0)
        for status in timeline.columns:
            fig.add_trace(go.Bar(x=timeline.index.astype(str), y=timeline[status], name=str(status)), row=1, col=1)
    if isinstance(inspection_df, pd.DataFrame) and {'start_date', 'audit_status'}.issubset(inspection_df.columns):
        ins = inspection_df.copy()
        ins['_m'] = pd.to_datetime(ins['start_date'], errors='coerce').dt.to_period('M')
        totals = ins.groupby(ins['_m']).size()
        fig.add_trace(go.Bar(x=totals.index.astype(str), y=totals.values, name='Inspections Total'), row=2, col=1)
        timeline = ins.groupby([ins['_m'], 'audit_status']).size().unstack(fill_value=0)
        for status in timeline.columns:
            fig.add_trace(go.Bar(x=timeline.index.astype(str), y=timeline[status], name=str(status)), row=2, col=1)
    fig.update_layout(barmode='stack', title='Audit & Inspection Compliance Tracking')
    return fig


def create_location_risk_treemap(df: Optional[pd.DataFrame]):
    if df is None or not {'location', 'sublocation'}.issubset(df.columns):
        return go.Figure()
    cp = df.copy()
    if 'incident_id' not in cp.columns:
        cp['incident_id'] = 1
    for c in ['severity_score', 'risk_score', 'estimated_cost_impact']:
        if c not in cp.columns:
            cp[c] = np.nan
    location_data = cp.groupby(['location', 'sublocation']).agg({'incident_id': 'count', 'severity_score': 'mean', 'risk_score': 'mean', 'estimated_cost_impact': 'sum'}).reset_index()
    location_data['size'] = location_data['incident_id']
    location_data['hover_text'] = (
        'Count: ' + location_data['incident_id'].astype(str) +
        '<br>Avg Severity: ' + location_data['severity_score'].round(2).astype(str) +
        '<br>Total Cost: ' + location_data['estimated_cost_impact'].round(0).astype(str)
    )
    fig = px.treemap(location_data, path=['location', 'sublocation'], values='size', color='risk_score', hover_data={'hover_text': True}, color_continuous_scale='RdYlGn_r', title='Location Risk Map (Size=Count, Color=Risk)')
    return fig


def create_department_spider(df: Optional[pd.DataFrame]):
    if df is None or 'department' not in df.columns:
        return go.Figure()
    cp = df.copy()
    for col in ['severity_score', 'risk_score', 'reporting_delay_days', 'resolution_time_days', 'root_cause_is_missing', 'corrective_actions_is_missing']:
        if col not in cp.columns:
            cp[col] = np.nan
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
    for col in dept_metrics.columns:
        m = dept_metrics[col].max()
        if m and m > 0:
            dept_metrics[col] = (dept_metrics[col] / m) * 100
    fig = go.Figure()
    labels = ['Low Severity', 'Low Risk', 'Fast Reporting', 'Quick Resolution', 'Root Cause ID', 'Actions Taken']
    for dept in dept_metrics.index[:5]:
        fig.add_trace(go.Scatterpolar(r=dept_metrics.loc[dept].values, theta=labels, fill='toself', name=str(dept)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title='Department HSE Performance Radar', showlegend=True)
    return fig


def create_violation_analysis(hazard_df: Optional[pd.DataFrame]):
    if hazard_df is None:
        return go.Figure()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Violation Types', 'Consequences Distribution', 'Reporting Delays', 'Department Violations'],
        specs=[[{'type': 'xy'}, {'type': 'domain'}], [{'type': 'xy'}, {'type': 'xy'}]]
    )
    if 'violation_type_hazard_id' in hazard_df.columns:
        vc = hazard_df['violation_type_hazard_id'].value_counts()
        fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation='h', name='Violation Types'), row=1, col=1)
    if 'worst_case_consequence_potential_hazard_id' in hazard_df.columns:
        wc = hazard_df['worst_case_consequence_potential_hazard_id'].value_counts()
        fig.add_trace(go.Pie(labels=wc.index, values=wc.values, name='Consequences', hole=0.3), row=1, col=2)
    if 'reporting_delay_days' in hazard_df.columns:
        fig.add_trace(go.Histogram(x=_to_days(hazard_df['reporting_delay_days']), nbinsx=30, name='Reporting Delay'), row=2, col=1)
    if {'department', 'violation_type_hazard_id'}.issubset(hazard_df.columns):
        heat = hazard_df.pivot_table(index='department', columns='violation_type_hazard_id', values='violation_type_hazard_id', aggfunc='count').fillna(0)
        fig.add_trace(go.Heatmap(z=heat.to_numpy(), x=list(heat.columns.astype(str)), y=list(heat.index.astype(str)), colorscale='YlOrRd', name='Dept x Violation'), row=2, col=2)
    fig.update_layout(title='Hazard Violation Analysis')
    return fig


# ---------- Additional charts ported from streamlit.py ----------
def create_cost_prediction_analysis(df: Optional[pd.DataFrame]):
    if df is None or 'estimated_cost_impact' not in df.columns:
        return go.Figure()
    numeric_cols = [c for c in ['severity_score','risk_score','reporting_delay_days','resolution_time_days','estimated_manhours_impact'] if c in df.columns]
    sub = df[numeric_cols + ['estimated_cost_impact']].dropna() if numeric_cols else pd.DataFrame()
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


# Facility zones used for facility layout heatmap
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
    'Karachi': {'x': 3, 'y': 3, 'area': 'Main'},
}


def _zone_heatmap_data(df: pd.DataFrame, zones: dict, data_type: str):
    zone_counts = {zn: {'count': 0, 'severity_sum': 0, 'risk_sum': 0, 'x': info['x'], 'y': info['y'], 'area': info['area']} for zn, info in zones.items()}
    for zone_name in zones.keys():
        count = 0
        severity_sum = 0
        risk_sum = 0
        for col in ['location.1', 'sublocation', 'location']:
            if col in df.columns:
                matches = df[df[col].astype(str).str.contains(zone_name, case=False, na=False)]
                count += len(matches)
                if 'severity_score' in df.columns:
                    severity_sum += pd.to_numeric(matches['severity_score'], errors='coerce').fillna(0).sum()
                if 'risk_score' in df.columns:
                    risk_sum += pd.to_numeric(matches['risk_score'], errors='coerce').fillna(0).sum()
        zone_counts[zone_name]['count'] = int(count)
        zone_counts[zone_name]['severity_sum'] = float(severity_sum)
        zone_counts[zone_name]['risk_sum'] = float(risk_sum)
    x, y, intensity, size, text, hover, labels = [], [], [], [], [], [], []
    for zone_name, data in zone_counts.items():
        x.append(data['x'])
        y.append(data['y'])
        intensity.append(data['count'])
        size.append(max(30, min(100, data['count'] * 8 + 20)))
        text.append(f"{data['count']}" if data['count'] > 0 else "")
        avg_severity = (data['severity_sum'] / data['count']) if data['count'] > 0 else 0
        avg_risk = (data['risk_sum'] / data['count']) if data['count'] > 0 else 0
        hover.append(f"{zone_name}<br>Area: {data['area']}<br>{data_type}: {data['count']}<br>Avg Severity: {avg_severity:.1f}<br>Avg Risk: {avg_risk:.1f}")
        labels.append(zone_name)
    return {'x': x, 'y': y, 'intensity': intensity, 'size': size, 'text': text, 'hover': hover, 'labels': labels}


def create_facility_layout_heatmap(incident_df: Optional[pd.DataFrame], hazard_df: Optional[pd.DataFrame]):
    inc = incident_df.copy() if isinstance(incident_df, pd.DataFrame) else pd.DataFrame()
    haz = hazard_df.copy() if isinstance(hazard_df, pd.DataFrame) else pd.DataFrame()
    incident_heatmap = _zone_heatmap_data(inc, FACILITY_ZONES, 'Incidents')
    hazard_heatmap = _zone_heatmap_data(haz, FACILITY_ZONES, 'Hazards')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('🔴 Incident Heat Map', '⚠️ Hazard Heat Map'), specs=[[{'type': 'scatter'}, {'type': 'scatter'}]], horizontal_spacing=0.15)
    fig.add_trace(go.Scatter(x=incident_heatmap['x'], y=incident_heatmap['y'], mode='markers', marker=dict(size=incident_heatmap['size'], color=incident_heatmap['intensity'], colorscale='Reds', showscale=True, colorbar=dict(x=0.45, title='Incidents', len=0.8), line=dict(width=2, color='darkred'), sizemode='diameter', sizeref=2, sizemin=20), hovertemplate='<b>%{hovertext}</b><br>Count: %{marker.color}<extra></extra>', hovertext=incident_heatmap['hover'], showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=incident_heatmap['x'], y=incident_heatmap['y'], mode='text', text=incident_heatmap['labels'], textposition='top center', textfont=dict(color='#374151', size=11), hoverinfo='skip', showlegend=False), row=1, col=1)
    inc_vals = incident_heatmap['intensity']
    inc_thr = (max(inc_vals) * 0.55) if inc_vals else 0
    inc_hi_idx = [i for i, v in enumerate(inc_vals) if v >= inc_thr]
    inc_lo_idx = [i for i, v in enumerate(inc_vals) if v < inc_thr]
    if inc_hi_idx:
        fig.add_trace(go.Scatter(x=[incident_heatmap['x'][i] for i in inc_hi_idx], y=[incident_heatmap['y'][i] for i in inc_hi_idx], mode='text', text=[incident_heatmap['text'][i] for i in inc_hi_idx], textposition='middle center', textfont=dict(color='white', size=12, family='Arial Black'), hoverinfo='skip', showlegend=False), row=1, col=1)
    if inc_lo_idx:
        fig.add_trace(go.Scatter(x=[incident_heatmap['x'][i] for i in inc_lo_idx], y=[incident_heatmap['y'][i] for i in inc_lo_idx], mode='text', text=[incident_heatmap['text'][i] for i in inc_lo_idx], textposition='middle center', textfont=dict(color='#111827', size=12, family='Arial Black'), hoverinfo='skip', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=hazard_heatmap['x'], y=hazard_heatmap['y'], mode='markers', marker=dict(size=hazard_heatmap['size'], color=hazard_heatmap['intensity'], colorscale='YlOrRd', showscale=True, colorbar=dict(x=1.0, title='Hazards', len=0.8), line=dict(width=2, color='darkorange'), sizemode='diameter', sizeref=2, sizemin=20), hovertemplate='<b>%{hovertext}</b><br>Count: %{marker.color}<extra></extra>', hovertext=hazard_heatmap['hover'], showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=hazard_heatmap['x'], y=hazard_heatmap['y'], mode='text', text=hazard_heatmap['labels'], textposition='top center', textfont=dict(color='#374151', size=11), hoverinfo='skip', showlegend=False), row=1, col=2)
    haz_vals = hazard_heatmap['intensity']
    haz_thr = (max(haz_vals) * 0.55) if haz_vals else 0
    haz_hi_idx = [i for i, v in enumerate(haz_vals) if v >= haz_thr]
    haz_lo_idx = [i for i, v in enumerate(haz_vals) if v < haz_thr]
    if haz_hi_idx:
        fig.add_trace(go.Scatter(x=[hazard_heatmap['x'][i] for i in haz_hi_idx], y=[hazard_heatmap['y'][i] for i in haz_hi_idx], mode='text', text=[hazard_heatmap['text'][i] for i in haz_hi_idx], textposition='middle center', textfont=dict(color='white', size=12, family='Arial Black'), hoverinfo='skip', showlegend=False), row=1, col=2)
    if haz_lo_idx:
        fig.add_trace(go.Scatter(x=[hazard_heatmap['x'][i] for i in haz_lo_idx], y=[hazard_heatmap['y'][i] for i in haz_lo_idx], mode='text', text=[hazard_heatmap['text'][i] for i in haz_lo_idx], textposition='middle center', textfont=dict(color='#111827', size=12, family='Arial Black'), hoverinfo='skip', showlegend=False), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False, showticklabels=False, range=[0, 6], title='', row=1, col=c)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False, showticklabels=False, range=[0, 6], title='', row=1, col=c)
    fig.update_layout(height=500, title_text="🏭 Facility Risk Heat Map - Real-time HSE Status", title_font_size=16, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white', margin=dict(t=60, l=40, r=40, b=40))
    return fig


def create_3d_facility_heatmap(df: Optional[pd.DataFrame], event_type: str = 'Incidents'):
    if df is None or df.empty:
        return go.Figure()
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    zone_centers = {'EVCM': (3, 7), 'PVC': (7, 7), 'HPO': (3, 3), 'HTDC': (7, 3), 'Admin': (1, 8), 'Default': (5, 5)}
    for _, row in df.iterrows():
        loc = str(row.get('location.1', row.get('sublocation', '')))
        cx, cy = zone_centers['Default']
        for zone_key, coords in zone_centers.items():
            if zone_key.lower() in loc.lower():
                cx, cy = coords
                break
        severity = row.get('severity_score', row.get('risk_score', 1))
        if pd.notna(severity):
            Z += float(severity) * np.exp(-((X - cx)**2 + (Y - cy)**2) / 2)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Hot', name=event_type, showscale=True, colorbar=dict(title=f"{event_type} Intensity"), contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))))])
    fig.update_layout(title=f'3D {event_type} Heat Map - Facility Risk Visualization', scene=dict(xaxis_title='Facility Width', yaxis_title='Facility Length', zaxis_title='Risk Intensity', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), height=600, margin=dict(t=40, l=20, r=20, b=20))
    return fig

