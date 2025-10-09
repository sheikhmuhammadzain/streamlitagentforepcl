import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta

try:
    import networkx as nx
    _NX_AVAILABLE = True
except Exception:
    _NX_AVAILABLE = False


def _first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_sheet(workbook: dict, token: str):
    if not workbook:
        return None
    for name, df in workbook.items():
        if token.lower() in str(name).lower():
            return df
    return None


def _coerce_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass
    return df


class HazardIncidentAnalyzer:
    """Analyze the relationship between hazards and incidents"""

    def __init__(self, incident_df: pd.DataFrame, hazard_df: pd.DataFrame, relationships_df: pd.DataFrame | None = None):
        self.incident_df = incident_df.copy() if incident_df is not None else pd.DataFrame()
        self.hazard_df = hazard_df.copy() if hazard_df is not None else pd.DataFrame()
        self.relationships_df = relationships_df.copy() if relationships_df is not None else pd.DataFrame()
        self.links_df = pd.DataFrame()
        self.prepare_data()

    def _detect_columns(self):
        # Incident columns
        self.inc_date = _first_present(self.incident_df, ['occurrence_date', 'reported_date', 'entered_date', 'completion_date'])
        self.inc_loc = _first_present(self.incident_df, ['location', 'sublocation', 'location.1'])
        self.inc_dept = _first_present(self.incident_df, ['department', 'sub_department'])
        self.inc_id = _first_present(self.incident_df, ['incident_id'])
        self.inc_sev = _first_present(self.incident_df, ['severity_score'])
        self.inc_type = _first_present(self.incident_df, ['incident_type', 'category'])
        self.inc_status = _first_present(self.incident_df, ['status'])

        # Hazard columns
        self.haz_date = _first_present(self.hazard_df, ['occurrence_date', 'reported_date', 'entered_date', 'entered_closed'])
        self.haz_loc = _first_present(self.hazard_df, ['location', 'sublocation', 'location.1'])
        self.haz_dept = _first_present(self.hazard_df, ['department', 'sub_department'])
        # Hazard unique id might be named incident_id in some exports
        self.haz_id = _first_present(self.hazard_df, ['hazard_id', 'incident_id'])
        self.haz_sev = _first_present(self.hazard_df, ['severity_score'])
        self.haz_type = _first_present(self.hazard_df, ['violation_type_hazard_id', 'category'])
        self.haz_status = _first_present(self.hazard_df, ['status'])

    def prepare_data(self):
        """Prepare and link hazard-incident data"""
        if self.incident_df.empty or self.hazard_df.empty:
            self.links_df = pd.DataFrame()
            return

        self._detect_columns()

        # Ensure datetime columns
        if self.inc_date:
            _coerce_datetime(self.incident_df, [self.inc_date])
        if self.haz_date:
            _coerce_datetime(self.hazard_df, [self.haz_date])

        self.create_hazard_incident_links()

    def create_hazard_incident_links(self, window_days: int = 30):
        """Identify potential hazard-to-incident conversions by location+department proximity and time."""
        if not all([self.inc_date, self.haz_date, self.inc_loc, self.haz_loc, self.inc_dept, self.haz_dept]):
            self.links_df = pd.DataFrame()
            return

        links: list[dict] = []
        # Index incidents by (location, dept) for speed
        inc = self.incident_df[[self.inc_id, self.inc_date, self.inc_loc, self.inc_dept, self.inc_sev, self.inc_type]].copy()
        inc = inc.dropna(subset=[self.inc_date])

        # Iterate hazards and select incident window
        for _, haz in self.hazard_df.iterrows():
            hz_dt = haz.get(self.haz_date)
            hz_loc = haz.get(self.haz_loc)
            hz_dept = haz.get(self.haz_dept)
            if pd.isna(hz_dt) or pd.isna(hz_loc) or pd.isna(hz_dept):
                continue
            dt_start = hz_dt
            dt_end = hz_dt + timedelta(days=window_days)
            candid = inc[
                (inc[self.inc_loc] == hz_loc) &
                (inc[self.inc_dept] == hz_dept) &
                (inc[self.inc_date] >= dt_start) &
                (inc[self.inc_date] <= dt_end)
            ]
            if candid.empty:
                continue
            for _, inc_row in candid.iterrows():
                links.append({
                    'hazard_id': haz.get(self.haz_id),
                    'incident_id': inc_row.get(self.inc_id),
                    'hazard_date': hz_dt,
                    'incident_date': inc_row.get(self.inc_date),
                    'days_to_incident': (inc_row.get(self.inc_date) - hz_dt).days,
                    'location': hz_loc,
                    'department': hz_dept,
                    'hazard_severity': haz.get(self.haz_sev, pd.NA),
                    'incident_severity': inc_row.get(self.inc_sev, pd.NA),
                    'hazard_type': haz.get(self.haz_type, 'Unknown'),
                    'incident_type': inc_row.get(self.inc_type, 'Unknown')
                })

        self.links_df = pd.DataFrame(links)

    # ---------- Charts ----------
    def create_conversion_funnel(self) -> go.Figure:
        """Show the funnel from hazard identification to incident prevention"""
        if self.hazard_df.empty:
            return _empty_chart("No hazard data available")

        total_hazards = len(self.hazard_df)
        hazards_closed = 0
        hazards_open = total_hazards
        if self.haz_status and self.haz_status in self.hazard_df.columns:
            hazards_closed = (self.hazard_df[self.haz_status].astype(str).str.lower() == 'closed').sum()
            hazards_open = total_hazards - hazards_closed

        if not self.links_df.empty and 'hazard_id' in self.links_df.columns:
            hazards_became_incidents = self.links_df['hazard_id'].nunique()
            prevented_hazards = max(0, hazards_closed - hazards_became_incidents)
        else:
            hazards_became_incidents = 0
            prevented_hazards = hazards_closed

        fig = go.Figure()
        stages = [
            ('Total Hazards Identified', total_hazards, '#FFA500'),
            ('Hazards Addressed', hazards_closed, '#FFD700'),
            ('Successfully Prevented', prevented_hazards, '#90EE90'),
            ('Became Incidents', hazards_became_incidents, '#FF6B6B'),
            ('Open Hazards (Risk)', hazards_open, '#FF4444')
        ]
        fig.add_trace(go.Funnel(
            y=[s[0] for s in stages],
            x=[s[1] for s in stages],
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.85,
            marker={"color": [s[2] for s in stages]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        prevention_rate = (prevented_hazards / hazards_closed * 100) if hazards_closed > 0 else 0
        fig.update_layout(
            title=f"Hazard to Incident Conversion Funnel<br><sub>Prevention Success Rate: {prevention_rate:.1f}%</sub>",
            height=500,
            showlegend=False
        )
        return fig

    def create_time_lag_analysis(self) -> go.Figure:
        if self.links_df.empty:
            return _empty_chart("No hazard-incident links found")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Days from Hazard to Incident',
                'Conversion Rate by Hazard Type',
                'Severity Escalation',
                'Monthly Conversion Trend'
            ],
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        # 1. Time lag histogram
        fig.add_trace(go.Histogram(x=self.links_df['days_to_incident'], nbinsx=20, marker_color='indianred', name='Days to Incident'), row=1, col=1)

        # 2. Conversion rate by hazard type
        if self.haz_type and self.haz_type in self.hazard_df.columns:
            hazard_types = self.hazard_df[self.haz_type].value_counts()
            converted_types = self.links_df['hazard_type'].value_counts()
            conversion_rates = (converted_types / hazard_types * 100).fillna(0)
            fig.add_trace(go.Bar(x=conversion_rates.values, y=conversion_rates.index, orientation='h', marker_color=conversion_rates.values, marker=dict(colorscale='RdYlGn_r'), text=[f'{v:.1f}%' for v in conversion_rates.values], textposition='outside'), row=1, col=2)

        # 3. Severity escalation scatter
        fig.add_trace(go.Scatter(x=self.links_df['hazard_severity'], y=self.links_df['incident_severity'], mode='markers', marker=dict(size=10, color=self.links_df['days_to_incident'], colorscale='Viridis', showscale=True, colorbar=dict(title="Days", x=0.45)), text=[f"Days: {d}" for d in self.links_df['days_to_incident']], hovertemplate='Hazard Severity: %{x}<br>Incident Severity: %{y}<br>%{text}<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=[1, 5], y=[1, 5], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False), row=2, col=1)

        # 4. Monthly trend
        if self.haz_date and self.haz_date in self.hazard_df.columns:
            monthly_hazards = self.hazard_df.groupby(self.hazard_df[self.haz_date].dt.to_period('M')).size()
            monthly_conversions = self.links_df.groupby(pd.to_datetime(self.links_df['hazard_date']).dt.to_period('M')).size()
            conversion_rate_monthly = (monthly_conversions / monthly_hazards * 100).fillna(0)
            fig.add_trace(go.Scatter(x=conversion_rate_monthly.index.astype(str), y=conversion_rate_monthly.values, mode='lines+markers', marker=dict(size=10), line=dict(width=3), name='Conversion Rate %'), row=2, col=2)

        fig.update_xaxes(title_text="Days", row=1, col=1)
        fig.update_xaxes(title_text="Conversion Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="Hazard Severity", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Hazard Type", row=1, col=2)
        fig.update_yaxes(title_text="Incident Severity", row=2, col=1)
        fig.update_yaxes(title_text="Conversion Rate (%)", row=2, col=2)
        fig.update_layout(height=800, showlegend=False, title_text="Hazard to Incident Conversion Analysis")
        return fig

    def create_sankey_flow(self) -> go.Figure:
        if self.hazard_df.empty or self.incident_df.empty:
            return _empty_chart("Insufficient data")

        haz_type_series = self.hazard_df[self.haz_type] if self.haz_type and self.haz_type in self.hazard_df.columns else pd.Series(dtype=object)
        inc_type_series = self.incident_df[self.inc_type] if self.inc_type and self.inc_type in self.incident_df.columns else pd.Series(dtype=object)
        hazard_types = haz_type_series.value_counts().head(5)
        incident_types = inc_type_series.value_counts().head(5)

        labels, colors = [], []
        for ht in hazard_types.index:
            labels.append(f"Hazard: {ht}")
            colors.append('#FFA500')
        labels.append("Conversion Process")
        colors.append('#FFD700')
        for it in incident_types.index:
            labels.append(f"Incident: {it}")
            colors.append('#FF6B6B')
        labels.append("Successfully Prevented")
        colors.append('#90EE90')

        source, target, value, link_colors = [], [], [], []
        process_idx = len(hazard_types)
        for i, (ht, count) in enumerate(hazard_types.items()):
            source.append(i)
            target.append(process_idx)
            value.append(int(count))
            link_colors.append('rgba(255, 165, 0, 0.4)')

        if not self.links_df.empty:
            incident_start_idx = process_idx + 1
            n_types = max(1, len(incident_types))
            for i, (it, count) in enumerate(incident_types.items()):
                source.append(process_idx)
                target.append(incident_start_idx + i)
                converted = min(int(count), int(len(self.links_df) / n_types))
                value.append(converted)
                link_colors.append('rgba(255, 107, 107, 0.4)')
            prevented_idx = len(labels) - 1
            source.append(process_idx)
            target.append(prevented_idx)
            value.append(max(0, int(hazard_types.sum() - len(self.links_df))))
            link_colors.append('rgba(144, 238, 144, 0.4)')

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=24,
                line=dict(color="#374151", width=1.2),
                label=labels,
                color=colors,
                hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                hovertemplate='Flow: %{value}<br>From: %{source.label}<br>To: %{target.label}<extra></extra>'
            )
        )])
        fig.update_layout(
            title="Hazard to Incident Flow Analysis",
            height=600,
            font=dict(family="Segoe UI, Arial, sans-serif", size=14, color="#111827"),
            paper_bgcolor="white"
        )
        return fig

    def create_department_conversion_matrix(self) -> go.Figure:
        if self.hazard_df.empty or self.incident_df.empty:
            return _empty_chart("Insufficient data")

        dept_col_h = self.haz_dept
        dept_col_i = self.inc_dept
        if not dept_col_h or not dept_col_i:
            return _empty_chart("Department column missing")

        dept_metrics: list[dict] = []
        for dept in pd.Series(self.hazard_df[dept_col_h].dropna().unique()).astype(str):
            dept_hazards = self.hazard_df[self.hazard_df[dept_col_h].astype(str) == dept]
            dept_incidents = self.incident_df[self.incident_df[dept_col_i].astype(str) == dept]
            if not self.links_df.empty:
                dept_conversions = self.links_df[self.links_df['department'].astype(str) == dept]
                conversion_rate = len(dept_conversions) / len(dept_hazards) * 100 if len(dept_hazards) > 0 else 0
            else:
                conversion_rate = 0
            dept_metrics.append({
                'Department': dept,
                'Total Hazards': len(dept_hazards),
                'Total Incidents': len(dept_incidents),
                'Conversion Rate (%)': conversion_rate,
                'Avg Hazard Severity': dept_hazards[self.haz_sev].mean() if self.haz_sev in dept_hazards.columns else 0,
                'Avg Incident Severity': dept_incidents[self.inc_sev].mean() if self.inc_sev in dept_incidents.columns else 0,
                'Prevention Success (%)': 100 - conversion_rate
            })
        dept_df = pd.DataFrame(dept_metrics)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Conversion Rate by Department',
                'Hazard vs Incident Count',
                'Prevention Success Rate',
                'Severity Comparison'
            ],
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        fig.add_trace(go.Bar(x=dept_df['Department'], y=dept_df['Conversion Rate (%)'], marker_color=dept_df['Conversion Rate (%)'], marker=dict(colorscale='RdYlGn_r', cmin=0, cmax=100), text=[f'{v:.1f}%' for v in dept_df['Conversion Rate (%)']], textposition='outside', name='Conversion Rate'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dept_df['Total Hazards'], y=dept_df['Total Incidents'], mode='markers+text', marker=dict(size=dept_df['Conversion Rate (%)'], color=dept_df['Conversion Rate (%)'], colorscale='RdYlGn_r', showscale=True, colorbar=dict(title="Conv %", x=1.15)), text=dept_df['Department'], textposition='top center', hovertemplate='%{text}<br>Hazards: %{x}<br>Incidents: %{y}<extra></extra>'), row=1, col=2)
        fig.add_trace(go.Bar(x=dept_df['Department'], y=dept_df['Prevention Success (%)'], marker_color=dept_df['Prevention Success (%)'], marker=dict(colorscale='RdYlGn', cmin=0, cmax=100), text=[f'{v:.1f}%' for v in dept_df['Prevention Success (%)']], textposition='outside', name='Prevention Success'), row=2, col=1)
        fig.add_trace(go.Bar(x=dept_df['Department'], y=dept_df['Avg Hazard Severity'], name='Hazard Severity', marker_color='orange'), row=2, col=2)
        fig.add_trace(go.Bar(x=dept_df['Department'], y=dept_df['Avg Incident Severity'], name='Incident Severity', marker_color='red'), row=2, col=2)
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(height=800, showlegend=False, title_text="Department-wise Hazard to Incident Analysis", barmode='group')
        return fig

    def create_risk_network(self) -> go.Figure:
        if not _NX_AVAILABLE:
            return _empty_chart("networkx is not installed (pip install networkx)")
        if self.links_df.empty:
            return _empty_chart("No hazard-incident links found")

        G = nx.Graph()
        # Limit nodes for readability
        haz_ids = list(pd.Series(self.links_df['hazard_id']).dropna().unique())[:20]
        inc_ids = list(pd.Series(self.links_df['incident_id']).dropna().unique())[:20]
        for h in haz_ids:
            G.add_node(f"H_{h}", node_type='hazard')
        for i in inc_ids:
            G.add_node(f"I_{i}", node_type='incident')
        # Add edges and ensure nodes exist with proper node_type
        for _, link in self.links_df.head(200).iterrows():
            haz_id = link.get('hazard_id')
            inc_id = link.get('incident_id')
            if pd.isna(haz_id) or pd.isna(inc_id):
                continue
            u = f"H_{haz_id}"
            v = f"I_{inc_id}"
            if u not in G:
                G.add_node(u, node_type='hazard')
            if v not in G:
                G.add_node(v, node_type='incident')
            G.add_edge(u, v, weight=link.get('days_to_incident', None))

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'))

        hazard_x, hazard_y, incident_x, incident_y = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_type = G.nodes[n].get('node_type')
            if node_type is None:
                # Fallback based on naming convention
                if str(n).startswith('H_'):
                    node_type = 'hazard'
                    G.nodes[n]['node_type'] = 'hazard'
                elif str(n).startswith('I_'):
                    node_type = 'incident'
                    G.nodes[n]['node_type'] = 'incident'
            if node_type == 'hazard':
                hazard_x.append(x); hazard_y.append(y)
            else:
                incident_x.append(x); incident_y.append(y)

        fig = go.Figure()
        for tr in edge_traces:
            fig.add_trace(tr)
        fig.add_trace(go.Scatter(x=hazard_x, y=hazard_y, mode='markers', marker=dict(size=15, color='orange', symbol='diamond'), name='Hazards', text=['Hazard' for _ in hazard_x], hovertemplate='%{text}<extra></extra>'))
        fig.add_trace(go.Scatter(x=incident_x, y=incident_y, mode='markers', marker=dict(size=15, color='red', symbol='circle'), name='Incidents', text=['Incident' for _ in incident_x], hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(title="Hazard-Incident Relationship Network", showlegend=True, hovermode='closest', height=600, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        return fig

    def create_prevention_effectiveness(self) -> go.Figure:
        if self.hazard_df.empty:
            return _empty_chart("No hazard data")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Hazard Closure Time vs Conversion',
                'Root Cause Analysis Impact',
                'Corrective Actions Effectiveness',
                'Time-based Prevention Success'
            ],
            specs=[[{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        haz = self.hazard_df.copy()
        closure_col = _first_present(haz, ['entered_closed'])
        if self.haz_date and closure_col:
            _coerce_datetime(haz, [self.haz_date, closure_col])
            haz['closure_time'] = (haz[closure_col] - haz[self.haz_date]).dt.days
        else:
            haz['closure_time'] = pd.NA

        if not self.links_df.empty and self.haz_id and self.haz_id in haz.columns:
            converted_ids = set(self.links_df['hazard_id'].dropna().unique().tolist())
            haz['converted'] = haz[self.haz_id].isin(converted_ids)
        else:
            haz['converted'] = False

        fig.add_trace(go.Box(y=haz.loc[~haz['converted'], 'closure_time'], name='Prevented', marker_color='green'), row=1, col=1)
        fig.add_trace(go.Box(y=haz.loc[haz['converted'], 'closure_time'], name='Became Incident', marker_color='red'), row=1, col=1)

        # 2. Root cause analysis impact (using incident data flags if present)
        rc_col = _first_present(self.incident_df, ['root_cause_is_missing'])
        if rc_col:
            rc_df = pd.DataFrame({
                'Category': ['With Root Cause', 'Without Root Cause'],
                'Count': [len(self.incident_df[~self.incident_df[rc_col]]), len(self.incident_df[self.incident_df[rc_col]])]
            })
            fig.add_trace(go.Bar(x=rc_df['Category'], y=rc_df['Count'], marker_color=['green', 'red'], text=rc_df['Count'], textposition='outside'), row=1, col=2)

        # 3. Corrective actions
        ca_col = _first_present(self.incident_df, ['corrective_actions_is_missing'])
        if ca_col:
            ca_df = pd.DataFrame({
                'Category': ['With Corrective Actions', 'Without Corrective Actions'],
                'Count': [len(self.incident_df[~self.incident_df[ca_col]]), len(self.incident_df[self.incident_df[ca_col]])]
            })
            fig.add_trace(go.Bar(x=ca_df['Category'], y=ca_df['Count'], marker_color=['green', 'orange'], text=ca_df['Count'], textposition='outside'), row=2, col=1)

        # 4. Time-based prevention trend
        if self.haz_date and self.haz_date in self.hazard_df.columns:
            period = self.hazard_df[self.haz_date].dt.to_period('M')
            monthly_total = self.hazard_df.groupby(period).size()
            status_col = self.haz_status
            if status_col:
                monthly_closed = self.hazard_df[self.hazard_df[status_col].astype(str).str.lower() == 'closed'].groupby(period).size()
            else:
                monthly_closed = pd.Series(0, index=monthly_total.index)
            rate = (monthly_closed / monthly_total * 100).reindex(monthly_total.index).fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=rate.index.astype(str),
                    y=rate.values,
                    mode='lines+markers',
                    marker=dict(size=10),
                    line=dict(width=3, color='green'),
                    name='Prevention Rate'
                ),
                row=2, col=2
            )

        fig.update_yaxes(title_text="Days to Closure", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Prevention Rate (%)", row=2, col=2)
        fig.update_layout(height=800, title_text="Prevention Effectiveness Analysis", showlegend=False)
        return fig


def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False), height=400)
    return fig


def render_conversion_page(workbook: dict):
    """Render the full Hazard→Incident analysis page using workbook data."""
    import streamlit as st

    incident_df = _get_sheet(workbook, 'incident')
    hazard_df = _get_sheet(workbook, 'hazard')
    relationships_df = _get_sheet(workbook, 'relationship')

    if incident_df is None or hazard_df is None:
        st.info("Incident or Hazard sheet not found.")
        return

    analyzer = HazardIncidentAnalyzer(incident_df, hazard_df, relationships_df)

    st.subheader("Hazard to Incident Conversion Analysis")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Hazards", f"{len(hazard_df):,}")
    with col2:
        st.metric("Total Incidents", f"{len(incident_df):,}")
    with col3:
        conv = (len(analyzer.links_df) / len(hazard_df) * 100) if len(hazard_df) > 0 else 0
        st.metric("Conversion Rate", f"{conv:.1f}%")
    with col4:
        prevention_rate = 100 - ((len(analyzer.links_df) / len(hazard_df) * 100) if len(hazard_df) > 0 else 0)
        st.metric("Prevention Success", f"{prevention_rate:.1f}%")
    with col5:
        avg_days = analyzer.links_df['days_to_incident'].mean() if not analyzer.links_df.empty else 0
        st.metric("Avg Days to Incident", f"{(0 if pd.isna(avg_days) else avg_days):.1f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Conversion Funnel",
        "⏱️ Time Analysis",
        "🌊 Flow Diagram",
        "🏢 Department Matrix",
        "🕸️ Risk Network",
        "🛡️ Prevention Effectiveness"
    ])

    with tab1:
        fig1 = analyzer.create_conversion_funnel()
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = analyzer.create_time_lag_analysis()
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = analyzer.create_sankey_flow()
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = analyzer.create_department_conversion_matrix()
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        fig5 = analyzer.create_risk_network()
        st.plotly_chart(fig5, use_container_width=True)

    with tab6:
        fig6 = analyzer.create_prevention_effectiveness()
        st.plotly_chart(fig6, use_container_width=True)


def create_conversion_metrics_card(workbook: dict) -> go.Figure:
    """Create a compact prevention gauge metric using real data from workbook."""
    incident_df = _get_sheet(workbook, 'incident')
    hazard_df = _get_sheet(workbook, 'hazard')
    if incident_df is None or hazard_df is None or hazard_df.empty:
        return _empty_chart("Insufficient data for prevention gauge")

    # Ensure datetime
    haz_date = _first_present(hazard_df, ['occurrence_date', 'reported_date', 'entered_date'])
    inc_date = _first_present(incident_df, ['occurrence_date', 'reported_date', 'entered_date'])
    if haz_date:
        _coerce_datetime(hazard_df, [haz_date])
    if inc_date:
        _coerce_datetime(incident_df, [inc_date])

    total_hazards = len(hazard_df)
    status_col = _first_present(hazard_df, ['status'])
    closed_hazards = (hazard_df[status_col].astype(str).str.lower() == 'closed').sum() if status_col else 0

    # Heuristic for estimated conversions: incidents after first hazard date
    if haz_date and inc_date:
        earliest_haz = hazard_df[haz_date].min()
        similar_incidents = incident_df[incident_df[inc_date] >= earliest_haz]
        estimated_conversions = min(len(similar_incidents), total_hazards)
    else:
        estimated_conversions = 0

    prevention_rate = max(0.0, (1 - (estimated_conversions / total_hazards)) * 100) if total_hazards > 0 else 100.0

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=prevention_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hazard Prevention Rate"},
        delta={'reference': 90, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if prevention_rate > 80 else "orange" if prevention_rate > 60 else "red"},
            'steps': [
                {'range': [0, 60], 'color': "#FDE68A"},
                {'range': [60, 80], 'color': "#FACC15"},
                {'range': [80, 100], 'color': "#86EFAC"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


