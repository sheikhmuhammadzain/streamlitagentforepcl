from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models.schemas import ConversionRequest, PlotlyFigureResponse
from ..services.excel import payload_to_df, get_incident_df, get_hazard_df
from ..services.json_utils import to_native_json

from analytics.hazard_incident import HazardIncidentAnalyzer
import pandas as pd


router = APIRouter(prefix="/analytics/conversion", tags=["analytics-conversion"])


def _analyzer_from_req(req: ConversionRequest) -> HazardIncidentAnalyzer:
    inc = payload_to_df(req.incident.records) if req.incident else None
    haz = payload_to_df(req.hazard.records) if req.hazard else None
    rel = payload_to_df(req.relationships.records) if req.relationships else None
    return HazardIncidentAnalyzer(inc, haz, rel)


# Removed POST /funnel endpoint (use GET /funnel)


@router.get("/funnel", response_model=PlotlyFigureResponse)
async def conversion_funnel_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_conversion_funnel()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /time-lag endpoint (use GET /time-lag)


@router.get("/time-lag", response_model=PlotlyFigureResponse)
async def time_lag_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_time_lag_analysis()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /sankey endpoint (use GET /sankey)


@router.get("/sankey", response_model=PlotlyFigureResponse)
async def sankey_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_sankey_flow()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /department-matrix endpoint (use GET /department-matrix)


@router.get("/department-matrix", response_model=PlotlyFigureResponse)
async def department_matrix_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_department_conversion_matrix()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /risk-network endpoint (use GET /risk-network)


@router.get("/risk-network", response_model=PlotlyFigureResponse)
async def risk_network_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_risk_network()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /prevention-effectiveness endpoint (use GET /prevention-effectiveness)


@router.get("/prevention-effectiveness", response_model=PlotlyFigureResponse)
async def prevention_effectiveness_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_prevention_effectiveness()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# Removed POST /metrics-gauge endpoint (use GET /metrics-gauge)


@router.get("/metrics-gauge", response_model=PlotlyFigureResponse)
async def metrics_gauge_auto():
    from analytics.hazard_incident import create_conversion_metrics_card
    workbook = {
        'Incidents': get_incident_df(),
        'Hazards': get_hazard_df(),
    }
    fig = create_conversion_metrics_card(workbook)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


# ---------- Relationship Data Endpoints (JSON) ----------

@router.get("/links")
async def hazard_incident_links():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    df = analyzer.links_df
    if df is None or df.empty:
        payload = {"total": 0, "unique_hazards": 0, "unique_incidents": 0}
        return JSONResponse(content=payload)
    # Return only counts
    total = int(len(df))
    uniq_h = int(df['hazard_id'].dropna().nunique()) if 'hazard_id' in df.columns else 0
    uniq_i = int(df['incident_id'].dropna().nunique()) if 'incident_id' in df.columns else 0
    payload = {"total": total, "unique_hazards": uniq_h, "unique_incidents": uniq_i}
    return JSONResponse(content=to_native_json(payload))


@router.get("/metrics")
async def hazard_incident_metrics():
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    links = analyzer.links_df

    total_hazards = 0 if haz is None else int(len(haz))
    total_incidents = 0 if inc is None else int(len(inc))

    hazards_became_incidents = 0
    avg_days_to_incident = 0.0
    if links is not None and not links.empty:
        if 'hazard_id' in links.columns:
            hazards_became_incidents = int(links['hazard_id'].dropna().nunique())
        if 'days_to_incident' in links.columns:
            avg = pd.to_numeric(links['days_to_incident'], errors='coerce').dropna().mean()
            if pd.notna(avg):
                avg_days_to_incident = float(avg)

    hazards_closed = 0
    hazards_open = total_hazards
    if haz is not None and hasattr(analyzer, 'haz_status') and analyzer.haz_status and analyzer.haz_status in haz.columns:
        s = haz[analyzer.haz_status].astype(str).str.lower()
        hazards_closed = int((s == 'closed').sum())
        hazards_open = int(total_hazards - hazards_closed)

    prevented_hazards = int(max(0, hazards_closed -hazards_became_incidents))
    conversion_rate = (hazards_became_incidents / total_hazards * 100.0) if total_hazards > 0 else 0.0
    prevention_rate = 100.0 - conversion_rate

    payload = {
        "total_hazards": total_hazards,
        "total_incidents": total_incidents,
        "hazards_became_incidents": hazards_became_incidents,
        "hazards_closed": hazards_closed,
        "hazards_open":hazards_open,
        "prevented_hazards": prevented_hazards,
        "conversion_rate_pct": round(conversion_rate, 2),
        "prevention_rate_pct": round(prevention_rate, 2),
        "avg_days_to_incident": round(avg_days_to_incident, 2),
    }
    return JSONResponse(content=to_native_json(payload))


@router.get("/department-metrics-data")
async def department_metrics_data():
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)

    if inc is None or haz is None or inc.empty or haz.empty:
        return JSONResponse(content=[])

    dept_col_h = analyzer.haz_dept
    dept_col_i = analyzer.inc_dept
    if not dept_col_h or not dept_col_i or dept_col_h not in haz.columns or dept_col_i not in inc.columns:
        return JSONResponse(content=[])

    metrics: list[dict] = []
    links = analyzer.links_df if analyzer.links_df is not None else pd.DataFrame()
    for dept in pd.Series(haz[dept_col_h].dropna().unique()).astype(str):
        dept_hazards = haz[haz[dept_col_h].astype(str) == dept]
        dept_incidents = inc[inc[dept_col_i].astype(str) == dept]
        if not links.empty and 'department' in links.columns:
            dept_conversions = links[links['department'].astype(str) == dept]
            conversion_rate = (len(dept_conversions) / len(dept_hazards) * 100.0) if len(dept_hazards) > 0 else 0.0
        else:
            conversion_rate = 0.0
        avg_haz_sev = 0.0
        if hasattr(analyzer, 'haz_sev') and analyzer.haz_sev and analyzer.haz_sev in dept_hazards.columns:
            avg_haz_sev = float(pd.to_numeric(dept_hazards[analyzer.haz_sev], errors='coerce').dropna().mean() or 0)
        avg_inc_sev = 0.0
        if hasattr(analyzer, 'inc_sev') and analyzer.inc_sev and analyzer.inc_sev in dept_incidents.columns:
            avg_inc_sev = float(pd.to_numeric(dept_incidents[analyzer.inc_sev], errors='coerce').dropna().mean() or 0)
        metrics.append({
            "department": dept,
            "total_hazards": int(len(dept_hazards)),
            "total_incidents": int(len(dept_incidents)),
            "conversion_rate_pct": round(conversion_rate, 2),
            "avg_hazard_severity": round(avg_haz_sev, 2),
            "avg_incident_severity": round(avg_inc_sev, 2),
            "prevention_success_pct": round(100.0 - conversion_rate, 2),
        })

    return JSONResponse(content=to_native_json(metrics))
