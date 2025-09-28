from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from ..models.schemas import (
    PlotlyFigureResponse,
    ChartInsightsRequest,
    ChartInsightsResponse,
    DataInsightsRequest,
)
from ..services.excel import (
    get_incident_df,
    get_hazard_df,
    get_audit_df,
    get_inspection_df,
)
from ..services import plots as plot_service
from ..services.json_utils import to_native_json
from ..services.agent import ask_openai
from ..services.excel import payload_to_df


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/hse-scorecard", response_model=PlotlyFigureResponse)
async def hse_scorecard():
    inc = get_incident_df()
    haz = get_hazard_df()
    aud = get_audit_df()
    ins = get_inspection_df()
    fig = plot_service.create_unified_hse_scorecard(inc, haz, aud, ins)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/hse-scorecard/insights", response_model=ChartInsightsResponse)
async def hse_scorecard_insights():
    inc = get_incident_df()
    haz = get_hazard_df()
    aud = get_audit_df()
    ins = get_inspection_df()
    fig = plot_service.create_unified_hse_scorecard(inc, haz, aud, ins)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="HSE Scorecard")
    return await generate_chart_insights(payload)


@router.get("/hse-performance-index", response_model=PlotlyFigureResponse)
async def hse_performance_index(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_hse_performance_index(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/hse-performance-index/insights", response_model=ChartInsightsResponse)
async def hse_performance_index_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_hse_performance_index(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="HSE Performance Index")
    return await generate_chart_insights(payload)


@router.get("/risk-calendar-heatmap", response_model=PlotlyFigureResponse)
async def risk_calendar_heatmap(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_risk_calendar_heatmap(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/risk-calendar-heatmap/insights", response_model=ChartInsightsResponse)
async def risk_calendar_heatmap_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_risk_calendar_heatmap(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Risk Calendar Heatmap")
    return await generate_chart_insights(payload)


@router.get("/psm-breakdown", response_model=PlotlyFigureResponse)
async def psm_breakdown(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_psm_breakdown(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/psm-breakdown/insights", response_model=ChartInsightsResponse)
async def psm_breakdown_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_psm_breakdown(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="PSM Breakdown")
    return await generate_chart_insights(payload)


@router.get("/consequence-matrix", response_model=PlotlyFigureResponse)
async def consequence_matrix(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_consequence_matrix(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/consequence-matrix/insights", response_model=ChartInsightsResponse)
async def consequence_matrix_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_consequence_matrix(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Consequence Matrix")
    return await generate_chart_insights(payload)


@router.get("/data-quality-metrics", response_model=PlotlyFigureResponse)
async def data_quality_metrics(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_data_quality_metrics(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/data-quality-metrics/insights", response_model=ChartInsightsResponse)
async def data_quality_metrics_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_data_quality_metrics(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Data Quality Metrics")
    return await generate_chart_insights(payload)


@router.get("/comprehensive-timeline", response_model=PlotlyFigureResponse)
async def comprehensive_timeline(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_comprehensive_timeline(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/comprehensive-timeline/insights", response_model=ChartInsightsResponse)
async def comprehensive_timeline_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_comprehensive_timeline(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Comprehensive Timeline")
    return await generate_chart_insights(payload)


@router.get("/audit-inspection-tracker", response_model=PlotlyFigureResponse)
async def audit_inspection_tracker():
    audit_df = get_audit_df()
    inspection_df = get_inspection_df()
    fig = plot_service.create_audit_inspection_tracker(audit_df, inspection_df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/audit-inspection-tracker/insights", response_model=ChartInsightsResponse)
async def audit_inspection_traker_insights():
    audit_df = get_audit_df()
    inspection_df = get_inspection_df()
    fig = plot_service.create_audit_inspection_tracker(audit_df, inspection_df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Audit & Inspection Tracker")
    return await generate_chart_insights(payload)


@router.get("/location-risk-treemap", response_model=PlotlyFigureResponse)
async def location_risk_treemap(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_location_risk_treemap(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/location-risk-treemap/insights", response_model=ChartInsightsResponse)
async def location_risk_treemap_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_location_risk_treemap(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Location Risk Treemap")
    return await generate_chart_insights(payload)


@router.get("/department-spider", response_model=PlotlyFigureResponse)
async def department_spider(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_department_spider(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/department-spider/insights", response_model=ChartInsightsResponse)
async def department_spider_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_department_spider(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Department Spider")
    return await generate_chart_insights(payload)


@router.get("/violation-analysis", response_model=PlotlyFigureResponse)
async def violation_analysis(dataset: str = Query("hazard", description="Dataset to use: incident or hazard")):
    df = get_hazard_df() if (dataset or "hazard").lower() == "hazard" else get_incident_df()
    fig = plot_service.create_violation_analysis(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/violation-analysis/insights", response_model=ChartInsightsResponse)
async def violation_analysis_insights(dataset: str = Query("hazard", description="Dataset to use: incident or hazard")):
    df = get_hazard_df() if (dataset or "hazard").lower() == "hazard" else get_incident_df()
    fig = plot_service.create_violation_analysis(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Violation Analysis")
    return await generate_chart_insights(payload)


@router.get("/cost-prediction-analysis", response_model=PlotlyFigureResponse)
async def cost_prediction_analysis(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_cost_prediction_analysis(df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/cost-prediction-analysis/insights", response_model=ChartInsightsResponse)
async def cost_prediction_analysis_insights(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_cost_prediction_analysis(df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Cost Prediction Analysis")
    return await generate_chart_insights(payload)


@router.get("/facility-layout-heatmap", response_model=PlotlyFigureResponse)
async def facility_layout_heatmap():
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    fig = plot_service.create_facility_layout_heatmap(inc_df, haz_df)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/facility-layout-heatmap/insights", response_model=ChartInsightsResponse)
async def facility_layout_heatmap_insights():
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    fig = plot_service.create_facility_layout_heatmap(inc_df, haz_df)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Facility Layout Heatmap")
    return await generate_chart_insights(payload)


@router.get("/facility-3d-heatmap", response_model=PlotlyFigureResponse)
async def facility_3d_heatmap(
    dataset: str = Query("incident", description="Dataset to use: incident or hazard"),
    event_type: str = Query("Incidents", description="Label for the 3D surface legend/title"),
):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_3d_facility_heatmap(df, event_type=event_type)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/facility-3d-heatmap/insights", response_model=ChartInsightsResponse)
async def facility_3d_heatmap_insights(
    dataset: str = Query("incident", description="Dataset to use: incident or hazard"),
    event_type: str = Query("Incidents", description="Label for the 3D surface legend/title"),
):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    fig = plot_service.create_3d_facility_heatmap(df, event_type=event_type)
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title="Facility 3D Heatmap")
    return await generate_chart_insights(payload)


@router.post("/insights", response_model=ChartInsightsResponse)
async def generate_chart_insights(payload: ChartInsightsRequest) -> ChartInsightsResponse:
    """Generate layman-friendly insights from a Plotly figure JSON.
    Heuristic summary first, then optionally refined with LLM if available.
    """
    fig = payload.figure or {}
    title = payload.title or fig.get("layout", {}).get("title") or "Chart"

    def _coerce_num(arr):
        try:
            s = pd.Series(arr)
            # Handle numeric-like strings such as "1,234" or "10%"
            s = pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
            return s
        except Exception:
            return pd.Series([], dtype=float)

    def _top_n_pairs(x, y, n=5):
        s = pd.Series(y, index=pd.Index(x, name="label"))
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        top = s.sort_values(ascending=False).head(n)
        return list(top.items())

    def _safe_get(tr: dict, key: str, alt_key: str | None = None):
        v = tr.get(key)
        if v is None and alt_key is not None:
            v = tr.get(alt_key)
        return v if v is not None else []

    def _get_bar_numeric_and_labels(tr):
        orientation = (tr.get("orientation") or "v").lower()
        x = _safe_get(tr, "x")
        y = _safe_get(tr, "y")
        if orientation == "h":
            nums = _coerce_num(x)
            labels = y
        else:
            nums = _coerce_num(y)
            labels = x
        # Fallbacks if nums are all NaN
        if nums.dropna().empty:
            nums = _coerce_num(_safe_get(tr, "text"))
        if nums.dropna().empty:
            nums = _coerce_num(_safe_get(tr, "values"))
        cd = tr.get("customdata")
        if nums.dropna().empty and isinstance(cd, (list, tuple, np.ndarray)):
            try:
                # Try first column of customdata if it's 2D
                cd_list = cd.tolist() if isinstance(cd, np.ndarray) else cd
                if cd_list and isinstance(cd_list[0], (list, tuple)):
                    nums = _coerce_num([row[0] for row in cd_list])
                else:
                    nums = _coerce_num(cd_list)
            except Exception:
                pass
        return nums, labels

    # Heuristic extraction across traces
    data = fig.get("data", []) or []
    findings: list[str] = []
    recs: list[str] = []

    if not data:
        insights_md = f"## {title}\n\n- **Summary**: No data provided. Please render a chart first."
        return ChartInsightsResponse(insights_md=insights_md)

    # Aggregate stats across numeric series
    series_stats = []
    all_trace_names: list[str] = []
    for tr in data:
        ttype = (tr.get("type") or tr.get("_type") or "").lower()
        name = tr.get("name") or ttype or "series"
        x = _safe_get(tr, "x")
        y = _safe_get(tr, "y")
        all_trace_names.append(name)
        if ttype in ("bar", "funnel"):
            nums, labels = _get_bar_numeric_and_labels(tr)
            if not nums.dropna().empty:
                series_stats.append({
                    "name": name,
                    "count": int(nums.count()),
                    "sum": float(nums.sum()),
                    "mean": float(nums.mean()),
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                })
                if labels and len(labels) == len(nums):
                    top = _top_n_pairs(labels, list(nums.fillna(0).values), n=3)
                    if top:
                        findings.append(
                            f"Top contributors in {name}: " + ", ".join([f"{lbl} ({val:.0f})" for lbl, val in top])
                        )
        elif ttype in ("scatter", "line", "scattergl"):
            ys = _coerce_num(y)
            if ys.dropna().empty and x:
                # Some lines may be horizontal with numeric x
                ys = _coerce_num(x)
            if not ys.dropna().empty:
                series_stats.append({
                    "name": name,
                    "count": int(ys.count()),
                    "sum": float(ys.sum()),
                    "mean": float(ys.mean()),
                    "min": float(ys.min()),
                    "max": float(ys.max()),
                })
        elif ttype == "histogram":
            xs = _coerce_num(x)
            if not xs.dropna().empty:
                series_stats.append({
                    "name": name,
                    "count": int(xs.count()),
                    "mean": float(xs.mean()),
                    "min": float(xs.min()),
                    "max": float(xs.max()),
                    "sum": float(xs.sum()),
                })
        elif ttype in ("box", "violin"):
            ys = _coerce_num(y)
            if not ys.dropna().empty:
                series_stats.append({
                    "name": name,
                    "count": int(ys.count()),
                    "mean": float(ys.mean()),
                    "min": float(ys.min()),
                    "max": float(ys.max()),
                })
        elif ttype == "heatmap":
            z = _safe_get(tr, "z")
            try:
                zarr = np.array(z, dtype=float)
                if np.isfinite(zarr).any():
                    total = float(np.nansum(zarr))
                    mx = float(np.nanmax(zarr)) if np.isfinite(zarr).any() else 0.0
                    series_stats.append({
                        "name": name,
                        "sum": total,
                        "max": mx,
                    })
                    # Hotspot labels if x/y provided
                    xi = _safe_get(tr, "x")
                    yi = _safe_get(tr, "y")
                    if isinstance(zarr, np.ndarray) and zarr.ndim == 2 and xi and yi:
                        # Find top 3 cells
                        idx = np.dstack(np.unravel_index(np.argsort(-zarr, axis=None)[:3], zarr.shape))[0]
                        labels = []
                        for r, c in idx:
                            try:
                                labels.append(f"{yi[r]} / {xi[c]} ({zarr[r, c]:.0f})")
                            except Exception:
                                pass
                        if labels:
                            findings.append("Heatmap hotspots: " + ", ".join(labels))
            except Exception:
                pass
        elif ttype in ("scatterpolar", "barpolar"):
            # Polar/radar charts: values in r, labels in theta
            r_vals = _safe_get(tr, "r")
            theta_vals = _safe_get(tr, "theta", alt_key="labels")
            rs = _coerce_num(r_vals)
            if not rs.dropna().empty:
                series_stats.append({
                    "name": name,
                    "count": int(rs.count()),
                    "sum": float(rs.sum()),
                    "mean": float(rs.mean()),
                    "min": float(rs.min()),
                    "max": float(rs.max()),
                })
                if theta_vals and len(theta_vals) == len(rs):
                    top = _top_n_pairs(theta_vals, list(rs.fillna(0).values), n=3)
                    if top:
                        findings.append(
                            f"Top categories in {name}: " + ", ".join([f"{lbl} ({val:.0f})" for lbl, val in top])
                        )
        elif ttype == "pie":
            labels = _safe_get(tr, "labels")
            values = _safe_get(tr, "values")
            if labels and values and len(labels) == len(values):
                top = _top_n_pairs(labels, values, n=3)
                total = float(pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).sum())
                share = ", ".join([f"{lbl} ({(val/total*100 if total>0 else 0):.1f}%)" for lbl, val in top])
                findings.append(f"Largest slices: {share}")
        elif ttype == "indicator":
            val = tr.get("value")
            try:
                v = float(val)
                findings.append(f"Current value: {v:,.0f}")
            except Exception:
                pass

    # Trend detection for lines/scatters (first numeric series)
    for tr in data:
        ttype = (tr.get("type") or "").lower()
        if ttype in ("scatter", "line", "scattergl"):
            y = tr.get("y") or []
            ys = _coerce_num(y)
            if ys.dropna().empty or len(ys) < 2:
                continue
            slope = ys.diff().mean()
            if pd.notna(slope):
                if slope > 0:
                    findings.append("Upward trend observed over time.")
                elif slope < 0:
                    findings.append("Downward trend observed over time.")
            break

    # If nothing numeric was found, still return a helpful summary
    if not series_stats:
        parts = [f"## {title}"]
        parts.append(f"- **Traces detected**: {len(data)} — " + ", ".join([str(n) for n in all_trace_names[:10]]))
        parts.append("- **Summary**: The chart data does not contain numeric values I can analyze reliably (y/x/text/values/customdata are non-numeric or empty).")
        parts.append("\n### Recommendations")
        parts.append("- **Action**: Ensure numeric fields are provided for metrics like Count, Severity (as scores/buckets mapped to scores), Risk, Cost, or Manhours.")
        parts.append("- **Action**: If values are strings (e.g., '10'), keep them numeric or pass as numbers in Plotly traces.")
        parts.append("- **Action**: For stacked bars, include numeric arrays in `y` (vertical) or `x` (horizontal). Optionally set `orientation` to clarify.")
        base_md = "\n".join(parts)

    # Optional LLM refinement to produce layman-friendly prose
    try:
        context_chunks = [
            f"Title: {title}",
            f"User context: {(payload.context or '').strip() or 'N/A'}",
            "Heuristic summary:\n" + base_md,
        ]
        llm_md = ask_openai(
            "Rewrite the heuristic chart summary into a clear, layman-friendly report. Use Markdown with sections: Summary, Key Insights (bullets), Recommendations (bullets). Keep it concise and avoid jargon.",
            context="\n\n".join(context_chunks),
            model="gpt-4o",
            code_mode=False,
            multi_df=False,
        )
        if llm_md and not llm_md.lower().startswith("openai") and "not installed" not in llm_md.lower():
            return ChartInsightsResponse(insights_md=llm_md)
    except Exception:
        pass

    return ChartInsightsResponse(insights_md=base_md)


# ---------- Chart-specific AI insights (GET) ----------

def _build_chart_figure(chart: str, dataset: str = "incident", event_type: str = "Incidents"):
    chart_l = (chart or "").strip().lower()
    ds_l = (dataset or "incident").strip().lower()
    # Load common dataframes
    inc = get_incident_df()
    haz = get_hazard_df()
    aud = get_audit_df()
    ins = get_inspection_df()

    # Map chart name to figure
    if chart_l in ("hse-scorecard", "scorecard"):
        fig = plot_service.create_unified_hse_scorecard(inc, haz, aud, ins)
        title = "HSE Scorecard"
        return fig, title
    if chart_l in ("hse-performance-index", "performance-index"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_hse_performance_index(df)
        title = "HSE Performance Index"
        return fig, title
    if chart_l in ("risk-calendar-heatmap", "risk-calendar"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_risk_calendar_heatmap(df)
        title = "Risk Calendar Heatmap"
        return fig, title
    if chart_l in ("psm-breakdown", "psm"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_psm_breakdown(df)
        title = "PSM Breakdown"
        return fig, title
    if chart_l in ("consequence-matrix", "consequence"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_consequence_matrix(df)
        title = "Consequence Matrix"
        return fig, title
    if chart_l in ("data-quality-metrics", "data-quality"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_data_quality_metrics(df)
        title = "Data Quality Metrics"
        return fig, title
    if chart_l in ("comprehensive-timeline", "timeline"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_comprehensive_timeline(df)
        title = "Comprehensive Timeline"
        return fig, title
    if chart_l in ("audit-inspection-tracker", "audit-tracker"):
        fig = plot_service.create_audit_inspection_tracker(aud, ins)
        title = "Audit & Inspection Tracker"
        return fig, title
    if chart_l in ("location-risk-treemap", "location-treemap"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_location_risk_treemap(df)
        title = "Location Risk Treemap"
        return fig, title
    if chart_l in ("department-spider", "spider"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_department_spider(df)
        title = "Department Spider"
        return fig, title
    if chart_l in ("violation-analysis", "violation"):
        # default hazard
        df = haz if ds_l == "hazard" else inc
        fig = plot_service.create_violation_analysis(df)
        title = "Violation Analysis"
        return fig, title
    if chart_l in ("cost-prediction-analysis", "cost-prediction"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_cost_prediction_analysis(df)
        title = "Cost Prediction Analysis"
        return fig, title
    if chart_l in ("facility-layout-heatmap", "layout-heatmap"):
        fig = plot_service.create_facility_layout_heatmap(inc, haz)
        title = "Facility Layout Heatmap"
        return fig, title
    if chart_l in ("facility-3d-heatmap", "3d-heatmap"):
        df = inc if ds_l == "incident" else haz
        fig = plot_service.create_3d_facility_heatmap(df, event_type=event_type or ("Incidents" if ds_l == "incident" else "Hazards"))
        title = "Facility 3D Heatmap"
        return fig, title
    raise ValueError(f"Unknown chart: {chart}")


@router.get("/insights/{chart}", response_model=ChartInsightsResponse)
async def insights_for_chart(chart: str, dataset: str = Query("incident"), event_type: str = Query("Incidents")) -> ChartInsightsResponse:
    """Build the specified chart using current data, then generate AI insights for it.
    Returns only the AI-generated insights (Markdown) to the client.
    """
    try:
        fig, title = _build_chart_figure(chart, dataset=dataset, event_type=event_type)
    except Exception as e:
        return ChartInsightsResponse(insights_md=f"## {chart}\n\n- **Error**: {e}")

    # Reuse the chart-based insights generator
    payload = ChartInsightsRequest(figure=fig.to_plotly_json(), title=title)
    resp = await generate_chart_insights(payload)  # type: ignore[arg-type]
    return resp

    # Default recommendations
    if not recs:
        recs = [
            "Highlight the top contributors to focus improvement efforts.",
            "Investigate outliers and recent changes for root causes.",
            "Track this metric monthly and set targets for the next quarter.",
        ]

    # Build initial markdown
    parts = [f"## {title}"]
    if series_stats:
        for s in series_stats[:3]:
            parts.append(
                f"- **{s['name']}** — min: {s['min']:.1f}, max: {s['max']:.1f}, mean: {s['mean']:.1f}, total: {s['sum']:.1f}"
            )
    if findings:
        parts.append("\n### Key insights")
        for f in findings:
            parts.append(f"- **Insight**: {f}")
    parts.append("\n### Recommendations")
    for r in recs:
        parts.append(f"- **Action**: {r}")

    base_md = "\n".join(parts)

    # Optionally refine with LLM if available
    try:
        context_chunks = [
            f"Title: {title}",
            f"User context: {payload.context or 'N/A'}",
            "Heuristic summary:\n" + base_md,
        ]
        llm_md = ask_openai(
            "Rewrite the heuristic chart summary into a clear, layman-friendly report. Use Markdown with sections: Summary, Key Insights (bullets), Recommendations (bullets). Keep it concise and avoid jargon.",
            context="\n\n".join(context_chunks),
            model="gpt-4o",
            code_mode=False,
            multi_df=False,
        )
        if llm_md and not llm_md.lower().startswith("openai") and "not installed" not in llm_md.lower():
            return ChartInsightsResponse(insights_md=llm_md)
    except Exception:
        pass

    return ChartInsightsResponse(insights_md=base_md)


@router.post("/insights/from-data", response_model=ChartInsightsResponse)
async def generate_data_insights(payload: DataInsightsRequest) -> ChartInsightsResponse:
    """Data-driven insights that do not rely on chart structure. Deterministic and robust.
    Accepts records plus optional column mappings.
    """
    title = payload.title or "Insights"
    df = payload_to_df(payload.data.records) if payload and payload.data else pd.DataFrame()
    if df is None or df.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No data provided.")

    def _pick(colnames: list[str]) -> str | None:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    # Infer columns if not provided
    time_col = payload.time_col or _pick([
        "date", "occurrence_date", "entered_date", "reported_date", "created_date", "start_date", "scheduled_date",
    ])
    cat_col = payload.category_col or _pick([
        "department", "dept", "category", "location", "site", "area",
    ])

    metrics = [m.lower() for m in (payload.metrics or ["count", "severity", "risk", "cost", "manhours"])]
    value_cols = payload.value_cols or {}
    sev_col = value_cols.get("severity") or _pick(["severity_score", "severity", "severity_level"])  # numeric preferred
    risk_col = value_cols.get("risk") or _pick(["risk_score", "risk", "risk_level"])  # numeric preferred
    cost_col = value_cols.get("cost") or _pick(["cost", "estimated_cost", "cost_usd", "cost_inr"])  # numeric
    mh_col = value_cols.get("manhours") or _pick(["manhours", "man_hours", "man_hrs", "hours"])  # numeric

    # Coerce numeric columns safely
    for c in [sev_col, risk_col, cost_col, mh_col]:
        if c and c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
            except Exception:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce time column
    if time_col and time_col in df.columns:
        try:
            df["__dt"] = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            df["__dt"] = pd.NaT
    else:
        df["__dt"] = pd.NaT

    parts: list[str] = [f"## {title}"]

    # Summary block
    parts.append(f"- **Rows**: {len(df):,}")
    if df["__dt"].notna().any():
        dmin = df["__dt"].min()
        dmax = df["__dt"].max()
        try:
            parts.append(f"- **Date range**: {dmin.date().isoformat()} to {dmax.date().isoformat()}")
        except Exception:
            parts.append("- **Date range**: available")

    # Top contributors by category
    if cat_col and cat_col in df.columns:
        vc = (
            df[cat_col].astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(payload.top_n)
        )
        if not vc.empty:
            parts.append("\n### Top contributors")
            for idx, val in vc.items():
                parts.append(f"- **{idx}**: {int(val):,}")

    # Metric stats
    def _metric_block(name: str, col: str | None):
        if not col or col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return
        parts.append(f"\n### {name.title()} statistics")
        parts.append(f"- **Mean**: {float(s.mean()):,.2f}")
        parts.append(f"- **Min/Max**: {float(s.min()):,.2f} / {float(s.max()):,.2f}")
        if cat_col and cat_col in df.columns:
            grp = s.groupby(df[cat_col].astype(str).str.strip().replace({"": "Unknown"})).mean().sort_values(ascending=False).head(3)
            if not grp.empty:
                tops = ", ".join([f"{k} ({v:,.2f})" for k, v in grp.items()])
                parts.append(f"- **Top by {name} (avg)**: {tops}")

    if "severity" in metrics:
        _metric_block("severity", sev_col)
    if "risk" in metrics:
        _metric_block("risk", risk_col)
    if "cost" in metrics:
        _metric_block("cost", cost_col)
    if "manhours" in metrics:
        _metric_block("manhours", mh_col)

    # Trend analysis (monthly counts)
    if df["__dt"].notna().any():
        monthly = df.dropna(subset=["__dt"]).copy()
        monthly["__ym"] = monthly["__dt"].dt.to_period("M").dt.to_timestamp()
        cnt = monthly.groupby("__ym").size().sort_index()
        if len(cnt) >= 3:
            last3 = cnt.tail(3).values
            if len(set(last3)) > 1:
                pct = (last3[-1] - last3[0]) / max(1, last3[0]) * 100.0
                direction = "up" if pct > 0 else "down"
                parts.append(f"\n### Trend")
                parts.append(f"- **Monthly count trend**: {direction} {abs(pct):.1f}% over last 3 months")

    # Data quality
    dq_notes: list[str] = []
    for name, col in [("severity", sev_col), ("risk", risk_col), ("cost", cost_col), ("manhours", mh_col)]:
        if col and col in df.columns:
            miss = df[col].isna().sum()
            rate = miss / len(df) * 100.0 if len(df) else 0.0
            if rate >= 30.0:
                dq_notes.append(f"- **{name}** missing {rate:.0f}% — improve capture to strengthen insights.")
    if dq_notes:
        parts.append("\n### Data quality")
        parts.extend(dq_notes)

    return ChartInsightsResponse(insights_md="\n".join(parts))
