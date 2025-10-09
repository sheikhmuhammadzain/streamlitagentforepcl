from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json

from ..models.schemas import ConversionRequest, PlotlyFigureResponse, ChartInsightsResponse
from ..services.excel import payload_to_df, get_incident_df, get_hazard_df
from ..services.json_utils import to_native_json
from ..services.agent import ask_openai
from ..analytics.hazard_incident import HazardIncidentAnalyzer

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


@router.get("/funnel/insights", response_model=ChartInsightsResponse)
async def conversion_funnel_insights():
    """Generate insights for conversion funnel"""
    title = "Conversion Funnel Analysis"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    
    total_hazards = 0 if haz is None else int(len(haz))
    total_incidents = 0 if inc is None else int(len(inc))
    
    if total_hazards == 0:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No hazard data available for funnel analysis.")
    
    links = analyzer.links_df
    hazards_became_incidents = 0
    if links is not None and not links.empty and 'hazard_id' in links.columns:
        hazards_became_incidents = int(links['hazard_id'].dropna().nunique())
    
    # Calculate funnel stages
    hazards_closed = 0
    hazards_open = total_hazards
    if haz is not None and hasattr(analyzer, 'haz_status') and analyzer.haz_status and analyzer.haz_status in haz.columns:
        s = haz[analyzer.haz_status].astype(str).str.lower()
        hazards_closed = int((s == 'closed').sum())
        hazards_open = int(total_hazards - hazards_closed)
    
    prevented = max(0, hazards_closed - hazards_became_incidents)
    conversion_rate = (hazards_became_incidents / total_hazards * 100.0) if total_hazards > 0 else 0.0
    closure_rate = (hazards_closed / total_hazards * 100.0) if total_hazards > 0 else 0.0
    
    summary = {
        "title": title,
        "stage_1_reported_hazards": total_hazards,
        "stage_2_closed_hazards": hazards_closed,
        "stage_3_prevented": prevented,
        "stage_4_became_incidents": hazards_became_incidents,
        "total_incidents": total_incidents,
        "conversion_rate_pct": round(conversion_rate, 2),
        "closure_rate_pct": round(closure_rate, 2),
        "prevention_rate_pct": round(100.0 - conversion_rate, 2),
    }
    
    try:
        prompt = (
            "Analyze the conversion funnel showing the flow from reported hazards through closure to prevention or incidents. "
            "Provide concise markdown insights on: funnel drop-off points, conversion efficiency at each stage, "
            "bottlenecks in the process, and 4-5 recommendations to optimize the funnel and improve prevention rates."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n### Funnel Stages")
    parts.append(f"- **Stage 1 - Hazards Reported**: {total_hazards}")
    parts.append(f"- **Stage 2 - Hazards Closed**: {hazards_closed} ({closure_rate:.1f}%)")
    parts.append(f"- **Stage 3 - Successfully Prevented**: {prevented}")
    parts.append(f"- **Stage 4 - Became Incidents**: {hazards_became_incidents}")
    parts.append(f"- **Final Incidents Count**: {total_incidents}")
    parts.append(f"\n### Conversion Metrics")
    parts.append(f"- **Conversion Rate**: {conversion_rate:.1f}%")
    parts.append(f"- **Prevention Success**: {100.0 - conversion_rate:.1f}%")
    parts.append(f"\n### Key Findings")
    if closure_rate < 70:
        parts.append("- ⚠️ Low closure rate - many hazards remain open")
    if conversion_rate < 15:
        parts.append("- ✓ Strong prevention performance - most closed hazards don't become incidents")
    else:
        parts.append("- ⚠️ High conversion rate - need to improve hazard resolution quality")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Accelerate hazard closure to reduce open hazard backlog")
    parts.append("- **Action**: Improve closure verification to ensure true prevention")
    parts.append("- **Action**: Analyze characteristics of hazards that convert vs those that don't")
    parts.append("- **Action**: Implement earlier intervention for high-risk hazards")
    parts.append("- **Action**: Set funnel stage targets and monitor weekly")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /time-lag endpoint (use GET /time-lag)


@router.get("/time-lag", response_model=PlotlyFigureResponse)
async def time_lag_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_time_lag_analysis()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/time-lag/insights", response_model=ChartInsightsResponse)
async def time_lag_insights():
    """Generate insights for time lag between hazards and incidents"""
    title = "Time Lag Analysis"
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    links = analyzer.links_df
    
    if links is None or links.empty or 'days_to_incident' not in links.columns:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Not enough linked data to analyze time lag.")
    
    days = pd.to_numeric(links['days_to_incident'], errors='coerce').dropna()
    if days.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No valid time lag data available.")
    
    summary = {
        "title": title,
        "total_conversions": int(len(days)),
        "avg_days": round(float(days.mean()), 2),
        "median_days": round(float(days.median()), 2),
        "min_days": round(float(days.min()), 2),
        "max_days": round(float(days.max()), 2),
        "std_days": round(float(days.std()), 2),
    }
    
    try:
        prompt = (
            "Analyze the time lag between hazard reports and resulting incidents. "
            "Provide concise markdown insights on: average conversion time, distribution patterns, "
            "risks of delayed action, and 3-4 actionable recommendations to reduce time-to-incident."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Average Time to Incident**: {summary['avg_days']} days")
    parts.append(f"- **Median Time**: {summary['median_days']} days")
    parts.append(f"- **Range**: {summary['min_days']} to {summary['max_days']} days")
    parts.append(f"\n### Key Findings")
    parts.append(f"- Analyzed {summary['total_conversions']} hazard-to-incident conversions")
    if summary['avg_days'] > 30:
        parts.append("- ⚠️ Long average time lag suggests delayed hazard resolution")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Prioritize hazards with high severity to prevent conversion")
    parts.append("- **Action**: Implement early warning system for at-risk hazards")
    parts.append("- **Action**: Reduce time-to-closure for open hazards")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /sankey endpoint (use GET /sankey)


@router.get("/sankey", response_model=PlotlyFigureResponse)
async def sankey_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_sankey_flow()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/sankey/insights", response_model=ChartInsightsResponse)
async def sankey_insights():
    """Generate insights for sankey flow diagram"""
    title = "Hazard-to-Incident Flow Analysis"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    
    total_hazards = 0 if haz is None else int(len(haz))
    total_incidents = 0 if inc is None else int(len(inc))
    
    if total_hazards == 0 or total_incidents == 0:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Insufficient data for flow analysis.")
    
    links = analyzer.links_df
    conversions = 0 if links is None or links.empty else int(len(links))
    conversion_rate = (conversions / total_hazards * 100.0) if total_hazards > 0 else 0.0
    
    summary = {
        "title": title,
        "total_hazards": total_hazards,
        "total_incidents": total_incidents,
        "conversions": conversions,
        "conversion_rate_pct": round(conversion_rate, 2),
        "prevention_rate_pct": round(100.0 - conversion_rate, 2),
    }
    
    try:
        prompt = (
            "Analyze the flow from hazards to incidents using the sankey diagram data. "
            "Provide concise markdown insights on: conversion patterns, prevention effectiveness, "
            "flow bottlenecks, and 3-4 recommendations to improve hazard prevention."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Hazards**: {total_hazards}")
    parts.append(f"- **Total Incidents**: {total_incidents}")
    parts.append(f"- **Conversions**: {conversions} ({conversion_rate:.1f}%)")
    parts.append(f"- **Prevention Success**: {100.0 - conversion_rate:.1f}%")
    parts.append(f"\n### Key Findings")
    if conversion_rate > 20:
        parts.append("- ⚠️ High conversion rate indicates room for improvement in hazard prevention")
    else:
        parts.append("- ✓ Good prevention rate - most hazards are resolved before becoming incidents")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Analyze characteristics of converted hazards vs prevented ones")
    parts.append("- **Action**: Strengthen follow-up processes for high-risk hazards")
    parts.append("- **Action**: Share prevention best practices across departments")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /department-matrix endpoint (use GET /department-matrix)


@router.get("/department-matrix", response_model=PlotlyFigureResponse)
async def department_matrix_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_department_conversion_matrix()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/department-matrix/insights", response_model=ChartInsightsResponse)
async def department_matrix_insights():
    """Generate insights for department conversion matrix"""
    title = "Department Conversion Matrix"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    
    if inc is None or haz is None or inc.empty or haz.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Insufficient data for department analysis.")
    
    dept_col_h = analyzer.haz_dept
    dept_col_i = analyzer.inc_dept
    if not dept_col_h or not dept_col_i or dept_col_h not in haz.columns or dept_col_i not in inc.columns:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Department columns not found in data.")
    
    links = analyzer.links_df if analyzer.links_df is not None else pd.DataFrame()
    dept_stats = []
    
    for dept in pd.Series(haz[dept_col_h].dropna().unique()).astype(str):
        dept_hazards = haz[haz[dept_col_h].astype(str) == dept]
        if not links.empty and 'department' in links.columns:
            dept_conversions = links[links['department'].astype(str) == dept]
            conversion_rate = (len(dept_conversions) / len(dept_hazards) * 100.0) if len(dept_hazards) > 0 else 0.0
        else:
            conversion_rate = 0.0
        dept_stats.append({
            "department": dept,
            "hazards": int(len(dept_hazards)),
            "conversion_rate_pct": round(conversion_rate, 2)
        })
    
    dept_stats.sort(key=lambda x: x['conversion_rate_pct'], reverse=True)
    
    summary = {
        "title": title,
        "total_departments": len(dept_stats),
        "departments": dept_stats[:10],
    }
    
    try:
        prompt = (
            "Analyze department-wise hazard conversion rates. "
            "Provide concise markdown insights on: which departments have highest/lowest conversion rates, "
            "patterns across departments, and 3-4 targeted recommendations for high-risk departments."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    if dept_stats:
        highest = dept_stats[0]
        lowest = dept_stats[-1]
        parts.append(f"\n- **Highest Conversion**: {highest['department']} ({highest['conversion_rate_pct']}%)")
        parts.append(f"- **Lowest Conversion**: {lowest['department']} ({lowest['conversion_rate_pct']}%)")
        parts.append(f"- **Total Departments**: {len(dept_stats)}")
    parts.append(f"\n### Key Findings")
    parts.append("- Department performance varies significantly in hazard prevention")
    parts.append("- High conversion rates indicate need for targeted interventions")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Conduct root cause analysis in high-conversion departments")
    parts.append("- **Action**: Share best practices from low-conversion departments")
    parts.append("- **Action**: Implement department-specific prevention programs")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /risk-network endpoint (use GET /risk-network)


@router.get("/risk-network", response_model=PlotlyFigureResponse)
async def risk_network_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_risk_network()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/risk-network/insights", response_model=ChartInsightsResponse)
async def risk_network_insights():
    """Generate insights for risk network analysis"""
    title = "Risk Network Analysis"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    
    links = analyzer.links_df
    if links is None or links.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No linked data available for network analysis.")
    
    total_nodes = 0
    total_edges = int(len(links))
    
    if haz is not None:
        total_nodes += int(len(haz))
    if inc is not None:
        total_nodes += int(len(inc))
    
    dept_links = {}
    if 'department' in links.columns:
        dept_links = links['department'].value_counts().head(5).to_dict()
    
    summary = {
        "title": title,
        "total_nodes": total_nodes,
        "total_connections": total_edges,
        "top_departments": {str(k): int(v) for k, v in dept_links.items()},
    }
    
    try:
        prompt = (
            "Analyze the risk network showing connections between hazards and incidents. "
            "Provide concise markdown insights on: network density, key connection patterns, "
            "high-risk clusters, and 3-4 recommendations to break negative chains."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Network Nodes**: {total_nodes}")
    parts.append(f"- **Total Connections**: {total_edges}")
    if dept_links:
        top_dept = list(dept_links.keys())[0]
        parts.append(f"- **Most Connected Department**: {top_dept}")
    parts.append(f"\n### Key Findings")
    parts.append("- Network reveals patterns of recurring risk pathways")
    parts.append("- Connected hazards indicate systemic issues requiring attention")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Identify and break high-frequency hazard-incident chains")
    parts.append("- **Action**: Focus on departments with dense connection clusters")
    parts.append("- **Action**: Implement systemic controls for recurring risk patterns")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /prevention-effectiveness endpoint (use GET /prevention-effectiveness)


@router.get("/prevention-effectiveness", response_model=PlotlyFigureResponse)
async def prevention_effectiveness_auto():
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    fig = analyzer.create_prevention_effectiveness()
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/prevention-effectiveness/insights", response_model=ChartInsightsResponse)
async def prevention_effectiveness_insights():
    """Generate insights for prevention effectiveness analysis"""
    title = "Prevention Effectiveness Analysis"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    
    total_hazards = 0 if haz is None else int(len(haz))
    if total_hazards == 0:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No hazard data available for analysis.")
    
    links = analyzer.links_df
    hazards_became_incidents = 0
    if links is not None and not links.empty and 'hazard_id' in links.columns:
        hazards_became_incidents = int(links['hazard_id'].dropna().nunique())
    
    hazards_closed = 0
    if haz is not None and hasattr(analyzer, 'haz_status') and analyzer.haz_status and analyzer.haz_status in haz.columns:
        s = haz[analyzer.haz_status].astype(str).str.lower()
        hazards_closed = int((s == 'closed').sum())
    
    prevented = max(0, hazards_closed - hazards_became_incidents)
    prevention_rate = (prevented / hazards_closed * 100.0) if hazards_closed > 0 else 0.0
    conversion_rate = (hazards_became_incidents / total_hazards * 100.0) if total_hazards > 0 else 0.0
    
    summary = {
        "title": title,
        "total_hazards": total_hazards,
        "hazards_closed": hazards_closed,
        "successfully_prevented": prevented,
        "became_incidents": hazards_became_incidents,
        "prevention_rate_pct": round(prevention_rate, 2),
        "conversion_rate_pct": round(conversion_rate, 2),
    }
    
    try:
        prompt = (
            "Analyze prevention effectiveness metrics showing how many hazards were successfully prevented vs converted to incidents. "
            "Provide concise markdown insights on: overall prevention performance, success rate trends, "
            "areas for improvement, and 3-4 actionable recommendations to enhance prevention effectiveness."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Hazards Closed**: {hazards_closed}")
    parts.append(f"- **Successfully Prevented**: {prevented}")
    parts.append(f"- **Became Incidents**: {hazards_became_incidents}")
    parts.append(f"- **Prevention Success Rate**: {prevention_rate:.1f}%")
    parts.append(f"\n### Key Findings")
    if prevention_rate > 70:
        parts.append("- ✓ Strong prevention effectiveness - majority of hazards resolved safely")
    elif prevention_rate > 50:
        parts.append("- Moderate prevention effectiveness - room for improvement")
    else:
        parts.append("- ⚠️ Low prevention effectiveness - urgent action needed")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Analyze root causes of hazards that became incidents")
    parts.append("- **Action**: Improve hazard closure verification processes")
    parts.append("- **Action**: Implement earlier intervention for high-risk hazards")
    parts.append("- **Action**: Track prevention metrics monthly to identify trends")
    return ChartInsightsResponse(insights_md="\n".join(parts))


# Removed POST /metrics-gauge endpoint (use GET /metrics-gauge)


@router.get("/metrics-gauge", response_model=PlotlyFigureResponse)
async def metrics_gauge_auto():
    from ..analytics.hazard_incident import create_conversion_metrics_card
    workbook = {
        'Incidents': get_incident_df(),
        'Hazards': get_hazard_df(),
    }
    fig = create_conversion_metrics_card(workbook)
    return JSONResponse(content={"figure": to_native_json(fig.to_plotly_json())})


@router.get("/metrics-gauge/insights", response_model=ChartInsightsResponse)
async def metrics_gauge_insights():
    """Generate insights for conversion metrics gauge"""
    title = "Conversion Metrics Overview"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)
    links = analyzer.links_df

    total_hazards = 0 if haz is None else int(len(haz))
    total_incidents = 0 if inc is None else int(len(inc))

    hazards_became_incidents = 0
    if links is not None and not links.empty and 'hazard_id' in links.columns:
        hazards_became_incidents = int(links['hazard_id'].dropna().nunique())

    conversion_rate = (hazards_became_incidents / total_hazards * 100.0) if total_hazards > 0 else 0.0
    prevention_rate = 100.0 - conversion_rate

    summary = {
        "title": title,
        "total_hazards": total_hazards,
        "total_incidents": total_incidents,
        "hazards_became_incidents": hazards_became_incidents,
        "conversion_rate_pct": round(conversion_rate, 2),
        "prevention_rate_pct": round(prevention_rate, 2),
    }
    
    try:
        prompt = (
            "Analyze key conversion metrics displayed in the gauge chart. "
            "Provide concise markdown insights on: overall conversion and prevention rates, "
            "performance assessment, benchmark comparison, and 3-4 strategic recommendations."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Hazards**: {total_hazards}")
    parts.append(f"- **Total Incidents**: {total_incidents}")
    parts.append(f"- **Conversion Rate**: {conversion_rate:.1f}%")
    parts.append(f"- **Prevention Rate**: {prevention_rate:.1f}%")
    parts.append(f"\n### Performance Assessment")
    if conversion_rate < 10:
        parts.append("- ✓ Excellent prevention performance")
    elif conversion_rate < 20:
        parts.append("- Good prevention performance with room for improvement")
    else:
        parts.append("- ⚠️ High conversion rate requires immediate attention")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Set target to reduce conversion rate by 20% over next quarter")
    parts.append("- **Action**: Monitor metrics weekly to detect early trends")
    parts.append("- **Action**: Benchmark against industry standards")
    return ChartInsightsResponse(insights_md="\n".join(parts))


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


@router.get("/links/insights", response_model=ChartInsightsResponse)
async def links_insights():
    """Generate insights for hazard-incident links"""
    title = "Hazard-Incident Link Analysis"
    analyzer = HazardIncidentAnalyzer(get_incident_df(), get_hazard_df())
    df = analyzer.links_df
    
    if df is None or df.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: No linked data available.")
    
    total = int(len(df))
    uniq_h = int(df['hazard_id'].dropna().nunique()) if 'hazard_id' in df.columns else 0
    uniq_i = int(df['incident_id'].dropna().nunique()) if 'incident_id' in df.columns else 0
    
    summary = {
        "title": title,
        "total_links": total,
        "unique_hazards_linked": uniq_h,
        "unique_incidents_linked": uniq_i,
    }
    
    try:
        prompt = (
            "Analyze the links between hazards and incidents. "
            "Provide concise markdown insights on: linkage patterns, data quality, "
            "relationship strength, and 3-4 recommendations to improve hazard-incident tracking."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Links**: {total}")
    parts.append(f"- **Unique Hazards Linked**: {uniq_h}")
    parts.append(f"- **Unique Incidents Linked**: {uniq_i}")
    parts.append(f"\n### Key Findings")
    parts.append("- Links established between hazards and resulting incidents")
    parts.append("- Strong linkage data enables conversion analysis")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Validate linkage accuracy through periodic audits")
    parts.append("- **Action**: Ensure consistent hazard-incident relationship tracking")
    parts.append("- **Action**: Use link data to identify prevention opportunities")
    return ChartInsightsResponse(insights_md="\n".join(parts))


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


@router.get("/metrics/insights", response_model=ChartInsightsResponse)
async def metrics_insights():
    """Generate insights for hazard-incident metrics"""
    title = "Hazard-Incident Metrics Summary"
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

    prevented_hazards = int(max(0, hazards_closed - hazards_became_incidents))
    conversion_rate = (hazards_became_incidents / total_hazards * 100.0) if total_hazards > 0 else 0.0
    prevention_rate = 100.0 - conversion_rate

    summary = {
        "title": title,
        "total_hazards": total_hazards,
        "total_incidents": total_incidents,
        "hazards_became_incidents": hazards_became_incidents,
        "hazards_closed": hazards_closed,
        "hazards_open": hazards_open,
        "prevented_hazards": prevented_hazards,
        "conversion_rate_pct": round(conversion_rate, 2),
        "prevention_rate_pct": round(prevention_rate, 2),
        "avg_days_to_incident": round(avg_days_to_incident, 2),
    }
    
    try:
        prompt = (
            "Analyze comprehensive hazard-incident metrics. "
            "Provide concise markdown insights on: overall performance, open vs closed hazards, "
            "prevention success rate, time-to-incident trends, and 4-5 strategic recommendations."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n### Hazard Status")
    parts.append(f"- **Total Hazards**: {total_hazards}")
    parts.append(f"- **Open Hazards**: {hazards_open}")
    parts.append(f"- **Closed Hazards**: {hazards_closed}")
    parts.append(f"\n### Conversion Metrics")
    parts.append(f"- **Hazards → Incidents**: {hazards_became_incidents}")
    parts.append(f"- **Successfully Prevented**: {prevented_hazards}")
    parts.append(f"- **Conversion Rate**: {conversion_rate:.1f}%")
    parts.append(f"- **Prevention Rate**: {prevention_rate:.1f}%")
    parts.append(f"- **Avg Days to Incident**: {avg_days_to_incident:.1f}")
    parts.append(f"\n### Key Findings")
    if hazards_open > hazards_closed:
        parts.append("- ⚠️ More open hazards than closed - focus on resolution")
    if conversion_rate < 15:
        parts.append("- ✓ Good prevention performance")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Prioritize closure of open hazards")
    parts.append("- **Action**: Reduce average time-to-incident through early intervention")
    parts.append("- **Action**: Maintain prevention rate above 85%")
    return ChartInsightsResponse(insights_md="\n".join(parts))


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


@router.get("/department-metrics-data/insights", response_model=ChartInsightsResponse)
async def department_metrics_data_insights():
    """Generate insights for department metrics data"""
    title = "Department Performance Metrics"
    inc = get_incident_df()
    haz = get_hazard_df()
    analyzer = HazardIncidentAnalyzer(inc, haz)

    if inc is None or haz is None or inc.empty or haz.empty:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Insufficient data for department analysis.")

    dept_col_h = analyzer.haz_dept
    dept_col_i = analyzer.inc_dept
    if not dept_col_h or not dept_col_i or dept_col_h not in haz.columns or dept_col_i not in inc.columns:
        return ChartInsightsResponse(insights_md=f"## {title}\n\n- **Summary**: Department columns not found in data.")

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

    metrics.sort(key=lambda x: x['conversion_rate_pct'], reverse=True)
    
    summary = {
        "title": title,
        "total_departments": len(metrics),
        "top_departments": metrics[:5],
    }
    
    try:
        prompt = (
            "Analyze detailed department performance metrics including hazards, incidents, conversion rates, and severity. "
            "Provide concise markdown insights on: department rankings, severity trends, "
            "conversion patterns by department, and 4-5 targeted recommendations for improvement."
        )
        md = ask_openai(prompt, context=json.dumps(summary, ensure_ascii=False), model="gpt-4o", code_mode=False, multi_df=False)
        if md and not md.lower().startswith("openai") and "not installed" not in md.lower():
            return ChartInsightsResponse(insights_md=md)
    except Exception:
        pass
    
    # Fallback
    parts = [f"## {title}"]
    parts.append(f"\n- **Total Departments Analyzed**: {len(metrics)}")
    if metrics:
        best = metrics[-1]  # Lowest conversion rate
        worst = metrics[0]   # Highest conversion rate
        parts.append(f"\n### Top Performer")
        parts.append(f"- **{best['department']}**: {best['prevention_success_pct']}% prevention success")
        parts.append(f"\n### Needs Attention")
        parts.append(f"- **{worst['department']}**: {worst['conversion_rate_pct']}% conversion rate")
    parts.append(f"\n### Key Findings")
    parts.append("- Significant variation in department performance")
    parts.append("- Both hazard severity and incident severity tracked per department")
    parts.append("- Conversion rates vary by department culture and practices")
    parts.append("\n### Recommendations")
    parts.append("- **Action**: Conduct best practice sharing sessions between top and bottom performers")
    parts.append("- **Action**: Implement targeted training in high-conversion departments")
    parts.append("- **Action**: Monitor department metrics monthly for trends")
    parts.append("- **Action**: Set department-specific improvement goals")
    return ChartInsightsResponse(insights_md="\n".join(parts))
