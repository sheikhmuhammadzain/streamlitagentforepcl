"""
Advanced Analytics Router
Implements critical HSE analytics including Heinrich's Pyramid, KPIs, and predictive analytics.
Built with professional data science practices and best engineering standards.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ..services.excel import get_incident_df, get_hazard_df, get_audit_df, get_inspection_df
from ..services.json_utils import to_native_json


router = APIRouter(prefix="/analytics/advanced", tags=["advanced-analytics"])


# ======================= HELPER FUNCTIONS =======================

def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Resolve column name from list of candidates (case-insensitive, flexible matching)."""
    if df is None or df.empty:
        return None
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in col_map:
            return col_map[key]
    # Relaxed contains match
    for candidate in candidates:
        key = str(candidate).strip().lower()
        for lk, orig in col_map.items():
            if key in lk:
                return orig
    return None


def _apply_filters(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    location: Optional[str] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
) -> pd.DataFrame:
    """Apply common filters to dataframe."""
    if df is None or df.empty:
        return df
    
    filtered = df.copy()
    
    # Date range filter
    if start_date or end_date:
        date_col = _resolve_column(filtered, ["occurrence_date", "date", "start_date", "reported_date"])
        if date_col:
            dates = pd.to_datetime(filtered[date_col], errors="coerce")
            if start_date:
                mask = dates >= pd.to_datetime(start_date)
                filtered = filtered.loc[mask]
            if end_date:
                mask = dates <= pd.to_datetime(end_date)
                filtered = filtered.loc[mask]
    
    # Location filter
    if location is not None and location != "":
        loc_col = _resolve_column(filtered, ["location", "audit_location", "finding_location"])
        if loc_col:
            filtered = filtered[
                filtered[loc_col].astype(str).str.contains(str(location), case=False, na=False)
            ]
    
    # Department filter
    if department is not None and department != "":
        dept_col = _resolve_column(filtered, ["department", "section", "sub_department"])
        if dept_col:
            filtered = filtered[
                filtered[dept_col].astype(str).str.contains(str(department), case=False, na=False)
            ]
    
    # Status filter
    if status is not None and status != "":
        status_col = _resolve_column(filtered, ["status", "incident_status", "audit_status", "hazard_status"])
        if status_col:
            filtered = filtered[
                filtered[status_col].astype(str).str.contains(str(status), case=False, na=False)
            ]
    
    return filtered


def _classify_severity_level(severity_score: Any, severity_text: Any = None) -> str:
    """
    Classify severity into Heinrich Pyramid layers:
    - Layer 1: Fatality/Serious Injury (severity >= 4 or 'Critical'/'Severe')
    - Layer 2: Minor Injury (severity 2-3 or 'Serious'/'High')
    - Layer 3: First Aid/Near Miss (severity 1 or 'Minor'/'Low')
    - Layer 4: Unsafe Condition (from hazards)
    - Layer 5: At-Risk Behavior (from observations/audits)
    """
    # Try text-based classification first
    if severity_text and pd.notna(severity_text):
        text = str(severity_text).lower().strip()
        if any(k in text for k in ["critical", "fatal", "severe", "c4", "c3"]):
            return "Serious Injury/Fatality"
        if any(k in text for k in ["serious", "high", "c2", "moderate"]):
            return "Minor Injury"
        if any(k in text for k in ["minor", "low", "c1", "first aid", "near miss"]):
            return "First Aid/Near Miss"
    
    # Numeric classification
    try:
        score = float(severity_score)
        if score >= 3:
            return "Serious Injury/Fatality"
        elif score >= 2:
            return "Minor Injury"
        else:
            return "First Aid/Near Miss"
    except (ValueError, TypeError):
        pass
    
    return "First Aid/Near Miss"  # Default to least severe


def _calculate_near_miss_ratio(incidents_df: pd.DataFrame, hazards_df: pd.DataFrame) -> float:
    """Calculate near-miss to incident ratio (industry standard: 1 incident : 10 near-misses)."""
    if incidents_df is None or incidents_df.empty:
        return 0.0
    
    incident_count = len(incidents_df)
    near_miss_count = 0
    
    # Count near-misses from incidents
    if "incident_type" in incidents_df.columns:
        near_miss_count += incidents_df["incident_type"].astype(str).str.contains(
            "near miss|near-miss|nearmiss", case=False, na=False
        ).sum()
    
    # Count hazards as potential near-misses
    if hazards_df is not None and not hazards_df.empty:
        near_miss_count += len(hazards_df)
    
    if incident_count == 0:
        return 0.0
    
    return round(near_miss_count / incident_count, 2)


# ======================= HEINRICH'S SAFETY PYRAMID =======================

@router.get("/heinrich-pyramid")
async def heinrich_safety_pyramid(
    start_date: Optional[str] = Query(None, description="Filter start date (YYYY-MM-DD)", example="2024-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date (YYYY-MM-DD)", example="2024-12-31"),
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi"),
    department: Optional[str] = Query(None, description="Filter by department", example="Process"),
):
    """
    Heinrich's Safety Pyramid - The foundational safety analytics chart.
    
    Returns a hierarchical pyramid structure showing:
    - Layer 1 (Top): Serious injuries/fatalities
    - Layer 2: Minor injuries
    - Layer 3: First aid cases / near misses
    - Layer 4: Unsafe conditions (hazards)
    - Layer 5 (Bottom): At-risk behaviors (audit/inspection findings)
    
    Industry standard ratios: 1 : 10 : 30 : 600 : 3000
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Apply filters
    inc_df = _apply_filters(inc_df, start_date, end_date, location, department)
    haz_df = _apply_filters(haz_df, start_date, end_date, location, department)
    
    pyramid_data = {
        "Serious Injury/Fatality": 0,
        "Minor Injury": 0,
        "First Aid/Near Miss": 0,
        "Unsafe Conditions": 0,
        "At-Risk Behaviors": 0,
    }
    
    # Layer 1-3: Classify incidents by severity
    if inc_df is not None and not inc_df.empty:
        sev_score_col = _resolve_column(inc_df, ["severity_score", "severity", "risk_score"])
        sev_text_col = _resolve_column(inc_df, [
            "actual_consequence_incident", "worst_case_consequence_incident",
            "relevant_consequence_incident", "severity_level"
        ])
        
        for idx, row in inc_df.iterrows():
            sev_score = row[sev_score_col] if sev_score_col else None
            sev_text = row[sev_text_col] if sev_text_col else None
            level = _classify_severity_level(sev_score, sev_text)
            pyramid_data[level] += 1
    
    # Layer 4: Unsafe conditions from hazards
    if haz_df is not None and not haz_df.empty:
        pyramid_data["Unsafe Conditions"] = len(haz_df)
    
    # Layer 5: At-risk behaviors from audits + inspections
    at_risk_count = 0
    if aud_df is not None and not aud_df.empty:
        finding_col = _resolve_column(aud_df, ["finding", "findings"])
        if finding_col:
            at_risk_count += aud_df[finding_col].notna().sum()
    
    if insp_df is not None and not insp_df.empty:
        finding_col = _resolve_column(insp_df, ["finding", "findings"])
        if finding_col:
            at_risk_count += insp_df[finding_col].notna().sum()
    
    pyramid_data["At-Risk Behaviors"] = at_risk_count
    
    # Calculate ratios (normalized to serious injuries)
    serious = pyramid_data["Serious Injury/Fatality"]
    ratios = {}
    if serious > 0:
        for key, val in pyramid_data.items():
            ratios[key] = round(val / serious, 2)
    else:
        ratios = {key: 0 for key in pyramid_data.keys()}
    
    # Build response
    layers = []
    for idx, (label, count) in enumerate(pyramid_data.items()):
        layers.append({
            "level": 5 - idx,  # 5 = top, 1 = bottom
            "label": label,
            "count": int(count),
            "ratio": ratios[label],
            "color": [
                "#d32f2f",  # Red - Serious
                "#f57c00",  # Orange - Minor
                "#fbc02d",  # Yellow - First Aid
                "#7cb342",  # Light Green - Unsafe Conditions
                "#66bb6a"   # Green - At-Risk Behaviors
            ][idx]
        })
    
    return JSONResponse(content=to_native_json({
        "layers": layers,
        "total_events": sum(pyramid_data.values()),
        "near_miss_ratio": _calculate_near_miss_ratio(inc_df, haz_df),
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "location": location,
            "department": department,
        }
    }))


# ======================= SITE SAFETY INDEX =======================

@router.get("/site-safety-index")
async def site_safety_index(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2024-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
    location: Optional[str] = Query(None, description="Filter by location", example="Manufacturing Facility"),
):
    """
    Site Safety Index (0-100 score) - Real-time safety health indicator.
    
    Calculation methodology:
    - Base score: 100
    - Deductions:
      - Serious injuries: -10 points each
      - Minor injuries: -3 points each
      - Hazards (high risk): -2 points each
      - Open corrective actions: -1 point each
    - Bonuses:
      - Days since last incident: +0.1 per day (max +10)
      - Completed audits: +0.5 each (max +5)
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    
    # Apply filters
    inc_df = _apply_filters(inc_df, start_date, end_date, location)
    haz_df = _apply_filters(haz_df, start_date, end_date, location)
    
    base_score = 100.0
    deductions = 0.0
    bonuses = 0.0
    breakdown = []
    
    # Deductions from incidents
    if inc_df is not None and not inc_df.empty:
        sev_score_col = _resolve_column(inc_df, ["severity_score", "risk_score"])
        sev_text_col = _resolve_column(inc_df, ["actual_consequence_incident", "severity"])
        
        serious_count = 0
        minor_count = 0
        
        for idx, row in inc_df.iterrows():
            sev_score = row[sev_score_col] if sev_score_col else None
            sev_text = row[sev_text_col] if sev_text_col else None
            level = _classify_severity_level(sev_score, sev_text)
            
            if level == "Serious Injury/Fatality":
                serious_count += 1
            elif level == "Minor Injury":
                minor_count += 1
        
        serious_deduction = serious_count * 10
        minor_deduction = minor_count * 3
        deductions += serious_deduction + minor_deduction
        
        if serious_count > 0:
            breakdown.append({"factor": f"Serious Injuries ({serious_count})", "impact": -serious_deduction})
        if minor_count > 0:
            breakdown.append({"factor": f"Minor Injuries ({minor_count})", "impact": -minor_deduction})
    
    # Deductions from high-risk hazards
    if haz_df is not None and not haz_df.empty:
        risk_col = _resolve_column(haz_df, ["risk_score", "risk_level"])
        if risk_col:
            high_risk = haz_df[risk_col].apply(
                lambda x: (pd.notna(x) and (
                    (isinstance(x, (int, float)) and x >= 3) or 
                    (isinstance(x, str) and any(k in str(x).lower() for k in ["high", "critical", "severe"]))
                ))
            ).sum()
            hazard_deduction = high_risk * 2
            deductions += hazard_deduction
            if high_risk > 0:
                breakdown.append({"factor": f"High-Risk Hazards ({high_risk})", "impact": -hazard_deduction})
    
    # Open corrective actions deduction
    if inc_df is not None and not inc_df.empty:
        status_col = _resolve_column(inc_df, ["status"])
        if status_col:
            open_count = inc_df[status_col].astype(str).str.contains(
                "open|pending|progress|review", case=False, na=False
            ).sum()
            open_deduction = open_count * 1
            deductions += open_deduction
            if open_count > 0:
                breakdown.append({"factor": f"Open Corrective Actions ({open_count})", "impact": -open_deduction})
    
    # Bonus: Days since last incident
    if inc_df is not None and not inc_df.empty:
        date_col = _resolve_column(inc_df, ["occurrence_date", "date"])
        if date_col:
            dates = pd.to_datetime(inc_df[date_col], errors="coerce")
            last_incident = dates.max()
            if pd.notna(last_incident):
                days_since = (pd.Timestamp.now() - last_incident).days
                days_bonus = min(days_since * 0.1, 10.0)
                bonuses += days_bonus
                breakdown.append({"factor": f"Days Since Last Incident ({days_since})", "impact": round(days_bonus, 2)})
    
    # Bonus: Completed audits
    if aud_df is not None and not aud_df.empty:
        status_col = _resolve_column(aud_df, ["audit_status", "status"])
        if status_col:
            completed = aud_df[status_col].astype(str).str.contains("closed|complete", case=False, na=False).sum()
            audit_bonus = min(completed * 0.5, 5.0)
            bonuses += audit_bonus
            if completed > 0:
                breakdown.append({"factor": f"Completed Audits ({completed})", "impact": round(audit_bonus, 2)})
    
    final_score = max(0.0, min(100.0, base_score - deductions + bonuses))
    
    # Determine rating
    if final_score >= 90:
        rating = "Excellent"
        color = "#4caf50"
    elif final_score >= 75:
        rating = "Good"
        color = "#8bc34a"
    elif final_score >= 60:
        rating = "Fair"
        color = "#ffc107"
    elif final_score >= 40:
        rating = "Poor"
        color = "#ff9800"
    else:
        rating = "Critical"
        color = "#f44336"
    
    return JSONResponse(content=to_native_json({
        "score": round(final_score, 2),
        "rating": rating,
        "color": color,
        "base_score": base_score,
        "total_deductions": round(deductions, 2),
        "total_bonuses": round(bonuses, 2),
        "breakdown": breakdown,
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "location": location,
        }
    }))


# ======================= KPI METRICS =======================

@router.get("/kpis/trir")
async def kpi_trir(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
    total_hours_worked: int = Query(2000000, description="Total hours worked (default: 2M for estimation)", example=2000000),
):
    """
    TRIR - Total Recordable Incident Rate
    Formula: (Number of recordable incidents × 200,000) / Total hours worked
    Industry benchmark: < 1.0 is excellent, < 3.0 is good
    """
    inc_df = get_incident_df()
    inc_df = _apply_filters(inc_df, start_date, end_date)
    
    if inc_df is None or inc_df.empty:
        recordable_count = 0
    else:
        # Recordable = severity >= 2 or medical treatment required
        sev_col = _resolve_column(inc_df, ["severity_score", "risk_score"])
        if sev_col:
            recordable_count = (pd.to_numeric(inc_df[sev_col], errors="coerce") >= 2).sum()
        else:
            recordable_count = len(inc_df)
    
    trir = (recordable_count * 200000) / total_hours_worked if total_hours_worked > 0 else 0
    
    # Benchmark assessment
    if trir < 1.0:
        benchmark = "Excellent"
        color = "#4caf50"
    elif trir < 3.0:
        benchmark = "Good"
        color = "#8bc34a"
    elif trir < 5.0:
        benchmark = "Average"
        color = "#ffc107"
    else:
        benchmark = "Needs Improvement"
        color = "#f44336"
    
    return JSONResponse(content=to_native_json({
        "value": round(trir, 2),
        "recordable_incidents": int(recordable_count),
        "total_hours_worked": total_hours_worked,
        "benchmark": benchmark,
        "color": color,
        "industry_standard": "< 1.0 Excellent, < 3.0 Good, < 5.0 Average",
    }))


@router.get("/kpis/ltir")
async def kpi_ltir(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
    total_hours_worked: int = Query(2000000, description="Total hours worked", example=2000000),
):
    """
    LTIR - Lost Time Incident Rate
    Formula: (Number of lost-time incidents × 200,000) / Total hours worked
    """
    inc_df = get_incident_df()
    inc_df = _apply_filters(inc_df, start_date, end_date)
    
    if inc_df is None or inc_df.empty:
        lost_time_count = 0
    else:
        # Lost time = severity >= 3
        sev_col = _resolve_column(inc_df, ["severity_score", "risk_score"])
        if sev_col:
            lost_time_count = (pd.to_numeric(inc_df[sev_col], errors="coerce") >= 3).sum()
        else:
            lost_time_count = 0
    
    ltir = (lost_time_count * 200000) / total_hours_worked if total_hours_worked > 0 else 0
    
    return JSONResponse(content=to_native_json({
        "value": round(ltir, 2),
        "lost_time_incidents": int(lost_time_count),
        "total_hours_worked": total_hours_worked,
    }))


@router.get("/kpis/pstir")
async def kpi_pstir(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
    total_hours_worked: int = Query(2000000, description="Total hours worked", example=2000000),
):
    """
    PSTIR - Process Safety Total Incident Rate
    Formula: (Number of PSM incidents × 200,000) / Total hours worked
    """
    inc_df = get_incident_df()
    inc_df = _apply_filters(inc_df, start_date, end_date)
    
    if inc_df is None or inc_df.empty:
        psm_count = 0
    else:
        # PSM incidents
        psm_col = _resolve_column(inc_df, ["psm", "pse_category"])
        if psm_col:
            psm_count = inc_df[psm_col].notna().sum()
        else:
            psm_count = 0
    
    pstir = (psm_count * 200000) / total_hours_worked if total_hours_worked > 0 else 0
    
    return JSONResponse(content=to_native_json({
        "value": round(pstir, 2),
        "psm_incidents": int(psm_count),
        "total_hours_worked": total_hours_worked,
    }))


@router.get("/kpis/near-miss-ratio")
async def kpi_near_miss_ratio(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
):
    """
    Near-Miss to Incident Ratio
    Industry benchmark: 10:1 (10 near-misses per incident indicates good reporting culture)
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    
    inc_df = _apply_filters(inc_df, start_date, end_date)
    haz_df = _apply_filters(haz_df, start_date, end_date)
    
    ratio = _calculate_near_miss_ratio(inc_df, haz_df)
    
    incident_count = len(inc_df) if inc_df is not None else 0
    near_miss_count = len(haz_df) if haz_df is not None else 0
    
    # Benchmark
    if ratio >= 10:
        benchmark = "Excellent reporting culture"
        color = "#4caf50"
    elif ratio >= 5:
        benchmark = "Good"
        color = "#8bc34a"
    elif ratio >= 2:
        benchmark = "Fair"
        color = "#ffc107"
    else:
        benchmark = "Under-reporting likely"
        color = "#f44336"
    
    return JSONResponse(content=to_native_json({
        "ratio": ratio,
        "near_misses": int(near_miss_count),
        "incidents": int(incident_count),
        "benchmark": benchmark,
        "color": color,
        "industry_standard": "10:1 indicates healthy reporting culture",
    }))


@router.get("/kpis/summary")
async def kpis_summary(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
):
    """Unified dashboard KPI summary with all critical metrics."""
    # Call individual KPI endpoints with proper parameters
    trir_resp = await kpi_trir(start_date, end_date, total_hours_worked=2000000)
    ltir_resp = await kpi_ltir(start_date, end_date, total_hours_worked=2000000)
    pstir_resp = await kpi_pstir(start_date, end_date, total_hours_worked=2000000)
    nmr_resp = await kpi_near_miss_ratio(start_date, end_date)
    safety_index_resp = await site_safety_index(start_date, end_date, location=None)
    
    # Extract JSON data from responses
    import json
    
    return JSONResponse(content=to_native_json({
        "trir": json.loads(trir_resp.body.decode()) if hasattr(trir_resp, 'body') else {},
        "ltir": json.loads(ltir_resp.body.decode()) if hasattr(ltir_resp, 'body') else {},
        "pstir": json.loads(pstir_resp.body.decode()) if hasattr(pstir_resp, 'body') else {},
        "near_miss_ratio": json.loads(nmr_resp.body.decode()) if hasattr(nmr_resp, 'body') else {},
        "safety_index": json.loads(safety_index_resp.body.decode()) if hasattr(safety_index_resp, 'body') else {},
    }))
