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
    Heinrich's Safety Pyramid aligned to the provided tiers and names:
    - 1: Fatality
    - 30: Lost Workday Cases
    - 300: Recordable Injuries
    - 3,000: Near Misses (estimated)
    - 300,000: At-Risk Behaviors (estimated)

    Calculation (from your data):
    - Fatality: incidents whose consequence mentions 'fatal' OR highest severity (fallback)
    - Lost Workday Cases: incidents with severity_score >= 3 (LTIR logic), excluding fatalities
    - Recordable Injuries: incidents with severity_score >= 2 (TRIR logic), excluding lost-time and fatalities
    - Near Misses (estimated): hazards count + incidents with incident_type containing 'near miss'
    - At-Risk Behaviors (estimated): non-null findings in audits and inspections

    We also compute Heinrich-expected counts using the 1:30:300:3000:300000 ratio anchored on the
    top-most available actual tier (prefer Fatalities; fallback to Lost Workday, then Recordable).
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Apply filters
    inc_df = _apply_filters(inc_df, start_date, end_date, location, department)
    haz_df = _apply_filters(haz_df, start_date, end_date, location, department)
    
    # ---------- Derive actual tier counts from data ----------
    tiers = {
        "Fatality": 0,
        "Lost Workday Cases": 0,
        "Recordable Injuries": 0,
        "Near Misses (estimated)": 0,
        "At-Risk Behaviors (estimated)": 0,
    }

    # Incidents mapping
    if inc_df is not None and not inc_df.empty:
        sev_col = _resolve_column(inc_df, ["severity_score", "severity"])  # numeric preferred
        act_cons = _resolve_column(inc_df, ["actual_consequence_incident"])  # text consequence
        worst_cons = _resolve_column(inc_df, ["worst_case_consequence_incident"])  # fallback text
        type_col = _resolve_column(inc_df, ["incident_type", "category"])  # near miss text

        # Fatalities by explicit text OR highest severity bucket
        fat_mask = pd.Series([False] * len(inc_df))
        if act_cons and act_cons in inc_df.columns:
            fat_mask |= inc_df[act_cons].astype(str).str.contains("fatal", case=False, na=False)
        if worst_cons and worst_cons in inc_df.columns:
            fat_mask |= inc_df[worst_cons].astype(str).str.contains("fatal", case=False, na=False)
        if sev_col and sev_col in inc_df.columns:
            sev_vals = pd.to_numeric(inc_df[sev_col], errors="coerce")
            # If scale is 0-5, consider 5 as fatal proxy; if 0-4, consider >=4
            max_val = sev_vals.max(skipna=True)
            if pd.notna(max_val):
                thr = 5 if max_val >= 5 else 4
                fat_mask |= sev_vals >= thr
        tiers["Fatality"] = int(fat_mask.sum())

        # Lost Workday Cases: severity >= 3 (exclude fatalities)
        lti_mask = pd.Series([False] * len(inc_df))
        if sev_col and sev_col in inc_df.columns:
            sev_vals = pd.to_numeric(inc_df[sev_col], errors="coerce")
            lti_mask = sev_vals >= 3
        if lti_mask.any():
            lti_mask &= ~fat_mask
        tiers["Lost Workday Cases"] = int(lti_mask.sum())

        # Recordable Injuries: severity >= 2 (exclude lost-time and fatalities)
        rec_mask = pd.Series([False] * len(inc_df))
        if sev_col and sev_col in inc_df.columns:
            sev_vals = pd.to_numeric(inc_df[sev_col], errors="coerce")
            rec_mask = sev_vals >= 2
        if rec_mask.any():
            rec_mask &= ~(lti_mask | fat_mask)
        tiers["Recordable Injuries"] = int(rec_mask.sum())

        # Near-miss incidents text
        inc_near = 0
        if type_col and type_col in inc_df.columns:
            inc_near = int(inc_df[type_col].astype(str).str.contains("near miss|near-miss|nearmiss", case=False, na=False).sum())
    else:
        inc_near = 0

    # Hazards as near misses
    haz_count = 0
    if haz_df is not None and not haz_df.empty:
        haz_count = int(len(haz_df))
    tiers["Near Misses (estimated)"] = int(haz_count + inc_near)

    # At-risk behaviors from audits and inspections (non-null findings)
    at_risk = 0
    if aud_df is not None and not aud_df.empty:
        fcol = _resolve_column(aud_df, ["finding", "findings"]) or None
        if fcol:
            at_risk += int(aud_df[fcol].notna().sum())
        else:
            at_risk += int(len(aud_df))
    if insp_df is not None and not insp_df.empty:
        fcol = _resolve_column(insp_df, ["finding", "findings"]) or None
        if fcol:
            at_risk += int(insp_df[fcol].notna().sum())
        else:
            at_risk += int(len(insp_df))
    tiers["At-Risk Behaviors (estimated)"] = at_risk

    # ---------- Heinrich expected counts (projection) ----------
    # Try to anchor on Fatality; fallback to Lost Workday, then Recordable.
    base = tiers["Fatality"]
    anchor = "Fatality"
    ratio = {
        "Fatality": 1,
        "Lost Workday Cases": 30,
        "Recordable Injuries": 300,
        "Near Misses (estimated)": 3000,
        "At-Risk Behaviors (estimated)": 300000,
    }
    if base == 0:
        if tiers["Lost Workday Cases"] > 0:
            base = tiers["Lost Workday Cases"] / ratio["Lost Workday Cases"]
            anchor = "Lost Workday Cases"
        elif tiers["Recordable Injuries"] > 0:
            base = tiers["Recordable Injuries"] / ratio["Recordable Injuries"]
            anchor = "Recordable Injuries"
        else:
            base = 0

    expected = {k: (int(round(base * v)) if base else 0) for k, v in ratio.items()}

    # ---------- Build response ----------
    order = ["Fatality", "Lost Workday Cases", "Recordable Injuries", "Near Misses (estimated)", "At-Risk Behaviors (estimated)"]
    colors = ["#616161", "#9e9e9e", "#a3d977", "#7cc7c3", "#7fbf7f"]
    layers = []
    for i, label in enumerate(order):
        layers.append({
            "level": len(order) - i,  # 5 = top, 1 = bottom
            "label": label,
            "count": int(tiers[label]),
            "heinrich_expected": int(expected[label]),
            "anchor": anchor if i == 0 else None,
            "color": colors[i],
        })
    
    return JSONResponse(content=to_native_json({
        "layers": layers,
        "totals": {
            "fatalities": tiers["Fatality"],
            "lost_workday_cases": tiers["Lost Workday Cases"],
            "recordable_injuries": tiers["Recordable Injuries"],
            "near_misses": tiers["Near Misses (estimated)"],
            "at_risk_behaviors": tiers["At-Risk Behaviors (estimated)"],
        },
        "heinrich_expected": expected,
        "near_miss_ratio": _calculate_near_miss_ratio(inc_df, haz_df),
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "location": location,
            "department": department,
        }
    }))


@router.get("/heinrich-pyramid-breakdown")
async def heinrich_pyramid_breakdown(
    start_date: Optional[str] = Query(None, description="Filter start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter end date (YYYY-MM-DD)"),
):
    """
    Detailed breakdown of Heinrich's Pyramid by Department and Location.
    
    Shows contribution of each department and location to pyramid layers:
    - Fatality (Incident sheet: severity_score >= 4-5 OR consequence contains 'fatal')
    - Lost Workday Cases (Incident sheet: severity_score >= 3)
    - Recordable Injuries (Incident sheet: severity_score >= 2)
    - Near Misses (Hazard ID sheet + Incident sheet with 'near miss' type)
    - At-Risk Behaviors (Audit + Inspection sheets with findings)
    
    Data Sources:
    - Incidents: Incident sheet
    - Hazards: Hazard ID sheet
    - Audits: Audit Findings sheet
    - Inspections: Inspection Findings sheet
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Apply date filters
    inc_df = _apply_filters(inc_df, start_date, end_date, None, None)
    haz_df = _apply_filters(haz_df, start_date, end_date, None, None)
    
    breakdown = {
        "by_department": [],
        "by_location": [],
        "data_sources": {
            "fatalities": "Incident sheet (severity_score >= 4-5 OR actual/worst_consequence contains 'fatal')",
            "lost_workday_cases": "Incident sheet (severity_score >= 3, excluding fatalities)",
            "recordable_injuries": "Incident sheet (severity_score >= 2, excluding LTI and fatalities)",
            "near_misses": "Hazard ID sheet + Incident sheet (incident_type contains 'near miss')",
            "at_risk_behaviors": "Audit Findings sheet + Inspection Findings sheet (non-null findings)"
        }
    }
    
    # Department breakdown
    if inc_df is not None and not inc_df.empty:
        dept_col = _resolve_column(inc_df, ["department", "sub_department"])
        sev_col = _resolve_column(inc_df, ["severity_score", "severity"])
        act_cons = _resolve_column(inc_df, ["actual_consequence_incident"])
        worst_cons = _resolve_column(inc_df, ["worst_case_consequence_incident"])
        type_col = _resolve_column(inc_df, ["incident_type", "category"])
        
        if dept_col and dept_col in inc_df.columns:
            for dept in inc_df[dept_col].dropna().unique():
                dept_inc = inc_df[inc_df[dept_col] == dept]
                
                # Calculate layers for this department
                fatalities = 0
                lost_workday = 0
                recordable = 0
                
                if sev_col and sev_col in dept_inc.columns:
                    sev_vals = pd.to_numeric(dept_inc[sev_col], errors="coerce")
                    
                    # Fatalities
                    fat_mask = pd.Series([False] * len(dept_inc))
                    if act_cons and act_cons in dept_inc.columns:
                        fat_mask |= dept_inc[act_cons].astype(str).str.contains("fatal", case=False, na=False)
                    if worst_cons and worst_cons in dept_inc.columns:
                        fat_mask |= dept_inc[worst_cons].astype(str).str.contains("fatal", case=False, na=False)
                    max_val = sev_vals.max(skipna=True)
                    if pd.notna(max_val):
                        thr = 5 if max_val >= 5 else 4
                        fat_mask |= sev_vals >= thr
                    fatalities = int(fat_mask.sum())
                    
                    # Lost Workday Cases
                    lti_mask = (sev_vals >= 3) & ~fat_mask
                    lost_workday = int(lti_mask.sum())
                    
                    # Recordable Injuries
                    rec_mask = (sev_vals >= 2) & ~(lti_mask | fat_mask)
                    recordable = int(rec_mask.sum())
                
                # Near misses from incidents
                near_miss_inc = 0
                if type_col and type_col in dept_inc.columns:
                    near_miss_inc = int(dept_inc[type_col].astype(str).str.contains("near miss|near-miss", case=False, na=False).sum())
                
                # Near misses from hazards
                near_miss_haz = 0
                if haz_df is not None and not haz_df.empty:
                    haz_dept_col = _resolve_column(haz_df, ["department", "sub_department"])
                    if haz_dept_col and haz_dept_col in haz_df.columns:
                        near_miss_haz = int((haz_df[haz_dept_col] == dept).sum())
                
                # At-risk behaviors from audits
                at_risk_aud = 0
                if aud_df is not None and not aud_df.empty:
                    aud_loc_col = _resolve_column(aud_df, ["finding_location", "location", "audit_location"])
                    if aud_loc_col and aud_loc_col in aud_df.columns:
                        at_risk_aud = int(aud_df[aud_loc_col].astype(str).str.contains(str(dept), case=False, na=False).sum())
                
                # At-risk behaviors from inspections
                at_risk_insp = 0
                if insp_df is not None and not insp_df.empty:
                    insp_loc_col = _resolve_column(insp_df, ["finding_location", "location", "audit_location"])
                    if insp_loc_col and insp_loc_col in insp_df.columns:
                        at_risk_insp = int(insp_df[insp_loc_col].astype(str).str.contains(str(dept), case=False, na=False).sum())
                
                breakdown["by_department"].append({
                    "department": str(dept),
                    "fatalities": fatalities,
                    "lost_workday_cases": lost_workday,
                    "recordable_injuries": recordable,
                    "near_misses": near_miss_inc + near_miss_haz,
                    "at_risk_behaviors": at_risk_aud + at_risk_insp,
                    "total_incidents": int(len(dept_inc))
                })
    
    # Location breakdown
    if inc_df is not None and not inc_df.empty:
        loc_col = _resolve_column(inc_df, ["location", "sublocation", "location.1"])
        
        if loc_col and loc_col in inc_df.columns:
            for loc in inc_df[loc_col].dropna().unique():
                loc_inc = inc_df[inc_df[loc_col] == loc]
                
                # Calculate layers for this location
                fatalities = 0
                lost_workday = 0
                recordable = 0
                
                if sev_col and sev_col in loc_inc.columns:
                    sev_vals = pd.to_numeric(loc_inc[sev_col], errors="coerce")
                    
                    # Fatalities
                    fat_mask = pd.Series([False] * len(loc_inc))
                    if act_cons and act_cons in loc_inc.columns:
                        fat_mask |= loc_inc[act_cons].astype(str).str.contains("fatal", case=False, na=False)
                    if worst_cons and worst_cons in loc_inc.columns:
                        fat_mask |= loc_inc[worst_cons].astype(str).str.contains("fatal", case=False, na=False)
                    max_val = sev_vals.max(skipna=True)
                    if pd.notna(max_val):
                        thr = 5 if max_val >= 5 else 4
                        fat_mask |= sev_vals >= thr
                    fatalities = int(fat_mask.sum())
                    
                    # Lost Workday Cases
                    lti_mask = (sev_vals >= 3) & ~fat_mask
                    lost_workday = int(lti_mask.sum())
                    
                    # Recordable Injuries
                    rec_mask = (sev_vals >= 2) & ~(lti_mask | fat_mask)
                    recordable = int(rec_mask.sum())
                
                # Near misses from incidents
                near_miss_inc = 0
                if type_col and type_col in loc_inc.columns:
                    near_miss_inc = int(loc_inc[type_col].astype(str).str.contains("near miss|near-miss", case=False, na=False).sum())
                
                # Near misses from hazards
                near_miss_haz = 0
                if haz_df is not None and not haz_df.empty:
                    haz_loc_col = _resolve_column(haz_df, ["location", "sublocation", "location.1"])
                    if haz_loc_col and haz_loc_col in haz_df.columns:
                        near_miss_haz = int((haz_df[haz_loc_col] == loc).sum())
                
                # At-risk behaviors from audits
                at_risk_aud = 0
                if aud_df is not None and not aud_df.empty:
                    aud_loc_col = _resolve_column(aud_df, ["location", "finding_location", "audit_location"])
                    if aud_loc_col and aud_loc_col in aud_df.columns:
                        at_risk_aud = int((aud_df[aud_loc_col] == loc).sum())
                
                # At-risk behaviors from inspections
                at_risk_insp = 0
                if insp_df is not None and not insp_df.empty:
                    insp_loc_col = _resolve_column(insp_df, ["location", "finding_location", "audit_location"])
                    if insp_loc_col and insp_loc_col in insp_df.columns:
                        at_risk_insp = int((insp_df[insp_loc_col] == loc).sum())
                
                breakdown["by_location"].append({
                    "location": str(loc),
                    "fatalities": fatalities,
                    "lost_workday_cases": lost_workday,
                    "recordable_injuries": recordable,
                    "near_misses": near_miss_inc + near_miss_haz,
                    "at_risk_behaviors": at_risk_aud + at_risk_insp,
                    "total_incidents": int(len(loc_inc))
                })
    
    return JSONResponse(content=to_native_json(breakdown))


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
