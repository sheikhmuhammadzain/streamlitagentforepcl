"""
Filters Router
Provides dropdown options for location, department, status, and other filter fields.
"""
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..services.excel import get_incident_df, get_hazard_df, get_audit_df, get_inspection_df
from ..services.json_utils import to_native_json


router = APIRouter(prefix="/filters", tags=["filters"])


def _get_unique_values(df: pd.DataFrame, column_candidates: List[str]) -> List[str]:
    """Extract unique non-null values from a column."""
    if df is None or df.empty:
        return []
    
    # Find the column
    col_map = {str(c).strip().lower(): c for c in df.columns}
    column = None
    for candidate in column_candidates:
        key = str(candidate).strip().lower()
        if key in col_map:
            column = col_map[key]
            break
    
    if not column:
        return []
    
    # Get unique values, filter out NaN/None/empty
    values = df[column].dropna().astype(str).unique()
    values = [v for v in values if v and v.strip() and v.lower() not in ['nan', 'none', 'not specified', 'not assigned']]
    return sorted(values)


@router.get("/locations")
async def get_locations():
    """
    Get all unique locations from all datasets for dropdown filters.
    
    Returns:
        List of unique location names sorted alphabetically
    """
    all_locations = set()
    
    # Get locations from incidents
    inc_df = get_incident_df()
    if inc_df is not None and not inc_df.empty:
        locations = _get_unique_values(inc_df, ["location", "sublocation"])
        all_locations.update(locations)
    
    # Get locations from hazards
    haz_df = get_hazard_df()
    if haz_df is not None and not haz_df.empty:
        locations = _get_unique_values(haz_df, ["location", "sublocation"])
        all_locations.update(locations)
    
    # Get locations from audits
    aud_df = get_audit_df()
    if aud_df is not None and not aud_df.empty:
        locations = _get_unique_values(aud_df, ["location", "audit_location"])
        all_locations.update(locations)
    
    # Get locations from inspections
    insp_df = get_inspection_df()
    if insp_df is not None and not insp_df.empty:
        locations = _get_unique_values(insp_df, ["location", "audit_location"])
        all_locations.update(locations)
    
    return JSONResponse(content=to_native_json({
        "locations": sorted(list(all_locations)),
        "count": len(all_locations)
    }))


@router.get("/departments")
async def get_departments():
    """
    Get all unique departments from all datasets for dropdown filters.
    
    Returns:
        List of unique department names sorted alphabetically
    """
    all_departments = set()
    
    # Get departments from incidents
    inc_df = get_incident_df()
    if inc_df is not None and not inc_df.empty:
        departments = _get_unique_values(inc_df, ["department", "sub_department", "section"])
        all_departments.update(departments)
    
    # Get departments from hazards
    haz_df = get_hazard_df()
    if haz_df is not None and not haz_df.empty:
        departments = _get_unique_values(haz_df, ["department", "sub_department", "section"])
        all_departments.update(departments)
    
    return JSONResponse(content=to_native_json({
        "departments": sorted(list(all_departments)),
        "count": len(all_departments)
    }))


@router.get("/statuses")
async def get_statuses():
    """
    Get all unique statuses from all datasets for dropdown filters.
    
    Returns:
        List of unique status values sorted alphabetically
    """
    all_statuses = set()
    
    # Get statuses from incidents
    inc_df = get_incident_df()
    if inc_df is not None and not inc_df.empty:
        statuses = _get_unique_values(inc_df, ["status"])
        all_statuses.update(statuses)
    
    # Get statuses from hazards
    haz_df = get_hazard_df()
    if haz_df is not None and not haz_df.empty:
        statuses = _get_unique_values(haz_df, ["status"])
        all_statuses.update(statuses)
    
    # Get statuses from audits
    aud_df = get_audit_df()
    if aud_df is not None and not aud_df.empty:
        statuses = _get_unique_values(aud_df, ["audit_status", "status"])
        all_statuses.update(statuses)
    
    return JSONResponse(content=to_native_json({
        "statuses": sorted(list(all_statuses)),
        "count": len(all_statuses)
    }))


@router.get("/incident-types")
async def get_incident_types():
    """
    Get all unique incident types for dropdown filters.
    
    Returns:
        List of unique incident type values
    """
    inc_df = get_incident_df()
    incident_types = _get_unique_values(inc_df, ["incident_type", "category"])
    
    return JSONResponse(content=to_native_json({
        "incident_types": incident_types,
        "count": len(incident_types)
    }))


@router.get("/violation-types")
async def get_violation_types():
    """
    Get all unique violation types from hazards for dropdown filters.
    
    Returns:
        List of unique violation type values
    """
    haz_df = get_hazard_df()
    violation_types = _get_unique_values(haz_df, ["violation_type_hazard_id"])
    
    return JSONResponse(content=to_native_json({
        "violation_types": violation_types,
        "count": len(violation_types)
    }))


@router.get("/companies")
async def get_companies():
    """
    Get all unique companies for dropdown filters.
    
    Returns:
        List of unique company names
    """
    all_companies = set()
    
    # Get companies from incidents
    inc_df = get_incident_df()
    if inc_df is not None and not inc_df.empty:
        companies = _get_unique_values(inc_df, ["company"])
        all_companies.update(companies)
    
    # Get companies from hazards
    haz_df = get_hazard_df()
    if haz_df is not None and not haz_df.empty:
        companies = _get_unique_values(haz_df, ["company"])
        all_companies.update(companies)
    
    return JSONResponse(content=to_native_json({
        "companies": sorted(list(all_companies)),
        "count": len(all_companies)
    }))


@router.get("/all")
async def get_all_filters():
    """
    Get all filter options in a single API call for efficiency.
    
    Returns:
        Dictionary containing all filter options:
        - locations
        - departments
        - statuses
        - incident_types
        - violation_types
        - companies
    """
    # Call all individual endpoints
    locations_resp = await get_locations()
    departments_resp = await get_departments()
    statuses_resp = await get_statuses()
    incident_types_resp = await get_incident_types()
    violation_types_resp = await get_violation_types()
    companies_resp = await get_companies()
    
    # Parse responses
    import json
    
    return JSONResponse(content=to_native_json({
        "locations": json.loads(locations_resp.body.decode())["locations"],
        "departments": json.loads(departments_resp.body.decode())["departments"],
        "statuses": json.loads(statuses_resp.body.decode())["statuses"],
        "incident_types": json.loads(incident_types_resp.body.decode())["incident_types"],
        "violation_types": json.loads(violation_types_resp.body.decode())["violation_types"],
        "companies": json.loads(companies_resp.body.decode())["companies"],
    }))


@router.get("/date-range")
async def get_date_range():
    """
    Get the min and max dates from all datasets to set date picker ranges.
    
    Returns:
        Dictionary with earliest_date and latest_date
    """
    all_dates = []
    
    # Get dates from incidents
    inc_df = get_incident_df()
    if inc_df is not None and not inc_df.empty:
        date_cols = ["occurrence_date", "date", "reported_date"]
        for col_candidate in date_cols:
            col_map = {str(c).strip().lower(): c for c in inc_df.columns}
            if col_candidate.lower() in col_map:
                col = col_map[col_candidate.lower()]
                dates = pd.to_datetime(inc_df[col], errors="coerce").dropna()
                all_dates.extend(dates.tolist())
                break
    
    # Get dates from hazards
    haz_df = get_hazard_df()
    if haz_df is not None and not haz_df.empty:
        date_cols = ["occurrence_date", "date", "reported_date"]
        for col_candidate in date_cols:
            col_map = {str(c).strip().lower(): c for c in haz_df.columns}
            if col_candidate.lower() in col_map:
                col = col_map[col_candidate.lower()]
                dates = pd.to_datetime(haz_df[col], errors="coerce").dropna()
                all_dates.extend(dates.tolist())
                break
    
    if not all_dates:
        return JSONResponse(content=to_native_json({
            "earliest_date": None,
            "latest_date": None,
            "message": "No dates found in datasets"
        }))
    
    earliest = min(all_dates)
    latest = max(all_dates)
    
    return JSONResponse(content=to_native_json({
        "earliest_date": earliest.strftime("%Y-%m-%d"),
        "latest_date": latest.strftime("%Y-%m-%d"),
        "total_days": (latest - earliest).days
    }))
