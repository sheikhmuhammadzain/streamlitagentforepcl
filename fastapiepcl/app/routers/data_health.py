"""
Data Health & Validation Router
Provides endpoints to verify data quality, view raw data samples, and check data sources.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..services.excel import get_incident_df, get_hazard_df, get_audit_df, get_inspection_df
from ..services.json_utils import to_native_json


router = APIRouter(prefix="/data-health", tags=["data-health"])


# ======================= DATA HEALTH SUMMARY =======================

@router.get("/summary")
async def get_data_health_summary():
    """
    Get overall data health summary including record counts, date ranges, and quality metrics.
    
    Returns:
        - Total record counts for each dataset
        - Date range (earliest to latest)
        - Data quality metrics
        - Last sync timestamp
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Count records
    total_records = {
        "incidents": len(inc_df) if inc_df is not None else 0,
        "hazards": len(haz_df) if haz_df is not None else 0,
        "audits": len(aud_df) if aud_df is not None else 0,
        "inspections": len(insp_df) if insp_df is not None else 0,
    }
    
    # Get date ranges
    all_dates = []
    
    if inc_df is not None and not inc_df.empty:
        date_col = _find_date_column(inc_df)
        if date_col:
            dates = pd.to_datetime(inc_df[date_col], errors="coerce").dropna()
            all_dates.extend(dates.tolist())
    
    if haz_df is not None and not haz_df.empty:
        date_col = _find_date_column(haz_df)
        if date_col:
            dates = pd.to_datetime(haz_df[date_col], errors="coerce").dropna()
            all_dates.extend(dates.tolist())
    
    date_range = {}
    if all_dates:
        date_range = {
            "earliest": min(all_dates).strftime("%Y-%m-%d"),
            "latest": max(all_dates).strftime("%Y-%m-%d"),
            "total_days": (max(all_dates) - min(all_dates)).days
        }
    
    # Calculate data quality metrics
    data_quality = _calculate_data_quality(inc_df, haz_df, aud_df, insp_df)
    
    return JSONResponse(content=to_native_json({
        "total_records": total_records,
        "grand_total": sum(total_records.values()),
        "date_range": date_range,
        "last_sync": datetime.now().isoformat(),
        "data_quality": data_quality,
        "status": "healthy" if sum(total_records.values()) > 0 else "no_data"
    }))


# ======================= RAW DATA SAMPLES =======================

@router.get("/sample/incidents")
async def get_incident_sample(
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    start_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)", example="2024-01-01"),
    end_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)", example="2024-12-31"),
    search: Optional[str] = Query(None, description="Search in title, description, department", example="catalyst"),
    status: Optional[str] = Query(None, description="Filter by status", example="Closed"),
    department: Optional[str] = Query(None, description="Filter by department", example="Process"),
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi")
):
    """
    Get sample raw data from Incidents sheet with search and filters.
    
    Query Parameters:
        - limit: Number of records (1-100, default: 10)
        - offset: Skip N records (default: 0)
        - start_date: Filter from date (YYYY-MM-DD)
        - end_date: Filter to date (YYYY-MM-DD)
        - search: Search text in title, description, department
        - status: Filter by status
        - department: Filter by department
        - location: Filter by location
    """
    inc_df = get_incident_df()
    
    if inc_df is None or inc_df.empty:
        return JSONResponse(content={"records": [], "total_count": 0, "message": "No incident data"})
    
    # Apply filters
    filtered_df = inc_df.copy()
    
    # Date range filter
    if start_date or end_date:
        date_col = _find_date_column(filtered_df)
        if date_col:
            dates = pd.to_datetime(filtered_df[date_col], errors="coerce")
            if start_date:
                mask = dates >= pd.to_datetime(start_date)
                filtered_df = filtered_df.loc[mask]
            if end_date:
                mask = dates <= pd.to_datetime(end_date)
                filtered_df = filtered_df.loc[mask]
    
    # Search filter
    if search and search.strip():
        search_cols = ["title", "description", "department", "incident_type"]
        search_mask = pd.Series([False] * len(filtered_df))
        for col in search_cols:
            if col in filtered_df.columns:
                search_mask |= filtered_df[col].astype(str).str.contains(search, case=False, na=False)
        filtered_df = filtered_df[search_mask]
    
    # Status filter
    if status and status.strip():
        if "status" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["status"].astype(str).str.contains(status, case=False, na=False)
            ]
    
    # Department filter
    if department and department.strip():
        dept_cols = ["department", "sub_department", "section"]
        dept_mask = pd.Series([False] * len(filtered_df))
        for col in dept_cols:
            if col in filtered_df.columns:
                dept_mask |= filtered_df[col].astype(str).str.contains(department, case=False, na=False)
        filtered_df = filtered_df[dept_mask]
    
    # Location filter
    if location and location.strip():
        loc_cols = ["location", "sublocation"]
        loc_mask = pd.Series([False] * len(filtered_df))
        for col in loc_cols:
            if col in filtered_df.columns:
                loc_mask |= filtered_df[col].astype(str).str.contains(location, case=False, na=False)
        filtered_df = filtered_df[loc_mask]
    
    # Get sample with pagination
    total_filtered = len(filtered_df)
    sample = filtered_df.iloc[offset:offset+limit]
    
    # Convert to records (limit columns for readability)
    key_columns = [
        "incident_id", "occurrence_date", "incident_type", "title", 
        "status", "department", "location", "severity_score", "risk_score"
    ]
    
    available_columns = [col for col in key_columns if col in sample.columns]
    sample_data = sample[available_columns].to_dict('records')
    
    return JSONResponse(content=to_native_json({
        "records": sample_data,
        "total_count": len(inc_df),
        "filtered_count": total_filtered,
        "returned_count": len(sample_data),
        "offset": offset,
        "sheet_name": "Incident",
        "columns_shown": available_columns,
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "search": search,
            "status": status,
            "department": department,
            "location": location
        }
    }))


@router.get("/sample/hazards")
async def get_hazard_sample(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    start_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)", example="2024-01-01"),
    end_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)", example="2024-12-31"),
    search: Optional[str] = Query(None, description="Search in title, description", example="PPE"),
    status: Optional[str] = Query(None, description="Filter by status", example="Closed"),
    department: Optional[str] = Query(None, description="Filter by department", example="PVC"),
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi")
):
    """Get sample raw data from Hazard ID sheet with search and filters."""
    haz_df = get_hazard_df()
    
    if haz_df is None or haz_df.empty:
        return JSONResponse(content={"records": [], "total_count": 0, "message": "No hazard data"})
    
    # Apply filters
    filtered_df = haz_df.copy()
    
    # Date range filter
    if start_date or end_date:
        date_col = _find_date_column(filtered_df)
        if date_col:
            dates = pd.to_datetime(filtered_df[date_col], errors="coerce")
            if start_date:
                mask = dates >= pd.to_datetime(start_date)
                filtered_df = filtered_df.loc[mask]
            if end_date:
                mask = dates <= pd.to_datetime(end_date)
                filtered_df = filtered_df.loc[mask]
    
    # Search filter
    if search and search.strip():
        search_cols = ["title", "description", "department", "incident_type", "violation_type_hazard_id"]
        search_mask = pd.Series([False] * len(filtered_df))
        for col in search_cols:
            if col in filtered_df.columns:
                search_mask |= filtered_df[col].astype(str).str.contains(search, case=False, na=False)
        filtered_df = filtered_df[search_mask]
    
    # Status filter
    if status and status.strip():
        if "status" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["status"].astype(str).str.contains(status, case=False, na=False)
            ]
    
    # Department filter
    if department and department.strip():
        dept_cols = ["department", "sub_department", "section"]
        dept_mask = pd.Series([False] * len(filtered_df))
        for col in dept_cols:
            if col in filtered_df.columns:
                dept_mask |= filtered_df[col].astype(str).str.contains(department, case=False, na=False)
        filtered_df = filtered_df[dept_mask]
    
    # Location filter
    if location and location.strip():
        loc_cols = ["location", "sublocation"]
        loc_mask = pd.Series([False] * len(filtered_df))
        for col in loc_cols:
            if col in filtered_df.columns:
                loc_mask |= filtered_df[col].astype(str).str.contains(location, case=False, na=False)
        filtered_df = filtered_df[loc_mask]
    
    total_filtered = len(filtered_df)
    sample = filtered_df.iloc[offset:offset+limit]
    
    key_columns = [
        "incident_id", "occurrence_date", "incident_type", "title",
        "status", "department", "location", "violation_type_hazard_id", "risk_score"
    ]
    
    available_columns = [col for col in key_columns if col in sample.columns]
    sample_data = sample[available_columns].to_dict('records')
    
    return JSONResponse(content=to_native_json({
        "records": sample_data,
        "total_count": len(haz_df),
        "filtered_count": total_filtered,
        "returned_count": len(sample_data),
        "offset": offset,
        "sheet_name": "Hazard ID",
        "columns_shown": available_columns,
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "search": search,
            "status": status,
            "department": department,
            "location": location
        }
    }))


@router.get("/sample/audits")
async def get_audit_sample(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get sample raw data from Audit sheet."""
    aud_df = get_audit_df()
    
    if aud_df is None or aud_df.empty:
        return JSONResponse(content={"records": [], "total_count": 0, "message": "No audit data"})
    
    sample = aud_df.iloc[offset:offset+limit]
    
    key_columns = [
        "audit_id", "start_date", "audit_title", "audit_status",
        "audit_location", "auditor", "audit_type_epcl", "audit_rating"
    ]
    
    available_columns = [col for col in key_columns if col in sample.columns]
    sample_data = sample[available_columns].to_dict('records')
    
    return JSONResponse(content=to_native_json({
        "records": sample_data,
        "total_count": len(aud_df),
        "returned_count": len(sample_data),
        "offset": offset,
        "sheet_name": "Audit",
        "columns_shown": available_columns
    }))


@router.get("/sample/inspections")
async def get_inspection_sample(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get sample raw data from Inspection sheet."""
    insp_df = get_inspection_df()
    
    if insp_df is None or insp_df.empty:
        return JSONResponse(content={"records": [], "total_count": 0, "message": "No inspection data"})
    
    sample = insp_df.iloc[offset:offset+limit]
    
    key_columns = [
        "audit_id", "start_date", "audit_title", "audit_status",
        "audit_location", "auditor", "audit_type_epcl"
    ]
    
    available_columns = [col for col in key_columns if col in sample.columns]
    sample_data = sample[available_columns].to_dict('records')
    
    return JSONResponse(content=to_native_json({
        "records": sample_data,
        "total_count": len(insp_df),
        "returned_count": len(sample_data),
        "offset": offset,
        "sheet_name": "Inspection",
        "columns_shown": available_columns
    }))


# ======================= DATA SOURCE INFO =======================

@router.get("/source-info")
async def get_data_source_info():
    """
    Get information about the Excel data source file.
    
    Returns:
        - File path and name
        - File size
        - Last modified date
        - Sheet information (names, row counts, columns)
    """
    # Try to find the Excel file
    excel_path = Path("EPCL_VEHS_Data_Processed.xlsx")
    
    if not excel_path.exists():
        # Try parent directory
        excel_path = Path("../EPCL_VEHS_Data_Processed.xlsx")
    
    file_info = {}
    if excel_path.exists():
        stat = excel_path.stat()
        file_info = {
            "filename": excel_path.name,
            "path": str(excel_path.absolute()),
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "exists": True
        }
    else:
        file_info = {
            "filename": "EPCL_VEHS_Data_Processed.xlsx",
            "path": "Not found",
            "exists": False,
            "message": "Excel file not found in expected location"
        }
    
    # Get sheet information
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    sheets = []
    
    if inc_df is not None:
        sheets.append({
            "name": "Incident",
            "row_count": len(inc_df),
            "column_count": len(inc_df.columns),
            "columns": list(inc_df.columns[:10]) + ["..."] if len(inc_df.columns) > 10 else list(inc_df.columns)
        })
    
    if haz_df is not None:
        sheets.append({
            "name": "Hazard ID",
            "row_count": len(haz_df),
            "column_count": len(haz_df.columns),
            "columns": list(haz_df.columns[:10]) + ["..."] if len(haz_df.columns) > 10 else list(haz_df.columns)
        })
    
    if aud_df is not None:
        sheets.append({
            "name": "Audit",
            "row_count": len(aud_df),
            "column_count": len(aud_df.columns),
            "columns": list(aud_df.columns[:10]) + ["..."] if len(aud_df.columns) > 10 else list(aud_df.columns)
        })
    
    if insp_df is not None:
        sheets.append({
            "name": "Inspection",
            "row_count": len(insp_df),
            "column_count": len(insp_df.columns),
            "columns": list(insp_df.columns[:10]) + ["..."] if len(insp_df.columns) > 10 else list(insp_df.columns)
        })
    
    return JSONResponse(content=to_native_json({
        "excel_file": file_info,
        "sheets": sheets,
        "total_sheets": len(sheets),
        "total_rows": sum(sheet["row_count"] for sheet in sheets)
    }))


@router.get("/selected-sheets")
async def get_selected_sheets():
    """Return the sheet names that are currently selected for each dataset.
    Useful to verify that 'inspection' and 'audit' are mapped to the correct Excel sheets.
    """
    from ..services.excel import get_dataset_selection_names
    names = get_dataset_selection_names()
    return JSONResponse(content=to_native_json(names))


@router.get("/counts/all")
async def get_all_counts():
    """Return counts for incidents, hazards, audits, audit findings, inspections, inspection findings.
    Uses dataset accessors for the four primary datasets and name-based matching for findings sheets.
    """
    from ..services.excel import load_default_sheets

    # Primary datasets via accessors
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()

    counts = {
        "incident": int(len(inc_df)) if inc_df is not None else 0,
        "hazard": int(len(haz_df)) if haz_df is not None else 0,
        "audit": int(len(aud_df)) if aud_df is not None else 0,
        "inspection": int(len(insp_df)) if insp_df is not None else 0,
    }

    # Findings via sheet name tokens
    sheets = load_default_sheets()
    selected_sheet_names = {
        "audit_findings": None,
        "inspection_findings": None,
    }

    def _pick_sheet_by_tokens(tokens):
        best_name = None
        best_score = -1
        for name in sheets.keys():
            ln = str(name).strip().lower()
            # must include all tokens
            if all(tok in ln for tok in tokens):
                score = 0
                # prefer exact phrase matches
                phrase = " ".join(tokens)
                if ln == phrase:
                    score += 5
                if ln.startswith(phrase) or ln.endswith(phrase):
                    score += 2
                # prefer names containing 'findings' over 'finding'
                if "findings" in ln:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_name = name
        return best_name

    aud_find_name = _pick_sheet_by_tokens(["audit", "find"])
    insp_find_name = _pick_sheet_by_tokens(["inspection", "find"])

    aud_find_df = sheets.get(aud_find_name) if aud_find_name else None
    insp_find_df = sheets.get(insp_find_name) if insp_find_name else None

    counts.update({
        "audit_findings": int(len(aud_find_df)) if aud_find_df is not None else 0,
        "inspection_findings": int(len(insp_find_df)) if insp_find_df is not None else 0,
    })

    selected_sheet_names.update({
        "audit_findings": aud_find_name,
        "inspection_findings": insp_find_name,
    })

    payload = {
        "counts": counts,
        "selected_sheets": selected_sheet_names,
        "timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    return JSONResponse(content=to_native_json(payload))


# ======================= DATA VALIDATION =======================

@router.get("/validation/check")
async def check_data_validation():
    """
    Validate data quality and identify issues.
    
    Returns:
        - Validation results for each dataset
        - List of issues found
        - Data quality scores
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    validation_results = {}
    
    # Validate incidents
    if inc_df is not None and not inc_df.empty:
        validation_results["incidents"] = _validate_dataframe(
            inc_df, 
            required_columns=["incident_id", "occurrence_date", "status"],
            name="Incidents"
        )
    
    # Validate hazards
    if haz_df is not None and not haz_df.empty:
        validation_results["hazards"] = _validate_dataframe(
            haz_df,
            required_columns=["incident_id", "occurrence_date", "status"],
            name="Hazards"
        )
    
    # Validate audits
    if aud_df is not None and not aud_df.empty:
        validation_results["audits"] = _validate_dataframe(
            aud_df,
            required_columns=["audit_id", "start_date", "audit_status"],
            name="Audits"
        )
    
    # Validate inspections
    if insp_df is not None and not insp_df.empty:
        validation_results["inspections"] = _validate_dataframe(
            insp_df,
            required_columns=["audit_id", "start_date", "audit_status"],
            name="Inspections"
        )
    
    # Calculate overall health score
    total_issues = sum(len(v.get("issues", [])) for v in validation_results.values())
    total_rows = sum(v.get("total_rows", 0) for v in validation_results.values())
    
    health_score = 100 - min(100, (total_issues / max(total_rows, 1)) * 100)
    
    return JSONResponse(content=to_native_json({
        "validation_results": validation_results,
        "overall_health_score": round(health_score, 2),
        "total_issues": total_issues,
        "status": "healthy" if health_score >= 90 else "needs_attention" if health_score >= 70 else "critical"
    }))


# ======================= HELPER FUNCTIONS =======================

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the primary date column in a dataframe."""
    date_candidates = ["occurrence_date", "date", "start_date", "reported_date"]
    for col in date_candidates:
        if col in df.columns:
            return col
    return None


def _calculate_data_quality(inc_df, haz_df, aud_df, insp_df) -> Dict[str, Any]:
    """Calculate data quality metrics."""
    all_dfs = [
        ("incidents", inc_df),
        ("hazards", haz_df),
        ("audits", aud_df),
        ("inspections", insp_df)
    ]
    
    total_cells = 0
    non_null_cells = 0
    
    for name, df in all_dfs:
        if df is not None and not df.empty:
            total_cells += df.size
            non_null_cells += df.count().sum()
    
    completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
    
    return {
        "completeness_percentage": round(completeness, 2),
        "total_cells": int(total_cells),
        "non_null_cells": int(non_null_cells),
        "null_cells": int(total_cells - non_null_cells)
    }


def _validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str) -> Dict[str, Any]:
    """Validate a dataframe and return issues."""
    issues = []
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append({
            "type": "missing_columns",
            "severity": "high",
            "message": f"Missing required columns: {', '.join(missing_cols)}"
        })
    
    # Check for duplicate IDs
    id_col = required_columns[0] if required_columns else None
    if id_col and id_col in df.columns:
        duplicates = df[id_col].duplicated().sum()
        if duplicates > 0:
            issues.append({
                "type": "duplicate_ids",
                "severity": "medium",
                "count": int(duplicates),
                "message": f"Found {duplicates} duplicate IDs in {id_col}"
            })
    
    # Check for null values in critical columns
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append({
                    "type": "null_values",
                    "severity": "medium",
                    "column": col,
                    "count": int(null_count),
                    "message": f"Found {null_count} null values in {col}"
                })
    
    # Check date validity
    date_col = _find_date_column(df)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        invalid_dates = dates.isnull().sum() - df[date_col].isnull().sum()
        if invalid_dates > 0:
            issues.append({
                "type": "invalid_dates",
                "severity": "high",
                "column": date_col,
                "count": int(invalid_dates),
                "message": f"Found {invalid_dates} invalid date formats in {date_col}"
            })
    
    valid_rows = len(df) - len(issues)
    
    return {
        "dataset": name,
        "total_rows": len(df),
        "valid_rows": max(0, valid_rows),
        "issues_found": len(issues),
        "issues": issues,
        "status": "valid" if len(issues) == 0 else "has_issues"
    }


# ======================= CHART DATA TRACING =======================

@router.get("/trace/heinrich-pyramid")
async def trace_heinrich_pyramid_data(
    start_date: Optional[str] = Query(None, example="2024-01-01"),
    end_date: Optional[str] = Query(None, example="2024-12-31"),
    location: Optional[str] = Query(None, example="Karachi"),
    department: Optional[str] = Query(None, example="Process")
):
    """
    Trace data sources for Heinrich's Safety Pyramid chart.
    Shows exactly which sheets, columns, and records are used.
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Apply same filters as the actual endpoint
    from ..routers.analytics_advanced import _apply_filters
    
    inc_filtered = _apply_filters(inc_df, start_date, end_date, location, department) if inc_df is not None else None
    haz_filtered = _apply_filters(haz_df, start_date, end_date, location, department) if haz_df is not None else None
    aud_filtered = _apply_filters(aud_df, start_date, end_date, location) if aud_df is not None else None
    insp_filtered = _apply_filters(insp_df, start_date, end_date, location) if insp_df is not None else None
    
    trace_info = {
        "chart_name": "Heinrich's Safety Pyramid",
        "endpoint": "/analytics/advanced/heinrich-pyramid",
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "location": location,
            "department": department
        },
        "data_sources": [
            {
                "layer": "Layer 1-2 (Serious/Minor Injuries)",
                "excel_sheet": "Incident",
                "columns_used": ["severity_score", "severity_level", "incident_type"],
                "total_records_in_sheet": len(inc_df) if inc_df is not None else 0,
                "records_after_filter": len(inc_filtered) if inc_filtered is not None else 0,
                "sample_ids": inc_filtered["incident_id"].head(3).tolist() if inc_filtered is not None and "incident_id" in inc_filtered.columns else []
            },
            {
                "layer": "Layer 3 (Near Misses)",
                "excel_sheet": "Incident",
                "columns_used": ["incident_type"],
                "filter_criteria": "incident_type contains 'near miss'",
                "total_records_in_sheet": len(inc_df) if inc_df is not None else 0,
                "records_after_filter": len(inc_filtered) if inc_filtered is not None else 0
            },
            {
                "layer": "Layer 4 (Unsafe Conditions)",
                "excel_sheet": "Hazard ID",
                "columns_used": ["incident_id", "violation_type_hazard_id"],
                "total_records_in_sheet": len(haz_df) if haz_df is not None else 0,
                "records_after_filter": len(haz_filtered) if haz_filtered is not None else 0,
                "sample_ids": haz_filtered["incident_id"].head(3).tolist() if haz_filtered is not None and "incident_id" in haz_filtered.columns else []
            },
            {
                "layer": "Layer 5 (At-Risk Behaviors)",
                "excel_sheet": "Audit + Inspection",
                "columns_used": ["audit_id", "audit_status"],
                "total_records_audit": len(aud_df) if aud_df is not None else 0,
                "total_records_inspection": len(insp_df) if insp_df is not None else 0,
                "records_after_filter": (len(aud_filtered) if aud_filtered is not None else 0) + (len(insp_filtered) if insp_filtered is not None else 0)
            }
        ],
        "calculation_method": "Classifies incidents by severity_score: >=4 (serious), 2-3 (minor), <2 (near-miss)"
    }
    
    return JSONResponse(content=to_native_json(trace_info))


@router.get("/trace/kpi-trir")
async def trace_trir_data(
    start_date: Optional[str] = Query(None, example="2023-01-01"),
    end_date: Optional[str] = Query(None, example="2024-12-31")
):
    """
    Trace data sources for TRIR (Total Recordable Incident Rate) KPI.
    """
    inc_df = get_incident_df()
    
    from ..routers.analytics_advanced import _apply_filters
    inc_filtered = _apply_filters(inc_df, start_date, end_date) if inc_df is not None else None
    
    # Count recordable incidents (severity >= 2)
    recordable_count = 0
    if inc_filtered is not None and not inc_filtered.empty:
        severity_col = None
        for col in ["severity_score", "severity_level"]:
            if col in inc_filtered.columns:
                severity_col = col
                break
        
        if severity_col:
            recordable_count = len(inc_filtered[pd.to_numeric(inc_filtered[severity_col], errors="coerce") >= 2])
    
    trace_info = {
        "chart_name": "TRIR (Total Recordable Incident Rate)",
        "endpoint": "/analytics/advanced/kpis/trir",
        "formula": "(Number of recordable incidents × 200,000) / Total hours worked",
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date
        },
        "data_sources": [
            {
                "excel_sheet": "Incident",
                "columns_used": ["severity_score", "severity_level"],
                "filter_criteria": "severity_score >= 2 (recordable incidents)",
                "total_records_in_sheet": len(inc_df) if inc_df is not None else 0,
                "records_after_date_filter": len(inc_filtered) if inc_filtered is not None else 0,
                "recordable_incidents_count": recordable_count,
                "sample_recordable_ids": inc_filtered[pd.to_numeric(inc_filtered.get("severity_score", pd.Series()), errors="coerce") >= 2]["incident_id"].head(5).tolist() if inc_filtered is not None and "incident_id" in inc_filtered.columns and "severity_score" in inc_filtered.columns else []
            }
        ],
        "parameters": {
            "total_hours_worked": 2000000,
            "note": "Default value used for calculation"
        }
    }
    
    return JSONResponse(content=to_native_json(trace_info))


@router.get("/trace/incident-forecast")
async def trace_incident_forecast_data(
    location: Optional[str] = Query(None, example="Karachi"),
    department: Optional[str] = Query(None, example="Process")
):
    """
    Trace data sources for Incident Forecast chart.
    """
    inc_df = get_incident_df()
    
    # Apply filters
    filtered_df = inc_df.copy() if inc_df is not None else None
    
    if filtered_df is not None and location:
        if "location" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["location"].astype(str).str.contains(location, case=False, na=False)]
    
    if filtered_df is not None and department:
        if "department" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["department"].astype(str).str.contains(department, case=False, na=False)]
    
    # Get date column
    date_col = None
    if filtered_df is not None:
        for col in ["occurrence_date", "date", "reported_date"]:
            if col in filtered_df.columns:
                date_col = col
                break
    
    # Get monthly distribution
    monthly_data = []
    if filtered_df is not None and date_col:
        dates = pd.to_datetime(filtered_df[date_col], errors="coerce")
        monthly_counts = dates.dt.to_period('M').value_counts().sort_index()
        monthly_data = [{"month": str(m), "count": int(c)} for m, c in monthly_counts.tail(12).items()]
    
    trace_info = {
        "chart_name": "Incident Likelihood Forecast",
        "endpoint": "/analytics/predictive/incident-forecast",
        "filters_applied": {
            "location": location,
            "department": department
        },
        "data_sources": [
            {
                "excel_sheet": "Incident",
                "columns_used": [date_col if date_col else "occurrence_date", "location", "department"],
                "date_column_used": date_col,
                "total_records_in_sheet": len(inc_df) if inc_df is not None else 0,
                "records_after_filter": len(filtered_df) if filtered_df is not None else 0,
                "historical_months_used": len(monthly_data),
                "monthly_distribution": monthly_data
            }
        ],
        "calculation_method": "Uses last 12 months of data, calculates moving average and trend, projects forward",
        "forecast_parameters": {
            "window_size": 3,
            "method": "Moving Average with Linear Trend"
        }
    }
    
    return JSONResponse(content=to_native_json(trace_info))


@router.get("/trace/site-safety-index")
async def trace_safety_index_data(
    start_date: Optional[str] = Query(None, example="2024-01-01"),
    end_date: Optional[str] = Query(None, example="2024-12-31"),
    location: Optional[str] = Query(None, example="Karachi")
):
    """
    Trace data sources for Site Safety Index calculation.
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    
    from ..routers.analytics_advanced import _apply_filters
    
    inc_filtered = _apply_filters(inc_df, start_date, end_date, location) if inc_df is not None else None
    haz_filtered = _apply_filters(haz_df, start_date, end_date, location) if haz_df is not None else None
    aud_filtered = _apply_filters(aud_df, start_date, end_date, location) if aud_df is not None else None
    
    trace_info = {
        "chart_name": "Site Safety Index (0-100 Score)",
        "endpoint": "/analytics/advanced/site-safety-index",
        "formula": "Base 100 - Deductions + Bonuses",
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "location": location
        },
        "data_sources": [
            {
                "component": "Serious Injuries (Deduction: -10 each)",
                "excel_sheet": "Incident",
                "columns_used": ["severity_score", "severity_level"],
                "filter_criteria": "severity_score >= 4",
                "records_used": len(inc_filtered[pd.to_numeric(inc_filtered.get("severity_score", pd.Series()), errors="coerce") >= 4]) if inc_filtered is not None and "severity_score" in inc_filtered.columns else 0
            },
            {
                "component": "Minor Injuries (Deduction: -3 each)",
                "excel_sheet": "Incident",
                "columns_used": ["severity_score"],
                "filter_criteria": "2 <= severity_score < 4",
                "records_used": len(inc_filtered[(pd.to_numeric(inc_filtered.get("severity_score", pd.Series()), errors="coerce") >= 2) & (pd.to_numeric(inc_filtered.get("severity_score", pd.Series()), errors="coerce") < 4)]) if inc_filtered is not None and "severity_score" in inc_filtered.columns else 0
            },
            {
                "component": "High Risk Hazards (Deduction: -2 each)",
                "excel_sheet": "Hazard ID",
                "columns_used": ["risk_score"],
                "filter_criteria": "risk_score >= 4",
                "records_used": len(haz_filtered[pd.to_numeric(haz_filtered.get("risk_score", pd.Series()), errors="coerce") >= 4]) if haz_filtered is not None and "risk_score" in haz_filtered.columns else 0
            },
            {
                "component": "Completed Audits (Bonus: +0.5 each, max +5)",
                "excel_sheet": "Audit",
                "columns_used": ["audit_status"],
                "filter_criteria": "audit_status contains 'closed' or 'complete'",
                "records_used": len(aud_filtered) if aud_filtered is not None else 0
            }
        ],
        "calculation_steps": [
            "1. Start with base score of 100",
            "2. Subtract points for serious injuries, minor injuries, hazards",
            "3. Add bonus points for completed audits and days since last incident",
            "4. Clamp result between 0 and 100"
        ]
    }
    
    return JSONResponse(content=to_native_json(trace_info))


@router.get("/trace/all-charts")
async def trace_all_charts():
    """
    Get a summary of data sources for all major charts.
    Quick reference for which sheets/columns each chart uses.
    """
    charts_summary = [
        {
            "chart_name": "Heinrich's Safety Pyramid",
            "endpoint": "/analytics/advanced/heinrich-pyramid",
            "sheets_used": ["Incident", "Hazard ID", "Audit", "Inspection"],
            "key_columns": ["severity_score", "incident_type", "violation_type_hazard_id"],
            "trace_endpoint": "/data-health/trace/heinrich-pyramid"
        },
        {
            "chart_name": "TRIR KPI",
            "endpoint": "/analytics/advanced/kpis/trir",
            "sheets_used": ["Incident"],
            "key_columns": ["severity_score", "severity_level"],
            "trace_endpoint": "/data-health/trace/kpi-trir"
        },
        {
            "chart_name": "LTIR KPI",
            "endpoint": "/analytics/advanced/kpis/ltir",
            "sheets_used": ["Incident"],
            "key_columns": ["severity_score", "lost_time_indicator"],
            "trace_endpoint": "Use /data-health/trace/kpi-trir (same logic)"
        },
        {
            "chart_name": "PSTIR KPI",
            "endpoint": "/analytics/advanced/kpis/pstir",
            "sheets_used": ["Incident"],
            "key_columns": ["incident_type", "category"],
            "trace_endpoint": "Use /data-health/trace/kpi-trir (same logic)"
        },
        {
            "chart_name": "Near-Miss Ratio",
            "endpoint": "/analytics/advanced/kpis/near-miss-ratio",
            "sheets_used": ["Incident", "Hazard ID"],
            "key_columns": ["incident_type"],
            "trace_endpoint": "Similar to TRIR"
        },
        {
            "chart_name": "Site Safety Index",
            "endpoint": "/analytics/advanced/site-safety-index",
            "sheets_used": ["Incident", "Hazard ID", "Audit"],
            "key_columns": ["severity_score", "risk_score", "audit_status"],
            "trace_endpoint": "/data-health/trace/site-safety-index"
        },
        {
            "chart_name": "Incident Forecast",
            "endpoint": "/analytics/predictive/incident-forecast",
            "sheets_used": ["Incident"],
            "key_columns": ["occurrence_date", "location", "department"],
            "trace_endpoint": "/data-health/trace/incident-forecast"
        },
        {
            "chart_name": "Risk Trend Projection",
            "endpoint": "/analytics/predictive/risk-trend-projection",
            "sheets_used": ["Incident", "Hazard ID"],
            "key_columns": ["occurrence_date", "risk_score", "severity_score"],
            "trace_endpoint": "Similar to incident forecast"
        },
        {
            "chart_name": "Leading vs Lagging Indicators",
            "endpoint": "/analytics/predictive/leading-vs-lagging",
            "sheets_used": ["Incident", "Hazard ID", "Audit", "Inspection"],
            "key_columns": ["incident_type", "audit_status", "occurrence_date"],
            "trace_endpoint": "Uses multiple sheets"
        }
    ]
    
    return JSONResponse(content=to_native_json({
        "total_charts": len(charts_summary),
        "charts": charts_summary,
        "note": "Use individual trace endpoints for detailed data source information"
    }))


# ======================= CHART DATA VALIDATION =======================

def _resolve_column_local(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    colmap = {str(c).strip().lower(): c for c in df.columns}
    # exact match first
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in colmap:
            return colmap[k]
    # relaxed contains
    for cand in candidates:
        k = str(cand).strip().lower()
        for lk, orig in colmap.items():
            if k in lk:
                return orig
    return None


@router.get("/validate/charts")
async def validate_charts():
    """Validate that each analytics chart can resolve its required columns against the current Excel sheets.
    Returns per-chart status with resolved columns and any missing ones.
    """
    inc = get_incident_df()
    haz = get_hazard_df()
    aud = get_audit_df()
    ins = get_inspection_df()

    results: List[Dict[str, Any]] = []

    def add_result(name: str, endpoint: str, df: Optional[pd.DataFrame], reqs: Dict[str, List[str]]):
        resolved: Dict[str, Optional[str]] = {}
        missing: List[str] = []
        for key, cands in reqs.items():
            col = _resolve_column_local(df, cands) if isinstance(df, pd.DataFrame) else None
            resolved[key] = col
            if col is None:
                missing.append(key)
        status = "ok" if not missing else "missing_columns"
        results.append({
            "chart": name,
            "endpoint": endpoint,
            "status": status,
            "resolved_columns": resolved,
            "missing": missing,
            "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        })

    # Incident charts
    add_result(
        "Incident Trend", "/analytics/data/incident-trend",
        inc, {"date": ["occurrence_date", "date of occurrence", "date reported", "date entered", "date"]}
    )
    add_result(
        "Incident Type Distribution", "/analytics/data/incident-type-distribution",
        inc, {"type": ["incident_type", "incident type(s)", "category", "accident type"]}
    )
    add_result(
        "Root Cause Pareto", "/analytics/data/root-cause-pareto",
        inc, {"root_cause": ["root_cause", "root cause"]}
    )
    add_result(
        "Injury Severity Pyramid", "/analytics/data/injury-severity-pyramid",
        inc, {"severity": [
            "injury_classification",
            "injury classification",
            "actual_consequence_incident",
            "actual consequence (incident)",
            "relevant_consequence_incident",
            "relevant consequence (incident)"
        ]}
    )
    add_result(
        "Department-Month Heatmap", "/analytics/data/department-month-heatmap",
        inc, {
            "department": ["department", "section"],
            "date": ["occurrence_date", "date of occurrence", "date reported", "date"]
        }
    )
    add_result(
        "Consequence Gap", "/analytics/data/consequence-gap",
        inc, {
            "actual": ["actual_consequence_incident", "actual consequence (incident)"],
            "worst": ["worst_case_consequence_incident", "worst case consequence (incident)"]
        }
    )
    add_result(
        "Incident Cost Trend", "/analytics/data/incident-cost-trend",
        inc, {
            "date": ["occurrence_date", "date of occurrence", "date reported"],
            "cost": ["total cost", "estimated_cost_impact"]
        }
    )
    
    add_result(
        "Repeated Incidents", "/analytics/data/repeated-incidents",
        inc, {
            "repeat_flag": ["repeated_incident", "repeated incident", "repeated_event", "repeated event"],
            "location": [
                "specific_location_of_occurrence",
                "specific location of occurrence",
                "sub_location",
                "sub-location",
                "sublocation",
                "location"
            ]
        }
    )

    # Hazard charts
    add_result(
        "Hazard Trend (via Incident Trend)", "/analytics/data/incident-trend?dataset=hazard",
        haz, {"date": ["occurrence_date", "date of occurrence", "date reported", "date entered", "date"]}
    )
    add_result(
        "Hazard Cost Trend", "/analytics/data/hazard-cost-trend",
        haz, {
            "date": ["occurrence_date", "date of occurrence", "date reported"],
            "cost": ["total cost", "estimated_cost_impact"]
        }
    )

    # Audit charts
    add_result(
        "Audit Status Distribution", "/analytics/data/audit-status-distribution",
        aud, {"status": ["audit_status", "audit status"]}
    )
    add_result(
        "Audit Rating Trend", "/analytics/data/audit-rating-trend",
        aud, {
            "date": ["start_date", "start date"],
            "rating": ["audit_rating", "audit rating"]
        }
    )

    # Inspection charts
    add_result(
        "Inspection Coverage", "/analytics/data/inspection-coverage",
        ins, {
            "date": ["start_date", "start date"],
            "status": ["audit_status", "audit status"]
        }
    )
    add_result(
        "Inspection Top Findings", "/analytics/data/inspection-top-findings",
        ins, {
            "category": ["checklist_category", "checklist category", "finding"]
        }
    )
    add_result(
        "Audit Top Findings", "/analytics/data/audit-top-findings",
        aud, {
            "category": ["checklist_category", "checklist category", "finding"]
        }
    )
    add_result(
        "Incident Top Findings", "/analytics/data/incident-top-findings",
        inc, {
            "category": ["description", "conclusion", "root_cause", "incident_type"]
        }
    )
    add_result(
        "Hazard Top Findings", "/analytics/data/hazard-top-findings",
        haz, {
            "category": ["description", "violation_type_hazard_id", "incident_type"]
        }
    )

    # Advanced analytics validations
    # Heinrich Pyramid
    add_result(
        "Heinrich - Incidents Layers",
        "/analytics/advanced/heinrich-pyramid",
        inc,
        {
            "severity_score_or_risk": ["severity_score", "severity", "risk_score"],
            "severity_text": [
                "actual_consequence_incident",
                "worst_case_consequence_incident",
                "relevant_consequence_incident",
                "severity_level"
            ],
        },
    )
    add_result(
        "Heinrich - Audits Findings",
        "/analytics/advanced/heinrich-pyramid",
        aud,
        {"finding": ["finding", "findings"]},
    )
    add_result(
        "Heinrich - Inspections Findings",
        "/analytics/advanced/heinrich-pyramid",
        ins,
        {"finding": ["finding", "findings"]},
    )
    add_result(
        "Heinrich Pyramid Breakdown - Incidents",
        "/analytics/advanced/heinrich-pyramid-breakdown",
        inc,
        {
            "department": ["department", "sub_department"],
            "location": ["location", "sublocation", "location.1"],
            "severity": ["severity_score", "severity"],
            "consequence": ["actual_consequence_incident", "worst_case_consequence_incident"],
            "type": ["incident_type", "category"]
        },
    )
    add_result(
        "Heinrich Pyramid Breakdown - Hazards",
        "/analytics/advanced/heinrich-pyramid-breakdown",
        haz,
        {
            "department": ["department", "sub_department"],
            "location": ["location", "sublocation", "location.1"]
        },
    )

    # Site Safety Index
    add_result(
        "Site Safety Index - Incidents",
        "/analytics/advanced/site-safety-index",
        inc,
        {
            "severity_score_or_risk": ["severity_score", "risk_score"],
            "severity_text": ["actual_consequence_incident", "severity"],
            "date": ["occurrence_date", "date"],
            "status": ["status"],
        },
    )
    add_result(
        "Site Safety Index - Hazards",
        "/analytics/advanced/site-safety-index",
        haz,
        {"risk": ["risk_score", "risk_level"]},
    )
    add_result(
        "Site Safety Index - Audits",
        "/analytics/advanced/site-safety-index",
        aud,
        {"status": ["audit_status", "status"]},
    )

    # Advanced KPIs
    add_result(
        "TRIR",
        "/analytics/advanced/kpis/trir",
        inc,
        {"severity_for_recordable": ["severity_score", "risk_score"]},
    )
    add_result(
        "LTIR",
        "/analytics/advanced/kpis/ltir",
        inc,
        {"severity_for_lost_time": ["severity_score", "risk_score"]},
    )
    add_result(
        "PSTIR",
        "/analytics/advanced/kpis/pstir",
        inc,
        {"psm_marker": ["psm", "pse_category"]},
    )
    add_result(
        "Near-Miss Ratio",
        "/analytics/advanced/kpis/near-miss-ratio",
        inc,
        {"incident_type_for_near_miss": ["incident_type"]},
    )

    # ---------------- Additional General Charts (Plot-based) ----------------
    # HSE Scorecard
    add_result(
        "HSE Scorecard",
        "/analytics/hse-scorecard",
        aud,
        {"audit_status": ["audit_status", "status"]}
    )

    # HSE Performance Index (requires department; others optional)
    add_result(
        "HSE Performance Index",
        "/analytics/hse-performance-index",
        inc,
        {"department": ["department"]}
    )

    # Risk Calendar Heatmap (strict)
    add_result(
        "Risk Calendar Heatmap",
        "/analytics/risk-calendar-heatmap",
        inc,
        {"date": ["occurrence_date", "date"], "department": ["department"], "risk": ["risk_score"]}
    )

    # PSM Breakdown (optional, but check useful columns)
    add_result(
        "PSM Breakdown",
        "/analytics/psm-breakdown",
        inc,
        {"psm": ["psm"], "pse_category": ["pse_category"]}
    )

    # Consequence Matrix (strict)
    add_result(
        "Consequence Matrix",
        "/analytics/consequence-matrix",
        inc,
        {"actual": ["actual_consequence_incident"], "worst": ["worst_case_consequence_incident"]}
    )

    # Data Quality Metrics (key fields for richer output)
    add_result(
        "Data Quality Metrics",
        "/analytics/data-quality-metrics",
        inc,
        {
            "department": ["department"],
            "root_cause_is_missing": ["root_cause_is_missing"],
            "corrective_actions_is_missing": ["corrective_actions_is_missing"],
            "reporting_delay_days": ["reporting_delay_days"],
            "resolution_time_days": ["resolution_time_days"],
            "status": ["status"],
        }
    )

    # Comprehensive Timeline (strict date)
    add_result(
        "Comprehensive Timeline",
        "/analytics/comprehensive-timeline",
        inc,
        {"date": ["occurrence_date"]}
    )

    # Audit & Inspection Tracker (needs start_date + audit_status)
    add_result(
        "Audit Inspection Tracker - Audits",
        "/analytics/audit-inspection-tracker",
        aud,
        {"start_date": ["start_date"], "audit_status": ["audit_status", "status"]}
    )
    add_result(
        "Audit Inspection Tracker - Inspections",
        "/analytics/audit-inspection-tracker",
        ins,
        {"start_date": ["start_date"], "audit_status": ["audit_status", "status"]}
    )

    # Location Risk Treemap (strict)
    add_result(
        "Location Risk Treemap",
        "/analytics/location-risk-treemap",
        inc,
        {"location": ["location"], "sublocation": ["sublocation", "sub_location", "location.1"]}
    )

    # Department Spider (strict department)
    add_result(
        "Department Spider",
        "/analytics/department-spider",
        inc,
        {"department": ["department"]}
    )

    # Violation Analysis (hazard dataset)
    add_result(
        "Violation Analysis",
        "/analytics/violation-analysis",
        haz,
        {"violation_type": ["violation_type_hazard_id"]}
    )

    # Cost Prediction Analysis (needs cost)
    add_result(
        "Cost Prediction Analysis",
        "/analytics/cost-prediction-analysis",
        inc,
        {"estimated_cost_impact": ["estimated_cost_impact"]}
    )

    # Facility Layout Heatmap (location fields)
    add_result(
        "Facility Layout Heatmap - Incidents",
        "/analytics/facility-layout-heatmap",
        inc,
        {"location": ["location", "sublocation", "location.1"]}
    )
    add_result(
        "Facility Layout Heatmap - Hazards",
        "/analytics/facility-layout-heatmap",
        haz,
        {"location": ["location", "sublocation", "location.1"]}
    )

    # Facility 3D Heatmap (location + severity/risk preferred)
    add_result(
        "Facility 3D Heatmap",
        "/analytics/facility-3d-heatmap",
        inc,
        {"location": ["location", "sublocation", "location.1"], "sev_or_risk": ["severity_score", "risk_score"]}
    )

    # ---------------- Filters & Summary ----------------
    # Filter Options (ensure dates and some dims exist)
    add_result(
        "Filter Options (Incident)",
        "/analytics/filter-options",
        inc,
        {"date_any": ["occurrence_date", "reported_date", "entered_date", "start_date"], "department": ["department"], "location": ["location"], "status": ["status"]}
    )
    add_result(
        "Filter Options (Hazard)",
        "/analytics/filter-options",
        haz,
        {"date_any": ["occurrence_date", "reported_date", "entered_date", "start_date"], "violation_type": ["violation_type_hazard_id"]}
    )
    add_result(
        "Filter Summary",
        "/analytics/filter-summary",
        inc,
        {"date": ["occurrence_date", "reported_date", "entered_date"]}
    )

    # ---------------- Predictive Analytics ----------------
    add_result(
        "Incident Forecast",
        "/analytics/predictive/incident-forecast",
        inc,
        {"date": ["occurrence_date", "date", "reported_date"]}
    )
    add_result(
        "Risk Trend Projection - Incidents",
        "/analytics/predictive/risk-trend-projection",
        inc,
        {"date": ["occurrence_date", "date"], "risk_or_sev": ["risk_score", "severity_score"]}
    )
    add_result(
        "Risk Trend Projection - Hazards",
        "/analytics/predictive/risk-trend-projection",
        haz,
        {"date": ["occurrence_date", "date"], "risk": ["risk_score"]}
    )
    add_result(
        "Leading vs Lagging - Audits",
        "/analytics/predictive/leading-vs-lagging",
        aud,
        {"status": ["audit_status", "status"]}
    )
    add_result(
        "Leading vs Lagging - Incidents",
        "/analytics/predictive/leading-vs-lagging",
        inc,
        {"type": ["incident_type", "category"], "severity_or_risk": ["severity_score", "risk_score"], "date": ["occurrence_date", "date", "reported_date"]}
    )
    add_result(
        "Leading vs Lagging - Hazards",
        "/analytics/predictive/leading-vs-lagging",
        haz,
        {"date": ["occurrence_date", "date", "reported_date"]}
    )
    add_result(
        "Leading vs Lagging - Inspections",
        "/analytics/predictive/leading-vs-lagging",
        ins,
        {"date": ["start_date", "date"]}
    )
    add_result(
        "Observation Lag Time - Incidents",
        "/analytics/predictive/observation-lag-time",
        inc,
        {"date": ["occurrence_date", "date"]}
    )
    add_result(
        "Observation Lag Time - Hazards",
        "/analytics/predictive/observation-lag-time",
        haz,
        {"date": ["occurrence_date", "date", "reported_date"]}
    )

    return JSONResponse(content=to_native_json({
        "validated": results,
        "summary": {
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "with_issues": sum(1 for r in results if r["status"] != "ok"),
            "timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
        }
    }))
