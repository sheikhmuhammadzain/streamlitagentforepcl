"""
Centralized filtering utilities for analytics data.
Provides flexible, reusable filtering logic for incidents, hazards, audits, and inspections.
"""
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime


def apply_analytics_filters(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    departments: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    sublocations: Optional[List[str]] = None,
    min_severity: Optional[float] = None,
    max_severity: Optional[float] = None,
    min_risk: Optional[float] = None,
    max_risk: Optional[float] = None,
    statuses: Optional[List[str]] = None,
    incident_types: Optional[List[str]] = None,
    violation_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply flexible filters to a DataFrame.
    
    Args:
        df: Input DataFrame (incident, hazard, audit, or inspection data)
        start_date: Filter records on or after this date (ISO format)
        end_date: Filter records on or before this date (ISO format)
        departments: List of departments to include
        locations: List of locations to include
        sublocations: List of sublocations to include
        min_severity: Minimum severity score (inclusive)
        max_severity: Maximum severity score (inclusive)
        min_risk: Minimum risk score (inclusive)
        max_risk: Maximum risk score (inclusive)
        statuses: List of status values to include
        incident_types: List of incident types to include
        violation_types: List of violation types to include (for hazards)
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    filtered = df.copy()
    
    # Date range filtering - try multiple common date column names
    if start_date or end_date:
        date_cols = ['occurrence_date', 'date_of_occurrence', 'date_reported', 
                     'entered_date', 'start_date', 'scheduled_date', 'created_date']
        date_col = None
        for col in date_cols:
            if col in filtered.columns:
                date_col = col
                break
        
        if date_col:
            try:
                filtered['__temp_date'] = pd.to_datetime(filtered[date_col], errors='coerce')
                
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    filtered = filtered[filtered['__temp_date'] >= start_dt]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    filtered = filtered[filtered['__temp_date'] <= end_dt]
                
                filtered = filtered.drop(columns=['__temp_date'])
            except Exception:
                pass  # If date parsing fails, skip date filtering
    
    # Department filtering
    if departments and 'department' in filtered.columns:
        # Case-insensitive matching
        filtered = filtered[filtered['department'].astype(str).str.lower().isin(
            [d.lower() for d in departments]
        )]
    
    # Location filtering
    if locations:
        loc_cols = ['location', 'location.1', 'site']
        for col in loc_cols:
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype(str).str.lower().isin(
                    [loc.lower() for loc in locations]
                )]
                break
    
    # Sublocation filtering
    if sublocations and 'sublocation' in filtered.columns:
        filtered = filtered[filtered['sublocation'].astype(str).str.lower().isin(
            [sl.lower() for sl in sublocations]
        )]
    
    # Severity filtering
    if min_severity is not None or max_severity is not None:
        sev_cols = ['severity_score', 'severity', 'severity_level']
        for col in sev_cols:
            if col in filtered.columns:
                try:
                    sev_values = pd.to_numeric(filtered[col], errors='coerce')
                    if min_severity is not None:
                        filtered = filtered[sev_values >= min_severity]
                    if max_severity is not None:
                        filtered = filtered[sev_values <= max_severity]
                    break
                except Exception:
                    pass
    
    # Risk filtering
    if min_risk is not None or max_risk is not None:
        risk_cols = ['risk_score', 'risk', 'risk_level']
        for col in risk_cols:
            if col in filtered.columns:
                try:
                    risk_values = pd.to_numeric(filtered[col], errors='coerce')
                    if min_risk is not None:
                        filtered = filtered[risk_values >= min_risk]
                    if max_risk is not None:
                        filtered = filtered[risk_values <= max_risk]
                    break
                except Exception:
                    pass
    
    # Status filtering
    if statuses and 'status' in filtered.columns:
        filtered = filtered[filtered['status'].astype(str).str.lower().isin(
            [s.lower() for s in statuses]
        )]
    
    # Incident type filtering
    if incident_types:
        type_cols = ['incident_type(s)', 'incident_type', 'category', 'accident_type']
        for col in type_cols:
            if col in filtered.columns:
                # Handle comma-separated values
                mask = filtered[col].astype(str).apply(
                    lambda x: any(it.lower() in x.lower() for it in incident_types)
                )
                filtered = filtered[mask]
                break
    
    # Violation type filtering (for hazards)
    if violation_types:
        viol_cols = ['violation_type_hazard_id', 'violation_type', 'violation_type_(incident)']
        for col in viol_cols:
            if col in filtered.columns:
                # Handle comma-separated values
                mask = filtered[col].astype(str).apply(
                    lambda x: any(vt.lower() in x.lower() for vt in violation_types)
                )
                filtered = filtered[mask]
                break
    
    return filtered


def get_filter_summary(
    df_original: pd.DataFrame,
    df_filtered: pd.DataFrame,
    filters_applied: dict
) -> dict:
    """
    Generate a summary of applied filters and their impact.
    
    Args:
        df_original: Original DataFrame before filtering
        df_filtered: DataFrame after filtering
        filters_applied: Dictionary of filter parameters that were applied
    
    Returns:
        Dictionary with filter summary statistics
    """
    original_count = len(df_original) if df_original is not None else 0
    filtered_count = len(df_filtered) if df_filtered is not None else 0
    
    active_filters = {k: v for k, v in filters_applied.items() if v is not None}
    
    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'records_removed': original_count - filtered_count,
        'retention_rate': (filtered_count / original_count * 100) if original_count > 0 else 0,
        'active_filters': active_filters,
        'filter_count': len(active_filters)
    }
