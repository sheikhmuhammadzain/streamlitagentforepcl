"""
Utility functions to extract available filter options from datasets.
Provides data for frontend dropdown menus and filter UI components.
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..models.schemas import (
    FilterOption,
    FilterOptionsResponse,
    DateRangeInfo,
    CombinedFilterOptionsResponse,
)


def _extract_unique_values(
    df: pd.DataFrame,
    column_candidates: List[str],
    explode_comma_separated: bool = False,
    min_count: int = 1,
    max_items: int = 100,
) -> List[FilterOption]:
    """
    Extract unique values from a DataFrame column with counts.
    
    Args:
        df: Input DataFrame
        column_candidates: List of possible column names to try
        explode_comma_separated: If True, split comma-separated values
        min_count: Minimum count to include in results
        max_items: Maximum number of items to return
    
    Returns:
        List of FilterOption objects sorted by count (descending)
    """
    if df is None or df.empty:
        return []
    
    # Find the first matching column
    column = None
    for col in column_candidates:
        if col in df.columns:
            column = col
            break
    
    if column is None:
        return []
    
    try:
        # Get the series
        series = df[column].dropna().astype(str)
        
        # Remove empty strings and common null representations
        series = series[~series.str.strip().isin(['', 'nan', 'NaN', 'None', 'null', 'N/A', 'n/a'])]
        
        if series.empty:
            return []
        
        # Handle comma-separated values if needed
        if explode_comma_separated:
            series = series.str.split(',').explode().str.strip()
            series = series[series != '']
        
        # Count occurrences
        value_counts = series.value_counts()
        
        # Filter by minimum count
        value_counts = value_counts[value_counts >= min_count]
        
        # Limit to max items
        value_counts = value_counts.head(max_items)
        
        # Convert to FilterOption objects
        options = [
            FilterOption(
                value=str(val),
                label=str(val).title() if len(str(val)) < 50 else str(val),
                count=int(count)
            )
            for val, count in value_counts.items()
        ]
        
        return options
    
    except Exception:
        return []


def _extract_date_range(
    df: pd.DataFrame,
    date_column_candidates: List[str]
) -> DateRangeInfo:
    """
    Extract date range information from DataFrame.
    
    Args:
        df: Input DataFrame
        date_column_candidates: List of possible date column names
    
    Returns:
        DateRangeInfo object with min/max dates
    """
    if df is None or df.empty:
        return DateRangeInfo(min_date=None, max_date=None, total_records=0)
    
    # Find the first matching date column
    date_col = None
    for col in date_column_candidates:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        return DateRangeInfo(min_date=None, max_date=None, total_records=0)
    
    try:
        # Convert to datetime
        dates = pd.to_datetime(df[date_col], errors='coerce')
        dates = dates.dropna()
        
        if dates.empty:
            return DateRangeInfo(min_date=None, max_date=None, total_records=0)
        
        min_date = dates.min().date().isoformat()
        max_date = dates.max().date().isoformat()
        
        return DateRangeInfo(
            min_date=min_date,
            max_date=max_date,
            total_records=len(dates)
        )
    
    except Exception:
        return DateRangeInfo(min_date=None, max_date=None, total_records=0)


def _extract_numeric_range(
    df: pd.DataFrame,
    column_candidates: List[str]
) -> Dict[str, float]:
    """
    Extract numeric range statistics (min, max, avg).
    
    Args:
        df: Input DataFrame
        column_candidates: List of possible column names
    
    Returns:
        Dictionary with min, max, avg, median values
    """
    if df is None or df.empty:
        return {}
    
    # Find the first matching column
    column = None
    for col in column_candidates:
        if col in df.columns:
            column = col
            break
    
    if column is None:
        return {}
    
    try:
        # Convert to numeric
        values = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if values.empty:
            return {}
        
        return {
            'min': float(values.min()),
            'max': float(values.max()),
            'avg': float(values.mean()),
            'median': float(values.median()),
            'count': int(len(values))
        }
    
    except Exception:
        return {}


def extract_filter_options(df: pd.DataFrame, dataset_name: str = "incident") -> FilterOptionsResponse:
    """
    Extract all available filter options from a dataset.
    
    Args:
        df: Input DataFrame (incident or hazard data)
        dataset_name: Name of the dataset ("incident" or "hazard")
    
    Returns:
        FilterOptionsResponse with all available filter options
    """
    if df is None or df.empty:
        return FilterOptionsResponse(
            dataset=dataset_name,
            date_range=DateRangeInfo(min_date=None, max_date=None, total_records=0),
            total_records=0
        )
    
    # Extract date range
    date_range = _extract_date_range(
        df,
        ['occurrence_date', 'date_of_occurrence', 'date_reported', 'entered_date', 
         'start_date', 'scheduled_date', 'created_date']
    )
    
    # Extract departments
    departments = _extract_unique_values(
        df,
        ['department', 'dept', 'department_name'],
        explode_comma_separated=False,
        min_count=1
    )
    
    # Extract locations
    locations = _extract_unique_values(
        df,
        ['location', 'location.1', 'site', 'facility'],
        explode_comma_separated=False,
        min_count=1
    )
    
    # Extract sublocations
    sublocations = _extract_unique_values(
        df,
        ['sublocation', 'sub_location', 'area', 'zone'],
        explode_comma_separated=False,
        min_count=1
    )
    
    # Extract statuses
    statuses = _extract_unique_values(
        df,
        ['status', 'incident_status', 'current_status'],
        explode_comma_separated=False,
        min_count=1
    )
    
    # Extract incident types
    incident_types = _extract_unique_values(
        df,
        ['incident_type(s)', 'incident_type', 'category', 'accident_type', 'type'],
        explode_comma_separated=True,  # Handle comma-separated values
        min_count=1
    )
    
    # Extract violation types (for hazards)
    violation_types = []
    if dataset_name.lower() == "hazard":
        violation_types = _extract_unique_values(
            df,
            ['violation_type_hazard_id', 'violation_type', 'violation_type_(incident)'],
            explode_comma_separated=True,
            min_count=1
        )
    
    # Extract severity range
    severity_range = _extract_numeric_range(
        df,
        ['severity_score', 'severity', 'severity_level']
    )
    
    # Extract risk range
    risk_range = _extract_numeric_range(
        df,
        ['risk_score', 'risk', 'risk_level']
    )
    
    return FilterOptionsResponse(
        dataset=dataset_name,
        date_range=date_range,
        departments=departments,
        locations=locations,
        sublocations=sublocations,
        statuses=statuses,
        incident_types=incident_types,
        violation_types=violation_types,
        severity_range=severity_range,
        risk_range=risk_range,
        total_records=len(df)
    )


def extract_combined_filter_options(
    incident_df: pd.DataFrame,
    hazard_df: pd.DataFrame
) -> CombinedFilterOptionsResponse:
    """
    Extract filter options from both incident and hazard datasets.
    
    Args:
        incident_df: Incident DataFrame
        hazard_df: Hazard DataFrame
    
    Returns:
        CombinedFilterOptionsResponse with options from both datasets
    """
    incident_options = extract_filter_options(incident_df, "incident")
    hazard_options = extract_filter_options(hazard_df, "hazard")
    
    return CombinedFilterOptionsResponse(
        incident=incident_options,
        hazard=hazard_options,
        last_updated=datetime.utcnow().isoformat()
    )
