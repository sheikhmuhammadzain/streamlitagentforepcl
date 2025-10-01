"""
Predictive Analytics Router
Implements forecasting, trend projection, and leading vs lagging indicators.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..services.excel import get_incident_df, get_hazard_df, get_audit_df, get_inspection_df
from ..services.json_utils import to_native_json


router = APIRouter(prefix="/analytics/predictive", tags=["predictive-analytics"])


# ======================= HELPER FUNCTIONS =======================

def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Resolve column name from list of candidates."""
    if df is None or df.empty:
        return None
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in col_map:
            return col_map[key]
    for candidate in candidates:
        key = str(candidate).strip().lower()
        for lk, orig in col_map.items():
            if key in lk:
                return orig
    return None


def _simple_forecast(historical_data: pd.Series, months_ahead: int = 3) -> List[float]:
    """
    Simple moving average forecast with trend adjustment.
    Uses last 6 months to predict next N months.
    """
    if historical_data.empty or len(historical_data) < 2:
        return [0.0] * months_ahead
    
    # Calculate moving average and trend
    window_size = min(6, len(historical_data))
    recent_data = historical_data.tail(window_size)
    
    # Simple linear trend
    x = np.arange(len(recent_data))
    y = recent_data.values
    
    if len(x) > 1:
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        intercept = recent_data.mean()
    else:
        slope = 0
        intercept = recent_data.iloc[0]
    
    # Forecast
    forecasts = []
    for i in range(1, months_ahead + 1):
        forecast_val = intercept + (slope * i)
        forecast_val = max(0, forecast_val)  # No negative forecasts
        forecasts.append(round(float(forecast_val), 2))
    
    return forecasts


# ======================= INCIDENT FORECAST =======================

@router.get("/incident-forecast")
async def incident_forecast(
    months_ahead: int = Query(4, ge=1, le=12, description="Number of months to forecast", example=4),
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi"),
    department: Optional[str] = Query(None, description="Filter by department", example="Process - EDC / VCM"),
):
    """
    Incident Likelihood Forecast (4-month outlook by default).
    
    Uses historical incident patterns to predict future incident counts.
    Methodology:
    - Analyzes last 12 months of incident data
    - Calculates moving average and trend
    - Projects forward with confidence intervals
    """
    inc_df = get_incident_df()
    
    if inc_df is None or inc_df.empty:
        return JSONResponse(content=to_native_json({
            "forecast": [],
            "historical": [],
            "message": "No incident data available"
        }))
    
    # Apply filters
    if location is not None and location != "":
        loc_col = _resolve_column(inc_df, ["location"])
        if loc_col:
            inc_df = inc_df[inc_df[loc_col].astype(str).str.contains(str(location), case=False, na=False)]
    
    if department is not None and department != "":
        dept_col = _resolve_column(inc_df, ["department", "section"])
        if dept_col:
            inc_df = inc_df[inc_df[dept_col].astype(str).str.contains(str(department), case=False, na=False)]
    
    # Extract monthly counts
    date_col = _resolve_column(inc_df, ["occurrence_date", "date", "reported_date"])
    if not date_col:
        return JSONResponse(content={"forecast": [], "historical": [], "message": "No date column found"})
    
    dates = pd.to_datetime(inc_df[date_col], errors="coerce")
    monthly_counts = dates.dt.to_period('M').value_counts().sort_index()
    
    # Get last 12 months
    if len(monthly_counts) > 12:
        monthly_counts = monthly_counts.tail(12)
    
    # Generate forecast
    forecast_values = _simple_forecast(monthly_counts, months_ahead)
    
    # Build historical data
    historical = []
    for period, count in monthly_counts.items():
        historical.append({
            "month": str(period),
            "count": int(count),
        })
    
    # Build forecast data
    last_period = monthly_counts.index[-1] if len(monthly_counts) > 0 else pd.Period.now(freq='M')
    forecast = []
    for i, val in enumerate(forecast_values, start=1):
        future_period = last_period + i
        forecast.append({
            "month": str(future_period),
            "predicted_count": val,
            "confidence_lower": round(max(0, val * 0.7), 2),
            "confidence_upper": round(val * 1.3, 2),
        })
    
    return JSONResponse(content=to_native_json({
        "historical": historical,
        "forecast": forecast,
        "months_ahead": months_ahead,
        "forecast_method": "Moving Average with Trend Adjustment",
    }))


# ======================= RISK TREND PROJECTION =======================

@router.get("/risk-trend-projection")
async def risk_trend_projection(
    months_ahead: int = Query(3, ge=1, le=12, description="Number of months to forecast", example=3),
    location: Optional[str] = Query(None, description="Filter by location", example="Manufacturing Facility"),
):
    """
    Risk Trend Lines with Future Projection.
    
    Analyzes average risk scores over time and projects future trends.
    Useful for identifying whether risk levels are increasing or decreasing.
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    
    combined_data = []
    
    # Process incidents
    if inc_df is not None and not inc_df.empty:
        if location is not None and location != "":
            loc_col = _resolve_column(inc_df, ["location"])
            if loc_col:
                inc_df = inc_df[inc_df[loc_col].astype(str).str.contains(str(location), case=False, na=False)]
        
        date_col = _resolve_column(inc_df, ["occurrence_date", "date"])
        risk_col = _resolve_column(inc_df, ["risk_score", "severity_score"])
        
        if date_col and risk_col:
            inc_df['month'] = pd.to_datetime(inc_df[date_col], errors="coerce").dt.to_period('M')
            inc_df['risk'] = pd.to_numeric(inc_df[risk_col], errors="coerce")
            
            inc_monthly = inc_df.groupby('month')['risk'].mean()
            for period, risk in inc_monthly.items():
                if pd.notna(risk):
                    combined_data.append({"month": str(period), "avg_risk": float(risk)})
    
    # Process hazards
    if haz_df is not None and not haz_df.empty:
        if location is not None and location != "":
            loc_col = _resolve_column(haz_df, ["location"])
            if loc_col:
                haz_df = haz_df[haz_df[loc_col].astype(str).str.contains(str(location), case=False, na=False)]
        
        date_col = _resolve_column(haz_df, ["occurrence_date", "date"])
        risk_col = _resolve_column(haz_df, ["risk_score"])
        
        if date_col and risk_col:
            haz_df['month'] = pd.to_datetime(haz_df[date_col], errors="coerce").dt.to_period('M')
            haz_df['risk'] = pd.to_numeric(haz_df[risk_col], errors="coerce")
            
            haz_monthly = haz_df.groupby('month')['risk'].mean()
            for period, risk in haz_monthly.items():
                if pd.notna(risk):
                    # Merge with incidents if same month exists
                    existing = next((d for d in combined_data if d["month"] == str(period)), None)
                    if existing:
                        existing["avg_risk"] = (existing["avg_risk"] + float(risk)) / 2
                    else:
                        combined_data.append({"month": str(period), "avg_risk": float(risk)})
    
    # Sort by month
    combined_data = sorted(combined_data, key=lambda x: x["month"])
    
    # Forecast
    if combined_data:
        risk_series = pd.Series([d["avg_risk"] for d in combined_data])
        forecast_values = _simple_forecast(risk_series, months_ahead)
        
        last_month = pd.Period(combined_data[-1]["month"])
        forecast_data = []
        for i, val in enumerate(forecast_values, start=1):
            future_period = last_month + i
            forecast_data.append({
                "month": str(future_period),
                "predicted_avg_risk": val,
            })
    else:
        forecast_data = []
    
    # Determine trend
    if len(combined_data) >= 2:
        recent_avg = np.mean([d["avg_risk"] for d in combined_data[-3:]])
        older_avg = np.mean([d["avg_risk"] for d in combined_data[:-3]]) if len(combined_data) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            trend = "Increasing"
            trend_color = "#f44336"
        elif recent_avg < older_avg * 0.9:
            trend = "Decreasing"
            trend_color = "#4caf50"
        else:
            trend = "Stable"
            trend_color = "#ffc107"
    else:
        trend = "Insufficient Data"
        trend_color = "#9e9e9e"
    
    return JSONResponse(content=to_native_json({
        "historical": combined_data,
        "forecast": forecast_data,
        "trend": trend,
        "trend_color": trend_color,
    }))


# ======================= LEADING VS LAGGING INDICATORS =======================

@router.get("/leading-vs-lagging")
async def leading_vs_lagging_indicators(
    start_date: Optional[str] = Query(None, description="Filter start date", example="2023-01-01"),
    end_date: Optional[str] = Query(None, description="Filter end date", example="2024-12-31"),
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi"),
):
    """
    Leading vs Lagging Indicators Comparison.
    
    Leading Indicators (Proactive):
    - Hazards identified
    - Audits completed
    - Inspections performed
    - Near-miss reports
    - Training hours
    
    Lagging Indicators (Reactive):
    - Incidents occurred
    - Lost time incidents
    - Medical treatment cases
    - Property damage
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    aud_df = get_audit_df()
    insp_df = get_inspection_df()
    
    # Apply date filters
    def filter_by_date(df, start_date, end_date):
        if df is None or df.empty:
            return df
        date_col = _resolve_column(df, ["occurrence_date", "date", "start_date", "reported_date"])
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if start_date:
                mask = dates >= pd.to_datetime(start_date)
                df = df.loc[mask]
            if end_date:
                mask = dates <= pd.to_datetime(end_date)
                df = df.loc[mask]
        return df
    
    inc_df = filter_by_date(inc_df, start_date, end_date)
    haz_df = filter_by_date(haz_df, start_date, end_date)
    aud_df = filter_by_date(aud_df, start_date, end_date)
    insp_df = filter_by_date(insp_df, start_date, end_date)
    
    # Leading indicators
    leading = {
        "hazards_identified": len(haz_df) if haz_df is not None else 0,
        "audits_completed": 0,
        "inspections_performed": len(insp_df) if insp_df is not None else 0,
        "near_miss_reports": 0,
    }
    
    if aud_df is not None and not aud_df.empty:
        status_col = _resolve_column(aud_df, ["audit_status", "status"])
        if status_col:
            leading["audits_completed"] = aud_df[status_col].astype(str).str.contains(
                "closed|complete", case=False, na=False
            ).sum()
    
    if inc_df is not None and not inc_df.empty:
        type_col = _resolve_column(inc_df, ["incident_type", "category"])
        if type_col:
            leading["near_miss_reports"] = inc_df[type_col].astype(str).str.contains(
                "near miss|near-miss", case=False, na=False
            ).sum()
    
    # Lagging indicators
    lagging = {
        "total_incidents": len(inc_df) if inc_df is not None else 0,
        "lost_time_incidents": 0,
        "medical_treatment_cases": 0,
        "serious_incidents": 0,
    }
    
    if inc_df is not None and not inc_df.empty:
        sev_col = _resolve_column(inc_df, ["severity_score", "risk_score"])
        if sev_col:
            severity = pd.to_numeric(inc_df[sev_col], errors="coerce")
            lagging["lost_time_incidents"] = (severity >= 3).sum()
            lagging["medical_treatment_cases"] = (severity == 2).sum()
            lagging["serious_incidents"] = (severity >= 4).sum()
    
    # Calculate ratio
    total_leading = sum(leading.values())
    total_lagging = sum(lagging.values())
    
    if total_lagging > 0:
        ratio = round(total_leading / total_lagging, 2)
        ratio_text = f"{ratio}:1"
    else:
        ratio = 0
        ratio_text = "N/A"
    
    # Assessment
    if ratio >= 10:
        assessment = "Excellent - Proactive safety culture"
        color = "#4caf50"
    elif ratio >= 5:
        assessment = "Good - Balanced approach"
        color = "#8bc34a"
    elif ratio >= 2:
        assessment = "Fair - Room for improvement"
        color = "#ffc107"
    else:
        assessment = "Poor - Too reactive"
        color = "#f44336"
    
    return JSONResponse(content=to_native_json({
        "leading_indicators": leading,
        "lagging_indicators": lagging,
        "total_leading": int(total_leading),
        "total_lagging": int(total_lagging),
        "ratio": ratio,
        "ratio_text": ratio_text,
        "assessment": assessment,
        "color": color,
        "recommendation": "Industry best practice: Leading indicators should be 5-10x lagging indicators",
    }))


# ======================= OBSERVATION TO INCIDENT LAG TIME =======================

@router.get("/observation-lag-time")
async def observation_lag_time(
    location: Optional[str] = Query(None, description="Filter by location", example="Karachi"),
    department: Optional[str] = Query(None, description="Filter by department", example="PVC"),
):
    """
    Observation-to-Incident Lag Time Analysis.
    
    Measures the time between when hazards/observations are identified
    and when related incidents occur. Helps assess intervention effectiveness.
    """
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    
    if inc_df is None or inc_df.empty or haz_df is None or haz_df.empty:
        return JSONResponse(content=to_native_json({
            "average_lag_days": 0,
            "lag_distribution": [],
            "message": "Insufficient data for analysis"
        }))
    
    # Get date columns
    inc_date_col = _resolve_column(inc_df, ["occurrence_date", "date"])
    haz_date_col = _resolve_column(haz_df, ["occurrence_date", "date", "reported_date"])
    
    if not inc_date_col or not haz_date_col:
        return JSONResponse(content={"message": "Date columns not found"})
    
    inc_dates = pd.to_datetime(inc_df[inc_date_col], errors="coerce")
    haz_dates = pd.to_datetime(haz_df[haz_date_col], errors="coerce")
    
    # Simple heuristic: for each incident, find nearest prior hazard
    lag_times = []
    
    for inc_date in inc_dates.dropna():
        # Find hazards reported before this incident
        prior_hazards = haz_dates[haz_dates < inc_date]
        if not prior_hazards.empty:
            nearest_hazard = prior_hazards.max()
            lag_days = (inc_date - nearest_hazard).days
            if 0 <= lag_days <= 365:  # Within 1 year
                lag_times.append(lag_days)
    
    if not lag_times:
        return JSONResponse(content=to_native_json({
            "average_lag_days": 0,
            "median_lag_days": 0,
            "lag_distribution": [],
            "message": "No clear observation-to-incident patterns found"
        }))
    
    avg_lag = np.mean(lag_times)
    median_lag = np.median(lag_times)
    
    # Distribution buckets
    buckets = {
        "0-7 days": 0,
        "8-30 days": 0,
        "31-90 days": 0,
        "91-180 days": 0,
        "180+ days": 0,
    }
    
    for lag in lag_times:
        if lag <= 7:
            buckets["0-7 days"] += 1
        elif lag <= 30:
            buckets["8-30 days"] += 1
        elif lag <= 90:
            buckets["31-90 days"] += 1
        elif lag <= 180:
            buckets["91-180 days"] += 1
        else:
            buckets["180+ days"] += 1
    
    distribution = [{"range": k, "count": v} for k, v in buckets.items()]
    
    return JSONResponse(content=to_native_json({
        "average_lag_days": round(avg_lag, 1),
        "median_lag_days": round(median_lag, 1),
        "lag_distribution": distribution,
        "total_correlations": len(lag_times),
        "interpretation": f"Average {round(avg_lag, 1)} days between hazard identification and related incident",
    }))
