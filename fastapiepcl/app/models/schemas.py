from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ---------- Generic data structures ----------
class DataFramePayload(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list, description="List of row dicts")


class SheetPreview(BaseModel):
    name: str
    n_rows: int
    n_cols: int
    columns: List[str]
    sample: List[Dict[str, Any]] = Field(default_factory=list)


class WorkbookSummaryResponse(BaseModel):
    sheets: List[SheetPreview] = Field(default_factory=list)


# ---------- Schema inference ----------
class InferredSchema(BaseModel):
    date_col: Optional[str] = None
    status_col: Optional[str] = None
    title_col: Optional[str] = None
    category_col: Optional[str] = None
    dept_col: Optional[str] = None
    loc_col: Optional[str] = None
    id_col: Optional[str] = None
    consequence_col: Optional[str] = None
    severity_col: Optional[str] = None
    risk_col: Optional[str] = None
    cost_col: Optional[str] = None
    manhours_col: Optional[str] = None
    reporting_delay_col: Optional[str] = None
    resolution_time_col: Optional[str] = None
    flags: List[str] = Field(default_factory=list)


class InferSchemaRequest(BaseModel):
    sheet_name: str
    data: DataFramePayload


# ---------- Wordclouds ----------
class DepartmentWordcloudRequest(BaseModel):
    incident: Optional[DataFramePayload] = None
    hazard: Optional[DataFramePayload] = None
    top_n: int = 50
    min_count: int = 1
    extra_stopwords: Optional[List[str]] = None


class WordItem(BaseModel):
    text: str
    value: int
    color: Optional[str] = None
    type: Optional[str] = None


class DepartmentWordcloudResponse(BaseModel):
    incident: List[WordItem] = Field(default_factory=list)
    hazard: List[WordItem] = Field(default_factory=list)
    html_incident: Optional[str] = None
    html_hazard: Optional[str] = None


# ---------- Maps ----------
class CombinedMapRequest(BaseModel):
    incident: Optional[DataFramePayload] = None
    hazard: Optional[DataFramePayload] = None
    location_coords: Optional[Dict[str, Dict[str, float]]] = None


class CombinedMapResponse(BaseModel):
    html: str


# ---------- General analytics (Plotly) ----------
class PlotlyFigureResponse(BaseModel):
    figure: Dict[str, Any]


class SingleDFRequest(BaseModel):
    data: DataFramePayload


class AuditInspectionRequest(BaseModel):
    audit: DataFramePayload
    inspection: DataFramePayload


class HSEScorecardRequest(BaseModel):
    incident: Optional[DataFramePayload] = None
    hazard: Optional[DataFramePayload] = None
    audit: Optional[DataFramePayload] = None
    inspection: Optional[DataFramePayload] = None


# Requests for additional analytics
class FacilityHeatmapRequest(BaseModel):
    incident: Optional[DataFramePayload] = None
    hazard: Optional[DataFramePayload] = None


class SingleMapRequest(BaseModel):
    data: DataFramePayload
    dataset: str = "incident"  # "incident" or "hazard"
    location_coords: Optional[Dict[str, Dict[str, float]]] = None


class SingleDFWithTypeRequest(BaseModel):
    data: DataFramePayload
    event_type: str = "Incidents"

class ConversionRequest(BaseModel):
    incident: DataFramePayload
    hazard: DataFramePayload
    relationships: Optional[DataFramePayload] = None


# ---------- Agent (LLM data assistant) ----------
class AgentRunResponse(BaseModel):
    code: str
    stdout: str
    error: str
    result_preview: List[Dict[str, Any]] = Field(default_factory=list)
    figure: Optional[Dict[str, Any]] = None
    mpl_png_base64: Optional[str] = None
    analysis: str
    attempts: int = Field(default=1, description="Number of self-correction attempts made")
    verification_score: float = Field(default=0.0, description="Confidence score (0.0-1.0) from result verification")
    correction_log: List[str] = Field(default_factory=list, description="Detailed log of self-correction process")


# ---------- Insights from charts ----------
class ChartInsightsRequest(BaseModel):
    figure: Dict[str, Any] = Field(description="Plotly figure JSON (data + layout)")
    title: Optional[str] = None
    context: Optional[str] = Field(default=None, description="Optional natural language context about the chart")


class ChartInsightsResponse(BaseModel):
    insights_md: str = Field(description="Layman-friendly insights in Markdown format")


# ---------- Data-driven insights (Option B) ----------
class DataInsightsRequest(BaseModel):
    title: Optional[str] = None
    data: DataFramePayload
    time_col: Optional[str] = Field(default=None, description="Column name containing date/time")
    category_col: Optional[str] = Field(default=None, description="Column used for top contributors (e.g., department)")
    metrics: Optional[List[str]] = Field(default=None, description="Subset of metrics to analyze: count, severity, risk, cost, manhours")
    value_cols: Optional[Dict[str, str]] = Field(default=None, description="Mapping of metric name to column name, e.g., { 'severity': 'severity_score', 'risk': 'risk_score' }")
    top_n: int = Field(default=5, ge=1, le=50)
    refine_with_llm: bool = Field(default=True, description="If true and LLM is available, rewrite insights in layman-friendly style")
    model: Optional[str] = Field(default="gpt-4o", description="LLM model to use if refinement is enabled")


# ---------- Flexible Filtering for Analytics ----------
class AnalyticsFilters(BaseModel):
    """Flexible filters for analytics endpoints to enable dynamic data filtering."""
    dataset: str = Field(default="incident", description="Dataset to use: 'incident' or 'hazard'")
    start_date: Optional[str] = Field(default=None, description="Start date filter (ISO format: YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date filter (ISO format: YYYY-MM-DD)")
    departments: Optional[List[str]] = Field(default=None, description="Filter by specific departments")
    locations: Optional[List[str]] = Field(default=None, description="Filter by specific locations")
    sublocations: Optional[List[str]] = Field(default=None, description="Filter by specific sublocations")
    min_severity: Optional[float] = Field(default=None, ge=0, le=5, description="Minimum severity score (0-5)")
    max_severity: Optional[float] = Field(default=None, ge=0, le=5, description="Maximum severity score (0-5)")
    min_risk: Optional[float] = Field(default=None, ge=0, le=5, description="Minimum risk score (0-5)")
    max_risk: Optional[float] = Field(default=None, ge=0, le=5, description="Maximum risk score (0-5)")
    statuses: Optional[List[str]] = Field(default=None, description="Filter by status values")
    incident_types: Optional[List[str]] = Field(default=None, description="Filter by incident types")
    violation_types: Optional[List[str]] = Field(default=None, description="Filter by violation types (for hazards)")


# ---------- Filter Options (for frontend dropdowns) ----------
class DateRangeInfo(BaseModel):
    """Date range information from dataset."""
    min_date: Optional[str] = Field(None, description="Earliest date in dataset (YYYY-MM-DD)")
    max_date: Optional[str] = Field(None, description="Latest date in dataset (YYYY-MM-DD)")
    total_records: int = Field(0, description="Total number of records with valid dates")


class FilterOption(BaseModel):
    """Single filter option with count."""
    value: str = Field(description="The actual value")
    label: str = Field(description="Display label for UI")
    count: int = Field(description="Number of records with this value")


class FilterOptionsResponse(BaseModel):
    """Available filter options for a specific dataset."""
    dataset: str = Field(description="Dataset name (incident or hazard)")
    date_range: DateRangeInfo = Field(description="Date range information")
    departments: List[FilterOption] = Field(default_factory=list, description="Available departments")
    locations: List[FilterOption] = Field(default_factory=list, description="Available locations")
    sublocations: List[FilterOption] = Field(default_factory=list, description="Available sublocations")
    statuses: List[FilterOption] = Field(default_factory=list, description="Available statuses")
    incident_types: List[FilterOption] = Field(default_factory=list, description="Available incident types")
    violation_types: List[FilterOption] = Field(default_factory=list, description="Available violation types (hazards only)")
    severity_range: Dict[str, float] = Field(default_factory=dict, description="Severity score range (min, max, avg)")
    risk_range: Dict[str, float] = Field(default_factory=dict, description="Risk score range (min, max, avg)")
    total_records: int = Field(0, description="Total records in dataset")


class CombinedFilterOptionsResponse(BaseModel):
    """Combined filter options from both incident and hazard datasets."""
    incident: FilterOptionsResponse = Field(description="Filter options for incidents")
    hazard: FilterOptionsResponse = Field(description="Filter options for hazards")
    last_updated: str = Field(description="Timestamp when options were generated")


# ---------- Enhanced Chart Tooltips ----------
class RecentItem(BaseModel):
    """Recent incident/hazard item for tooltip."""
    title: str = Field(description="Title or description of the item")
    department: str = Field(description="Department where it occurred")
    date: str = Field(description="Date of occurrence (YYYY-MM-DD)")
    severity: Optional[float] = Field(None, description="Severity score")


class CountItem(BaseModel):
    """Count item for departments or types."""
    name: str = Field(description="Name of the department/type")
    count: int = Field(description="Count of items")


class ScoreStats(BaseModel):
    """Statistics for severity or risk scores."""
    avg: float = Field(description="Average score")
    max: float = Field(description="Maximum score")
    min: float = Field(description="Minimum score")


class MonthDetailedData(BaseModel):
    """Detailed breakdown for a specific month."""
    month: str = Field(description="Month label (YYYY-MM)")
    total_count: int = Field(description="Total count for the month")
    departments: List[CountItem] = Field(default_factory=list, description="Top departments")
    types: List[CountItem] = Field(default_factory=list, description="Top incident/violation types")
    severity: Optional[ScoreStats] = Field(None, description="Severity statistics")
    risk: Optional[ScoreStats] = Field(None, description="Risk statistics")
    recent_items: List[RecentItem] = Field(default_factory=list, description="Recent items (up to 5)")


class ChartSeries(BaseModel):
    """Chart series data."""
    name: str = Field(description="Series name")
    data: List[int] = Field(description="Series data points")


class DetailedTrendResponse(BaseModel):
    """Response with detailed breakdown for trend charts."""
    labels: List[str] = Field(description="Month labels")
    series: List[ChartSeries] = Field(description="Chart series")
    details: List[MonthDetailedData] = Field(default_factory=list, description="Detailed breakdown per month")
