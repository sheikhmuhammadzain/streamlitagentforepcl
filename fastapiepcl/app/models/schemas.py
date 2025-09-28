from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
