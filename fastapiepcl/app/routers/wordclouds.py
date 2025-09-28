from fastapi import APIRouter, Query

from ..models.schemas import (
    DepartmentWordcloudResponse,
    WordItem,
)
from ..services.analytics_general import build_department_wordclouds
from ..services.excel import get_incident_df, get_hazard_df


router = APIRouter(prefix="/wordclouds", tags=["wordclouds"])


@router.get("/departments", response_model=DepartmentWordcloudResponse)
async def department_wordclouds(
    top_n: int = Query(50, ge=1, le=500, description="Top N words per dataset"),
    min_count: int = Query(1, ge=1, description="Minimum frequency to include a word"),
    extra_stopwords: str | None = Query(None, description="Comma-separated extra stopwords"),
):
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    extra = set([w.strip() for w in extra_stopwords.split(",") if w.strip()]) if extra_stopwords else None
    words, html_incident, html_hazard = build_department_wordclouds(
        inc_df,
        haz_df,
        top_n=top_n,
        min_count=min_count,
        extra_stopwords=extra,
    )
    inc_items = [WordItem(**w) for w in words.get("incident", [])]
    haz_items = [WordItem(**w) for w in words.get("hazard", [])]
    return DepartmentWordcloudResponse(
        incident=inc_items,
        hazard=haz_items,
        html_incident=html_incident,
        html_hazard=html_hazard,
    )

