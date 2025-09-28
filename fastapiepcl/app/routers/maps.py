from fastapi import APIRouter, Query

from ..models.schemas import CombinedMapResponse
from ..services.analytics_general import build_combined_map
from ..services.excel import get_incident_df, get_hazard_df


router = APIRouter(prefix="/maps", tags=["maps"])


@router.get("/combined", response_model=CombinedMapResponse)
async def combined_map():
    inc_df = get_incident_df()
    haz_df = get_hazard_df()
    html = build_combined_map(inc_df, haz_df, None)
    return CombinedMapResponse(html=html)


@router.get("/single", response_model=CombinedMapResponse)
async def single_map(dataset: str = Query("incident", description="Dataset to use: incident or hazard")):
    if (dataset or "incident").lower() == "incident":
        html = build_combined_map(get_incident_df(), None, None)
    else:
        html = build_combined_map(None, get_hazard_df(), None)
    return CombinedMapResponse(html=html)

