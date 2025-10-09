from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..analytics.wordclouds import (
    get_incident_hazard_department_words,
    create_modern_word_cloud_html,
)
from ..analytics.maps import add_coordinates_to_df, build_combined_map_html


def build_department_wordclouds(
    incident_df: Optional[pd.DataFrame],
    hazard_df: Optional[pd.DataFrame],
    *,
    top_n: int = 50,
    min_count: int = 1,
    extra_stopwords: Optional[set] = None,
) -> Tuple[Dict[str, List[Dict[str, object]]], Optional[str], Optional[str]]:
    words = get_incident_hazard_department_words(
        incident_df,
        hazard_df,
        top_n=top_n,
        min_count=min_count,
        extra_stopwords=extra_stopwords,
    )
    html_incident = create_modern_word_cloud_html(words.get("incident", []), title="Incidents by Department") if words else None
    html_hazard = create_modern_word_cloud_html(words.get("hazard", []), title="Hazards by Department") if words else None
    return words, html_incident, html_hazard


def build_combined_map(
    incident_df: Optional[pd.DataFrame],
    hazard_df: Optional[pd.DataFrame],
    location_coords: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    inc = incident_df.copy() if isinstance(incident_df, pd.DataFrame) else pd.DataFrame()
    haz = hazard_df.copy() if isinstance(hazard_df, pd.DataFrame) else pd.DataFrame()

    if location_coords:
        if not inc.empty:
            inc = add_coordinates_to_df(inc, location_coords)
        if not haz.empty:
            haz = add_coordinates_to_df(haz, location_coords)

    return build_combined_map_html(inc, haz)

