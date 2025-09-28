from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_schema(df: Optional[pd.DataFrame], sheet_name: str) -> Dict[str, Optional[str]]:
    """Infer columns of interest following the Streamlit logic for consistency."""
    if df is None or df.empty:
        return {
            'date_col': None, 'status_col': None, 'title_col': None, 'category_col': None,
            'dept_col': None, 'loc_col': None, 'id_col': None, 'consequence_col': None,
            'severity_col': None, 'risk_col': None, 'cost_col': None, 'manhours_col': None,
            'reporting_delay_col': None, 'resolution_time_col': None, 'flags': []
        }

    s = str(sheet_name).lower()
    if 'incident' in s:
        date_candidates = ['occurrence_date', 'reported_date', 'entered_date', 'completion_date']
        status_candidates = ['status']
        title_candidates = ['title']
        category_candidates = ['category', 'incident_type']
        consequence_candidates = ['worst_case_consequence_incident', 'actual_consequence_incident', 'relevant_consequence_incident']
    elif 'hazard' in s:
        date_candidates = ['occurrence_date', 'reported_date', 'entered_date', 'entered_closed']
        status_candidates = ['status']
        title_candidates = ['title']
        category_candidates = ['category', 'violation_type_hazard_id']
        consequence_candidates = ['worst_case_consequence_potential_hazard_id', 'relevant_consequence_hazard_id']
    elif 'audit findings' in s:
        date_candidates = ['start_date', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category']
        consequence_candidates = ['worst_case_consequence']
    elif 'audit' in s:
        date_candidates = ['start_date', 'entered_in_progress', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category', 'auditing_body']
        consequence_candidates = ['worst_case_consequence']
    elif 'inspection' in s:
        date_candidates = ['start_date', 'entered_in_progress', 'entered_review', 'entered_closed']
        status_candidates = ['audit_status']
        title_candidates = ['audit_title']
        category_candidates = ['audit_category']
        consequence_candidates = ['worst_case_consequence']
    else:
        date_candidates = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        status_candidates = ['status', 'audit_status']
        title_candidates = ['title', 'audit_title']
        category_candidates = ['category', 'audit_category']
        consequence_candidates = []

    date_col = _first_present(df, date_candidates) if date_candidates else None
    status_col = _first_present(df, status_candidates)
    title_col = _first_present(df, title_candidates)
    category_col = _first_present(df, category_candidates)

    dept_col = _first_present(df, ['department', 'sub_department'])
    loc_col = _first_present(df, ['location', 'sublocation', 'location.1'])
    id_col = _first_present(df, ['incident_id', 'audit_id'])
    consequence_col = _first_present(df, consequence_candidates) if consequence_candidates else None

    severity_col = _first_present(df, ['severity_score'])
    risk_col = _first_present(df, ['risk_score', 'department_avg_risk'])
    cost_col = _first_present(df, ['estimated_cost_impact'])
    manhours_col = _first_present(df, ['estimated_manhours_impact'])
    reporting_delay_col = _first_present(df, ['reporting_delay_days'])
    resolution_time_col = _first_present(df, ['resolution_time_days'])

    flags = [c for c in ['root_cause_is_missing', 'corrective_actions_is_missing'] if c in df.columns]

    return {
        'date_col': date_col,
        'status_col': status_col,
        'title_col': title_col,
        'category_col': category_col,
        'dept_col': dept_col,
        'loc_col': loc_col,
        'id_col': id_col,
        'consequence_col': consequence_col,
        'severity_col': severity_col,
        'risk_col': risk_col,
        'cost_col': cost_col,
        'manhours_col': manhours_col,
        'reporting_delay_col': reporting_delay_col,
        'resolution_time_col': resolution_time_col,
        'flags': flags,
    }

