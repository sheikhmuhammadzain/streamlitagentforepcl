from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        lc = str(col).lower()
        if (
            ("date" in lc)
            or ("time" in lc)
            or lc.startswith("entered_")
            or lc in ["start_date", "entered_closed"]
        ):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                # keep original on failure
                pass
    return df


def read_excel_to_sheets(content: bytes) -> Dict[str, pd.DataFrame]:
    """Read an Excel workbook (bytes) and return dict of sheet_name -> DataFrame.
    Datetime-like columns will be coerced using best-effort rules.
    """
    from io import BytesIO
    xls = pd.ExcelFile(BytesIO(content))
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        sheets[sheet] = _coerce_datetime_columns(df)
    return sheets


def df_to_payload(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None:
        return []
    # Convert timestamps to ISO strings for JSON safety
    safe = df.copy()
    for col in safe.columns:
        if np.issubdtype(safe[col].dtype, np.datetime64):
            safe[col] = pd.to_datetime(safe[col], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return safe.to_dict(orient="records")


def payload_to_df(records: Optional[List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
    if not records:
        return None
    try:
        df = pd.DataFrame.from_records(records)
    except Exception:
        return None
    return _coerce_datetime_columns(df)

def summarize_sheet(name: str, df: pd.DataFrame, sample_size: int = 10) -> Tuple[str, int, int, List[str], List[Dict[str, Any]]]:
    sample = df.head(sample_size).to_dict(orient="records")
    return name, int(len(df)), int(len(df.columns)), [str(c) for c in df.columns], sample


# -------------- Auto-loading default workbook placed in app/ -----------------
# This file resides in app/services/, but the Excel is placed in app/
DEFAULT_EXCEL_PATH = Path(__file__).resolve().parent.parent / "EPCL_VEHS_Data_Processed.xlsx"


@lru_cache(maxsize=1)
def load_default_sheets() -> Dict[str, pd.DataFrame]:
    """Load and cache sheets from the default Excel file in the app folder.
    Returns an empty dict if the file is not present or unreadable.
    """
    try:
        if not DEFAULT_EXCEL_PATH.exists():
            return {}
        content = DEFAULT_EXCEL_PATH.read_bytes()
        return read_excel_to_sheets(content)
    except Exception:
        return {}


def _indicator_columns() -> Dict[str, List[str]]:
    return {
        "incident": [
            "occurrence_date",
            "severity_score",
            "risk_score",
            "estimated_cost_impact",
            "estimated_manhours_impact",
            "department",
        ],
        "hazard": [
            "violation_type_hazard_id",
            "worst_case_consequence_potential_hazard_id",
            "department",
            "reporting_delay_days",
        ],
        "audit": [
            "audit_status",
            "start_date",
            "audit_title",
            "audit_id",
            "audit_type_epcl",
        ],
        "inspection": [
            "audit_status",
            "start_date",
            "checklist_category",
            "checklist category",
            "finding",
        ],
    }


def _score_sheet_for_dataset(df: pd.DataFrame, indicators: List[str]) -> int:
    cols = {str(c).lower() for c in df.columns}
    score = 0
    for ind in indicators:
        if ind.lower() in cols:
            score += 1
    return score


def _choose_by_name_token(sheets: Dict[str, pd.DataFrame], token: str) -> Optional[pd.DataFrame]:
    """Choose a sheet by name token with smarter matching.
    Preference order:
      1) Exact match on token (case-insensitive), or common variants like 'Total <token>'
      2) Name contains token but avoids conflicting dataset tokens (e.g., avoid 'audit' when selecting 'inspection')
      3) First name that contains the token
    """
    token_l = token.lower().strip()
    avoid = {"audit": ["inspection", "finding", "findings"], "inspection": ["audit", "finding", "findings"]}.get(token_l, [])

    # 1) Exact/clear variants
    for name, df in sheets.items():
        ln = str(name).strip().lower()
        if (
            ln == token_l
            or ln == f"total {token_l}"
            or ln == f"{token_l} total"
            or ln.startswith(f"{token_l} ")
            or ln.endswith(f" {token_l}")
        ) and (not any(a in ln for a in avoid)):
            return df

    # 2) Contains token, prefer without avoid tokens
    candidates: List[Tuple[str, pd.DataFrame]] = []
    for name, df in sheets.items():
        ln = str(name).strip().lower()
        if token_l in ln:
            candidates.append((ln, df))
    if candidates:
        for ln, df in candidates:
            if not any(a in ln for a in avoid):
                return df
        # fallback to first candidate
        return candidates[0][1]
    return None


def get_default_dataframes() -> Dict[str, Optional[pd.DataFrame]]:
    """Identify and return best-matching DataFrames for each dataset type.
    Keys: incident, hazard, audit, inspection
    Values may be None if not found.
    """
    sheets = load_default_sheets()
    if not sheets:
        return {"incident": None, "hazard": None, "audit": None, "inspection": None}

    indicators = _indicator_columns()
    best: Dict[str, Tuple[int, Optional[pd.DataFrame]]] = {k: (0, None) for k in indicators.keys()}

    # Prefer name-token matches up-front so scoring won't override clear intent
    for key in list(indicators.keys()):
        cand = _choose_by_name_token(sheets, key)
        if cand is not None:
            # Use a very high score to lock this selection unless a better explicit rule is added later
            best[key] = (10_000, cand)

    for name, df in sheets.items():
        for key, cols in indicators.items():
            s = _score_sheet_for_dataset(df, cols)
            if s > best[key][0]:
                best[key] = (s, df)

    selected: Dict[str, Optional[pd.DataFrame]] = {k: v for k, (s, v) in best.items()}

    # Secondary fallback by sheet name token if column heuristics fail
    if selected.get("incident") is None:
        cand = _choose_by_name_token(sheets, "incident")
        selected["incident"] = cand or selected.get("incident")
    if selected.get("hazard") is None:
        cand = _choose_by_name_token(sheets, "hazard")
        selected["hazard"] = cand or selected.get("hazard")
    if selected.get("audit") is None:
        cand = _choose_by_name_token(sheets, "audit")
        selected["audit"] = cand or selected.get("audit")
    if selected.get("inspection") is None:
        cand = _choose_by_name_token(sheets, "inspection")
        selected["inspection"] = cand or selected.get("inspection")

    # Final fallback: largest sheet by rows for any still-missing dataset
    if sheets:
        sizes = [(name, len(df) if isinstance(df, pd.DataFrame) else 0) for name, df in sheets.items()]
        sizes.sort(key=lambda x: x[1], reverse=True)
        largest_df = sheets[sizes[0][0]] if sizes else None
        for key in list(selected.keys()):
            if selected[key] is None:
                selected[key] = largest_df

    return selected


def get_dataset_selection_names() -> Dict[str, Optional[str]]:
    """Return the sheet names selected for each dataset based on current cache/heuristics."""
    sheets = load_default_sheets()
    if not sheets:
        return {"incident": None, "hazard": None, "audit": None, "inspection": None}
    selected = get_default_dataframes()
    name_map: Dict[str, Optional[str]] = {k: None for k in selected.keys()}
    for key, df in selected.items():
        if df is None:
            name_map[key] = None
            continue
        # find the sheet name for the df object
        for name, sdf in sheets.items():
            if sdf is df:
                name_map[key] = name
                break
    return name_map


def get_incident_df() -> Optional[pd.DataFrame]:
    return get_default_dataframes().get("incident")


def get_hazard_df() -> Optional[pd.DataFrame]:
    return get_default_dataframes().get("hazard")


def get_audit_df() -> Optional[pd.DataFrame]:
    return get_default_dataframes().get("audit")


def get_inspection_df() -> Optional[pd.DataFrame]:
    return get_default_dataframes().get("inspection")
