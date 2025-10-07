from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..services.json_utils import to_native_json

from ..services.excel import (
    get_incident_df,
    get_hazard_df,
    get_audit_df,
)

router = APIRouter(tags=["data"])


# ---------- Helpers ----------

def _coerce_date_str(series: Optional[pd.Series]) -> List[str]:
    if series is None:
        return []
    s = pd.to_datetime(series, errors="coerce")
    return [d.date().isoformat() if pd.notna(d) else "" for d in s]


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _ensure_str_list(series: Optional[pd.Series]) -> List[str]:
    if series is None:
        return []
    return ["" if pd.isna(v) else str(v) for v in series]


def _ensure_int_list(series: Optional[pd.Series], default: int = 0) -> List[int]:
    if series is None:
        return []
    s = pd.to_numeric(series, errors="coerce").fillna(default)
    return [int(v) for v in s]


def _score_to_bucket(val: Any, *, kind: str = "severity") -> str:
    try:
        x = float(val)
    except Exception:
        # if a string like "High" is present, normalize capitalization
        if isinstance(val, str) and val.strip():
            v = val.strip().lower()
            if v.startswith("low"):
                return "Low"
            if v.startswith("med"):
                return "Medium"
            if v.startswith("high") and not v.startswith("highl"):  # avoid 'highlight'
                return "High"
            if v.startswith("crit"):
                return "Critical"
        return "Medium"
    # numeric bucketing (tune thresholds as needed)
    if x <= 1.5:
        return "Low"
    if x <= 2.5:
        return "Medium"
    if x <= 3.5:
        return "High"
    return "Critical"


# ---------- Endpoints ----------

@router.get("/incidents")
async def list_incidents() -> List[Dict[str, Any]]:
    df = get_incident_df()
    if df is None or df.empty:
        return []
    # Resolve columns with fallbacks
    id_s = _first_existing(df, [
        "incident_id", "id", "Incident ID", "INCIDENT_ID", "IncidentID",
    ])
    if id_s is None:
        id_s = pd.Series([f"INC-{i+1:03d}" for i in range(len(df))])
    title_s = _first_existing(df, [
        "title", "incident_title", "description", "short_description", "remarks",
    ])
    dept_s = _first_existing(df, [
        "department", "dept", "Department",
    ])
    severity_s = _first_existing(df, [
        "severity", "Severity", "severity_level",
    ])
    severity_score_s = _first_existing(df, [
        "severity_score", "Severity Score", "severity_numeric",
    ])
    status_s = _first_existing(df, [
        "status", "incident_status", "workflow_status", "Status",
    ])
    date_s = _first_existing(df, [
        "occurrence_date", "date", "entered_date", "reported_date", "created_date",
    ])
    location_s = _first_existing(df, [
        "location.1", "sublocation", "location", "Location",
    ])

    # Build records
    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    depts = _ensure_str_list(dept_s)
    # severity prefer string col, else map from numeric
    if severity_s is not None:
        severities = [_score_to_bucket(v, kind="severity") for v in severity_s]
    else:
        fallback = severity_score_s if severity_score_s is not None else pd.Series([None] * len(df))
        severities = [_score_to_bucket(v, kind="severity") for v in fallback]
    statuses = _ensure_str_list(status_s)
    dates = _coerce_date_str(date_s)
    locations = _ensure_str_list(location_s)

    out: List[Dict[str, Any]] = []
    n = len(df)
    for i in range(n):
        out.append({
            "id": ids[i] if i < len(ids) else f"INC-{i+1:03d}",
            "title": titles[i] if i < len(titles) and titles[i] else f"Incident {ids[i] if i < len(ids) else i+1}",
            "department": depts[i] if i < len(depts) else "",
            "severity": severities[i] if i < len(severities) else "Medium",
            "status": statuses[i] if i < len(statuses) and statuses[i] else "Open",
            "date": dates[i] if i < len(dates) else "",
            "location": locations[i] if i < len(locations) else "",
        })
    return out


@router.get("/hazards")
async def list_hazards() -> List[Dict[str, Any]]:
    df = get_hazard_df()
    if df is None or df.empty:
        return []
    id_s = _first_existing(df, ["hazard_id", "id", "HAZARD_ID"])
    if id_s is None:
        id_s = pd.Series([f"HAZ-{i+1:03d}" for i in range(len(df))])
    title_s = _first_existing(df, ["title", "hazard_title", "description"]) 
    dept_s = _first_existing(df, ["department", "dept"]) 
    risk_level_s = _first_existing(df, ["risk_level", "Risk Level"]) 
    risk_score_s = _first_existing(df, ["risk_score", "Risk Score"]) 
    status_s = _first_existing(df, ["status", "hazard_status", "workflow_status"]) 
    date_s = _first_existing(df, ["occurrence_date", "date", "entered_date"]) 
    location_s = _first_existing(df, ["location.1", "sublocation", "location"]) 
    violation_type_s = _first_existing(df, ["violation_type_hazard_id", "violation_type"]) 

    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    depts = _ensure_str_list(dept_s)
    if risk_level_s is not None:
        risk_levels = [_score_to_bucket(v, kind="risk") for v in risk_level_s]
    else:
        fallback = risk_score_s if risk_score_s is not None else pd.Series([None] * len(df))
        risk_levels = [_score_to_bucket(v, kind="risk") for v in fallback]
    statuses = _ensure_str_list(status_s)
    dates = _coerce_date_str(date_s)
    locations = _ensure_str_list(location_s)
    violations = _ensure_str_list(violation_type_s)

    out: List[Dict[str, Any]] = []
    n = len(df)
    for i in range(n):
        out.append({
            "id": ids[i] if i < len(ids) else f"HAZ-{i+1:03d}",
            "title": titles[i] if i < len(titles) and titles[i] else f"Hazard {ids[i] if i < len(ids) else i+1}",
            "department": depts[i] if i < len(depts) else "",
            "riskLevel": risk_levels[i] if i < len(risk_levels) else "Medium",
            "status": statuses[i] if i < len(statuses) and statuses[i] else "Identified",
            "date": dates[i] if i < len(dates) else "",
            "location": locations[i] if i < len(locations) else "",
            "violationType": violations[i] if i < len(violations) else "",
        })
    return out


@router.get("/audits")
async def list_audits() -> List[Dict[str, Any]]:
    df = get_audit_df()
    if df is None or df.empty:
        return []
    id_s = _first_existing(df, ["audit_id", "id", "AUDIT_ID"])
    if id_s is None:
        id_s = pd.Series([f"AUD-{i+1:03d}" for i in range(len(df))])
    title_s = _first_existing(df, ["title", "audit_title"]) 
    auditor_s = _first_existing(df, ["auditor", "auditor_name"]) 
    dept_s = _first_existing(df, ["department", "dept"]) 
    status_s = _first_existing(df, ["audit_status", "status"]) 
    sched_s = _first_existing(df, ["scheduled_date", "start_date"]) 
    comp_s = _first_existing(df, ["completion_date", "end_date"]) 
    findings_s = _first_existing(df, ["findings", "n_findings"]) 
    score_s = _first_existing(df, ["score", "audit_score"]) 

    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    auditors = _ensure_str_list(auditor_s)
    depts = _ensure_str_list(dept_s)
    statuses = _ensure_str_list(status_s)
    sched = _coerce_date_str(sched_s)
    comp = _coerce_date_str(comp_s)
    findings = _ensure_int_list(findings_s, default=0)
    scores = _ensure_int_list(score_s, default=0)

    out: List[Dict[str, Any]] = []
    n = len(df)
    for i in range(n):
        out.append({
            "id": ids[i] if i < len(ids) else f"AUD-{i+1:03d}",
            "title": titles[i] if i < len(titles) and titles[i] else f"Audit {ids[i] if i < len(ids) else i+1}",
            "auditor": auditors[i] if i < len(auditors) else "",
            "department": depts[i] if i < len(depts) else "",
            "status": statuses[i] if i < len(statuses) else "Scheduled",
            "scheduledDate": sched[i] if i < len(sched) else "",
            "completionDate": comp[i] if i < len(comp) and comp[i] else None,
            "findings": findings[i] if i < len(findings) else 0,
            "score": scores[i] if i < len(scores) else 0,
        })
    return out


@router.get("/actions/outgoing")
async def list_actions_outgoing() -> List[Dict[str, Any]]:
    """Best-effort derive an action queue from open incidents/hazards.
    - severity: mapped from severity/risk
    - title: from title/description
    - date: from occurrence/entered
    - assignee: empty unless column exists
    - type: 'corrective' for hazards with violation, 'investigation' for incidents High/Critical else 'attention'
    """
    inc = get_incident_df()
    if inc is None:
        inc = pd.DataFrame()
    haz = get_hazard_df()
    if haz is None:
        haz = pd.DataFrame()

    items: List[Dict[str, Any]] = []

    def _from_df(df: pd.DataFrame, kind: str) -> None:
        if df is None or df.empty:
            return
        title_s = _first_existing(df, ["title", "description", "incident_title", "hazard_title"])
        if title_s is None:
            title_s = pd.Series([f"{kind.title()}" for _ in range(len(df))])
        status_s = _first_existing(df, ["status", "audit_status", "hazard_status", "workflow_status"])
        if status_s is None:
            status_s = pd.Series(["Open"] * len(df))
        date_s = _first_existing(df, ["occurrence_date", "entered_date", "date", "start_date"])
        if date_s is None:
            date_s = pd.Series([None] * len(df))
        assignee_s = _first_existing(df, ["assignee", "assigned_to"])
        if assignee_s is None:
            assignee_s = pd.Series([""] * len(df))
        sev_s = _first_existing(df, ["severity", "severity_level"])
        if sev_s is None:
            sev_s = pd.Series([None] * len(df))
        sev_score_s = _first_existing(df, ["severity_score", "risk_score"])
        if sev_score_s is None:
            sev_score_s = pd.Series([None] * len(df))
        viol_s = _first_existing(df, ["violation_type_hazard_id", "violation_type"])
        if viol_s is None:
            viol_s = pd.Series([None] * len(df))

        dates = _coerce_date_str(date_s)
        titles = _ensure_str_list(title_s)
        assignees = _ensure_str_list(assignee_s)
        status = _ensure_str_list(status_s)
        severities = [
            _score_to_bucket(a if pd.notna(a) and str(a).strip() else b)
            for a, b in zip(sev_s, sev_score_s)
        ]
        violations = _ensure_str_list(viol_s)

        for i in range(len(df)):
            # filter only open/in-progress like statuses
            st = (status[i] or "").lower()
            if any(k in st for k in ["open", "progress", "identified", "review", "mitigat"]):
                t = titles[i] or f"{kind.title()} Action"
                sev = severities[i] if i < len(severities) else "Medium"
                action_type = "corrective" if (kind == "hazard" and violations[i]) else ("investigation" if sev in ("High", "Critical") else "attention")
                items.append({
                    "id": f"{kind[:3].upper()}-{i+1}",
                    "severity": sev.lower(),
                    "title": t,
                    "date": dates[i] if i < len(dates) else "",
                    "assignee": assignees[i] if i < len(assignees) else "",
                    "type": action_type,
                })

    _from_df(inc, "incident")
    _from_df(haz, "hazard")

    # Limit to a reasonable number (UI likely shows a short queue)
    return items[:50]


def _derive_outgoing_items() -> List[Dict[str, Any]]:
    """Build outgoing action items from incidents and hazards using same logic as list_actions_outgoing."""
    inc = get_incident_df()
    if inc is None:
        inc = pd.DataFrame()
    haz = get_hazard_df()
    if haz is None:
        haz = pd.DataFrame()

    items: List[Dict[str, Any]] = []

    def _from_df(df: pd.DataFrame, kind: str) -> None:
        if df is None or df.empty:
            return
        title_s = _first_existing(df, ["title", "description", "incident_title", "hazard_title"])
        if title_s is None:
            title_s = pd.Series([f"{kind.title()}" for _ in range(len(df))])
        status_s = _first_existing(df, ["status", "audit_status", "hazard_status", "workflow_status"])
        if status_s is None:
            status_s = pd.Series(["Open"] * len(df))
        date_s = _first_existing(df, ["occurrence_date", "entered_date", "date", "start_date"])
        if date_s is None:
            date_s = pd.Series([None] * len(df))
        assignee_s = _first_existing(df, ["assignee", "assigned_to"])
        if assignee_s is None:
            assignee_s = pd.Series([""] * len(df))
        sev_s = _first_existing(df, ["severity", "severity_level"])
        if sev_s is None:
            sev_s = pd.Series([None] * len(df))
        sev_score_s = _first_existing(df, ["severity_score", "risk_score"])
        if sev_score_s is None:
            sev_score_s = pd.Series([None] * len(df))
        viol_s = _first_existing(df, ["violation_type_hazard_id", "violation_type"])
        if viol_s is None:
            viol_s = pd.Series([None] * len(df))

        dates = _coerce_date_str(date_s)
        titles = _ensure_str_list(title_s)
        assignees = _ensure_str_list(assignee_s)
        status = _ensure_str_list(status_s)
        severities = [
            _score_to_bucket(a if pd.notna(a) and str(a).strip() else b)
            for a, b in zip(sev_s, sev_score_s)
        ]
        violations = _ensure_str_list(viol_s)

        for i in range(len(df)):
            st = (status[i] or "").lower()
            if any(k in st for k in ["open", "progress", "identified", "review", "mitigat", "pending", "assigned", "todo", "to-do", "not started", "awaiting"]):
                t = titles[i] or f"{kind.title()} Action"
                sev = severities[i] if i < len(severities) else "Medium"
                action_type = "corrective" if (kind == "hazard" and (i < len(violations) and violations[i])) else ("investigation" if sev in ("High", "Critical") else "attention")
                items.append({
                    "id": f"{kind[:3].upper()}-{i+1}",
                    "severity": str(sev).lower(),
                    "title": t,
                    "date": dates[i] if i < len(dates) else "",
                    "assignee": assignees[i] if i < len(assignees) else "",
                    "type": action_type,
                })

    _from_df(inc, "incident")
    _from_df(haz, "hazard")
    return items


@router.get("/actions/outgoing/summary")
async def list_actions_outgoing_summary() -> Dict[str, Any]:
    """Aggregated summary for the Outgoing Actions card.
    Returns totals, priority counts, type breakdown, and a small sample list.
    """
    items = _derive_outgoing_items()
    total = len(items)
    high_priority = sum(1 for it in items if it.get("severity") in ("high", "critical"))
    by_type: Dict[str, int] = {"corrective": 0, "investigation": 0, "attention": 0}
    for it in items:
        t = it.get("type", "attention")
        if t not in by_type:
            by_type[t] = 0
        by_type[t] += 1

    payload = {
        "total": total,
        "highPriority": high_priority,
        "byType": by_type,
        "sample": items[:5],
        "lastUpdated": pd.Timestamp.utcnow().isoformat() + "Z",
    }
    return JSONResponse(content=to_native_json(payload))


# ---------- Recent items (Incidents, Hazards, Audits) ----------

@router.get("/incidents/recent")
async def recent_incidents(limit: int = 5) -> List[Dict[str, Any]]:
    df = get_incident_df()
    if df is None or df.empty:
        return []
    cp = df.copy()
    date_col = _first_existing(cp, [
        "occurrence_date", "date", "entered_date", "reported_date", "created_date",
    ])
    if date_col is not None:
        try:
            cp["__dt"] = pd.to_datetime(date_col, errors="coerce")
            cp = cp.sort_values("__dt", ascending=False)
        except Exception:
            pass
    # Map fields
    id_s = _first_existing(cp, ["incident_id", "id", "Incident ID", "INCIDENT_ID", "IncidentID"])
    if id_s is None:
        id_s = pd.Series([f"INC-{i+1:03d}" for i in range(len(cp))])
    title_s = _first_existing(cp, ["title", "incident_title", "description", "short_description", "remarks"])
    if title_s is None:
        title_s = pd.Series(["" for _ in range(len(cp))])
    dept_s = _first_existing(cp, ["department", "dept", "Department"])
    if dept_s is None:
        dept_s = pd.Series(["" for _ in range(len(cp))])
    status_s = _first_existing(cp, ["status", "incident_status", "workflow_status", "Status"])
    if status_s is None:
        status_s = pd.Series(["Open" for _ in range(len(cp))])
    sev_s = _first_existing(cp, ["severity", "Severity", "severity_level"])  # may be None
    sev_score_s = _first_existing(cp, ["severity_score", "Severity Score", "severity_numeric"])  # may be None
    date_s = date_col if date_col is not None else pd.Series([None] * len(cp))

    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    depts = _ensure_str_list(dept_s)
    statuses = _ensure_str_list(status_s)
    dates = _coerce_date_str(date_s)
    if sev_s is not None:
        severities = [_score_to_bucket(v, kind="severity") for v in sev_s]
    else:
        fallback = sev_score_s if sev_score_s is not None else pd.Series([None] * len(cp))
        severities = [_score_to_bucket(v, kind="severity") for v in fallback]

    out: List[Dict[str, Any]] = []
    for i in range(min(limit, len(cp))):
        out.append({
            "id": ids[i] if i < len(ids) else f"INC-{i+1:03d}",
            "title": titles[i] if i < len(titles) else "",
            "department": depts[i] if i < len(depts) else "",
            "status": statuses[i] if i < len(statuses) else "Open",
            "severity": severities[i] if i < len(severities) else "Medium",
            "date": dates[i] if i < len(dates) else "",
        })
    return out


@router.get("/hazards/recent")
async def recent_hazards(limit: int = 5) -> List[Dict[str, Any]]:
    df = get_hazard_df()
    if df is None or df.empty:
        return []
    cp = df.copy()
    date_col = _first_existing(cp, ["occurrence_date", "date", "entered_date", "reported_date"])
    if date_col is not None:
        try:
            cp["__dt"] = pd.to_datetime(date_col, errors="coerce")
            cp = cp.sort_values("__dt", ascending=False)
        except Exception:
            pass
    id_s = _first_existing(cp, ["hazard_id", "id", "HAZARD_ID"])
    if id_s is None:
        id_s = pd.Series([f"HAZ-{i+1:03d}" for i in range(len(cp))])
    title_s = _first_existing(cp, ["title", "hazard_title", "description"])
    if title_s is None:
        title_s = pd.Series(["" for _ in range(len(cp))])
    dept_s = _first_existing(cp, ["department", "dept"])
    if dept_s is None:
        dept_s = pd.Series(["" for _ in range(len(cp))])
    status_s = _first_existing(cp, ["status", "hazard_status", "workflow_status"])
    if status_s is None:
        status_s = pd.Series(["Identified" for _ in range(len(cp))])
    risk_level_s = _first_existing(cp, ["risk_level", "Risk Level"])  # may be None
    risk_score_s = _first_existing(cp, ["risk_score", "Risk Score"])  # may be None
    date_s = date_col if date_col is not None else pd.Series([None] * len(cp))

    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    depts = _ensure_str_list(dept_s)
    statuses = _ensure_str_list(status_s)
    dates = _coerce_date_str(date_s)
    if risk_level_s is not None:
        risk_levels = [_score_to_bucket(v, kind="risk") for v in risk_level_s]
    else:
        fallback = risk_score_s if risk_score_s is not None else pd.Series([None] * len(cp))
        risk_levels = [_score_to_bucket(v, kind="risk") for v in fallback]

    out: List[Dict[str, Any]] = []
    for i in range(min(limit, len(cp))):
        out.append({
            "id": ids[i] if i < len(ids) else f"HAZ-{i+1:03d}",
            "title": titles[i] if i < len(titles) else "",
            "department": depts[i] if i < len(depts) else "",
            "status": statuses[i] if i < len(statuses) else "Identified",
            "riskLevel": risk_levels[i] if i < len(risk_levels) else "Medium",
            "date": dates[i] if i < len(dates) else "",
        })
    return out


@router.get("/audits/recent")
async def recent_audits(limit: int = 5) -> List[Dict[str, Any]]:
    df = get_audit_df()
    if df is None or df.empty:
        return []
    cp = df.copy()
    date_col = _first_existing(cp, ["scheduled_date", "start_date", "completion_date", "end_date"])  # prefer start
    if date_col is not None:
        try:
            cp["__dt"] = pd.to_datetime(date_col, errors="coerce")
            cp = cp.sort_values("__dt", ascending=False)
        except Exception:
            pass

    id_s = _first_existing(cp, ["audit_id", "id", "AUDIT_ID"])
    if id_s is None:
        id_s = pd.Series([f"AUD-{i+1:03d}" for i in range(len(cp))])
    title_s = _first_existing(cp, ["title", "audit_title"])
    if title_s is None:
        title_s = pd.Series(["" for _ in range(len(cp))])
    status_s = _first_existing(cp, ["audit_status", "status"])
    if status_s is None:
        status_s = pd.Series(["Scheduled" for _ in range(len(cp))])
    sched_s = _first_existing(cp, ["scheduled_date", "start_date"])  # used as main date
    comp_s = _first_existing(cp, ["completion_date", "end_date"])  # optional

    ids = _ensure_str_list(id_s)
    titles = _ensure_str_list(title_s)
    statuses = _ensure_str_list(status_s)
    dates = _coerce_date_str(sched_s if sched_s is not None else comp_s)

    out: List[Dict[str, Any]] = []
    for i in range(min(limit, len(cp))):
        out.append({
            "id": ids[i] if i < len(ids) else f"AUD-{i+1:03d}",
            "title": titles[i] if i < len(titles) else "",
            "status": statuses[i] if i < len(statuses) else "Scheduled",
            "date": dates[i] if i < len(dates) else "",
        })
    return out
