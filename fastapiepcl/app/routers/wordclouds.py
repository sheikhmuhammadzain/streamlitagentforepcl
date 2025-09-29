from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
import re
from collections import Counter

from wordcloud import WordCloud, STOPWORDS

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


def _resolve_column(df, candidates):
    if df is None or df.empty:
        return None
    cmap = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        k = str(c).strip().lower()
        if k in cmap:
            return cmap[k]
    # relaxed contains
    for c in candidates:
        k = str(c).strip().lower()
        for lk, orig in cmap.items():
            if k in lk:
                return orig
    return None


def _collect_text(df, columns):
    if df is None or df.empty:
        return ""
    cols = []
    for group in columns:
        col = _resolve_column(df, group)
        if col is not None:
            cols.append(col)
    if not cols:
        return ""
    s = df[cols].astype(str).apply(lambda r: " ".join([x for x in r.values if x and x != 'nan']), axis=1)
    text = " ".join(s.tolist())
    # basic cleaning: strip urls and extra punctuation spacing
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@router.get("/image")
async def wordcloud_image(
    dataset: str = Query("incident", description="incident or hazard"),
    width: int = Query(960, ge=200, le=4096),
    height: int = Query(540, ge=200, le=4096),
    background_color: str = Query("white"),
    extra_stopwords: str | None = Query(None, description="Comma-separated extra stopwords"),
):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    # default text columns groups (each inner list are variants to match)
    column_groups = [
        ["title"],
        ["description"],
        ["finding", "findings"],
        ["recommendation", "recommendations"],
        ["response"],
        ["task or activity at time of incident"],
        ["root cause"],
        ["accident type"],
        ["injury classification", "injury"],
        ["pse category"],
        ["worst case consequence (incident)", "worst case consequence potential (hazard id)"],
        ["actual consequence (incident)", "relevant consequence (incident)"],
        ["violation type (incident)", "violation type (hazard id)"],
    ]
    text = _collect_text(df, column_groups)
    if not text:
        text = "No Data"
    stops = set(STOPWORDS)
    if extra_stopwords:
        stops |= {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=stops,
        collocations=True,
    ).generate(text)
    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/departments-image")
async def wordcloud_departments_image(
    dataset: str = Query("both", description="incident | hazard | both"),
    width: int = Query(960, ge=200, le=4096),
    height: int = Query(540, ge=200, le=4096),
    background_color: str = Query("white"),
    top_n: int = Query(100, ge=5, le=2000),
    min_count: int = Query(1, ge=1),
    exclude_other: bool = Query(True, description="Exclude generic 'Other/NA' buckets"),
):
    def collect_counts(df):
        if df is None or df.empty:
            return Counter()
        dep_col = _resolve_column(df, ["department"]) or _resolve_column(df, ["section"]) or None
        if dep_col is None:
            return Counter()
        s = df[dep_col].dropna().astype(str).str.strip()
        # remove NA-like and generic placeholders
        na_pat = re.compile(r"^(n/?a|nan|null|none|not\s*applicable)$", re.IGNORECASE)
        other_pat = re.compile(r"^(other|others|misc|not\s*assigned|unknown)$", re.IGNORECASE)
        def _ok(v: str) -> bool:
            if not v:
                return False
            if na_pat.match(v):
                return False
            if exclude_other and other_pat.match(v):
                return False
            return True
        s = s[s.map(_ok)]
        # normalize for display consistency
        s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.title()
        counts = s.value_counts()
        # apply min_count and top_n
        counts = counts[counts >= int(min_count)].head(int(top_n))
        return Counter(counts.to_dict())

    wanted = (dataset or "both").strip().lower()
    freq = Counter()
    if wanted in ("incident", "both"):
        freq += collect_counts(get_incident_df())
    if wanted in ("hazard", "both"):
        freq += collect_counts(get_hazard_df())

    if not freq:
        freq = Counter({"No Data": 1})

    wc = WordCloud(width=width, height=height, background_color=background_color, collocations=False)
    img = wc.generate_from_frequencies(freq).to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/department-image")
async def wordcloud_department_image(
    dataset: str = Query("incident"),
    department: str | None = Query(None, description="Filter by Department name"),
    width: int = Query(960, ge=200, le=4096),
    height: int = Query(540, ge=200, le=4096),
    background_color: str = Query("white"),
    extra_stopwords: str | None = Query(None),
):
    df = get_incident_df() if (dataset or "incident").lower() == "incident" else get_hazard_df()
    if df is None or df.empty:
        text = "No Data"
    else:
        if department:
            dep_col = _resolve_column(df, ["department"]) or _resolve_column(df, ["section"]) or None
            if dep_col is not None:
                df = df[df[dep_col].astype(str).str.strip().str.lower() == department.strip().lower()]
        column_groups = [
            ["title"],
            ["description"],
            ["finding", "findings"],
            ["recommendation", "recommendations"],
            ["response"],
            ["task or activity at time of incident"],
            ["root cause"],
            ["accident type"],
            ["injury classification", "injury"],
            ["pse category"],
            ["worst case consequence (incident)", "worst case consequence potential (hazard id)"],
            ["actual consequence (incident)", "relevant consequence (incident)"],
            ["violation type (incident)", "violation type (hazard id)"],
        ]
        text = _collect_text(df, column_groups)
        if not text:
            text = "No Data"
    stops = set(STOPWORDS)
    if extra_stopwords:
        stops |= {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}
    wc = WordCloud(width=width, height=height, background_color=background_color, stopwords=stops, collocations=True).generate(text)
    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
