from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..models.schemas import (
    InferredSchema,
    InferSchemaRequest,
    SheetPreview,
    WorkbookSummaryResponse,
)
from ..services.excel import (
    payload_to_df,
    read_excel_to_sheets,
    summarize_sheet,
    load_default_sheets,
    get_dataset_selection_names,
)
from ..services.schema import infer_schema


router = APIRouter(prefix="/workbooks", tags=["workbooks"])


@router.post("/upload")
async def upload_workbook(file: UploadFile = File(...)):
    try:
        content = await file.read()
        sheets = read_excel_to_sheets(content)
        out_sheets: List[dict] = []
        for name, df in sheets.items():
            n, rows, cols, cols_list, sample = summarize_sheet(name, df)
            out_sheets.append({
                "name": n,
                "columns": cols_list,
                "rowCount": rows,
                "sampleData": sample,
            })
        payload = {
            "fileName": file.filename or "uploaded.xlsx",
            "uploadDate": datetime.utcnow().isoformat() + "Z",
            "sheetCount": len(out_sheets),
            "sheets": out_sheets,
        }
        return JSONResponse(content=payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse workbook: {e}")


@router.get("/example")
async def load_example_workbook():
    # Resolve project root and expected example file location
    project_root = Path(__file__).resolve().parents[3]
    example_path = project_root / "EPCL_VEHS_Data_Processed.xlsx"
    if not example_path.exists():
        raise HTTPException(status_code=404, detail="Example workbook not found")
    try:
        content = example_path.read_bytes()
        sheets = read_excel_to_sheets(content)
        out_sheets: List[dict] = []
        for name, df in sheets.items():
            n, rows, cols, cols_list, sample = summarize_sheet(name, df)
            out_sheets.append({
                "name": n,
                "columns": cols_list,
                "rowCount": rows,
                "sampleData": sample,
            })
        # Determine file modified time as uploadDate, fallback to now
        try:
            mtime = datetime.utcfromtimestamp(example_path.stat().st_mtime).isoformat() + "Z"
        except Exception:
            mtime = datetime.utcnow().isoformat() + "Z"
        payload = {
            "fileName": example_path.name,
            "uploadDate": mtime,
            "sheetCount": len(out_sheets),
            "sheets": out_sheets,
        }
        return JSONResponse(content=payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load example workbook: {e}")


@router.get("/reload")
async def reload_default_workbook():
    """Clear the cached default Excel workbook and reload it from disk.
    Returns the list of sheet names and count after reload.
    """
    try:
        # Clear LRU cache and reload
        load_default_sheets.cache_clear()
        sheets = load_default_sheets()
        return {
            "reloaded": True,
            "sheet_count": len(sheets),
            "sheets": list(sheets.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload default workbook: {e}")


@router.get("/selection")
async def get_selection_mapping():
    """Return which sheet names are currently mapped to incident/hazard/audit/inspection."""
    try:
        mapping = get_dataset_selection_names()
        return mapping
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute selection mapping: {e}")


@router.post("/infer-schema", response_model=InferredSchema)
async def infer_sheet_schema(payload: InferSchemaRequest):
    df = payload_to_df(payload.data.records)
    schema = infer_schema(df, payload.sheet_name)
    return InferredSchema(**schema)

