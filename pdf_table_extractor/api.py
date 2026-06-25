"""FastAPI application exposing the table-extraction pipeline over HTTP.

Run locally with::

    uvicorn pdf_table_extractor.api:app --reload

Endpoints
---------
GET  /health                  liveness probe
POST /extract?page=<n>        upload a PDF, returns detected tables as JSON
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from . import __version__
from .pipeline import extract_tables

app = FastAPI(
    title="PDF Table Extractor",
    description="Detect and extract tables from PDF pages using YOLOv3 + Camelot.",
    version=__version__,
)


class TableModel(BaseModel):
    index: int
    area: str
    rows: list[dict]


class ExtractResponse(BaseModel):
    filename: str
    page: int
    num_tables: int
    tables: list[TableModel]


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(..., description="PDF file to parse"),
    page: int = Query(1, ge=1, description="1-indexed page to parse"),
) -> ExtractResponse:
    """Extract tables from a single page of an uploaded PDF."""
    if file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    } and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A PDF file is required.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.TemporaryDirectory(prefix="api_pdf_") as tmp:
        pdf_path = Path(tmp) / (file.filename or "upload.pdf")
        pdf_path.write_bytes(contents)
        try:
            result = extract_tables(str(pdf_path), page=page)
        except IndexError as exc:
            raise HTTPException(
                status_code=400, detail=f"Page {page} is out of range."
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    records = result.to_records()
    tables = [
        TableModel(index=i, area=result.areas[i], rows=rows)
        for i, rows in enumerate(records)
    ]
    return ExtractResponse(
        filename=file.filename or "upload.pdf",
        page=page,
        num_tables=result.num_tables,
        tables=tables,
    )
