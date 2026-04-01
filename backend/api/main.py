"""
main.py (FastAPI)
─────────────────
REST API for the SPEAR-RAG system.

Endpoints:
  POST /query          — Full RAG pipeline query
  GET  /health         — Health check
  GET  /modalities     — List available (indexed) modalities

Run:
    cd /Users/akshatsaraswat/Desktop/SPEAR-RAG/backend
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Resolve important paths
HERE = Path(__file__).resolve()
BACKEND_ROOT = HERE.parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

# Add backend root to import path so spear_rag resolves after move
sys.path.insert(0, str(BACKEND_ROOT))

from spear_rag.rag.geo_parser      import parse_query
from spear_rag.rag.retriever       import retrieve
from spear_rag.rag.context_builder import build_context
from spear_rag.rag.answer_generator import generate_answer
from spear_rag.viz.map_viz         import build_map_from_results

INDEX_ROOT = BACKEND_ROOT / "spear_index"

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SPEAR-RAG",
    description="Spatial RAG system on SPEAR satellite embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str
    modalities: Optional[List[str]] = None    # None = all available
    max_rows:   int = 50_000
    heatmap:    bool = True


class QueryResponse(BaseModel):
    answer:        str
    total_pixels:  int
    query_summary: str
    map_html:      str
    elapsed_s:     float
    modality_counts: Dict[str, int]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "system": "SPEAR-RAG"}


@app.get("/modalities")
def list_modalities():
    available = []
    for mod in ["climate", "s2", "s1", "planet"]:
        if (INDEX_ROOT / f"{mod}_meta.parquet").exists():
            available.append(mod)
    return {"available": available}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    t0 = time.time()

    # 1. Parse query
    try:
        spatial_query = parse_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query parsing failed: {e}")

    # 2. Retrieve
    try:
        results = retrieve(spatial_query, modalities=req.modalities, max_rows=req.max_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 3. Build context
    context = build_context(spatial_query, results)

    # 4. Generate answer
    generated = generate_answer(spatial_query, results, context)

    # 5. Build map
    bbox = spatial_query.bbox
    center_lat = (bbox[0] + bbox[1]) / 2
    center_lon = (bbox[2] + bbox[3]) / 2
    map_html = build_map_from_results(results, center_lat, center_lon)

    modality_counts = {
        mod: res.get("n_retrieved", 0)
        for mod, res in results.items()
    }

    return QueryResponse(
        answer        = generated["answer"],
        total_pixels  = generated["total_pixels"],
        query_summary = generated["query_summary"],
        map_html      = map_html,
        elapsed_s     = round(time.time() - t0, 2),
        modality_counts = modality_counts,
    )


# ─── Serve frontend ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    candidates = [
        PROJECT_ROOT / "frontend" / "dist" / "index.html",  # built React app
        PROJECT_ROOT / "frontend" / "index.html",             # dev fallback
    ]

    for path in candidates:
        if path.exists():
            # index files use UTF-8 encoding
            return path.read_text(encoding="utf-8")

    return "<h1>SPEAR-RAG API Running</h1><p>POST /query to query.</p>"
