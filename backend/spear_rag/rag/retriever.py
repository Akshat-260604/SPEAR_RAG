"""
retriever.py
────────────
Orchestrates the full retrieval pipeline:
  1. Takes a SpatialQuery
  2. Queries all available modalities via spatial bbox filter
  3. Returns aggregated results per modality
"""

from __future__ import annotations
from typing import Dict, List, Optional
import os

from spear_rag.rag.geo_parser import SpatialQuery
from spear_rag.index.spatial_query import query_all_modalities

INDEX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spear_index"
)

def retrieve(
    query: SpatialQuery,
    modalities: Optional[List[str]] = None,
    max_rows: int = 50_000,
) -> Dict[str, Dict]:
    """
    Run full retrieval for a SpatialQuery across all modalities.

    Args:
        query:      Parsed SpatialQuery (from geo_parser)
        modalities: List of modalities to query (default: all available)
        max_rows:   Max rows to load per modality for aggregation

    Returns:
        Dict keyed by modality name with summarize_modality results.
    """
    lat_min, lat_max, lon_min, lon_max = query.bbox

    print(f"\n[Retriever] Querying bbox: lat[{lat_min:.3f},{lat_max:.3f}] "
          f"lon[{lon_min:.3f},{lon_max:.3f}]")
    if query.year_start:
        print(f"[Retriever] Year range: {query.year_start}–{query.year_end}")
    if query.month:
        print(f"[Retriever] Month filter: {query.month}")

    # Determine which modalities have built indexes
    available = []
    for mod in (modalities or ["climate", "s2", "s1", "planet"]):
        meta_path = os.path.join(INDEX_ROOT, f"{mod}_meta.parquet")
        if os.path.exists(meta_path):
            available.append(mod)
        else:
            print(f"[Retriever] Skipping '{mod}' — index not built yet.")

    results = query_all_modalities(
        lat_min    = lat_min,
        lat_max    = lat_max,
        lon_min    = lon_min,
        lon_max    = lon_max,
        year_start = query.year_start,
        year_end   = query.year_end,
        month      = query.month,
        modalities = available,
        max_rows   = max_rows,
    )

    return results
