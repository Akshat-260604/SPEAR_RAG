"""
context_builder.py
──────────────────
Formats retrieval results into a structured LLM-readable context block.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from spear_rag.rag.geo_parser import SpatialQuery


MODALITY_DESCRIPTIONS = {
    "climate": "Climate & Meteorological Data (Elevation, Land Surface Temperature day/night, Precipitation)",
    "s2":      "Sentinel-2 Multispectral Optical Data (10 bands: Blue to SWIR)",
    "s1":      "Sentinel-1 SAR Radar Data (VV/VH backscatter — sensitive to water & soil moisture)",
    "planet":  "Planet Optical Data (High-frequency optical imagery)",
}


def _format_raw_stats(raw_stats: Dict) -> str:
    lines = []
    for col, stats in raw_stats.items():
        desc = stats.get("description", col)
        mean = stats.get("mean", "N/A")
        std  = stats.get("std",  "N/A")
        mn   = stats.get("min",  "N/A")
        mx   = stats.get("max",  "N/A")
        lines.append(f"    • {desc}: mean={mean}, std={std}, range=[{mn} – {mx}]")
    return "\n".join(lines) if lines else "    (no raw feature data)"


def build_context(
    query: SpatialQuery,
    retrieval_results: Dict[str, Dict],
) -> str:
    """
    Build a structured text context block for the LLM from retrieval results.

    Returns a formatted string to be injected into the LLM prompt.
    """
    lines: List[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    year_str = ""
    if query.year_start and query.year_end:
        year_str = f"{query.year_start}–{query.year_end}"
    elif query.year_start:
        year_str = str(query.year_start)

    month_str = f", Month: {query.month}" if query.month else ""
    event_str = f"Query type: {query.query_type.upper()}" if query.query_type != "general" else ""

    lines.append("=" * 70)
    lines.append("SPATIAL SATELLITE DATA CONTEXT")
    lines.append("=" * 70)
    lines.append(f"Region: {query.location}")
    lat_min, lat_max, lon_min, lon_max = query.bbox
    lines.append(f"BBox:   lat [{lat_min:.3f}° – {lat_max:.3f}°N]  "
                 f"lon [{lon_min:.3f}° – {lon_max:.3f}°E]")
    if year_str:
        lines.append(f"Time:   {year_str}{month_str}")
    if event_str:
        lines.append(event_str)
    lines.append("")

    total_pixels = sum(r.get("n_retrieved", 0) for r in retrieval_results.values())
    lines.append(f"Total pixels retrieved across all modalities: {total_pixels:,}")
    lines.append("")

    # ── Per-modality blocks ─────────────────────────────────────────────────
    for mod, result in retrieval_results.items():
        n = result.get("n_retrieved", 0)
        desc = MODALITY_DESCRIPTIONS.get(mod, mod)
        lines.append(f"─── {desc} ───")

        if n == 0:
            lines.append("  No data available for this region/time.")
            lines.append("")
            continue

        lines.append(f"  Pixels retrieved: {n:,}")

        # Spatial extent
        if "lat_range" in result:
            lr = result["lat_range"]
            lonr = result["lon_range"]
            lines.append(f"  Spatial coverage: lat [{lr[0]}–{lr[1]}°] lon [{lonr[0]}–{lonr[1]}°]")

        # Temporal distribution
        if result.get("year_counts"):
            yc = result["year_counts"]
            yc_str = ", ".join(f"{yr}: {cnt}" for yr, cnt in sorted(yc.items()))
            lines.append(f"  Year distribution: {yc_str}")

        # Raw feature statistics
        if result.get("raw_stats"):
            lines.append("  Feature statistics:")
            lines.append(_format_raw_stats(result["raw_stats"]))

        # Derived indices
        if result.get("ndvi") is not None:
            ndvi_interp = (
                "high vegetation" if result["ndvi"] > 0.5 else
                "moderate vegetation" if result["ndvi"] > 0.2 else
                "sparse/stressed vegetation or non-vegetated"
            )
            lines.append(f"  NDVI (avg): {result['ndvi']}  → {ndvi_interp}")

        if result.get("ndwi") is not None:
            ndwi_interp = (
                "strong water presence / flooding likely" if result["ndwi"] > 0.2 else
                "some moisture / marginal water" if result["ndwi"] > 0.0 else
                "dry / no significant water"
            )
            lines.append(f"  NDWI (avg): {result['ndwi']}  → {ndwi_interp}")

        # Embedding summary
        if result.get("embedding"):
            emb = result["embedding"]
            lines.append(f"  Embedding avg-norm: {emb['avg_norm']},  "
                         f"avg-std (spread): {emb['avg_std']}")

        lines.append("")

    lines.append("=" * 70)
    lines.append("NOTE: All data is derived from SPEAR self-supervised satellite "
                 "embeddings trained on actual Sentinel-1, Sentinel-2, Planet, "
                 "and climate data over India.")
    lines.append("=" * 70)

    return "\n".join(lines)
