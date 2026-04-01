"""
answer_generator.py
────────────────────
Sends the assembled context + user query to Google Gemini (or falls back to
a rule-based summary if no API key is configured).

Set GEMINI_API_KEY environment variable to enable LLM answers.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ─── Load .env automatically ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"   # project root/.env
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass   # python-dotenv not installed — rely on env vars being set externally

from spear_rag.rag.geo_parser import SpatialQuery


# ─── Gemini setup ─────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        _GEMINI_AVAILABLE = True
        _USE_LEGACY = True
    except ImportError:
        _GEMINI_AVAILABLE = False

_GEMINI_CLIENT = None

def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not _GEMINI_AVAILABLE:
        return None
    try:
        client = genai.Client(api_key=api_key)
        _GEMINI_CLIENT = client
        return client
    except Exception:
        return None


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert GIS analyst and remote sensing scientist specialising in \
India's land cover, climate, and agricultural systems.

IMPORTANT CONSTRAINTS:
- You MUST answer in ENGLISH ONLY. Never use Hindi, Hinglish, or any other language.
- The satellite data you have access to covers the period 2020–2024 ONLY. Do not make claims about data outside this range.
- Base your analysis strictly on the satellite context provided. Do not hallucinate events not supported by the data.

DATA YOU HAVE ACCESS TO:
- SPEAR embeddings from Sentinel-1 (SAR), Sentinel-2 (multispectral), Planet (optical), and Climate (LST, precipitation, elevation)
- Landcover classification codes: 0=Water, 1=Trees/Forest, 2=Grass, 3=Flooded Vegetation, 4=Crops/Agriculture, 5=Shrub & Scrub, 6=Built Area, 7=Bare Ground, 8=Snow & Ice
- NDVI: vegetation health (>0.5 dense, 0.2-0.5 moderate, <0.2 sparse/stressed)
- NDWI: water presence (>0.2 water/flooding, 0.0-0.2 moist, <0.0 dry)

YOUR TASK:
1. Identify the dominant landcover class(es) in the queried region from the classification distribution.
2. For crop/agricultural queries: analyse NDVI trend (health), NDWI (soil moisture), LST (heat stress).
3. For flood/water queries: focus on NDWI, Sentinel-1 SAR backscatter (VV/VH), precipitation.
4. Cite specific values from the context (e.g., "NDVI of 0.48 indicates moderate-to-good vegetation").
5. Give a spatial summary: which parts of the region show what conditions.
6. End with a concise 1–2 sentence key finding.
7. Keep total response under 300 words."""



def _rule_based_answer(
    query: SpatialQuery,
    retrieval_results: Dict[str, Dict],
    context: str,
) -> str:
    """Fallback answer when no LLM API key is available."""
    total = sum(r.get("n_retrieved", 0) for r in retrieval_results.values())
    if total == 0:
        return (
            f"No satellite data was found for {query.location} "
            f"in the specified time range. Try a broader region or time window."
        )

    lines = [
        f"📍 **{query.location}** — Satellite Analysis",
        f"Year: {query.year_start or 'all years'} | Event: {query.event_type or 'general'}",
        "",
    ]

    for mod, result in retrieval_results.items():
        n = result.get("n_retrieved", 0)
        if n == 0:
            continue
        lines.append(f"**{mod.upper()} ({n:,} pixels)**")
        if result.get("ndwi") is not None:
            ndwi = result["ndwi"]
            interp = ("Flood/water detected" if ndwi > 0.2 else
                      "Marginal moisture" if ndwi > 0 else "Dry conditions")
            lines.append(f"  • NDWI: {ndwi} → {interp}")
        if result.get("ndvi") is not None:
            ndvi = result["ndvi"]
            interp = ("Dense vegetation" if ndvi > 0.5 else
                      "Moderate vegetation" if ndvi > 0.2 else "Sparse/stressed vegetation")
            lines.append(f"  • NDVI: {ndvi} → {interp}")
        stats = result.get("raw_stats", {})
        for col in ["total_precipitation", "LST_Day_1km", "elevation"]:
            if col in stats:
                s = stats[col]
                lines.append(f"  • {stats[col]['description']}: {s['mean']}")
        lines.append("")

    lines.append("*(LLM analysis unavailable — set GEMINI_API_KEY for full interpretation)*")
    return "\n".join(lines)


# ─── Main answer function ──────────────────────────────────────────────────────

def generate_answer(
    query: SpatialQuery,
    retrieval_results: Dict[str, Dict],
    context: str,
) -> Dict:
    """
    Generate a text answer + structured map data from retrieval results.

    Returns:
        {
          "answer":     str,           # LLM or rule-based answer
          "map_points": list[dict],    # [{lat, lon, year, modality}, ...]
          "total_pixels": int,
          "query_summary": str,
        }
    """
    model = _get_gemini_client()

    # Collect map points from all modalities
    map_points: List[Dict] = []
    for mod, result in retrieval_results.items():
        if result.get("map_points"):
            pts = result["map_points"][:5000]   # cap for frontend
            for pt in pts:
                pt["modality"] = mod
            map_points.extend(pts)

    total_pixels = sum(r.get("n_retrieved", 0) for r in retrieval_results.values())

    # ── LLM answer ────────────────────────────────────────────────────────────
    if model is not None:
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- SATELLITE CONTEXT ---\n{context}\n\n"
            f"--- USER QUESTION ---\n{query.raw_query}"
        )
        try:
            response = model.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=full_prompt,
            )
            answer_text = response.text
        except Exception as e:
            print(f"[LLM] Error: {e}")
            answer_text = _rule_based_answer(query, retrieval_results, context)
    else:
        answer_text = _rule_based_answer(query, retrieval_results, context)

    return {
        "answer":        answer_text,
        "map_points":    map_points,
        "total_pixels":  total_pixels,
        "query_summary": str(query),
        "context":       context,
    }
