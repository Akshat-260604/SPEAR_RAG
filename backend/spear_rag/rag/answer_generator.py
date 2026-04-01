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
    _env_path = Path(__file__).resolve().parents[2] / ".env"   # backend/.env
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass   # rely on env vars being set externally

from spear_rag.rag.geo_parser import SpatialQuery


# ─── Gemini setup (uses google-generativeai legacy SDK) ───────────────────────
try:
    import google.generativeai as genai_sdk
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_GEMINI_MODEL = None

def _get_gemini_client():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is not None:
        return _GEMINI_MODEL
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not _GEMINI_AVAILABLE:
        print(f"[Gemini] API key present={bool(api_key)}, SDK available={_GEMINI_AVAILABLE}")
        return None
    try:
        genai_sdk.configure(api_key=api_key)
        _GEMINI_MODEL = genai_sdk.GenerativeModel("gemini-2.5-flash-lite")
        print("[Gemini] Client initialised ✅")
        return _GEMINI_MODEL
    except Exception as e:
        print(f"[Gemini] Setup failed: {e}")
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

    # 1. Aggregate signals across all modalities
    avg_ndvi = []
    avg_ndwi = []
    climate_stats = {}
    active_mods = []

    for mod, result in retrieval_results.items():
        n = result.get("n_retrieved", 0)
        if n == 0: continue
        active_mods.append(f"{mod.upper()} ({n:,}px)")
        
        if result.get("ndvi") is not None: avg_ndvi.append(result["ndvi"])
        if result.get("ndwi") is not None: avg_ndwi.append(result["ndwi"])
        
        if mod == "climate" and "raw_stats" in result:
            stats = result["raw_stats"]
            if "total_precipitation" in stats:
                climate_stats["precip"] = stats["total_precipitation"]["mean"]
            if "LST_Day_1km" in stats:
                climate_stats["lst_day"] = stats["LST_Day_1km"]["mean"]
            if "elevation" in stats:
                climate_stats["elev"] = stats["elevation"]["mean"]

    # 2. Build professional output
    lines = [
        f"**Satellite Insights: {query.location}**",
        f"Timeframe: {query.year_start or 'All years'} | Focus: {query.query_type.title()}",
        "",
        "**Key Environmental Signals**"
    ]

    # Vegetation
    if avg_ndvi:
        ndvi = sum(avg_ndvi) / len(avg_ndvi)
        interp = "Dense, healthy canopy" if ndvi > 0.5 else "Moderate vegetation" if ndvi > 0.2 else "Sparse/stressed vegetation or barren land"
        lines.append(f"• **Vegetation Health (NDVI):** {ndvi:.3f} — {interp}")

    # Moisture/Water
    if avg_ndwi:
        ndwi = sum(avg_ndwi) / len(avg_ndwi)
        interp = "High water presence / flooded" if ndwi > 0.2 else "Marginal moisture" if ndwi > 0 else "Dry conditions"
        lines.append(f"• **Surface Moisture (NDWI):** {ndwi:.3f} — {interp}")

    # Climate
    if climate_stats:
        lines.append("")
        lines.append("**Climate Profile**")
        if "lst_day" in climate_stats: lines.append(f"• **Surface Temp (Day):** {climate_stats['lst_day']} °C")
        if "precip" in climate_stats: lines.append(f"• **Precipitation:** {climate_stats['precip']} mm")
        if "elev" in climate_stats: lines.append(f"• **Elevation:** {climate_stats['elev']} m")

    lines.append("")
    lines.append("**Data Sources Utilized:**")
    lines.append(", ".join(active_mods))

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

    # First generate the beautiful rule-based insight box
    structured_insights = _rule_based_answer(query, retrieval_results, context)

    # ── LLM answer ────────────────────────────────────────────────────────────
    if model is not None:
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- SATELLITE CONTEXT ---\n{context}\n\n"
            f"--- USER QUESTION ---\n{query.raw_query}\n\n"
            f"(Provide a highly crisp, punchy summary in max 2 sentences. Do not hallucinate or use emojis.)"
        )
        try:
            response = model.generate_content(full_prompt)
            ai_text = response.text.strip()
            answer_text = f"{structured_insights}\n\n{ai_text}"
        except Exception as e:
            err_msg = str(e)
            print(f"[LLM] Error: {err_msg}")
            
            # Pass the error to the UI so it's obvious when a quota limit is hit
            answer_text = f"{structured_insights}\n\n*(Analysis failed: {err_msg[:120]}...)*"
    else:
        answer_text = structured_insights

    return {
        "answer":        answer_text,
        "map_points":    map_points,
        "total_pixels":  total_pixels,
        "query_summary": str(query),
        "context":       context,
    }
