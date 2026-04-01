"""
map_viz.py
──────────
Builds a Folium interactive map from retrieved satellite data points.
Returns an HTML string to embed in the frontend.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _FOLIUM = True
except ImportError:
    _FOLIUM = False


MODALITY_COLORS = {
    "climate": "#00b4d8",
    "s2":      "#52b788",
    "s1":      "#f4a261",
    "planet":  "#e76f51",
}

LANDCOVER_LABELS = {
    0: "Water",
    1: "Trees / Forest",
    2: "Grass",
    3: "Flooded Vegetation",
    4: "Crops / Agriculture",
    5: "Shrub & Scrub",
    6: "Built Area",
    7: "Bare Ground",
    8: "Snow & Ice",
}


def _make_popup(p: dict) -> str:
    """Build an HTML popup string for a single data point."""
    mod   = p.get("modality", "?").upper()
    year  = p.get("year", "?")
    cls   = p.get("classification")
    ndvi  = p.get("ndvi")
    ndwi  = p.get("ndwi")
    lat   = p.get("lat", 0)
    lon   = p.get("lon", 0)

    cls_label = LANDCOVER_LABELS.get(int(cls), f"Class {cls}") if cls is not None else "—"

    rows = [
        f"<tr><td><b>Sensor</b></td><td>{mod}</td></tr>",
        f"<tr><td><b>Year</b></td><td>{year}</td></tr>",
        f"<tr><td><b>Land cover</b></td><td>{cls_label}</td></tr>",
        f"<tr><td><b>Lat / Lon</b></td><td>{lat:.4f}, {lon:.4f}</td></tr>",
    ]
    if ndvi is not None:
        rows.append(f"<tr><td><b>NDVI</b></td><td>{ndvi:.3f}</td></tr>")
    if ndwi is not None:
        rows.append(f"<tr><td><b>NDWI</b></td><td>{ndwi:.3f}</td></tr>")

    return (
        "<table style='font-size:13px;border-collapse:collapse;min-width:180px'>"
        + "".join(rows)
        + "</table>"
    )


def build_map(
    map_points: List[Dict],
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: int = 8,
    heatmap: bool = True,
) -> str:
    """
    Build an interactive Folium map from retrieved spatial points.

    Args:
        map_points: List of {lat, lon, year, modality, ...} dicts
        center_lat/lon: Map center (auto-computed if None)
        zoom: Initial zoom level
        heatmap: If True, render HeatMap + clickable markers; else MarkerCluster only

    Returns:
        HTML string suitable for iframe embedding
    """
    if not _FOLIUM:
        return "<p>Install folium: <code>pip install folium</code></p>"

    if not map_points:
        return "<p>No spatial data points to display for this query.</p>"

    # Auto-center
    if center_lat is None:
        center_lat = sum(p["lat"] for p in map_points) / len(map_points)
    if center_lon is None:
        center_lon = sum(p["lon"] for p in map_points) / len(map_points)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )

    modalities_seen = list({p.get("modality", "unknown") for p in map_points})

    if heatmap:
        # ── Heatmap layer per modality ────────────────────────────────────────
        for mod in modalities_seen:
            pts = [[p["lat"], p["lon"]] for p in map_points if p.get("modality") == mod]
            if pts:
                HeatMap(
                    pts,
                    name=f"{mod.upper()} Heatmap",
                    radius=10,
                    blur=15,
                    max_zoom=13,
                    gradient={0.2: MODALITY_COLORS.get(mod, "#ffffff"),
                               1.0: "#ffffff"},
                    show=True,
                ).add_to(m)

    # ── Clickable circle markers (sampled for performance) ────────────────────
    for mod in modalities_seen:
        mod_pts = [p for p in map_points if p.get("modality") == mod]
        # Sample up to 1000 per modality to keep the page responsive
        sample = mod_pts[:1000]
        color  = MODALITY_COLORS.get(mod, "#ffffff")
        fg     = folium.FeatureGroup(name=f"{mod.upper()} Pixels", show=False)
        for p in sample:
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=1,
                popup=folium.Popup(_make_popup(p), max_width=260),
                tooltip=f"{mod.upper()} — click for details",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m._repr_html_()


def build_map_from_results(
    retrieval_results: Dict[str, Dict],
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: int = 8,
) -> str:
    """Convenience wrapper that extracts map_points from retrieve() output."""
    all_points: List[Dict] = []
    for mod, result in retrieval_results.items():
        pts = result.get("map_points", [])[:2500]
        for p in pts:
            p["modality"] = mod
        all_points.extend(pts)

    return build_map(all_points, center_lat, center_lon, zoom)
