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
        map_points: List of {lat, lon, year, modality} dicts
        center_lat/lon: Map center (auto-computed if None)
        zoom: Initial zoom level
        heatmap: If True, render HeatMap; else MarkerCluster

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

    if heatmap:
        # One HeatMap layer per modality
        modalities_seen = list({p.get("modality", "unknown") for p in map_points})
        for mod in modalities_seen:
            pts = [[p["lat"], p["lon"]] for p in map_points if p.get("modality") == mod]
            if pts:
                HeatMap(
                    pts,
                    name=mod.upper(),
                    radius=10,
                    blur=15,
                    max_zoom=13,
                    gradient={0.2: MODALITY_COLORS.get(mod, "#ffffff"),
                               1.0: "#ffffff"},
                ).add_to(m)
    else:
        cluster = MarkerCluster().add_to(m)
        for p in map_points[:2000]:   # cap markers for performance
            color = MODALITY_COLORS.get(p.get("modality", ""), "#ffffff")
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{p.get('modality','?')} | year={p.get('year','?')}",
            ).add_to(cluster)

    folium.LayerControl().add_to(m)

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
