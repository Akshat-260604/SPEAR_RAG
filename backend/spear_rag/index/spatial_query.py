"""
spatial_query.py
────────────────
Core spatial retrieval utilities:
  - bbox_filter:        pandas mask on metadata parquet
  - load_embeddings:    reads rows from original .npy files (mmap, no copy)
  - summarize_modality: aggregates embeddings + raw feature stats for LLM context

No Zarr store needed — reads directly from the original .npy files.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(ROOT, "Models", "Checkpoints_SPEAR", "data")
INDEX_ROOT = os.path.join(ROOT, "spear_index")

NPY_FILES = {
    "climate": os.path.join(DATA_DIR, "climate_cls_embeddings.npy"),
    "s2":      os.path.join(DATA_DIR, "s2_cls_embeddings.npy"),
    "s1":      os.path.join(DATA_DIR, "s1_cls_embeddings.npy"),
    "planet":  os.path.join(DATA_DIR, "planet_cls_full.npy"),
}

# Cache memmaps so we don't re-open on every call
_MMAP_CACHE: Dict[str, np.ndarray] = {}

MODALITIES = ["climate", "s2", "s1", "planet"]

# Human-readable feature descriptions for LLM context
FEATURE_DESCRIPTIONS = {
    "elevation":           "Elevation (meters above sea level)",
    "LST_Day_1km":         "Land Surface Temperature - Day (°C)",
    "LST_Night_1km":       "Land Surface Temperature - Night (°C)",
    "total_precipitation": "Total Precipitation (mm)",
    "VH":                  "Sentinel-1 VH backscatter (SAR)",
    "VV":                  "Sentinel-1 VV backscatter (SAR)",
    "B2":                  "S2 Blue (490nm)",
    "B3":                  "S2 Green (560nm)",
    "B4":                  "S2 Red (665nm)",
    "B5":                  "S2 Red Edge 1 (705nm)",
    "B6":                  "S2 Red Edge 2 (740nm)",
    "B7":                  "S2 Red Edge 3 (783nm)",
    "B8":                  "S2 NIR (842nm)",
    "B8A":                 "S2 Narrow NIR (865nm)",
    "B11":                 "S2 SWIR 1 (1610nm)",
    "B12":                 "S2 SWIR 2 (2190nm)",
}

# Derived indices that can be computed from retrieved values
def compute_ndvi(meta_rows: pd.DataFrame) -> Optional[float]:
    if "B8" in meta_rows.columns and "B4" in meta_rows.columns:
        nir = meta_rows["B8"]
        red = meta_rows["B4"]
        denom = (nir + red).replace(0, np.nan)
        ndvi = ((nir - red) / denom).mean()
        return round(float(ndvi), 4) if not np.isnan(ndvi) else None
    return None

def compute_ndwi(meta_rows: pd.DataFrame) -> Optional[float]:
    """Normalized Difference Water Index — high values indicate water/flood."""
    if "B3" in meta_rows.columns and "B8" in meta_rows.columns:
        green = meta_rows["B3"]
        nir   = meta_rows["B8"]
        denom = (green + nir).replace(0, np.nan)
        ndwi  = ((green - nir) / denom).mean()
        return round(float(ndwi), 4) if not np.isnan(ndwi) else None
    return None


# ─── Core filtering ───────────────────────────────────────────────────────────

def load_meta(modality: str) -> pd.DataFrame:
    """Load the lightweight metadata parquet for a modality."""
    path = os.path.join(INDEX_ROOT, f"{modality}_meta.parquet")
    return pd.read_parquet(path)


def bbox_filter(
    meta: pd.DataFrame,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    year_start: Optional[int] = None,
    year_end:   Optional[int] = None,
    month:      Optional[int] = None,
) -> pd.DataFrame:
    """
    Fast pandas filter on metadata parquet.
    Returns filtered DataFrame with row_id column for subsequent embedding lookup.
    """
    mask = (
        (meta["lat"] >= lat_min) & (meta["lat"] <= lat_max) &
        (meta["lon"] >= lon_min) & (meta["lon"] <= lon_max)
    )
    if year_start is not None:
        mask &= (meta["year"] >= year_start)
    if year_end is not None:
        mask &= (meta["year"] <= year_end)
    if month is not None:
        mask &= (meta["month"] == month)

    return meta[mask].copy()


def _get_mmap(modality: str) -> np.ndarray:
    """Return memory-mapped view of the .npy embedding file. Cached."""
    if modality not in _MMAP_CACHE:
        npy_path = NPY_FILES[modality]
        _MMAP_CACHE[modality] = np.load(npy_path, mmap_mode="r")
    return _MMAP_CACHE[modality]


def load_embeddings(modality: str, row_ids: np.ndarray) -> np.ndarray:
    """
    Load specific rows from the .npy embedding file via memory-mapped indexing.
    row_ids: 1D int array of indices into the modality's embedding array.
    Returns: (K, 32) float32 array.
    """
    emb_mmap = _get_mmap(modality)
    unique_ids = np.unique(row_ids.astype(np.int64))
    # Fancy indexing on memmap — reads only needed chunks from disk
    return emb_mmap[unique_ids].astype(np.float32)


def summarize_modality(
    modality: str,
    filtered_meta: pd.DataFrame,
    max_rows: int = 50_000,
) -> Dict:
    """
    Aggregate embedding stats + raw feature stats for a modality's retrieved rows.
    Returns a dict ready for context_builder.
    """
    if filtered_meta.empty:
        return {"modality": modality, "n_retrieved": 0}

    # Sample if too many rows (for speed)
    if len(filtered_meta) > max_rows:
        filtered_meta = filtered_meta.sample(n=max_rows, random_state=42)

    row_ids = filtered_meta["row_id"].values.astype(np.int32)
    embs    = load_embeddings(modality, row_ids)

    # Embedding statistics
    centroid = embs.mean(axis=0).tolist()          # 32-dim mean
    emb_std  = embs.std(axis=0).mean()             # average across dims (scalar)
    emb_norm = float(np.linalg.norm(embs, axis=1).mean())

    # Raw feature statistics (from what's stored in meta)
    raw_stats: Dict[str, Dict] = {}
    feature_cols = [c for c in filtered_meta.columns
                    if c not in ("row_id", "lat", "lon", "year", "month", "classification")]
    for col in feature_cols:
        vals = filtered_meta[col].dropna()
        if len(vals) > 0:
            raw_stats[col] = {
                "mean":   round(float(vals.mean()), 4),
                "std":    round(float(vals.std()),  4),
                "min":    round(float(vals.min()),  4),
                "max":    round(float(vals.max()),  4),
                "description": FEATURE_DESCRIPTIONS.get(col, col),
            }

    # Derived indices
    ndvi = compute_ndvi(filtered_meta)
    ndwi = compute_ndwi(filtered_meta)

    # Spatial extent
    lat_range = (round(float(filtered_meta["lat"].min()), 4),
                 round(float(filtered_meta["lat"].max()), 4))
    lon_range = (round(float(filtered_meta["lon"].min()), 4),
                 round(float(filtered_meta["lon"].max()), 4))

    # Temporal distribution
    year_counts = filtered_meta["year"].value_counts().sort_index().to_dict()

    return {
        "modality":    modality,
        "n_retrieved": len(filtered_meta),
        "embedding": {
            "centroid":   centroid,
            "avg_std":    round(float(emb_std), 4),
            "avg_norm":   round(emb_norm, 4),
        },
        "raw_stats":   raw_stats,
        "ndvi":        ndvi,
        "ndwi":        ndwi,
        "lat_range":   lat_range,
        "lon_range":   lon_range,
        "year_counts": year_counts,
        "map_points":  filtered_meta[["lat", "lon", "year"]].to_dict(orient="records"),
    }


def query_all_modalities(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    year_start: Optional[int] = None,
    year_end:   Optional[int] = None,
    month:      Optional[int] = None,
    modalities: List[str] = None,
    max_rows: int = 50_000,
) -> Dict[str, Dict]:
    """
    Run spatial query across all (or specified) modalities.
    Returns dict keyed by modality name with summarize_modality results.
    """
    if modalities is None:
        modalities = MODALITIES

    results = {}
    for mod in modalities:
        meta_path = os.path.join(INDEX_ROOT, f"{mod}_meta.parquet")
        if not os.path.exists(meta_path):
            print(f"  [WARN] No metadata found for '{mod}' — skipping.")
            continue

        meta     = load_meta(mod)
        filtered = bbox_filter(meta, lat_min, lat_max, lon_min, lon_max,
                               year_start, year_end, month)
        print(f"  [{mod}] {len(filtered):,} pixels in bbox")
        summary  = summarize_modality(mod, filtered, max_rows=max_rows)
        results[mod] = summary

    return results
