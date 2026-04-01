"""
build_zarr.py  (meta-only — no Zarr, no disk duplication)
──────────────────────────────────────────────────────────
Builds per-modality metadata parquets by joining:
  - pre-computed .npy embeddings  (to determine valid row count)
  - parquet source data           (Lat, Lon, randomDate, raw features)

Output: spear_index/{modality}_meta.parquet
  - Contains: row_id, lat, lon, year, month, classification, raw features
  - Used for fast spatial bbox+time filtering at query time
  - Original .npy files are read directly via memmap at retrieval time

Usage:
    python -m spear_rag.data.build_zarr                   # all modalities
    python -m spear_rag.data.build_zarr --modality s2     # single modality
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARQUET_PATH = os.path.join(ROOT, "merged_file.parquet")
DATA_DIR     = os.path.join(ROOT, "Models", "Checkpoints_SPEAR", "data")
INDEX_ROOT   = os.path.join(ROOT, "spear_index")

# ─── Modality configs ─────────────────────────────────────────────────────────
MODALITIES = {
    "climate": {
        "npy": os.path.join(DATA_DIR, "climate_cls_embeddings.npy"),
        "valid_cols":   ["elevation", "LST_Day_1km", "LST_Night_1km", "total_precipitation"],
        "feature_cols": ["elevation", "LST_Day_1km", "LST_Night_1km", "total_precipitation"],
    },
    "s2": {
        "npy": os.path.join(DATA_DIR, "s2_cls_embeddings.npy"),
        "valid_cols":   ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        "feature_cols": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    },
    "s1": {
        "npy": os.path.join(DATA_DIR, "s1_cls_embeddings.npy"),
        "valid_cols":   ["VH", "VV"],
        "feature_cols": ["VH", "VV"],
    },
    "planet": {
        "npy": os.path.join(DATA_DIR, "planet_cls_full.npy"),
        "valid_cols":   ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        "feature_cols": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    },
}


def build_meta(name: str, cfg: dict, df_full: pd.DataFrame) -> None:
    """Build and save the metadata parquet for a single modality."""
    print(f"\n{'='*60}")
    print(f"  Modality: {name.upper()}")
    print(f"{'='*60}")

    # Check if already built
    meta_path = os.path.join(INDEX_ROOT, f"{name}_meta.parquet")
    if os.path.exists(meta_path):
        existing = pd.read_parquet(meta_path)
        print(f"  Already exists ({len(existing):,} rows) → {meta_path}  [skip]")
        return

    # Load embedding shape (get N without loading full array)
    emb_header = np.load(cfg["npy"], mmap_mode="r")
    N_emb = emb_header.shape[0]
    print(f"  .npy rows: {N_emb:,}  embed_dim: {emb_header.shape[1]}")

    # Build valid-row mask
    valid_mask = df_full[cfg["valid_cols"]].notna().all(axis=1)
    df_valid   = df_full[valid_mask].reset_index(drop=True)
    N_valid    = len(df_valid)
    print(f"  Parquet valid rows for {name}: {N_valid:,}")

    if N_emb > N_valid:
        raise ValueError(
            f"[{name}] .npy has {N_emb} rows but only {N_valid} valid parquet rows."
        )

    # Row-aligned trim
    df_mod = df_valid.iloc[:N_emb].copy()

    # Parse date
    dates = pd.to_datetime(df_mod["randomDate"], errors="coerce")
    year  = dates.dt.year.fillna(0).astype(np.int16).values
    month = dates.dt.month.fillna(0).astype(np.int8).values
    lat   = df_mod["Lat"].astype(np.float32).values
    lon   = df_mod["Lon"].astype(np.float32).values
    cls   = df_mod["classification"].astype(np.int8).values

    # Assemble metadata
    meta_cols = {
        "row_id": np.arange(N_emb, dtype=np.int32),
        "lat":    lat,
        "lon":    lon,
        "year":   year,
        "month":  month,
        "classification": cls,
    }
    for fc in cfg["feature_cols"]:
        if fc in df_mod.columns:
            meta_cols[fc] = df_mod[fc].astype(np.float32).values

    os.makedirs(INDEX_ROOT, exist_ok=True)
    pd.DataFrame(meta_cols).to_parquet(meta_path, index=False)
    print(f"  ✅  Saved → {meta_path}  ({N_emb:,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Build SPEAR metadata parquets")
    parser.add_argument("--modality", default="all",
                        choices=["all", "climate", "s2", "s1", "planet"])
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if already exists")
    args = parser.parse_args()

    print(f"[IO] Loading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[IO] Parquet loaded: {df.shape}")

    targets = MODALITIES if args.modality == "all" else {args.modality: MODALITIES[args.modality]}

    # If force, remove existing
    if args.force:
        for name in targets:
            p = os.path.join(INDEX_ROOT, f"{name}_meta.parquet")
            if os.path.exists(p):
                os.remove(p)

    for name, cfg in targets.items():
        build_meta(name, cfg, df)

    print(f"\n✅  Metadata parquets ready in: {INDEX_ROOT}")
    for name in targets:
        p = os.path.join(INDEX_ROOT, f"{name}_meta.parquet")
        if os.path.exists(p):
            size = os.path.getsize(p) / 1e6
            print(f"   {name}_meta.parquet  {size:.1f} MB")


if __name__ == "__main__":
    main()
