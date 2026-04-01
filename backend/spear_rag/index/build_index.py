"""
build_index.py
──────────────
Builds a FAISS IndexFlatL2 per modality from the Zarr embedding store.
Saves: {modality}.faiss  (vector index, 32-dim)

The metadata parquet already built by build_zarr.py is used at query time
for spatial (bbox + year) filtering — no re-indexing needed.

Usage:
    python -m spear_rag.index.build_index                 # all modalities
    python -m spear_rag.index.build_index --modality s2   # single
"""

from __future__ import annotations
import argparse
import os
import time
import numpy as np
import zarr
import faiss

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZARR_ROOT  = os.path.join(ROOT, "spear_zarr")
INDEX_ROOT = os.path.join(ROOT, "spear_index")

MODALITIES = ["climate", "s2", "s1", "planet"]


def build_faiss(modality: str) -> None:
    zarr_path  = os.path.join(ZARR_ROOT, modality)
    index_path = os.path.join(INDEX_ROOT, f"{modality}.faiss")

    print(f"\n[{modality.upper()}] Building FAISS index…")

    store = zarr.open_group(zarr_path, mode="r")
    emb   = store["embeddings"][:]          # (N, 32) float32
    emb   = np.ascontiguousarray(emb, dtype=np.float32)

    N, D = emb.shape
    print(f"  Loaded embeddings: {N:,} × {D}")

    # Normalize for cosine similarity (optional, works well for cluster comparisons)
    # Using L2 flat index — exact search, fast enough at 32-dim with 3M rows
    index = faiss.IndexFlatL2(D)
    print(f"  Training/adding {N:,} vectors…")
    t0 = time.time()
    index.add(emb)
    elapsed = time.time() - t0
    print(f"  Added {index.ntotal:,} vectors in {elapsed:.1f}s")

    os.makedirs(INDEX_ROOT, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"  FAISS index saved → {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for SPEAR modalities")
    parser.add_argument("--modality", default="all",
                        choices=["all"] + MODALITIES)
    args = parser.parse_args()

    targets = MODALITIES if args.modality == "all" else [args.modality]
    for m in targets:
        build_faiss(m)

    print("\n✅  FAISS indexes saved to:", INDEX_ROOT)


if __name__ == "__main__":
    main()
