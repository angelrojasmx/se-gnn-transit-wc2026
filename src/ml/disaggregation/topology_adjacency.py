"""
ml/disaggregation/topology_adjacency.py — I3 (Reviewer 2.1) sensitivity analysis

Builds a track/line-topology adjacency for the Manhattan station graph from the
NYC subway GTFS feed (legacy/gtfsNYC/gtfs_feeds/gtfs_subway), as an alternative
to the purely geographic k-NN adjacency used in build_adjacency() (model.py).

Definition: two stations (sid_i, sid_j) are connected if, on at least one GTFS
trip, their corresponding stops are consecutive (adjacent) in stop_sequence —
i.e. there is a direct track segment between them on some subway line. This is
the standard notion of "line/track topology" as opposed to straight-line
geographic proximity, and directly answers Reviewer 2's request for a
sensitivity comparison.

station_lookup.csv "sid" values are NOT GTFS stop_ids (they come from a
separate MTA complex/station numbering). We join by station name (stripping
the "(N,R,W)"-style route suffix from station_lookup names) against GTFS
parent stops (location_type == 1).
"""

from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

GTFS_DIR = Path(__file__).resolve().parents[3] / "legacy" / "gtfsNYC" / "gtfs_feeds" / "gtfs_subway"


def _normalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # strip trailing "(N,R,W)"
    return name.strip().lower()


def build_sid_to_stopids(stations_df: pd.DataFrame, gtfs_dir: Path = GTFS_DIR) -> dict[int, list[str]]:
    """Map dataset sid -> list of GTFS parent stop_ids sharing the same station name."""
    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str)
    stops["location_type"] = pd.to_numeric(stops["location_type"], errors="coerce")
    parents = stops[stops["location_type"] == 1].copy()
    parents["norm_name"] = parents["stop_name"].apply(_normalize_name)

    name_to_stopids: dict[str, list[str]] = defaultdict(list)
    for _, row in parents.iterrows():
        name_to_stopids[row["norm_name"]].append(row["stop_id"])

    sid_to_stopids: dict[int, list[str]] = {}
    unmatched = []
    for _, row in stations_df.iterrows():
        sid = int(row["sid"])
        norm = _normalize_name(row["name"])
        stopids = name_to_stopids.get(norm, [])
        if not stopids:
            unmatched.append((sid, row["name"]))
        sid_to_stopids[sid] = stopids

    if unmatched:
        print(f"  [topology_adjacency] AVISO: {len(unmatched)}/{len(stations_df)} estaciones sin match GTFS por nombre:")
        for sid, name in unmatched[:10]:
            print(f"      sid={sid}  '{name}'")

    return sid_to_stopids


def build_track_edges(gtfs_dir: Path = GTFS_DIR) -> set[tuple[str, str]]:
    """Return set of (stop_id_a, stop_id_b) unordered pairs that are consecutive on
    some GTFS trip (i.e. connected by a direct track segment)."""
    stop_times = pd.read_csv(
        gtfs_dir / "stop_times.txt", dtype=str,
        usecols=["trip_id", "stop_id", "stop_sequence"],
    )
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"])
    # Strip N/S direction suffix to parent stop id (platforms share the same parent)
    stop_times["parent_stop"] = stop_times["stop_id"].str.replace(r"[NS]$", "", regex=True)

    edges: set[tuple[str, str]] = set()
    # Group by trip, look at consecutive rows sorted by stop_sequence
    for _, grp in stop_times.groupby("trip_id", sort=False):
        grp = grp.sort_values("stop_sequence")
        seq = grp["parent_stop"].tolist()
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return edges


def build_adjacency_topology(stations_df: pd.DataFrame, gtfs_dir: Path = GTFS_DIR) -> torch.Tensor:
    """
    Construye la matriz de adjacencia normalizada A_hat basada en topología real
    de líneas (track-adjacency), como alternativa al k-NN geográfico.

    Returns:
        A_hat_t: torch.FloatTensor (N, N)
    """
    N = len(stations_df)
    sids = stations_df["sid"].astype(int).tolist()
    sid_index = {sid: i for i, sid in enumerate(sids)}

    sid_to_stopids = build_sid_to_stopids(stations_df, gtfs_dir)
    stopid_to_sid: dict[str, int] = {}
    for sid, stopids in sid_to_stopids.items():
        for sp in stopids:
            stopid_to_sid[sp] = sid

    track_edges = build_track_edges(gtfs_dir)

    A = np.zeros((N, N), dtype=np.float32)
    n_edges = 0
    for a, b in track_edges:
        sid_a = stopid_to_sid.get(a)
        sid_b = stopid_to_sid.get(b)
        if sid_a is not None and sid_b is not None and sid_a != sid_b:
            i, j = sid_index[sid_a], sid_index[sid_b]
            if A[i, j] == 0:
                n_edges += 1
            A[i, j] = 1.0
            A[j, i] = 1.0

    print(f"  [topology_adjacency] {n_edges} aristas track-topology entre las {N} estaciones "
          f"({sum(1 for s in sid_to_stopids.values() if s)}/{N} con match GTFS)")

    # Aislar nodos sin ninguna arista: fallback a self-loop-only (degree 1, la
    # normalización de abajo los deja como "isla" conectada solo a sí misma).
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A_hat, dtype=torch.float32)


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[3]
    stations = pd.read_csv(ROOT / "data/nyc/manhattan/station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)
    A_topo = build_adjacency_topology(stations)
    print("A_topo shape:", A_topo.shape)

    from model import build_adjacency
    A_geo = build_adjacency(stations, k=8)

    # Compare edge overlap (off-diagonal, before self-loop, thresholded)
    topo_bin = (A_topo.numpy() > 0.01).astype(int)
    geo_bin  = (A_geo.numpy()  > 0.01).astype(int)
    np.fill_diagonal(topo_bin, 0)
    np.fill_diagonal(geo_bin, 0)
    overlap = (topo_bin & geo_bin).sum()
    print(f"Topology edges (binarized, incl. both directions): {topo_bin.sum()}")
    print(f"Geographic k-NN edges: {geo_bin.sum()}")
    print(f"Overlap: {overlap}")
