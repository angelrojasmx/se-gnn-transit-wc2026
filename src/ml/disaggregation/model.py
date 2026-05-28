"""
Spatial-temporal GNN for hourly transit demand disaggregation.

Given per-station daily ridership totals and a set of contextual features,
the model reconstructs the normalized 24-hour profile for each station:
a probability distribution over hours that sums to 1.0 per station.

Architecture:
    Feature projection  →  k-layer GCN with residual connections  →  MLP decoder  →  softmax
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree


def build_adjacency(stations_df: pd.DataFrame, k: int = 8) -> torch.Tensor:
    """
    Construct the symmetric normalised adjacency matrix A_hat for graph convolution.

    Uses k-nearest-neighbour geographic proximity (haversine distance) to define
    graph edges. Self-loops are added before symmetric normalisation
    (D^{-1/2} A D^{-1/2}) following Kipf & Welling (2017).

    Args:
        stations_df: DataFrame with columns 'latitude' and 'longitude',
                     rows aligned to the station ordering used throughout training.
        k:           Number of geographic neighbours per station.

    Returns:
        A_hat: FloatTensor (N, N) — normalised adjacency matrix.
    """
    N = len(stations_df)
    coords_rad = np.deg2rad(
        stations_df[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    )

    k = min(k, N - 1)
    tree = BallTree(coords_rad, metric="haversine")
    _, neighbours = tree.query(coords_rad, k=k + 1)   # k+1 includes self

    A = np.zeros((N, N), dtype=np.float32)
    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0

    A += np.eye(N, dtype=np.float32)                        # self-loops
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A_hat, dtype=torch.float32)


class DisaggregationGNN(nn.Module):
    """
    Graph neural network for temporal disaggregation of daily transit demand.

    Input node features (N_FEATURES = 7) per station per day:
        0  daily_total_norm     log-normalised total ridership relative to daily maximum
        1  is_event             binary event indicator
        2  sin_dow              sine encoding of day-of-week
        3  cos_dow              cosine encoding of day-of-week
        4  dist_to_venue_norm   normalised haversine distance to the primary event venue
        5  is_hub               1 if station median daily ridership exceeds the 80th percentile
                                of the baseline distribution (computed from normal days only)
        6  ridership_ratio      log(today_total + 1) / log(baseline_median + 1) by (sid, dow);
                                values > 1 indicate above-baseline demand
    """

    N_FEATURES = 7
    HORIZON    = 24

    def __init__(
        self,
        hidden:       int   = 64,
        dropout:      float = 0.2,
        n_gcn_layers: int   = 2,
    ):
        super().__init__()

        self.hidden       = hidden
        self.n_gcn_layers = n_gcn_layers

        self.input_proj = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_gcn_layers)]
        )
        self.gcn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_gcn_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.HORIZON),
        )

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N, N_FEATURES) node feature matrix for one day.
            A_hat: (N, N) normalised adjacency matrix (constant across days).

        Returns:
            profile: (N, 24) normalised hourly demand distribution.
                     Each row sums to 1.0 (softmax output).
        """
        h = self.input_proj(x)

        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_agg = A_hat @ h                        # spatial message passing
            h_new = self.dropout(torch.relu(gcn(h_agg)))
            h     = norm(h + h_new)                  # residual connection

        return torch.softmax(self.decoder(h), dim=-1)


def build_node_features(
    date,
    stations_df:         pd.DataFrame,
    baseline_df:         pd.DataFrame,
    daily_totals_series: pd.Series,
    venue_coords:        tuple,
    event_flag:          int               = 0,
    device:              str               = "cpu",
    baseline_daily_df:   Optional[pd.DataFrame] = None,
) -> torch.Tensor:
    """
    Construct the (N, 7) node feature tensor for a single day.

    Feature 5 (is_hub) is derived from the per-station median daily ridership
    computed on *non-event days only* (see build_dataset.py), so it does not
    introduce leakage on event days.

    Args:
        date:                datetime.date or ISO string.
        stations_df:         DataFrame with columns sid, latitude, longitude.
        baseline_df:         DataFrame with columns sid, dow, hour, mean_profile.
        daily_totals_series: Series mapping sid → observed ridership for this day.
        venue_coords:        (lat, lon) of the primary event venue.
        event_flag:          0 or 1.
        device:              Torch device string.
        baseline_daily_df:   DataFrame with columns sid, dow, median_daily_total
                             (output of build_dataset.py). Required for Feature 6.

    Returns:
        FloatTensor (N, 7) on the requested device.
    """
    N   = len(stations_df)
    ts  = pd.Timestamp(date)
    dow = ts.dayofweek

    features = np.zeros((N, DisaggregationGNN.N_FEATURES), dtype=np.float32)

    totals = (
        daily_totals_series
        .reindex(stations_df["sid"].values)
        .fillna(0.0)
        .values
        .astype(np.float32)
    )

    # Feature 0: log-normalised daily ridership
    log_totals = np.log1p(totals)
    features[:, 0] = log_totals / (log_totals.max() + 1e-8)

    # Feature 1: event indicator
    features[:, 1] = float(event_flag)

    # Features 2–3: cyclic day-of-week encoding
    features[:, 2] = np.sin(2 * math.pi * dow / 7)
    features[:, 3] = np.cos(2 * math.pi * dow / 7)

    # Feature 4: normalised haversine distance to venue
    vlat, vlon = venue_coords
    vlat_r, vlon_r = math.radians(vlat), math.radians(vlon)

    def _haversine_km(lat: float, lon: float) -> float:
        dlat = math.radians(lat) - vlat_r
        dlon = math.radians(lon) - vlon_r
        a = math.sin(dlat / 2) ** 2 + (
            math.cos(math.radians(lat)) * math.cos(vlat_r) * math.sin(dlon / 2) ** 2
        )
        return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dists = np.array(
        [_haversine_km(r.latitude, r.longitude) for r in stations_df.itertuples()],
        dtype=np.float32,
    )
    features[:, 4] = dists / (dists.max() + 1e-8)

    # Pre-compute baseline lookups once — shared by Features 5 and 6.
    if baseline_daily_df is not None:
        bl_map = (
            baseline_daily_df[baseline_daily_df["dow"] == dow]
            .set_index("sid")["median_daily_total"]
        )
        baseline_totals: Optional[np.ndarray] = np.array(
            [bl_map.get(sid, 0.0) for sid in stations_df["sid"].values],
            dtype=np.float32,
        )
    else:
        baseline_totals = None

    # Feature 5: hub indicator — based on baseline medians, not today's totals,
    # to avoid leakage on high-ridership event days.
    if baseline_totals is not None:
        hub_threshold = np.percentile(baseline_totals, 80)
        features[:, 5] = (baseline_totals >= hub_threshold).astype(np.float32)
    else:
        # Fallback: use today's totals if baseline is unavailable.
        # Note: this introduces mild leakage on event days.
        hub_threshold = np.percentile(totals, 80)
        features[:, 5] = (totals >= hub_threshold).astype(np.float32)

    # Feature 6: ratio of today's ridership to the day-of-week baseline.
    # Values > 1 indicate above-baseline demand (e.g., event days).
    if baseline_totals is not None:
        log_baseline = np.log1p(baseline_totals)
        features[:, 6] = np.clip(np.log1p(totals) / (log_baseline + 1e-8), 0.0, 3.0)
    else:
        features[:, 6] = features[:, 0]   # degrade gracefully to Feature 0

    return torch.tensor(features, dtype=torch.float32, device=device)
