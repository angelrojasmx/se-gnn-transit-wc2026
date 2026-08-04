"""
SE-GNN LOTO-CV for CDMX Metro spatial demand disaggregation.

Unlike NYC (temporal profile over 24 h per station), the CDMX problem is
spatial: given an event at a venue, which of the 196 Metro stations absorb
the additional ridership?  The model predicts a normalized spatial weight
vector w in R^N (summing to 1) rather than a 24-h temporal profile.

LOTO-CV split design:
  4 event types x [24, 54, 120, 48] event-days:
  Fold 1 - TEST: motorsport (F1, 24 days)
           TRAIN: concert + music_festival + soccer
  Fold 2 - TEST: music_festival (Vive Latino / Corona Capital, 54 days)
           TRAIN: motorsport + concert + soccer
  Fold 3 - TEST: concert (Foro Sol / Palacio de los Deportes, 120 days)
           TRAIN: motorsport + music_festival + soccer
  Fold 4 - TEST: soccer_domestic (Liga MX Azteca + Pumas UNAM, 48 days)
           TRAIN: motorsport + concert + music_festival

Evaluation metric: venue-aware MAE.
  Error is computed only over stations in the venue's catchment zone, not
  across all 196 stations.  Out-of-catchment error is noise, not signal.

Baselines:
  1. Historical DOW Average: same day-of-week, 4 weeks prior
  2. GNN backbone (no FiLM conditioning)
  3. SE-GNN with FiLM conditioning (proposed model)

Outputs (outputs/paper/cdmx/):
  cdmx_loto_results.csv     - per-fold per-event MAE
  cdmx_loto_summary.csv     - aggregated table for paper
  cdmx_wc2026_spatial.csv   - projected spatial distribution for WC2026 matches
  outputs/paper/comparative/figures/
    fig_cdmx_loto_cv.png
    fig_cdmx_wc2026_spatial.png
"""

from __future__ import annotations

import sys
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.ml.disaggregation.event_encoder import EventEncoder, EMBEDDING_DIM

# Paths
DATA_DIR   = ROOT / "data" / "cdmx" / "metro"
CATALOG_PATH = ROOT / "data" / "cdmx" / "event_catalog_2016_2025.csv"
OUT_CDMX   = ROOT / "outputs" / "paper" / "cdmx"
OUT_FIGS   = ROOT / "outputs" / "paper" / "comparative" / "figures"
EMB_CACHE  = OUT_CDMX / "cdmx_event_embeddings_cache.json"

OUT_CDMX.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Venue-station catchment mapping (station names, analogous to NYC VENUE_SIDS)
# For venue-aware MAE: evaluate only on stations within catchment
VENUE_CATCHMENT: dict[str, list[str]] = {
    "Ciudad Deportiva": [
        "ciudad deportiva",   # L9 - primary
        "veldromo",           # L9 - adjacent
        "mixiuhca",           # L9 - adjacent
        "jamaica",            # L4/L9 transfer
    ],
    "Azteca": [
        "mexicaltzingo",      # L12
        "ermita",             # L12/L2
        "atlalilco",          # L12/L8
        "tasquea",            # L2 terminus
    ],
    "Pumas/CU": [
        "copilco",                    # L3
        "universidad",                # L3
        "viveros/derechos humanos",   # L3 - adjacent (full name in station_lookup.csv)
        "miguel ngel de quevedo",     # L3 - adjacent (accent-stripped in station_lookup.csv)
    ],
}

# Event type → primary venue (for baseline and WC2026 inference)
TYPE_TO_VENUE: dict[str, str] = {
    "motorsport":       "Ciudad Deportiva",
    "music_festival":   "Ciudad Deportiva",
    "concert":          "Ciudad Deportiva",
    "soccer_domestic":  "Azteca",   # both Azteca and Pumas, primary = Azteca
}

# Event catalog - text descriptions for LLM encoder
# Parallel structure to data/event_catalog.py in NYC
CDMX_EVENT_DESCRIPTIONS: dict[str, str] = {
    "motorsport": (
        "Formula 1 Mexico City Grand Prix, Autodromo Hermanos Rodriguez, "
        "Magdalena Mixhuca sports complex, Mexico City. Capacity ~130,000 spectators. "
        "3-day race weekend: Friday practice, Saturday qualifying, Sunday race. "
        "Major international motorsport event with global TV audience. "
        "High proportion of local and international fans traveling by Metro Line 9 "
        "(Ciudad Deportiva station) and by private vehicle. "
        "Race day draws the largest crowd; qualifying Saturday often also very busy. "
        "Event generates significant outbound demand via Metro in the afternoon "
        "and return demand post-race in the evening."
    ),
    "music_festival": (
        "Major outdoor music festival, Autodromo Hermanos Rodriguez or Foro Aztlan, "
        "Magdalena Mixhuca sports complex, Mexico City. Vive Latino or Lollapalooza "
        "or Corona Capital. Capacity 70,000–100,000 per day. Multi-day event "
        "held over a weekend in March or November. "
        "Predominantly young local fans traveling by Metro Line 9. "
        "Adjacent to Ciudad Deportiva metro station (Line 9). "
        "Heavy inbound demand midday, heavy outbound demand late evening/night. "
        "International headliners attract attendees from across the metro area."
    ),
    "concert": (
        "Large-scale concert or sports event at Foro Sol amphitheater or "
        "Palacio de los Deportes arena, Ciudad Deportiva complex, Mexico City. "
        "Capacity 65,000 (Foro Sol outdoor) or 22,000 (Palacio Deportes indoor). "
        "Mix of national and international artists. "
        "Elevated Metro Line 9 ridership at Ciudad Deportiva station. "
        "Demand concentrated in afternoon and evening hours. "
        "Fans travel from across the metropolitan area."
    ),
    "soccer_domestic": (
        "Liga MX home match, Estadio Azteca, Mexico City. "
        "Club América (primary tenant) vs domestic rival. "
        "Capacity 87,000. Sunday afternoon or evening kickoff. "
        "Fans travel via Metro Line 12 (Mexicaltzingo, Ermita, Atlalilco stations) "
        "and Metro Line 2 (Tasqueña terminus). "
        "Strong local fan base concentrated in southern Mexico City districts. "
        "Return ridership peaks approximately 2 hours after kickoff. "
        "Elevated demand also at Pumas UNAM home games at Estadio Olimpico "
        "Universitario, served by Metro Line 3 (Copilco, Universidad stations)."
    ),
    # WC2026-specific description for forecast
    "wc2026_azteca": (
        "FIFA World Cup 2026, Estadio Azteca, Mexico City, Mexico. "
        "One of three host stadiums for Mexico (with Guadalajara and Monterrey). "
        "Capacity 87,000. Mexico national team home matches in Group A. "
        "Highest-attendance international soccer tournament in history. "
        "Extraordinary demand expected across all Metro lines, especially "
        "Line 12 (Mexicaltzingo, Ermita, Atlalilco) and Line 2 (Tasqueña). "
        "Mexico City has 22M metropolitan population with strong soccer culture. "
        "Match against South Africa on June 11 is the opening ceremony game. "
        "Demand expected to significantly exceed typical Liga MX match levels "
        "due to international visitors, tourist traffic, and national significance."
    ),
}

# Spatial GNN Architecture
# Analog of EventConditionedGNN but outputs spatial weights (N,1) instead of
# temporal profile (N,24). The "horizon" is 1 (daily demand weight per station).

class SpatialDisaggregationGNN(nn.Module):
    """
    GNN for spatial disaggregation of transit demand on event days.

    Given per-station features and an optional event embedding, predicts a
    spatial weight per station reflecting the distribution of event-induced
    demand across the Metro network.

    Args:
        x         : (N, F) - node feature matrix
        A_hat     : (N, N) - symmetrically normalised adjacency (CDMX Metro graph)
        event_emb : (emb_dim,) optional - sentence embedding for FiLM conditioning

    Returns:
        weights : (N,) - normalised spatial distribution (sums to 1.0)

    Input features (F = 6):
        0: log(baseline_riders + 1) normalised - station baseline ridership
        1: log(distance_km + 1)     - Haversine distance to event venue
        2: is_venue_line             - 1 if station is on the venue's primary Metro line
        3: sin(2*pi * dow / 7)       - cyclic day-of-week encoding
        4: cos(2*pi * dow / 7)
        5: log(scaled_baseline + 1) normalised - system-wide uniform uplift signal
    """
    N_FEATURES = 6

    def __init__(
        self,
        hidden:        int   = 64,
        dropout:       float = 0.2,
        n_gcn_layers:  int   = 2,
        event_emb_dim: int   = 384,   # all-MiniLM-L6-v2 output dim
        film_hidden:   int   = 64,
    ):
        super().__init__()
        self.hidden        = hidden
        self.event_emb_dim = event_emb_dim
        self.n_gcn_layers  = n_gcn_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM layers - only these are fine-tuned; backbone is frozen
        self.film_scale = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden),
        )
        self.film_shift = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden),
        )

        # GCN layers: h_new = ReLU(A_hat @ h @ W)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_gcn_layers)
        ])
        self.gcn_norms  = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_gcn_layers)
        ])
        self.dropout    = nn.Dropout(dropout)

        # Decoder: (N, hidden) → (N, 1) → softmax over N
        self.decoder = nn.Linear(hidden, 1)

    def film_parameters(self):
        """Return FiLM parameters only - used for fine-tuning with backbone frozen."""
        return list(self.film_scale.parameters()) + list(self.film_shift.parameters())

    def backbone_parameters(self):
        params = (
            list(self.input_proj.parameters()) +
            list(self.gcn_layers.parameters()) +
            list(self.gcn_norms.parameters()) +
            list(self.decoder.parameters())
        )
        return params

    def forward(
        self,
        x:         torch.Tensor,              # (N, F)
        A_hat:     torch.Tensor,              # (N, N)
        event_emb: torch.Tensor | None = None, # (emb_dim,)
    ) -> torch.Tensor:                         # (N,) spatial weights summing to 1
        h = self.input_proj(x)  # (N, hidden)

        # FiLM conditioning (optional)
        if event_emb is not None:
            emb = event_emb.unsqueeze(0)  # (1, emb_dim)
            gamma = self.film_scale(emb)  # (1, hidden) - broadcast
            beta  = self.film_shift(emb)
            h = h * gamma + beta

        # GCN message passing
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_new = torch.relu(gcn(A_hat @ h))
            h     = norm(h + self.dropout(h_new))  # residual

        # Decode to spatial weights
        logits  = self.decoder(h).squeeze(-1)  # (N,)
        weights = torch.softmax(logits, dim=0) # (N,) sums to 1
        return weights


# Graph construction

def build_metro_adjacency(stations_df: pd.DataFrame, k: int = 8) -> torch.Tensor:
    """
    Build the normalised adjacency matrix A_hat for the CDMX Metro graph.

    Edges are added in two passes:
      1. Topological: connect consecutive stations on the same line.
      2. Geographic: add k-NN edges (Haversine) to capture interchange stations
         that are geographically close but on different lines.

    The combined adjacency is symmetrically degree-normalised:
        A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    N = len(stations_df)
    A = np.zeros((N, N), dtype=np.float32)

    # Pass 1: line-topology edges (consecutive stations on the same line)
    for line, grp in stations_df.groupby('line'):
        idxs = grp.index.tolist()
        for i in range(len(idxs)-1):
            a, b = idxs[i], idxs[i+1]
            A[a, b] = 1.0
            A[b, a] = 1.0

    # Pass 2: geographic k-NN edges (catches interchanges not captured by topology)
    coords     = stations_df[["latitude","longitude"]].to_numpy()
    coords_rad = np.deg2rad(coords)
    k_actual   = min(k, N-1)
    tree       = BallTree(coords_rad, metric="haversine")
    _, ind     = tree.query(coords_rad, k=k_actual+1)

    for i in range(N):
        for j in ind[i]:
            if i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Self-loops and symmetric normalisation
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_hat, dtype=torch.float32)


# Feature construction

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


VENUE_COORDS: dict[str, tuple[float,float]] = {
    "Ciudad Deportiva": (19.4035, -99.0660),
    "Azteca":           (19.3023, -99.1490),
    "Pumas/CU":         (19.3152, -99.1869),
}


def build_node_features(
    date_str:        str,
    stations_df:     pd.DataFrame,
    baseline_df:     pd.DataFrame,   # sid × dow → baseline_riders
    daily_df:        pd.DataFrame,   # daily_totals (used only to compute system-wide scale)
    venue_name:      str,
    is_event:        int,
    venue_coords:    tuple[float, float] | None = None,
    proj_total:      float | None = None,
) -> torch.Tensor:
    """
    Build the (N, 6) node feature matrix for a given date without data leakage.

    F[0]: log(baseline_riders + 1) normalised
    F[1]: log(distance_to_venue_km + 1)
    F[2]: is_venue_line             - 1 if station is on the venue's primary line
    F[3]: sin(2*pi * dow / 7)
    F[4]: cos(2*pi * dow / 7)
    F[5]: log(scaled_baseline + 1) normalised
          - baseline scaled uniformly by (proj_total / baseline_total).
          This signals to the model "today Metro is busier than a typical day"
          without leaking which specific stations carry the elevated demand.
          Mirrors the NYC approach of scaling all station baselines by
          proj_mid / total_ref before feature construction.

    Args:
        proj_total: Projected system-wide Metro ridership for this day.
                    If None, the sum of same-DOW baselines is used (typical day).
                    For LOTO-CV validation: use the observed daily system total
                    as the scale reference (uniform scaling, no spatial leakage).
                    For WC2026 inference: use baseline_total + WC_extra_riders.
    """
    dt    = pd.Timestamp(date_str)
    dow   = dt.dayofweek
    venue = venue_coords if venue_coords else VENUE_COORDS.get(venue_name, (19.42, -99.14))

    N = len(stations_df)
    X = np.zeros((N, 6), dtype=np.float32)

    base_idx = baseline_df.set_index(['sid', 'dow'])['baseline_riders']

    venue_lines = {
        "Ciudad Deportiva": "linea 9",
        "Azteca":           "linea 12",
        "Pumas/CU":         "linea 3",
    }
    v_line = venue_lines.get(venue_name, "")

    baseline_vals = np.zeros(N)
    dist_vals     = np.zeros(N)
    is_vline      = np.zeros(N)

    for i, row in stations_df.reset_index(drop=True).iterrows():
        sid  = row['sid']
        base = float(base_idx.get((sid, dow), 0.0))
        baseline_vals[i] = base
        dist_vals[i]     = haversine_km(row['latitude'], row['longitude'], venue[0], venue[1])
        is_vline[i]      = 1.0 if row['line'] == v_line else 0.0

    # F[5]: system-wide scale factor, applied uniformly (no spatial leakage)
    # baseline_total = expected system ridership on a typical same-DOW day
    baseline_total = baseline_vals.sum()

    if proj_total is None:
        # For validation: use the observed system-wide total to compute the
        # scale factor, but apply it uniformly to baseline (no station-level leakage)
        today_total = float(
            daily_df[daily_df['date'] == date_str]['riders'].sum()
        ) if not daily_df[daily_df['date'] == date_str].empty else baseline_total
        proj_total = today_total

    scale_factor  = proj_total / max(baseline_total, 1.0)
    scaled_vals   = baseline_vals * scale_factor      # uniform uplift, no spatial signal

    b_max = baseline_vals.max() + 1e-8
    s_max = scaled_vals.max() + 1e-8

    X[:, 0] = np.log1p(baseline_vals) / np.log1p(b_max)
    X[:, 1] = np.log1p(dist_vals)
    X[:, 2] = is_vline
    X[:, 3] = np.sin(2 * np.pi * dow / 7)
    X[:, 4] = np.cos(2 * np.pi * dow / 7)
    X[:, 5] = np.log1p(scaled_vals) / np.log1p(s_max)

    return torch.tensor(X, dtype=torch.float32)


# Dataset loading

def load_data():
    """Load all CDMX Metro data files and construct the graph adjacency matrix."""
    print("Loading CDMX Metro dataset...")
    stations_df = pd.read_csv(DATA_DIR / "station_lookup.csv")
    daily_df    = pd.read_parquet(DATA_DIR / "daily_totals.parquet")
    baseline_df = pd.read_parquet(DATA_DIR / "baseline_daily.parquet")
    catalog_df  = pd.read_csv(CATALOG_PATH)

    # Normalize
    daily_df['date']   = pd.to_datetime(daily_df['date']).dt.strftime('%Y-%m-%d')
    catalog_df['date'] = pd.to_datetime(catalog_df['fecha']).dt.strftime('%Y-%m-%d')

    A_hat = build_metro_adjacency(stations_df, k=8)
    print(f"  Stations: {len(stations_df)}, Event days: {len(catalog_df)}")
    print(f"  Adjacency: {A_hat.shape}")
    return stations_df, daily_df, baseline_df, catalog_df, A_hat


# Target construction: observed spatial distribution on event day

def get_observed_spatial_distribution(
    date_str:    str,
    stations_df: pd.DataFrame,
    daily_df:    pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> torch.Tensor | None:
    """
    Compute the observed spatial demand distribution for a given event day.

    Returns weight_i = riders_i / sum(riders_all), i.e. the fraction of
    system-wide ridership at each station.  Stations with missing data are
    imputed from their same-DOW baseline.  Returns None if more than 20% of
    stations are missing (insufficient data quality for training).
    """
    today    = daily_df[daily_df['date']==date_str].set_index('sid')['riders']
    N        = len(stations_df)
    dow      = pd.Timestamp(date_str).dayofweek
    base_idx = baseline_df.set_index(['sid', 'dow'])['baseline_riders']

    weights   = np.zeros(N, dtype=np.float32)
    n_missing = 0
    for i, row in stations_df.reset_index(drop=True).iterrows():
        sid = row['sid']
        if sid in today.index:
            weights[i] = float(today[sid])
        else:
            n_missing += 1
            # impute with baseline for missing stations
            weights[i] = float(base_idx.get((sid, dow), 0.0))

    if n_missing > N * 0.20:
        return None

    total = weights.sum()
    if total <= 0:
        return None
    return torch.tensor(weights / total, dtype=torch.float32)


# Venue-aware MAE

def venue_aware_mae(
    pred:        torch.Tensor,  # (N,)
    target:      torch.Tensor,  # (N,)
    venue:       str,
    stations_df: pd.DataFrame,
) -> float:
    """MAE computed only over stations in the venue's catchment zone."""
    catchment_names = VENUE_CATCHMENT.get(venue, [])
    station_names   = stations_df['name'].reset_index(drop=True)
    mask = station_names.isin(catchment_names).values

    if mask.sum() == 0:
        # Fallback: MAE global
        return (pred - target).abs().mean().item()

    return (pred[mask] - target[mask]).abs().mean().item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# Prepare samples for a set of event days

def prepare_samples(
    events:      pd.DataFrame,
    stations_df: pd.DataFrame,
    daily_df:    pd.DataFrame,
    baseline_df: pd.DataFrame,
    A_hat:       torch.Tensor,
    encoder:     EventEncoder,
) -> list[dict]:
    """Pre-compute features, targets, and sentence embeddings for a set of events."""
    samples = []
    skipped = 0

    for _, ev in events.iterrows():
        date_str    = ev['date']
        ev_cat      = ev['event_category']
        # Use catalog venue directly - avoids misclassifying Pumas/CU events as Azteca
        # TYPE_TO_VENUE is only a fallback if catalog venue column is absent
        catalog_venue = ev.get('venue', None)
        if pd.notna(catalog_venue) and str(catalog_venue) in VENUE_CATCHMENT:
            venue_name = str(catalog_venue)
        else:
            venue_name = TYPE_TO_VENUE.get(ev_cat, "Ciudad Deportiva")

        target = get_observed_spatial_distribution(date_str, stations_df, daily_df, baseline_df)
        if target is None:
            skipped += 1
            continue

        x = build_node_features(
            date_str, stations_df, baseline_df, daily_df,
            venue_name, is_event=1,
        )

        desc      = CDMX_EVENT_DESCRIPTIONS.get(ev_cat, CDMX_EVENT_DESCRIPTIONS["concert"])
        event_emb = torch.tensor(encoder.encode(desc), dtype=torch.float32)

        samples.append({
            "date":       date_str,
            "venue":      venue_name,
            "event_cat":  ev_cat,
            "event_name": ev.get('event_name',''),
            "x":          x,
            "A_hat":      A_hat,
            "target":     target,
            "event_emb":  event_emb,
        })

    if skipped:
        print(f"  Skipped {skipped} events (insufficient data)")
    return samples


# Historical Average baseline

def historical_avg_baseline(
    date_str:    str,
    stations_df: pd.DataFrame,
    daily_df:    pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> torch.Tensor:
    """Same-DOW average spatial distribution over the 4 weeks preceding the event."""
    dt = pd.Timestamp(date_str)
    dow   = dt.dayofweek
    prior_dates = [(dt - pd.Timedelta(weeks=w)).strftime('%Y-%m-%d')
                   for w in range(1, 5)]

    N     = len(stations_df)
    accum = np.zeros(N, dtype=np.float32)
    count = 0

    for d in prior_dates:
        day = daily_df[daily_df['date']==d]
        if day.empty:
            continue
        today = day.set_index('sid')['riders']
        v = np.array([float(today.get(s, 0)) for s in stations_df['sid'].values],
                     dtype=np.float32)
        if v.sum() > 0:
            accum += v / v.sum()
            count += 1

    if count == 0:
        # fallback: DOW baseline distribution
        base_idx = baseline_df.set_index(['sid', 'dow'])['baseline_riders']
        accum = np.array([float(base_idx.get((s, dow), 1.0))
                          for s in stations_df['sid'].values], dtype=np.float32)

    total = accum.sum() + 1e-10
    return torch.tensor(accum / total, dtype=torch.float32)


# Training loop for FiLM layers

def train_film(
    model:        SpatialDisaggregationGNN,
    train_samples: list[dict],
    n_epochs:     int   = 50,
    lr:           float = 1e-3,
    device:       str   = "cpu",
) -> SpatialDisaggregationGNN:
    """
    Fine-tunes FiLM layers only; backbone is frozen.
    Loss: MSE on spatial distribution at venue catchment stations.
    """
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.film_parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s in train_samples:
            optimizer.zero_grad()
            pred = model(s["x"].to(device), s["A_hat"].to(device), s["event_emb"].to(device))
            loss = loss_fn(pred, s["target"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 25 == 0:
            print(f"    epoch {epoch+1}/{n_epochs} - loss: {epoch_loss/len(train_samples):.6f}")

    model.eval()
    return model


# LOTO-CV

def run_loto_cv(
    stations_df:  pd.DataFrame,
    daily_df:     pd.DataFrame,
    baseline_df:  pd.DataFrame,
    catalog_df:   pd.DataFrame,
    A_hat:        torch.Tensor,
    encoder:      EventEncoder,
    n_epochs:     int = 50,
    lr:           float = 1e-3,
    hidden:       int = 64,
    n_gcn_layers: int = 2,
    device:       str = "cpu",
) -> pd.DataFrame:
    """
    Leave-One-Type-Out CV:
      For each event type, train on all other types, evaluate on held-out type.
    """
    event_types = catalog_df['event_category'].unique().tolist()
    print(f"\nLOTO-CV: {len(event_types)} folds × {len(catalog_df)} total events")

    # Precompute all samples (sentence embeddings + node features)
    print("Preparing samples (sentence encodings + node features)...")
    all_samples = prepare_samples(
        catalog_df, stations_df, daily_df, baseline_df, A_hat, encoder
    )
    print(f"  {len(all_samples)} samples prepared")

    rows = []
    for fold_idx, test_type in enumerate(event_types):
        train_samples = [s for s in all_samples if s["event_cat"] != test_type]
        test_samples  = [s for s in all_samples if s["event_cat"] == test_type]

        if not train_samples or not test_samples:
            print(f"  Fold {fold_idx+1} ({test_type}): no data, skipping")
            continue

        print(f"\n  Fold {fold_idx+1}/{len(event_types)}: "
              f"TEST={test_type} ({len(test_samples)} events), "
              f"TRAIN={len(train_samples)} events")

        # Re-initialise from backbone each fold - prevents weight leakage across test types
        fold_model = SpatialDisaggregationGNN(
            hidden=hidden, dropout=0.2, n_gcn_layers=n_gcn_layers,
            event_emb_dim=EMBEDDING_DIM, film_hidden=64,
        ).to(device)

        # Phase 1: pre-train backbone only (no event_emb).
        # Keeping backbone training FiLM-free ensures the backbone baseline is
        # not contaminated by FiLM conditioning - otherwise the "no-FiLM" MAE
        # would be artificially inflated, overstating the FiLM improvement.
        optimizer_backbone = torch.optim.Adam(
            fold_model.backbone_parameters(), lr=lr,
        )
        loss_fn = nn.MSELoss()
        fold_model.train()
        print(f"    Pre-training backbone ({n_epochs} epochs, no FiLM)...")
        for epoch in range(n_epochs):
            for s in train_samples:
                optimizer_backbone.zero_grad()
                pred = fold_model(s["x"].to(device), s["A_hat"].to(device),
                                  event_emb=None)   # backbone only, no FiLM
                loss = loss_fn(pred, s["target"].to(device))
                loss.backward()
                optimizer_backbone.step()

        # Save backbone snapshot before FiLM fine-tuning.
        # This ensures the backbone baseline shares identical training history
        # with the FiLM model - the only difference is the conditioning.
        base_model = copy.deepcopy(fold_model)
        base_model.eval()

        # Phase 2: freeze backbone, fine-tune FiLM layers only
        for p in fold_model.backbone_parameters():
            p.requires_grad_(False)
        print(f"    Fine-tuning FiLM ({n_epochs} epochs)...")
        fold_model = train_film(fold_model, train_samples, n_epochs, lr, device)

        # Evaluation on test events
        fold_model.eval()
        test_rows = []

        with torch.no_grad():
            for s in test_samples:
                # Ground truth
                target  = s["target"]
                venue   = s["venue"]

                # 1. Historical average baseline
                hist_avg = historical_avg_baseline(s["date"], stations_df, daily_df, baseline_df)
                mae_hist = venue_aware_mae(hist_avg, target, venue, stations_df)

                # 2. GNN backbone (no FiLM)
                pred_base = base_model(s["x"], s["A_hat"], event_emb=None)
                mae_base  = venue_aware_mae(pred_base, target, venue, stations_df)
                cos_base  = cosine_similarity(pred_base, target)

                # 3. LLM-augmented GNN (with FiLM, out-of-sample)
                pred_llm  = fold_model(s["x"], s["A_hat"], s["event_emb"])
                mae_llm   = venue_aware_mae(pred_llm, target, venue, stations_df)
                cos_llm   = cosine_similarity(pred_llm, target)

                test_rows.append({
                    "fold":            fold_idx + 1,
                    "test_type":       test_type,
                    "date":            s["date"],
                    "event_name":      s["event_name"],
                    "venue":           venue,
                    "mae_hist_avg":    mae_hist,
                    "mae_baseline":    mae_base,
                    "mae_llm_oos":     mae_llm,
                    "cosine_baseline": cos_base,
                    "cosine_llm":      cos_llm,
                    "improvement_pct": (mae_base - mae_llm) / (mae_base + 1e-8) * 100,
                })

        fold_df = pd.DataFrame(test_rows)
        fold_mae_base = fold_df["mae_baseline"].mean()
        fold_mae_llm  = fold_df["mae_llm_oos"].mean()
        fold_imp      = (fold_mae_base - fold_mae_llm) / (fold_mae_base + 1e-8) * 100

        rows.extend(test_rows)
        print(f"    MAE baseline={fold_mae_base:.5f} | "
              f"MAE LLM-OOS={fold_mae_llm:.5f} | Δ={fold_imp:+.1f}%")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CDMX / "cdmx_loto_results.csv", index=False)

    # Aggregate summary (table for paper)
    summary = results_df.groupby("test_type").agg(
        n_events    = ("date",          "count"),
        mae_hist    = ("mae_hist_avg",  "mean"),
        mae_base    = ("mae_baseline",  "mean"),
        mae_llm_oos = ("mae_llm_oos",   "mean"),
        cos_base    = ("cosine_baseline","mean"),
        cos_llm     = ("cosine_llm",     "mean"),
        improvement = ("improvement_pct","mean"),
    ).round(5).reset_index()

    overall = pd.DataFrame([{
        "test_type":  "ALL (LOTO mean)",
        "n_events":   len(results_df),
        "mae_hist":   results_df["mae_hist_avg"].mean(),
        "mae_base":   results_df["mae_baseline"].mean(),
        "mae_llm_oos":results_df["mae_llm_oos"].mean(),
        "cos_base":   results_df["cosine_baseline"].mean(),
        "cos_llm":    results_df["cosine_llm"].mean(),
        "improvement":results_df["improvement_pct"].mean(),
    }])
    summary = pd.concat([summary, overall], ignore_index=True).round(5)
    summary.to_csv(OUT_CDMX / "cdmx_loto_summary.csv", index=False)

    print("\n=== LOTO-CV Summary ===")
    print(summary[["test_type","n_events","mae_base","mae_llm_oos","improvement","cos_llm"]].to_string(index=False))
    return results_df


# WC2026 Spatial Distribution Inference

def run_wc2026_inference(
    stations_df:  pd.DataFrame,
    daily_df:     pd.DataFrame,
    baseline_df:  pd.DataFrame,
    A_hat:        torch.Tensor,
    encoder:      EventEncoder,
    n_epochs:     int = 50,
    lr:           float = 1e-3,
    device:       str = "cpu",
) -> pd.DataFrame:
    """
    Project the spatial demand distribution for WC2026 matches at Estadio Azteca.

    Trains the SE-GNN on the full CDMX event catalog (all types) and applies it
    to each WC2026 match date.  Returns a per-station per-match DataFrame with
    projected ridership under both the backbone and FiLM-conditioned models.
    """
    from src.paper.config import WC2026_CDMX

    print("\n=== WC2026 Spatial Inference ===")

    # Train model on full catalog
    catalog_df = pd.read_csv(CATALOG_PATH)
    catalog_df['date'] = pd.to_datetime(catalog_df['fecha']).dt.strftime('%Y-%m-%d')

    all_samples = prepare_samples(
        catalog_df, stations_df, daily_df, baseline_df, A_hat, encoder
    )
    print(f"  Training on {len(all_samples)} catalog events...")

    model = SpatialDisaggregationGNN(
        hidden=64, dropout=0.2, n_gcn_layers=2, event_emb_dim=EMBEDDING_DIM
    ).to(device)

    loss_fn = nn.MSELoss()

    # Phase 1: train backbone only, no event_emb (mirrors run_loto_cv design).
    # Training the backbone without FiLM guarantees a neutral spatial estimator
    # as the comparison baseline before adding event conditioning.
    print(f"  Phase 1: pre-training backbone ({n_epochs} epochs, no FiLM)...")
    optimizer_backbone = torch.optim.Adam(model.backbone_parameters(), lr=lr)
    model.train()
    for epoch in range(n_epochs):
        for s in all_samples:
            optimizer_backbone.zero_grad()
            pred = model(s["x"].to(device), s["A_hat"].to(device), event_emb=None)
            loss = loss_fn(pred, s["target"].to(device))
            loss.backward()
            optimizer_backbone.step()

    # Phase 2: freeze backbone, fine-tune FiLM only
    print(f"  Phase 2: fine-tuning FiLM ({n_epochs} epochs)...")
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    model = train_film(model, all_samples, n_epochs, lr, device)

    model.eval()
    print("  Training complete (2-phase, no spatial leakage).")

    # WC2026 event description (more specific than training)
    wc_desc = CDMX_EVENT_DESCRIPTIONS["wc2026_azteca"]
    wc_emb  = torch.tensor(encoder.encode(wc_desc), dtype=torch.float32)

    # WC2026 system-wide ridership projection (mirrors NYC Tier 1 methodology):
    #   1. baseline_total: median system-wide ridership for that DOW in June-July,
    #      using 2019 (best pre-COVID analog) or 2022-2023 post-COVID reference.
    #   2. WC_extra: additional riders at the Azteca corridor
    #      = Azteca corridor baseline (2019 Sunday) x WC2026 uplift multiplier.
    #      Copa America NYC observed ~2x for top matches; WC2026 opening game
    #      (Mexico vs South Africa) is conservatively estimated at 3x Liga MX.
    #      Extra = baseline_azteca x (3.0 - 1.0) = baseline_azteca x 2.0
    #   3. proj_total = baseline_total + WC_extra, which correctly scales the
    #      spatial distribution to the full system.

    # Compute system baseline for a Sunday in June-July (WC2026 matches are Thu/Sun)
    base_idx_all = baseline_df.set_index(['sid', 'dow'])['baseline_riders']
    wc_matches_dow = {
        "2026-06-11": 3,   # Thursday
        "2026-06-17": 2,   # Wednesday
        "2026-06-24": 2,   # Wednesday
        "2026-06-30": 1,   # Tuesday
    }
    # Azteca corridor stations baseline (Sunday, 2019-era reference)
    AZTECA_CATCHMENT_NAMES = VENUE_CATCHMENT["Azteca"]
    azteca_sids = stations_df[stations_df['name'].isin(AZTECA_CATCHMENT_NAMES)]['sid'].values

    # WC uplift: from outputs/cdmx/event_catalog: Azteca baseline 120k, 1.21x uplift
    # WC2026 multiplier estimated 3.0x vs Liga MX baseline (FIFA > domestic)
    AZTECA_BASELINE_SUNDAY = 120_000   # daily riders at Azteca corridor, 2019 Sunday
    WC_MULTIPLIER          = 3.0       # WC2026 vs Liga MX (Copa América NYC used 2.0x)
    WC_EXTRA               = AZTECA_BASELINE_SUNDAY * (WC_MULTIPLIER - 1.0)  # 240k extra

    # Empirical Azteca uplift distribution (2019 Liga MX, ~18 events).
    # Used as a cross-check against the GNN prediction.
    # For each Azteca soccer event: uplift_i = riders_i - baseline_i_dow.
    # Average the normalised uplift distribution across all events.
    azteca_event_dates = (
        catalog_df[(catalog_df['event_category']=='soccer_domestic') &
                   (catalog_df.get('venue', pd.Series(['Ciudad Deportiva']*len(catalog_df))
                    ).isin(['Azteca']))]['date'].tolist()
        if 'venue' in catalog_df.columns else []
    )
    N = len(stations_df)
    sids_list = stations_df['sid'].values

    empirical_uplift_accum = np.zeros(N, dtype=np.float32)
    empirical_count = 0

    for ev_date in azteca_event_dates:
        ev_dow = pd.Timestamp(ev_date).dayofweek
        today  = daily_df[daily_df['date']==ev_date].set_index('sid')['riders']
        uplifts = np.zeros(N, dtype=np.float32)
        for i, sid in enumerate(sids_list):
            base   = float(base_idx_all.get((sid, ev_dow), 0.0))
            riders = float(today.get(sid, base))
            uplifts[i] = max(riders - base, 0.0)
        total_ul = uplifts.sum()
        if total_ul > 1000.0:
            empirical_uplift_accum += uplifts / total_ul
            empirical_count += 1

    has_empirical = empirical_count > 0
    if has_empirical:
        empirical_uplift_dist = empirical_uplift_accum / empirical_count
        print(f"\n  Empirical Azteca uplift pattern: {empirical_count} events (2019 Liga MX)")
        az_names_low = [n.lower() for n in VENUE_CATCHMENT["Azteca"]]
        az_mask = stations_df['name'].isin(az_names_low).values
        az_share = empirical_uplift_dist[az_mask].sum()
        print(f"  Azteca corridor share of empirical uplift: {az_share*100:.1f}%")
    else:
        print("  No empirical Azteca data available for cross-check.")

    # Per-match inference.
    # The model predicts the absolute spatial distribution (weight_i = fraction of
    # total system ridership at station i).  For WC2026:
    #   total_riders_i = weight_gnn_i * proj_total
    #   extra_riders_i = total_riders_i - baseline_i
    # This is equivalent to re-allocating the WC_EXTRA demand according to the
    # model's spatial weights rather than assuming a uniform distribution.
    records = []
    with torch.no_grad():
        for (date_str, kickoff, matchup, stage) in WC2026_CDMX:
            dt  = pd.Timestamp(date_str)
            dow = dt.dayofweek

            # System-wide baseline for this DOW
            baseline_total = float(sum(
                base_idx_all.get((sid, dow), 0.0)
                for sid in stations_df['sid'].values
            ))
            proj_total = baseline_total + WC_EXTRA

            # Build features (F[5] = uniform scale signal, no spatial leakage)
            x = build_node_features(
                date_str, stations_df, baseline_df, daily_df,
                "Azteca", is_event=1,
                proj_total=proj_total,
            )

            # Model predictions: absolute spatial distribution (sums to 1)
            weights_base = model(x, A_hat, event_emb=None)   # (N,) - no FiLM
            weights_llm  = model(x, A_hat, wc_emb)           # (N,) - FiLM conditioned

            # Convert to projected total riders per station
            total_riders_base = weights_base.numpy() * proj_total  # (N,)
            total_riders_llm  = weights_llm.numpy()  * proj_total  # (N,)

            # Compute per-station uplift vs DOW baseline
            for i, row in stations_df.reset_index(drop=True).iterrows():
                sid    = row['sid']
                base_i = float(base_idx_all.get((sid, dow), 0.0))

                total_b = float(total_riders_base[i])
                total_l = float(total_riders_llm[i])

                records.append({
                    "date":               date_str,
                    "matchup":            matchup,
                    "stage":              stage,
                    "kickoff":            kickoff,
                    "sid":                sid,
                    "station":            row['name'],
                    "line":               row['line'],
                    "baseline_riders":    base_i,
                    "weight_base":        float(weights_base[i]),
                    "weight_llm":         float(weights_llm[i]),
                    "total_riders_base":  total_b,
                    "total_riders_llm":   total_l,
                    "extra_riders_base":  total_b - base_i,
                    "extra_riders_llm":   total_l - base_i,
                    "uplift_ratio_base":  total_b / max(base_i, 1.0),
                    "uplift_ratio_llm":   total_l / max(base_i, 1.0),
                })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CDMX / "cdmx_wc2026_spatial.csv", index=False)
    print(f"  Saved: outputs/paper/cdmx/cdmx_wc2026_spatial.csv")

    # Azteca catchment analysis
    azteca_names_lower = [n.lower() for n in VENUE_CATCHMENT["Azteca"]]
    az_df = df[df["station"].str.lower().isin(azteca_names_lower)]

    print("\n  Azteca corridor (WC2026 model projection, avg per match):")
    az_summary = (az_df.groupby(["station","line"])
                       [["baseline_riders","total_riders_llm","extra_riders_llm","uplift_ratio_llm"]]
                       .mean().round(1).reset_index())
    print(az_summary.to_string(index=False))

    total_extra_az  = az_df.groupby("date")["extra_riders_llm"].sum().mean()
    total_extra_sys = df.groupby("date")["extra_riders_llm"].sum().mean()
    print(f"\n  Azteca corridor extra riders (LLM): {total_extra_az:.0f}"
          f"  /  total system extra: {total_extra_sys:.0f}")
    print(f"  Azteca corridor share: {total_extra_az/total_extra_sys*100:.1f}%"
          f"  (vs WC_EXTRA={WC_EXTRA:.0f})")

    # Add empirical comparison if available
    if has_empirical:
        emp_dist_t = torch.tensor(empirical_uplift_dist, dtype=torch.float32)
        az_emp_share = emp_dist_t[az_mask].sum().item()
        print(f"\n  Cross-check - empirical 2019 Liga MX at Azteca:")
        for i, row in stations_df[az_mask].reset_index(drop=True).iterrows():
            idx = stations_df[stations_df['sid']==row['sid']].index[0]
            print(f"    {row['name']} ({row['line']}): "
                  f"empirical uplift share = {empirical_uplift_dist[idx]*100:.2f}%")
        print(f"  → Azteca corridor total empirical share: {az_emp_share*100:.1f}%")
        print(f"  NOTE: Model trained on absolute distribution; empirical shows uplift "
              f"distribution. For WC2026, empirical analog may be more reliable for "
              f"Azteca corridor.")

    # Top 15 by projected total ridership (LLM model)
    top = (df.groupby(["station","line"])["total_riders_llm"]
             .mean()
             .nlargest(15)
             .reset_index())
    print("\n  Top 15 stations by projected total ridership WC2026 (SE-GNN):")
    print(top.to_string(index=False))

    return df


# Figures

def plot_loto_results(results_df: pd.DataFrame):
    """Bar chart of venue-aware MAE by event type fold (CDMX LOTO-CV)."""
    summary = results_df.groupby("test_type").agg(
        mae_hist   = ("mae_hist_avg",  "mean"),
        mae_base   = ("mae_baseline",  "mean"),
        mae_llm    = ("mae_llm_oos",   "mean"),
    ).reset_index()

    x   = np.arange(len(summary))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w,   summary["mae_hist"] * 1000, w, label="Historical DOW Avg", color="#adb5bd", alpha=0.85)
    ax.bar(x,       summary["mae_base"] * 1000, w, label="GNN Backbone (no FiLM)", color="#457b9d", alpha=0.85)
    ax.bar(x + w,   summary["mae_llm"]  * 1000, w, label="LLM-Augmented GNN (FiLM, OOS)", color="#e63946", alpha=0.85)

    # Improvement labels
    for i, row in summary.iterrows():
        imp = (row["mae_base"] - row["mae_llm"]) / (row["mae_base"] + 1e-8) * 100
        ax.text(i + w, row["mae_llm"] * 1000 + 0.02,
                f"{imp:+.0f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#2a9d8f")

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{t.replace('_',' ').title()}\n(n={len(results_df[results_df.test_type==t])})"
        for t in summary["test_type"]
    ], fontsize=9)
    ax.set_ylabel("Venue-Aware MAE × 1000 (spatial distribution)", fontsize=10)
    ax.set_title(
        "LOTO-CV: Spatial Demand Disaggregation - CDMX Metro\n"
        "GNN Backbone vs LLM-Augmented (FiLM) - Out-of-Sample by Event Type",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT_FIGS / "fig_cdmx_loto_cv.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Fig CDMX LOTO-CV → {path.name}")


def plot_wc2026_spatial(wc_df: pd.DataFrame, stations_df: pd.DataFrame):
    """
    Two-panel figure for the WC2026 spatial demand projection.

    Left panel:  top 20 stations by projected extra riders (uplift share).
    Right panel: Azteca catchment stations - uplift ratio vs. a normal day.
    """
    if wc_df.empty:
        return

    # Panel A: top 20 stations by extra riders (uplift)
    top = (wc_df.groupby(["station","line"])
               .agg(extra_llm  = ("extra_riders_llm",  "mean"),
                    extra_base = ("extra_riders_base", "mean"),
                    base_day   = ("baseline_riders",   "mean"))
               .reset_index()
               .nlargest(20, "extra_llm"))

    # Panel B: Azteca catchment uplift ratio
    azteca_names = ["mexicaltzingo","ermita","atlalilco","tasquea"]
    az = (wc_df[wc_df["station"].isin(azteca_names)]
            .groupby("station")
            .agg(uplift_base = ("uplift_ratio_base","mean"),
                 uplift_llm  = ("uplift_ratio_llm","mean"))
            .reset_index()
            .sort_values("uplift_llm", ascending=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A
    y  = np.arange(len(top))
    ax1.barh(y,       top["extra_base"], 0.4, label="GNN Backbone", color="#457b9d", alpha=0.8)
    ax1.barh(y + 0.4, top["extra_llm"],  0.4, label="LLM-Augmented (FiLM)", color="#e63946", alpha=0.85)
    ax1.set_yticks(y + 0.2)
    ax1.set_yticklabels([f"{r['station'].title()} ({r['line'].replace('linea ','L')})"
                         for _, r in top.iterrows()], fontsize=9)
    ax1.set_xlabel("Projected extra riders (above baseline, avg per WC2026 match)", fontsize=10)
    ax1.set_title("Top 20 Stations - WC2026 Extra Demand\nCDMX Metro (Uplift Distribution)",
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Panel B - Azteca catchment
    x2 = np.arange(len(az))
    ax2.bar(x2 - 0.2, az["uplift_base"], 0.35, label="GNN Backbone", color="#457b9d", alpha=0.8)
    ax2.bar(x2 + 0.2, az["uplift_llm"],  0.35, label="LLM-Augmented (FiLM)", color="#e63946", alpha=0.85)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Normal day (1.0x)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([s.title() for s in az["station"]], fontsize=10)
    ax2.set_ylabel("Uplift ratio (WC2026 / normal day)", fontsize=10)
    ax2.set_title("Azteca Corridor Stations - Demand Uplift\nMetro Lines 2 & 12 (WC2026 avg)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate uplift values
    for j, row in enumerate(az.itertuples()):
        ax2.text(j + 0.2, row.uplift_llm + 0.02,
                 f"{row.uplift_llm:.2f}x", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#e63946")

    plt.tight_layout()
    path = OUT_FIGS / "fig_cdmx_wc2026_spatial.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig WC2026 Spatial → {path.name}")


# Main runner

def set_seed(seed: int = 42):
    """Set global random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(
    phase:     str = "all",
    n_epochs:  int = 100,
    lr:        float = 1e-3,
    device:    str = "cpu",
    seed:      int = 42,
):
    set_seed(seed)
    stations_df, daily_df, baseline_df, catalog_df, A_hat = load_data()
    encoder = EventEncoder(cache_path=EMB_CACHE)

    if phase in ("loto", "all"):
        print("\n" + "="*55)
        print("Phase 1: LOTO-CV (out-of-sample evaluation by event type)")
        print("="*55)
        set_seed(seed)  # reset before each phase for reproducibility
        results = run_loto_cv(
            stations_df, daily_df, baseline_df, catalog_df, A_hat,
            encoder, n_epochs=n_epochs, lr=lr, device=device,
        )
        plot_loto_results(results)

    if phase in ("inference", "all"):
        print("\n" + "="*55)
        print("Phase 2: WC2026 inference - spatial demand distribution")
        print("="*55)
        set_seed(seed)
        wc_df = run_wc2026_inference(
            stations_df, daily_df, baseline_df, A_hat,
            encoder, n_epochs=n_epochs, lr=lr, device=device,
        )
        plot_wc2026_spatial(wc_df, stations_df)

    print("\nCDMX GNN LOTO-CV complete.")
    print(f"  Outputs: {OUT_CDMX}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",    choices=["loto","inference","all"], default="all")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()
    run(phase=args.phase, n_epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
