"""
k-Shot LOTO-CV Ablation — expanded catalog (v2) with NFL as a fifth event type.

Multi-seed: each (type, k) combination is trained N_SEEDS times with different
random seeds to quantify stochastic variance of FiLM optimisation.

Differences from run_kshot_ablation.py:
  - Uses event_catalog_expanded_v2.json (21 events: 12 original + 7 NFL + 2 from 2021)
  - Adds 'sports_nfl_metlife' as a fifth LOTO type
  - k=1 uses the k most recent examples (reverse=True)
  - N_SEEDS runs per (type, k) — reports mean ± std

Outputs:
  outputs/paper/nyc/kshot_ablation_v2_full.csv        (one row per type/k/event/seed)
  outputs/paper/nyc/kshot_ablation_v2_event_means.csv (seed-averaged per event)
  outputs/paper/nyc/kshot_ablation_v2_summary.csv     (mean ± std per type/k)
  outputs/paper/nyc/kshot_ablation_v2_aggregate.csv   (mean ± std per k)
  outputs/paper/comparative/figures/fig_kshot_ablation_v2.png
"""

from __future__ import annotations
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.disaggregation.model import build_node_features
from src.ml.disaggregation.model_v2 import EventConditionedGNN, numpy_to_event_tensor
from src.paper.tier2.llm_augmented_gnn import load_dataset_and_model

# Paths and constants
GNN_MODEL    = ROOT / "outputs/nyc/manhattan_model.pt"
CATALOG_V2   = ROOT / "data/nyc/event_catalog_expanded_v2.json"
EMB_CACHE    = ROOT / "outputs/paper/nyc/event_embeddings_cache.json"
OUT          = ROOT / "outputs/paper/nyc"
FIG          = ROOT / "outputs/paper/comparative/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

VENUE_COORDS = {
    "MetLife_Penn": (40.8135, -74.0745),
    "WVillage":     (40.7335, -74.0029),
    "CentralPark":  (40.7756, -73.9779),
    "MSG":          (40.7505, -73.9934),
}
VENUE_SIDS = {
    "MetLife_Penn": [164, 318, 607],
    "WVillage":     [323, 324, 601, 618],
    "CentralPark":  [158, 159, 160, 311, 313, 397, 476, 477, 614],
    "MSG":          [164, 318, 607, 319, 403, 404, 13],
}
TYPE_TO_VENUE = {
    "soccer_international": "MetLife_Penn",
    "concert_pop":          "MetLife_Penn",
    "parade_street":        "WVillage",
    "race_marathon":        "CentralPark",
    "sports_nfl_metlife":   "MetLife_Penn",
}
HOUR_COLS  = [f"h{h:02d}" for h in range(24)]

LOTO_TYPES = [
    "soccer_international",
    "concert_pop",
    "parade_street",
    "race_marathon",
    "sports_nfl_metlife",
]

K_VALUES   = [0, 1, 2, 3]
N_EPOCHS   = 60
LR         = 3e-4
SEEDS      = [42, 123, 456]   # multiple seeds to estimate stochastic variance
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Seeds:  {SEEDS}  ({len(SEEDS)} runs per fold×k)")

print("\nLoading dataset...")
dataset, A_hat, baseline_daily_df, _ = load_dataset_and_model(freeze_backbone=True)
A_hat = A_hat.to(DEVICE)

sid_to_idx = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
profiles_df = dataset.profiles
dates_set   = set(dataset.dates)

print("\nLoading expanded catalog v2...")
with open(CATALOG_V2) as f:
    catalog_v2 = json.load(f)

print(f"  Total events: {len(catalog_v2)}")
type_counts = {}
for e in catalog_v2:
    type_counts[e["event_type"]] = type_counts.get(e["event_type"], 0) + 1
for etype, cnt in sorted(type_counts.items()):
    print(f"    {etype:30s}: {cnt}")

print("\nEncoding event embeddings...")
from src.ml.disaggregation.event_encoder import EventEncoder, EMBEDDING_DIM
encoder = EventEncoder(cache_path=EMB_CACHE)

event_embs = {}   # date → embedding
for ev in catalog_v2:
    if ev["date"] not in dates_set:
        print(f"  SKIP: {ev['date']} not in Manhattan profiles")
        continue
    emb = encoder.encode(ev["description"])
    event_embs[ev["date"]] = emb

print(f"  Embeddings computed: {len(event_embs)}")

def get_event_sample(ev: dict, device: str = "cpu"):
    """
    Returns (x, target_profile, venue_idx_list) for one event.
    x: (N, 7) node features
    target_profile: (len(catchment), 24)
    """
    date_str     = ev["date"]
    venue_cluster = ev.get("venue_cluster") or TYPE_TO_VENUE.get(ev["event_type"], "MetLife_Penn")
    venue_coords  = VENUE_COORDS[venue_cluster]
    catchment_sids = [s for s in VENUE_SIDS.get(venue_cluster, []) if s in sid_to_idx]

    day_data = profiles_df[profiles_df["date"] == date_str]
    if len(day_data) == 0:
        return None, None, None

    daily_totals = day_data.set_index("sid")["daily_total"]

    x = build_node_features(
        date              = pd.Timestamp(date_str).date(),
        stations_df       = dataset.stations,
        baseline_df       = dataset.baseline,
        daily_totals_series = daily_totals,
        venue_coords      = venue_coords,
        event_flag        = 1,
        device            = device,
        baseline_daily_df = baseline_daily_df,
    )

    target_rows = []
    for sid in catchment_sids:
        row = day_data[day_data["sid"] == sid]
        if len(row) == 0:
            target_rows.append(np.ones(24) / 24.0)
        else:
            target_rows.append(row[HOUR_COLS].values[0].astype(np.float32))

    if not target_rows:
        return None, None, None

    target = torch.tensor(np.array(target_rows), dtype=torch.float32).to(device)
    catch_idxs = [sid_to_idx[s] for s in catchment_sids]
    return x, target, catch_idxs


print("\n" + "=" * 70)
print(f"k-Shot LOTO-CV  |  5 event types (v2)  |  {len(SEEDS)} seeds per fold x k")
print("=" * 70)

criterion = nn.MSELoss()
all_results = []   # one row per (type, k, event, seed)

for held_out_type in LOTO_TYPES:
    test_events  = [e for e in catalog_v2 if e["event_type"] == held_out_type
                    and e["date"] in event_embs]
    other_events = [e for e in catalog_v2 if e["event_type"] != held_out_type
                    and e["date"] in event_embs]

    if len(test_events) == 0:
        print(f"\n  SKIP {held_out_type}: no test events with embeddings")
        continue

    print(f"\nFold: {held_out_type}  ({len(test_events)} test events)")
    print(f"   Training pool (other types): {len(other_events)} events")

    # Backbone MAE is deterministic; compute once per held-out type
    base_model = EventConditionedGNN.from_pretrained(
        GNN_MODEL, event_emb_dim=EMBEDDING_DIM, film_hidden=64,
        freeze_backbone=True, device=DEVICE,
    )
    # Per-event backbone MAE dict for later per-event improvement computation
    backbone_per_event = {}
    base_model.eval()
    with torch.no_grad():
        for ev in test_events:
            x, target, catch_idxs = get_event_sample(ev, DEVICE)
            if x is None:
                continue
            pred = base_model(x, A_hat, event_emb=None)
            backbone_per_event[ev["date"]] = (pred[catch_idxs] - target).abs().mean().item()

    mae_backbone_mean = float(np.mean(list(backbone_per_event.values())))
    print(f"   Backbone MAE (mean): {mae_backbone_mean:.6f}")

    for k in K_VALUES:
        # Most recent k examples as training (reverse=True)
        k_examples = sorted(test_events, key=lambda e: e["date"], reverse=True)[:k]
        train_events = other_events + k_examples

        if len(train_events) == 0:
            continue

        k_dates = {e["date"] for e in k_examples}
        test_remaining = [e for e in test_events if e["date"] not in k_dates]

        if len(test_remaining) == 0:
            print(f"   k={k}: no test events remaining, skip")
            continue

        seed_means = []   # mean improvement per seed (for quick print)

        for seed in SEEDS:
            # Fix all sources of randomness
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Fresh FiLM-initialised model
            model_k = EventConditionedGNN.from_pretrained(
                GNN_MODEL, event_emb_dim=EMBEDDING_DIM, film_hidden=64,
                freeze_backbone=True, device=DEVICE,
            )

            optimizer = torch.optim.Adam(model_k.film_parameters(), lr=LR)
            model_k.train()

            for epoch in range(N_EPOCHS):
                shuffled = train_events.copy()
                random.shuffle(shuffled)
                for ev in shuffled:
                    if ev["date"] not in event_embs:
                        continue
                    x, target, catch_idxs = get_event_sample(ev, DEVICE)
                    if x is None:
                        continue
                    emb = numpy_to_event_tensor(event_embs[ev["date"]], DEVICE)
                    optimizer.zero_grad()
                    pred = model_k(x, A_hat, event_emb=emb)
                    loss = criterion(pred[catch_idxs], target)
                    loss.backward()
                    optimizer.step()

            # Eval per event for this seed
            seed_imps = []
            model_k.eval()
            with torch.no_grad():
                for ev in test_remaining:
                    if ev["date"] not in backbone_per_event:
                        continue
                    x, target, catch_idxs = get_event_sample(ev, DEVICE)
                    if x is None:
                        continue
                    emb_t = numpy_to_event_tensor(event_embs[ev["date"]], DEVICE)
                    pred_film = model_k(x, A_hat, event_emb=emb_t)
                    mae_f = (pred_film[catch_idxs] - target).abs().mean().item()
                    mae_b = backbone_per_event[ev["date"]]
                    imp   = (mae_b - mae_f) / mae_b * 100 if mae_b > 0 else 0.0
                    seed_imps.append(imp)
                    all_results.append({
                        "held_out_type": held_out_type,
                        "k":            k,
                        "event_date":   ev["date"],
                        "seed":         seed,
                        "mae_backbone": mae_b,
                        "mae_film":     mae_f,
                        "improvement":  imp,
                    })

            seed_mean = float(np.mean(seed_imps)) if seed_imps else float("nan")
            seed_means.append(seed_mean)

        print(f"   k={k}: seeds={[f'{m:+.1f}%' for m in seed_means]}  "
              f"mean={np.mean(seed_means):+.1f}%  std={np.std(seed_means):.1f}pp  "
              f"(n_test={len(test_remaining)})")

# Save raw results and build summary tables
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT / "kshot_ablation_v2_full.csv", index=False)

# Event-level means (averaged across seeds) — used by run_bootstrap_ci.py
event_means = (results_df
    .groupby(["held_out_type", "k", "event_date"])
    .agg(
        mae_backbone = ("mae_backbone", "mean"),
        mae_film     = ("mae_film",     "mean"),
        improvement  = ("improvement",  "mean"),
    )
    .reset_index()
)
event_means.to_csv(OUT / "kshot_ablation_v2_event_means.csv", index=False)

# Type-level summary: mean ± std across seeds, then averaged over events
# Step 1: per-event, per-seed improvement → seed-level mean per (type, k)
seed_type_means = (results_df
    .groupby(["held_out_type", "k", "seed"])["improvement"]
    .mean()
    .reset_index()
    .rename(columns={"improvement": "seed_mean_improvement"})
)

# Step 2: across-seed mean and std per (type, k)
summary = (seed_type_means
    .groupby(["held_out_type", "k"])
    .agg(
        n_seeds          = ("seed_mean_improvement", "count"),
        improvement_mean = ("seed_mean_improvement", "mean"),
        improvement_std  = ("seed_mean_improvement", "std"),
    )
    .reset_index()
)

# Also add backbone and film MAE (seed-averaged, event-averaged)
mae_summary = (event_means
    .groupby(["held_out_type", "k"])
    .agg(
        n_events     = ("event_date", "count"),
        mae_backbone = ("mae_backbone", "mean"),
        mae_film     = ("mae_film",     "mean"),
    )
    .reset_index()
)
summary = summary.merge(mae_summary, on=["held_out_type", "k"])
summary = summary.round(4)
summary.to_csv(OUT / "kshot_ablation_v2_summary.csv", index=False)

# Aggregate by k: macro-average over types (each type equally weighted)
agg = (seed_type_means
    .groupby(["k", "seed"])["seed_mean_improvement"]
    .mean()                              # macro-avg over types for this seed
    .reset_index()
    .groupby("k")["seed_mean_improvement"]
    .agg(improvement_mean="mean", improvement_std="std")
    .reset_index()
    .round(3)
)
agg.to_csv(OUT / "kshot_ablation_v2_aggregate.csv", index=False)

print("\n" + "=" * 70)
print("RESULTS SUMMARY — v2 (5 event types, multi-seed)")
print("=" * 70)
print(f"{'Type':<25} {'k':>2} {'n_ev':>4} {'Backbone':>10} {'Film':>10} "
      f"{'Mean Imp':>10} {'Std':>8}")
print("-" * 75)
for _, row in summary.sort_values(["held_out_type", "k"]).iterrows():
    print(f"  {row['held_out_type']:<23} {int(row['k']):>2} {int(row['n_events']):>4} "
          f"  {row['mae_backbone']:>8.4f}   {row['mae_film']:>8.4f} "
          f"  {row['improvement_mean']:>+8.1f}%  ±{row['improvement_std']:>5.1f}pp")

print("\nAggregate by k (macro-avg over 5 types):")
for _, row in agg.iterrows():
    print(f"  k={int(row['k'])}: {row['improvement_mean']:+.1f}% ± {row['improvement_std']:.1f}pp")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: improvement per fold per k
ax = axes[0]
colors = plt.cm.Set2.colors
fold_labels = summary["held_out_type"].unique()
x_base = np.arange(len(K_VALUES))
bar_w  = 0.15
for i, ftype in enumerate(fold_labels):
    sub = summary[summary["held_out_type"] == ftype]
    ks  = sub["k"].tolist()
    imp = sub["improvement_mean"].tolist()
    ax.bar(np.array(ks) + (i - len(fold_labels)/2 + 0.5) * bar_w,
           imp, bar_w, label=ftype.replace("_"," "), color=colors[i], alpha=0.8)
ax.axhline(0, color="black", lw=1, ls="--")
ax.set_xlabel("k (in-domain training examples)", fontsize=11)
ax.set_ylabel("Improvement over backbone (%)", fontsize=11)
ax.set_title("k-Shot LOTO by Event Type (5 types, v2)", fontsize=11)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=8, loc="upper left")

# Right: aggregate curve
ax2 = axes[1]
ks  = agg["k"].tolist()
imp = agg["improvement_mean"].tolist()
std = agg["improvement_std"].tolist()
ax2.fill_between(ks,
                 [i - s for i, s in zip(imp, std)],
                 [i + s for i, s in zip(imp, std)],
                 alpha=0.2, color="#1f77b4")
ax2.plot(ks, imp, "o-", color="#1f77b4", lw=2, ms=8)
ax2.axhline(0, color="black", lw=1.5, ls="--")
for k_i, imp_i in zip(ks, imp):
    ax2.text(k_i, imp_i + 1.5, f"{imp_i:+.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("k (in-domain training examples)", fontsize=11)
ax2.set_ylabel("Mean improvement over backbone (%)", fontsize=11)
ax2.set_title("Aggregate k-Shot Curve (5 types, mean ± std)", fontsize=11)
ax2.set_xticks(ks)

plt.suptitle(f"LLM-FiLM Few-Shot Adaptation — Catalog v2 (21 events, 5 types, {len(SEEDS)} seeds)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "fig_kshot_ablation_v2.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved -> {FIG / 'fig_kshot_ablation_v2.png'}")
