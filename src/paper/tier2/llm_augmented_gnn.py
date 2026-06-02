"""
Tier 2 NYC: full LLM-augmented GNN pipeline.

Four phases (run individually or together with --phase all):

  1. finetune:  Fine-tune FiLM layers on Copa América 2024 events.
  2. loocv:     Leave-one-out cross-validation — the primary out-of-sample
                evaluation reported in the paper.
  3. ablation:  Compare baseline vs. LLM-augmented on Copa América (in-sample
                reference for maximum model capacity).
  4. inference: Generate WC2026 hourly profiles with the trained model.

Usage:
    python src/paper/tier2/llm_augmented_gnn.py --phase loocv
    python src/paper/tier2/llm_augmented_gnn.py --phase all

Outputs (outputs/paper/nyc/):
  loocv_results.csv
  finetune_log.csv
  ablation_results.csv
  hourly_profiles_wc2026_llm.csv

Outputs (outputs/paper/comparative/figures/):
  fig5_loocv_copa_america.png
  fig5_ablation_copa_america.png
  fig6_wc2026_llm_vs_baseline.png
"""

import sys
import copy
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/paper

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.disaggregation.model   import build_adjacency, build_node_features
from src.ml.disaggregation.model_v2 import EventConditionedGNN, numpy_to_event_tensor
from src.ml.disaggregation.event_encoder import EventEncoder, EMBEDDING_DIM
from src.ml.disaggregation.train import DailyProfileDataset

from config import (
    GNN_MODEL, OUT_NYC, OUT_FIGS, DATA_DIR,
    PENN_CORRIDOR,
)

MANHATTAN_DIR   = DATA_DIR / "nyc" / "manhattan"
METLIFE_COORDS  = (40.8135, -74.0745)
PENN_COORDS     = (40.7527, -73.9772)
FILM_MODEL_PATH = OUT_NYC / "manhattan_model_film.pt"


def load_dataset_and_model(freeze_backbone: bool = True):
    """
    Load the Manhattan daily-profile dataset and an EventConditionedGNN
    initialised with backbone weights from the pretrained checkpoint.

    Args:
        freeze_backbone: If True, only FiLM parameters are trainable.

    Returns:
        (dataset, A_hat, baseline_daily_df, model)
    """
    print("Loading Manhattan dataset...")
    profiles_df       = pd.read_parquet(MANHATTAN_DIR / "daily_profiles.parquet")
    labels_df         = pd.read_csv(MANHATTAN_DIR / "event_labels.csv")
    stations_df       = pd.read_csv(MANHATTAN_DIR / "station_lookup.csv")
    baseline_df       = pd.read_parquet(MANHATTAN_DIR / "baseline_profile.parquet")
    bdt_path          = MANHATTAN_DIR / "baseline_daily_totals.parquet"
    baseline_daily_df = pd.read_parquet(bdt_path) if bdt_path.exists() else None

    stations_df["sid"] = stations_df["sid"].astype(int)
    profiles_df["sid"] = profiles_df["sid"].astype(int)

    # Normalise date columns to ISO strings for consistent comparisons
    profiles_df["date"] = pd.to_datetime(profiles_df["date"]).dt.strftime("%Y-%m-%d")
    labels_df["date"]   = pd.to_datetime(labels_df["date"]).dt.strftime("%Y-%m-%d")

    dataset = DailyProfileDataset(
        profiles_df       = profiles_df,
        labels_df         = labels_df,
        stations_df       = stations_df,
        baseline_df       = baseline_df,
        venue_coords      = PENN_COORDS,
        baseline_daily_df = baseline_daily_df,
    )
    A_hat = build_adjacency(dataset.stations, k=8)
    print(f"  {len(dataset)} days, {len(dataset.stations)} stations")

    print("Loading EventConditionedGNN from backbone checkpoint...")
    model = EventConditionedGNN.from_pretrained(
        checkpoint_path = GNN_MODEL,
        event_emb_dim   = EMBEDDING_DIM,
        film_hidden     = 64,
        freeze_backbone = freeze_backbone,
    )
    return dataset, A_hat, baseline_daily_df, model


def get_encoder() -> EventEncoder:
    cache_path = OUT_NYC / "event_embeddings_cache.json"
    return EventEncoder(cache_path=cache_path)


def run_finetune(
    dataset,
    A_hat:            torch.Tensor,
    baseline_daily_df,
    model:            EventConditionedGNN,
    encoder:          EventEncoder,
    n_epochs:         int   = 40,
    lr:               float = 1e-3,
    device:           str   = "cpu",
) -> pd.DataFrame:
    """
    Fine-tune the FiLM layers using all Copa América 2024 events.

    Training signal: for each event day, minimise MSE between the model's
    predicted Penn Station corridor hourly profile and the observed profile.
    Only FiLM parameters (γ and β MLPs) are updated; the backbone is frozen.

    Returns a DataFrame with per-epoch training loss.
    """
    from data.event_catalog import COPA_AMERICA_2024

    copa_dates         = [ev["date"] for ev in COPA_AMERICA_2024]
    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()
    sid_to_idx         = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    penn_idxs          = [sid_to_idx[s] for s in PENN_CORRIDOR if s in sid_to_idx]
    hour_cols          = [f"h{h:02d}" for h in range(24)]

    train_samples = []
    for ev in COPA_AMERICA_2024:
        date_str = ev["date"]
        if date_str not in dataset.dates:
            print(f"  WARNING: {date_str} not found in dataset, skipping")
            continue

        ref_totals = daily_totals_table[daily_totals_table["date"] == date_str]
        if ref_totals.empty:
            continue
        totals_series = ref_totals.set_index("sid")["daily_total"]

        x = build_node_features(
            date                = date_str,
            stations_df         = dataset.stations,
            baseline_df         = dataset.baseline,
            daily_totals_series = totals_series,
            venue_coords        = METLIFE_COORDS,
            event_flag          = 1,
            device              = "cpu",
            baseline_daily_df   = baseline_daily_df,
        )

        day_profiles  = dataset.profiles[dataset.profiles["date"] == date_str]
        penn_profiles = []
        for sid in PENN_CORRIDOR:
            if sid in sid_to_idx:
                row = day_profiles[day_profiles["sid"] == sid]
                if not row.empty:
                    penn_profiles.append(row[hour_cols].values[0])
        if not penn_profiles:
            continue
        target = np.mean(penn_profiles, axis=0).astype(np.float32)
        t_sum  = target.sum()
        if t_sum > 0:
            target = target / t_sum

        train_samples.append({
            "date":      date_str,
            "match":     ev["match"],
            "x":         torch.tensor(x.numpy(), dtype=torch.float32),
            "target":    torch.tensor(target, dtype=torch.float32),
            "event_emb": torch.tensor(encoder.encode(ev["description"]), dtype=torch.float32),
        })

    if not train_samples:
        print("  ERROR: no training samples found (Copa América dates absent from dataset)")
        return pd.DataFrame()

    print(f"  Fine-tuning on {len(train_samples)} Copa América matches")

    optimizer = torch.optim.Adam(model.film_parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    model.train()
    A_hat_dev = A_hat.to(device)

    log_rows = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s in train_samples:
            optimizer.zero_grad()
            profiles  = model(s["x"].to(device), A_hat_dev, s["event_emb"].to(device))
            penn_pred = profiles[penn_idxs].mean(dim=0)
            loss      = loss_fn(penn_pred, s["target"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_samples)
        log_rows.append({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}")

    model.eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": {"hidden": model.hidden, "event_emb_dim": model.event_emb_dim},
        },
        FILM_MODEL_PATH,
    )
    print(f"  FiLM model saved -> {FILM_MODEL_PATH}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUT_NYC / "finetune_log.csv", index=False)
    return log_df


def run_ablation(
    dataset,
    A_hat:            torch.Tensor,
    baseline_daily_df,
    model_film:       EventConditionedGNN,
    encoder:          EventEncoder,
) -> pd.DataFrame:
    """
    In-sample comparison: backbone GNN vs. FiLM-conditioned GNN on Copa América.

    This is a reference experiment showing the model's maximum capacity on seen
    data.  The LOO-CV result (run_loocv) is the primary out-of-sample metric
    reported in the paper.

    Metric: MAE over the 24-hour normalised hourly profile of the Penn Station
    corridor (mean of stations in PENN_CORRIDOR).
    """
    from data.event_catalog import COPA_AMERICA_2024

    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()
    sid_to_idx         = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    penn_idxs          = [sid_to_idx[s] for s in PENN_CORRIDOR if s in sid_to_idx]
    hour_cols          = [f"h{h:02d}" for h in range(24)]

    # Backbone-only model (FiLM at identity initialisation)
    model_base = copy.deepcopy(model_film)
    model_base.eval()
    model_film.eval()
    rows = []

    with torch.no_grad():
        for ev in COPA_AMERICA_2024:
            date_str = ev["date"]
            if date_str not in dataset.dates:
                continue

            ref_totals = daily_totals_table[daily_totals_table["date"] == date_str]
            if ref_totals.empty:
                continue
            totals_series = ref_totals.set_index("sid")["daily_total"]

            x = build_node_features(
                date                = date_str,
                stations_df         = dataset.stations,
                baseline_df         = dataset.baseline,
                daily_totals_series = totals_series,
                venue_coords        = METLIFE_COORDS,
                event_flag          = 1,
                device              = "cpu",
                baseline_daily_df   = baseline_daily_df,
            )

            day_profiles = dataset.profiles[dataset.profiles["date"] == date_str]
            penn_real    = []
            for sid in PENN_CORRIDOR:
                if sid in sid_to_idx:
                    row = day_profiles[day_profiles["sid"] == sid]
                    if not row.empty:
                        penn_real.append(row[hour_cols].values[0])
            if not penn_real:
                continue
            target = np.mean(penn_real, axis=0).astype(np.float32)
            t_sum  = target.sum()
            if t_sum > 0:
                target = target / t_sum
            target_t = torch.tensor(target)

            event_emb = numpy_to_event_tensor(encoder.encode(ev["description"]))

            prof_base = model_base(x, A_hat, event_emb=None)[penn_idxs].mean(dim=0)
            mae_base  = (prof_base - target_t).abs().mean().item()

            prof_film = model_film(x, A_hat, event_emb=event_emb)[penn_idxs].mean(dim=0)
            mae_film  = (prof_film - target_t).abs().mean().item()

            rows.append({
                "date":            date_str,
                "match":           ev["match"],
                "stage":           ev["stage"],
                "mae_baseline":    mae_base,
                "mae_llm":         mae_film,
                "improvement_pct": (mae_base - mae_film) / (mae_base + 1e-8) * 100,
                "peak_hour_real":  int(target.argmax()),
                "peak_hour_base":  int(prof_base.argmax().item()),
                "peak_hour_llm":   int(prof_film.argmax().item()),
                "profile_real":    target.tolist(),
                "profile_base":    prof_base.numpy().tolist(),
                "profile_llm":     prof_film.numpy().tolist(),
            })

    ablation_df = pd.DataFrame(rows)
    if ablation_df.empty:
        print(f"  ERROR: ablation_df is empty — Copa América dates not in dataset")
        print(f"  Sample dataset dates: {dataset.dates[:5]}")
        return ablation_df

    profile_cols     = [c for c in ["profile_real", "profile_base", "profile_llm"]
                        if c in ablation_df.columns]
    ablation_summary = ablation_df.drop(columns=profile_cols)
    ablation_summary.to_csv(OUT_NYC / "ablation_results.csv", index=False)

    print("\n=== Ablation — Copa América 2024 ===")
    print(ablation_summary[[
        "match", "mae_baseline", "mae_llm", "improvement_pct",
        "peak_hour_real", "peak_hour_llm",
    ]].to_string(index=False))
    return ablation_df


def run_loocv(
    dataset,
    A_hat:            torch.Tensor,
    baseline_daily_df,
    encoder:          EventEncoder,
    n_epochs:         int   = 40,
    lr:               float = 1e-3,
    device:           str   = "cpu",
) -> pd.DataFrame:
    """
    Leave-one-out cross-validation over the three Copa América 2024 matches.

    For each fold k:
      - Train FiLM layers on the two remaining matches.
      - Evaluate on match k (out-of-sample).

    This is the primary evaluation metric reported in the paper.  With only
    three validation events, LOO-CV is the only valid out-of-sample protocol.

    Returns a DataFrame with one row per fold.
    """
    from data.event_catalog import COPA_AMERICA_2024

    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()
    sid_to_idx         = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    penn_idxs          = [sid_to_idx[s] for s in PENN_CORRIDOR if s in sid_to_idx]
    hour_cols          = [f"h{h:02d}" for h in range(24)]
    loss_fn            = nn.MSELoss()

    # Pre-compute node features, targets, and embeddings for all three matches
    all_samples = []
    for ev in COPA_AMERICA_2024:
        date_str = ev["date"]
        if date_str not in dataset.dates:
            continue
        ref_totals = daily_totals_table[daily_totals_table["date"] == date_str]
        if ref_totals.empty:
            continue
        totals_series = ref_totals.set_index("sid")["daily_total"]
        x             = build_node_features(
            date                = date_str,
            stations_df         = dataset.stations,
            baseline_df         = dataset.baseline,
            daily_totals_series = totals_series,
            venue_coords        = METLIFE_COORDS,
            event_flag          = 1,
            device              = "cpu",
            baseline_daily_df   = baseline_daily_df,
        )
        day_profiles = dataset.profiles[dataset.profiles["date"] == date_str]
        penn_real    = []
        for sid in PENN_CORRIDOR:
            if sid in sid_to_idx:
                row = day_profiles[day_profiles["sid"] == sid]
                if not row.empty:
                    penn_real.append(row[hour_cols].values[0])
        if not penn_real:
            continue
        target = np.mean(penn_real, axis=0).astype(np.float32)
        t_sum  = target.sum()
        if t_sum > 0:
            target = target / t_sum

        all_samples.append({
            "date":      date_str,
            "match":     ev["match"],
            "stage":     ev["stage"],
            "x":         torch.tensor(x.numpy(), dtype=torch.float32),
            "target":    torch.tensor(target, dtype=torch.float32),
            "event_emb": torch.tensor(encoder.encode(ev["description"]), dtype=torch.float32),
        })

    if len(all_samples) < 2:
        print(f"  ERROR: only {len(all_samples)} samples available — LOO-CV requires >=2")
        return pd.DataFrame()

    print(f"  LOO-CV: {len(all_samples)} folds")
    rows = []

    for k in range(len(all_samples)):
        test_s   = all_samples[k]
        train_s  = [s for i, s in enumerate(all_samples) if i != k]

        # Re-initialise a fresh FiLM model for each fold
        fold_model = EventConditionedGNN.from_pretrained(
            GNN_MODEL, event_emb_dim=EMBEDDING_DIM, film_hidden=64,
            freeze_backbone=True, device=device,
        )
        fold_model.train()
        optimizer = torch.optim.Adam(fold_model.film_parameters(), lr=lr)
        A_hat_dev = A_hat.to(device)

        for _epoch in range(n_epochs):
            for s in train_s:
                optimizer.zero_grad()
                profiles = fold_model(s["x"].to(device), A_hat_dev, s["event_emb"].to(device))
                pred     = profiles[penn_idxs].mean(dim=0)
                loss     = loss_fn(pred, s["target"].to(device))
                loss.backward()
                optimizer.step()

        fold_model.eval()
        with torch.no_grad():
            # Backbone only (event_emb=None gives identity FiLM)
            prof_base = fold_model(test_s["x"].to(device), A_hat_dev, event_emb=None)[penn_idxs].mean(dim=0)
            # FiLM-conditioned
            prof_llm  = fold_model(test_s["x"].to(device), A_hat_dev, test_s["event_emb"].to(device))[penn_idxs].mean(dim=0)
            tgt       = test_s["target"]

            mae_base = (prof_base - tgt).abs().mean().item()
            mae_llm  = (prof_llm  - tgt).abs().mean().item()

        rows.append({
            "fold":              k + 1,
            "held_out_date":     test_s["date"],
            "held_out_match":    test_s["match"],
            "train_matches":     " | ".join(s["match"] for s in train_s),
            "mae_baseline":      mae_base,
            "mae_llm_oos":       mae_llm,
            "improvement_pct":   (mae_base - mae_llm) / (mae_base + 1e-8) * 100,
            "peak_hour_real":    int(tgt.numpy().argmax()),
            "peak_hour_llm_oos": int(prof_llm.numpy().argmax()),
        })
        print(
            f"  Fold {k+1} | {test_s['match'][:30]:<30} | "
            f"MAE base={mae_base:.5f}  MAE LLM={mae_llm:.5f}  "
            f"delta={rows[-1]['improvement_pct']:+.1f}%"
        )

    loocv_df = pd.DataFrame(rows)
    loocv_df.to_csv(OUT_NYC / "loocv_results.csv", index=False)
    print(f"\n  LOO-CV mean improvement: {loocv_df['improvement_pct'].mean():+.1f}%")
    print(f"  MAE baseline (avg):      {loocv_df['mae_baseline'].mean():.5f}")
    print(f"  MAE LLM OOS (avg):       {loocv_df['mae_llm_oos'].mean():.5f}")
    return loocv_df


def plot_loocv(loocv_df: pd.DataFrame):
    """Bar chart: out-of-sample MAE per LOO-CV fold."""
    if loocv_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(loocv_df))
    w = 0.35

    ax.bar(x - w/2, loocv_df["mae_baseline"] * 100, w,
           label="Baseline GNN", color="#457b9d", alpha=0.85)
    ax.bar(x + w/2, loocv_df["mae_llm_oos"] * 100, w,
           label="LLM-Augmented (OOS)", color="#e63946", alpha=0.85)

    for i, row in loocv_df.iterrows():
        ax.text(
            i,
            max(row["mae_baseline"], row["mae_llm_oos"]) * 100 + 0.01,
            f"{row['improvement_pct']:+.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2a9d8f",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Fold {r['fold']}\n{r['held_out_match'][:22]}" for _, r in loocv_df.iterrows()],
        fontsize=8,
    )
    ax.set_ylabel("MAE × 100 (hourly profile)", fontsize=10)
    ax.set_title(
        "Leave-One-Out CV: Baseline GNN vs LLM-Augmented (FiLM)\n"
        "Copa América 2024 at MetLife — Penn Station Corridor (out-of-sample)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUT_FIGS / "fig5_loocv_copa_america.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {path.name}")


def plot_ablation(ablation_df: pd.DataFrame):
    """Line plot: observed vs. predicted hourly profiles per Copa América match."""
    if ablation_df.empty:
        return

    n_games = len(ablation_df)
    fig, axes = plt.subplots(1, n_games, figsize=(6 * n_games, 5), sharey=True)
    if n_games == 1:
        axes = [axes]

    hours = list(range(24))
    for ax, (_, row) in zip(axes, ablation_df.iterrows()):
        ax.plot(hours, [v * 100 for v in row["profile_real"]], "k-",
                lw=2.5, label="Observed", zorder=3)
        ax.plot(hours, [v * 100 for v in row["profile_base"]], "b--",
                lw=1.8, label="Baseline GNN", alpha=0.8)
        ax.plot(hours, [v * 100 for v in row["profile_llm"]], "r-",
                lw=1.8, label="LLM-augmented", alpha=0.9)

        ax.axvspan(18, 23, alpha=0.07, color="gold")
        ax.set_title(
            f"{row['match']}\n"
            f"MAE base={row['mae_baseline']:.4f}  LLM={row['mae_llm']:.4f} "
            f"({row['improvement_pct']:+.1f}%)",
            fontsize=9,
        )
        ax.set_xlabel("Hour of day (ET)", fontsize=9)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)], fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Share of daily ridership (%)", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Ablation Study: Copa América 2024 at MetLife\n"
        "Baseline GNN vs LLM-Augmented (FiLM) — Penn Station Corridor",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    path = OUT_FIGS / "fig5_ablation_copa_america.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {path.name}")


def run_wc2026_inference(
    dataset,
    A_hat:            torch.Tensor,
    baseline_daily_df,
    model_film:       EventConditionedGNN,
    encoder:          EventEncoder,
    wc_proj_df:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate WC2026 hourly ridership profiles using the fine-tuned FiLM model.

    For each WC2026 match, the method:
      1. Retrieves the match's semantic embedding from the event catalog.
      2. Scales a reference day's ridership to match the projected mid-scenario total.
      3. Feeds the scaled features and embedding through the FiLM-conditioned GNN.
      4. Extracts the Penn Station corridor profile.

    Returns a DataFrame with one row per (match, hour).
    """
    from data.event_catalog import WC2026_NYC_CATALOG

    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()
    sid_to_idx         = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    penn_idxs          = [sid_to_idx[s] for s in PENN_CORRIDOR if s in sid_to_idx]
    wc_catalog_by_date = {ev["date"]: ev for ev in WC2026_NYC_CATALOG}

    records    = []
    model_film.eval()

    with torch.no_grad():
        for _, proj_row in wc_proj_df.iterrows():
            date_str = proj_row["date"]
            proj_mid = float(proj_row["proj_mid"])

            ev = wc_catalog_by_date.get(date_str)
            if ev is None:
                print(f"  WARNING: {date_str} not in WC2026 catalog; using generic embedding")
                event_text = f"FIFA World Cup 2026 match at MetLife Stadium on {date_str}"
                event_id   = "unknown"
            else:
                event_text = ev["description"]
                event_id   = ev["id"]

            event_emb = numpy_to_event_tensor(encoder.encode(event_text))

            # Choose a reference day with the same weekday in June–July
            wday       = pd.Timestamp(date_str).dayofweek
            candidates = [
                d for d in dataset.dates
                if pd.Timestamp(d).dayofweek == wday and pd.Timestamp(d).month in [6, 7]
            ]
            if not candidates:
                candidates = [d for d in dataset.dates if pd.Timestamp(d).dayofweek == wday]
            if not candidates:
                continue

            ref_date   = candidates[len(candidates) // 2]
            ref_totals = daily_totals_table[daily_totals_table["date"] == ref_date]
            if ref_totals.empty:
                continue

            total_ref     = ref_totals["daily_total"].sum()
            scale_factor  = proj_mid / max(total_ref, 1.0)
            scaled_totals = ref_totals.set_index("sid")["daily_total"] * scale_factor

            x = build_node_features(
                date                = date_str,
                stations_df         = dataset.stations,
                baseline_df         = dataset.baseline,
                daily_totals_series = scaled_totals,
                venue_coords        = METLIFE_COORDS,
                event_flag          = 1,
                device              = "cpu",
                baseline_daily_df   = baseline_daily_df,
            )

            profiles     = model_film(x, A_hat, event_emb=event_emb)
            penn_profile = profiles[penn_idxs].mean(dim=0).numpy()

            for h, w in enumerate(penn_profile):
                records.append({
                    "date":       date_str,
                    "matchup":    proj_row["matchup"],
                    "stage":      proj_row["stage"],
                    "hour":       h,
                    "weight_llm": float(w),
                    "riders_est": float(w * proj_mid),
                    "event_id":   event_id,
                })

    df = pd.DataFrame(records)
    df.to_csv(OUT_NYC / "hourly_profiles_wc2026_llm.csv", index=False)
    return df


def plot_llm_vs_baseline(wc_llm: pd.DataFrame, wc_baseline_path: Path):
    """Hourly profile difference: LLM-augmented minus baseline GNN for WC2026."""
    if not wc_baseline_path.exists():
        print("  Skipping Fig 6: baseline hourly profiles file not found")
        return

    baseline = pd.read_csv(wc_baseline_path).rename(
        columns={"weight_gnn": "weight_base"}
    )
    merged = wc_llm.merge(
        baseline[["date", "hour", "weight_base"]], on=["date", "hour"], how="inner"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = {"Final": "#000", "Round of 16": "#b5838d", "Round of 32": "#6d6875"}

    for stage, grp in merged.groupby("stage"):
        llm_hourly  = grp.groupby("hour")["weight_llm"].mean()
        base_hourly = grp.groupby("hour")["weight_base"].mean()
        diff        = (llm_hourly - base_hourly) * 100
        ax.plot(
            diff.index, diff.values,
            label=f"{stage} (LLM - base)",
            color=colors.get(stage, "#aaa"), lw=1.8, alpha=0.85,
        )

    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.axvspan(18, 21, alpha=0.07, color="gold")
    ax.set_xlabel("Hour of day (ET)", fontsize=11)
    ax.set_ylabel("Delta share of daily ridership (pp)", fontsize=11)
    ax.set_title(
        "LLM-Augmented vs Baseline GNN — Hourly Profile Difference WC2026\n"
        "(Penn Station Corridor; positive = LLM shifts more demand to that hour)",
        fontsize=11,
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)], rotation=45)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = OUT_FIGS / "fig6_wc2026_llm_vs_baseline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {path.name}")


def run(phase: str = "all"):
    OUT_NYC.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    dataset, A_hat, baseline_daily_df, model = load_dataset_and_model(freeze_backbone=True)
    encoder = get_encoder()

    if phase in ("finetune", "all"):
        print("\n" + "=" * 55)
        print("Phase 1: FiLM fine-tuning on Copa América 2024")
        print("=" * 55)
        run_finetune(dataset, A_hat, baseline_daily_df, model, encoder)

    if phase in ("loocv", "all"):
        print("\n" + "=" * 55)
        print("Phase 2: Leave-one-out cross-validation (out-of-sample)")
        print("=" * 55)
        loocv_df = run_loocv(dataset, A_hat, baseline_daily_df, encoder)
        plot_loocv(loocv_df)

    # Load the full fine-tuned model for ablation and inference
    if phase in ("ablation", "inference", "all") and FILM_MODEL_PATH.exists():
        print(f"\nLoading fine-tuned FiLM model from {FILM_MODEL_PATH}")
        ck = torch.load(FILM_MODEL_PATH, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        model.eval()

    if phase in ("ablation", "all"):
        print("\n" + "=" * 55)
        print("Phase 3: Ablation — in-sample Copa América 2024")
        print("=" * 55)
        ablation_df = run_ablation(dataset, A_hat, baseline_daily_df, model, encoder)
        plot_ablation(ablation_df)

    if phase in ("inference", "all"):
        print("\n" + "=" * 55)
        print("Phase 4: WC2026 inference with LLM-augmented model")
        print("=" * 55)
        wc_proj_path = OUT_NYC / "wc2026_projections_nyc.csv"
        if not wc_proj_path.exists():
            print("  ERROR: run nyc_shocks.py first to generate projection inputs")
            return
        wc_proj = pd.read_csv(wc_proj_path)
        wc_llm  = run_wc2026_inference(
            dataset, A_hat, baseline_daily_df, model, encoder, wc_proj
        )
        plot_llm_vs_baseline(wc_llm, OUT_NYC / "hourly_profiles_wc2026.csv")

        print("\n=== WC2026 LLM-augmented inference complete ===")
        summary = (
            wc_llm.groupby(["date", "matchup", "stage"])
                  .apply(
                      lambda g: pd.Series({
                          "peak_hour_llm":      int(g.loc[g["weight_llm"].idxmax(), "hour"]),
                          "peak_share_pct_llm": float(g["weight_llm"].max() * 100),
                          "evening_share_llm":  float(
                              g[g["hour"].between(18, 23)]["weight_llm"].sum() * 100
                          ),
                      })
                  )
                  .reset_index()
        )
        print(summary[["matchup", "stage", "peak_hour_llm",
                        "peak_share_pct_llm", "evening_share_llm"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-augmented GNN pipeline for WC2026 NYC demand projection."
    )
    parser.add_argument(
        "--phase",
        choices=["finetune", "loocv", "ablation", "inference", "all"],
        default="all",
        help="Pipeline phase to execute (default: all).",
    )
    args = parser.parse_args()
    run(phase=args.phase)
