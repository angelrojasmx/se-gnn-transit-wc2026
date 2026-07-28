"""
ml/disaggregation/train_dcrnn.py -- trains DiffusionConvGNN (B4 baseline) with the
SAME data, split, loss, optimizer config, and target epoch budget (60) as train.py
(production DisaggregationGNN), for a fair, direct comparison.

Sandbox note: build_node_features() recomputes per-day features on every __getitem__
(true of the original train.py too, not a change introduced here) at ~46ms/day, which
makes a full from-scratch epoch loop too slow for a single 45s shell call. This script
precomputes every day's (x, target, is_event) ONCE into memory before training (a
harmless caching optimization -- features are deterministic per day, independent of
epoch), then supports --resume so training can be chunked across multiple short
invocations while producing IDENTICAL results to one uninterrupted run (same seed,
same data order given shuffle uses torch's global RNG state which is saved/restored).

Usage:
    python src/ml/disaggregation/train_dcrnn.py --max_epochs_this_run 8          # first chunk
    python src/ml/disaggregation/train_dcrnn.py --max_epochs_this_run 8 --resume # subsequent chunks
"""
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ml.disaggregation.train import DailyProfileDataset
from ml.disaggregation.model_dcrnn import DiffusionConvGNN, build_diffusion_matrices


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_cfg(seed=42):
    return {
        "data_dir":    str(ROOT / "data/nyc/manhattan"),
        "train_years": [2022, 2023],
        "val_year":    2024,
        "hidden":      64,
        "dropout":     0.2,
        "n_layers":    2,
        "K":           2,
        "k_neighbors": 8,
        "epochs":      60,
        "lr":          1e-3,
        "weight_decay": 1e-4,
        "batch_size":  32,
        "event_alpha": 0.4,
        "patience":    8,
        "venue_coords": (40.8135, -74.0745),
        "checkpoint": str(ROOT / f"outputs/nyc/manhattan_model_dcrnn_seed{seed}.pt"),
        "resume_state": str(ROOT / f"outputs/nyc/manhattan_dcrnn_resume_state_seed{seed}.pt"),
        "train_log":  str(ROOT / f"outputs/nyc/manhattan_train_log_dcrnn_seed{seed}.csv"),
    }


# Kept for backward compatibility with any external import of CFG (seed=42 default paths).
CFG = make_cfg(42)


def precompute(ds):
    """Materialize every (x, target, is_event) once -- avoids recomputing
    build_node_features() on every epoch (the actual bottleneck, not the model)."""
    xs, ys, evs = [], [], []
    for i in range(len(ds)):
        x, y, ev = ds[i]
        xs.append(x)
        ys.append(y)
        evs.append(ev)
    return torch.stack(xs), torch.stack(ys), torch.stack(evs)


def batched_epoch(model, X, Y, EV, P_f, P_b, loss_fn, event_alpha, batch_size,
                   optimizer=None, shuffle=False):
    """X:(D,N,7) Y:(D,N,24) EV:(D,) -- D = number of days. If optimizer is None, eval mode."""
    D = X.shape[0]
    idx = torch.randperm(D) if shuffle else torch.arange(D)
    losses = []
    training = optimizer is not None
    model.train() if training else model.eval()
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, D, batch_size):
            batch_idx = idx[start:start + batch_size]
            x_batch, y_batch, ev_batch = X[batch_idx], Y[batch_idx], EV[batch_idx]
            preds = torch.stack([model(x_batch[b], P_f, P_b) for b in range(x_batch.shape[0])])
            loss_elem = loss_fn(preds, y_batch)
            loss_per_day = loss_elem.mean(dim=(1, 2))
            weights = 1.0 + event_alpha * ev_batch
            loss = (loss_per_day * weights).mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train(max_epochs_this_run, resume, seed=42):
    CFG = make_cfg(seed)
    device = "cpu"
    data_dir = Path(CFG["data_dir"])
    profiles = pd.read_parquet(data_dir / "daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"]).dt.date
    labels = pd.read_csv(data_dir / "event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    baseline = pd.read_parquet(data_dir / "baseline_profile.parquet")
    stations = pd.read_csv(data_dir / "station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)
    bdt_path = data_dir / "baseline_daily_totals.parquet"
    baseline_daily = pd.read_parquet(bdt_path) if bdt_path.exists() else None

    P_f, P_b = build_diffusion_matrices(stations, k=CFG["k_neighbors"])

    train_ds = DailyProfileDataset(profiles, labels, stations, baseline, CFG["venue_coords"],
                                    years=CFG["train_years"], baseline_daily_df=baseline_daily)
    val_ds = DailyProfileDataset(profiles, labels, stations, baseline, CFG["venue_coords"],
                                  years=[CFG["val_year"]], baseline_daily_df=baseline_daily)

    tr_cache_path = ROOT / "outputs/nyc/manhattan_dcrnn_feature_cache_train.pt"
    val_cache_path = ROOT / "outputs/nyc/manhattan_dcrnn_feature_cache_val.pt"
    if tr_cache_path.exists():
        cache = torch.load(tr_cache_path, weights_only=False)
        Xtr, Ytr, EVtr = cache["Xtr"], cache["Ytr"], cache["EVtr"]
        print(f"Loaded cached TRAIN features from {tr_cache_path}")
    else:
        print("Precomputing TRAIN features (one-time cost, chunked)...")
        Xtr, Ytr, EVtr = precompute(train_ds)
        torch.save({"Xtr": Xtr, "Ytr": Ytr, "EVtr": EVtr}, tr_cache_path)
        print(f"Cached TRAIN -> {tr_cache_path}")

    if val_cache_path.exists():
        cache = torch.load(val_cache_path, weights_only=False)
        Xval, Yval, EVval = cache["Xval"], cache["Yval"], cache["EVval"]
        print(f"Loaded cached VAL features from {val_cache_path}")
    else:
        print("Precomputing VAL features (one-time cost, chunked)...")
        Xval, Yval, EVval = precompute(val_ds)
        torch.save({"Xval": Xval, "Yval": Yval, "EVval": EVval}, val_cache_path)
        print(f"Cached VAL -> {val_cache_path}")
    print(f"Train: {Xtr.shape[0]} dias | Val: {Xval.shape[0]} dias")

    resume_path = Path(CFG["resume_state"])
    fresh_start = not (resume and resume_path.exists())
    if fresh_start:
        # Seed BEFORE model instantiation so weight init is also seed-controlled
        # (previously this was seeded only after model creation, so init never varied by seed).
        set_seed(seed)

    model = DiffusionConvGNN(hidden=CFG["hidden"], dropout=CFG["dropout"],
                              n_layers=CFG["n_layers"], K=CFG["K"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    loss_fn = nn.MSELoss(reduction="none")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log_rows = []

    if resume and resume_path.exists():
        state = torch.load(resume_path, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        torch.set_rng_state(state["rng_state"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        patience_counter = state["patience_counter"]
        log_rows = state["log_rows"]
        print(f"Resumed from epoch {state['epoch']} (best_val_loss={best_val_loss:.5f}, patience={patience_counter})")

    ckpt_path = Path(CFG["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, min(start_epoch + max_epochs_this_run, CFG["epochs"] + 1)):
        train_loss = batched_epoch(model, Xtr, Ytr, EVtr, P_f, P_b, loss_fn,
                                    CFG["event_alpha"], CFG["batch_size"], optimizer=optimizer, shuffle=True)
        val_loss = batched_epoch(model, Xval, Yval, EVval, P_f, P_b, loss_fn,
                                  CFG["event_alpha"], CFG["batch_size"], optimizer=None, shuffle=False)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best_val_loss": float(best_val_loss), "cfg": CFG}, ckpt_path)
        else:
            patience_counter += 1

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "best_val_loss": best_val_loss})
        print(f"Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f} | best={best_val_loss:.5f} | patience={patience_counter}")
        last_epoch = epoch

        if patience_counter >= CFG["patience"]:
            print(f"Early stopping en epoca {epoch}")
            stopped_early = True
            break

    pd.DataFrame(log_rows).to_csv(CFG["train_log"], index=False)

    done = stopped_early or last_epoch >= CFG["epochs"]
    if not done:
        torch.save({"epoch": last_epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(), "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter, "log_rows": log_rows}, resume_path)
        print(f"CHUNK DONE, not finished -- resume state saved -> {resume_path} (last_epoch={last_epoch}/{CFG['epochs']})")
    else:
        print(f"TRAINING COMPLETE at epoch {last_epoch}. Best val_loss: {best_val_loss:.5f}")
        if resume_path.exists():
            resume_path.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_epochs_this_run", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(max_epochs_this_run=args.max_epochs_this_run, resume=args.resume, seed=args.seed)
