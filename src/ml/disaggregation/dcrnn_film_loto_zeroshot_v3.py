"""
ml/disaggregation/dcrnn_film_loto_zeroshot_v3.py -- redirect check (2026-07-08).

WHY THIS SCRIPT EXISTS:
classical_ml_loto_zeroshot_v3.py showed that a plain RandomForest (NO semantic
conditioning at all) beats the production GNN backbone + FiLM in most zero-shot
LOTO-CV folds, using the 63-event catalog (event_catalog_expanded_v3.json).
That result used the SIMPLE GCN backbone (model_v2.py's EventConditionedGNN) --
the same backbone the response letter already documents (R3.3/R6.4) as WORSE
than a day-of-week average on overall MAE.

Separately, R3.3/R6.4 already reports a strong positive FiLM result on a more
competitive backbone -- DiffusionConvGNN (model_dcrnn_v2.py's
EventConditionedDiffusionConvGNN), +57.3% +/- 15.5%, t=30.4, p<1e-6, 27 obs
(dcrnn_film_loocv_multiseed_b4.py). BUT that result used the SAME small n=3
Copa-America-only LOO-CV protocol reviewers already flagged as too small
(R3.2/R6.3) -- it was NEVER tested zero-shot across event TYPES the way RF was.

THE QUESTION THIS SCRIPT ANSWERS: does RF still beat FiLM when FiLM conditions
the competitive DiffusionConvGNN backbone instead of the weak plain GCN, under
the EXACT SAME zero-shot LOTO-CV protocol (8 real folds, 63-event catalog) used
for the RF comparison? If yes, the "RF beats semantic conditioning" finding is
robust to backbone choice and the redirect fails. If no -- if FiLM-on-DCRNN
beats RF -- that is a much more defensible headline than the original
FiLM-on-simple-GCN story, and worth rewriting the paper's framing around.

METHOD: identical sample construction to the other _v3 diagnostics (CLEAN_TEXT
leakage-stripped descriptions via time_conditioning_loto_test.build_sample,
same MSG venue patch for the 3 new event types). For each of the 9 available
DiffusionConvGNN backbone-training checkpoints (seeds 1-8, 42 -- the same 9
used in dcrnn_film_loocv_multiseed_b4.py, so results are directly comparable
to that "stability across backbone-training-seed" framing) and each of the 8
EVAL_TYPES, FiLM is fine-tuned zero-shot (trained on the other 7 types, never
seeing the held-out type) and evaluated against both the no-FiLM backbone and
RF's already-computed per-type MAE (classical_ml_loto_zeroshot_v3.csv).

RUN THIS LOCALLY, NOT IN A CONSTRAINED SANDBOX: 9 seeds x 8 folds x 40 FiLM
fine-tuning epochs each, all on CPU, is too slow for a ~35-45s call budget.
It IS resumable -- re-running the script skips (backbone_seed, held_out_type)
pairs already present in the output CSV -- so you can stop and resume anytime,
but a single uninterrupted local run should be far more practical than the
sandbox's short-call pattern used for the RF/Mantel/kNN diagnostics.

Quick sanity check first (recommended): run with --seeds 42 to do just ONE
backbone seed across all 8 types before committing to the full 9-seed run --
this alone (8 folds instead of 72) already tells you whether the redirect has
legs, and takes ~1/9th the time.

Usage:
    python src/ml/disaggregation/dcrnn_film_loto_zeroshot_v3.py --seeds 42
    python src/ml/disaggregation/dcrnn_film_loto_zeroshot_v3.py
    python src/ml/disaggregation/dcrnn_film_loto_zeroshot_v3.py --minutes 15

Outputs:
    outputs/paper/nyc/dcrnn_film_loto_zeroshot_v3.csv          (per event_id row)
    outputs/paper/nyc/dcrnn_vs_rf_loto_comparison_v3.csv       (final per-type comparison,
                                                                 written once ALL requested
                                                                 (seed, type) pairs are done)

Requires: classical_ml_loto_zeroshot_v3.py must have been run already (its
output CSV is read for the RF comparison column; if missing, this script still
runs and reports FiLM's own numbers, just without the RF column).
"""
from __future__ import annotations
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "paper"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.disaggregation.model_dcrnn import build_diffusion_matrices
from ml.disaggregation.model_dcrnn_v2 import EventConditionedDiffusionConvGNN
from ml.disaggregation.event_encoder import EventEncoder
from paper.tier2.llm_augmented_gnn import load_dataset_and_model
from paper.tier2.time_conditioning_loto_test import build_sample
import paper.tier2.time_conditioning_loto_test as tct

OUT_NYC = ROOT / "outputs" / "paper" / "nyc"
OUT_NYC.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_NYC / "dcrnn_film_loto_zeroshot_v3.csv"
COMPARISON_CSV = OUT_NYC / "dcrnn_vs_rf_loto_comparison_v3.csv"
RF_REF = OUT_NYC / "classical_ml_loto_zeroshot_v3.csv"
CATALOG_V3 = ROOT / "data" / "nyc" / "event_catalog_expanded_v3.json"

ALL_BACKBONE_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 42]
# Same 8 real zero-shot folds as classical_ml_loto_zeroshot_v3.py.
# sports_other_msg (2 events: boxing + Jimmy V Classic) excluded from evaluation
# for the same reason as there (too few to hold out meaningfully), but its rows
# remain in the training pool for every other fold via ordinary catalog membership.
EVAL_TYPES = ["soccer_international", "concert_pop", "parade_street", "race_marathon",
              "sports_nfl_metlife", "sports_nba_msg", "sports_nhl_msg", "concert_msg"]
CATALOG_TYPES = EVAL_TYPES + ["sports_other_msg"]
N_EPOCHS, LR = 40, 3e-4
DEVICE = "cpu"


def patch_msg_venue_support():
    """Same patch applied in semantic_outcome_correlation_test_v3.py /
    embedding_knn_test_v3.py so the shared time_conditioning_loto_test module
    knows about the MSG venue and the 3 new event types."""
    tct.VENUE_SIDS.setdefault("MSG", [164, 318, 607, 319, 403, 404, 13])
    tct.VENUE_COORDS.setdefault("MSG", (40.7505, -73.9934))
    tct.TYPE_TO_VENUE["sports_nfl_metlife"] = "MetLife_Penn"
    tct.TYPE_TO_VENUE["sports_nba_msg"] = "MSG"
    tct.TYPE_TO_VENUE["sports_nhl_msg"] = "MSG"
    tct.TYPE_TO_VENUE["concert_msg"] = "MSG"
    tct.TYPE_TO_VENUE["sports_other_msg"] = "MSG"  # training-only, excluded from EVAL_TYPES


def load_samples(dataset, baseline_daily_df, encoder):
    with open(CATALOG_V3, encoding="utf-8") as f:
        catalog_raw = json.load(f)
    catalog = [e for e in catalog_raw if e["event_type"] in CATALOG_TYPES]
    for e in catalog:
        e.setdefault("event_id", f"{e['event_type']}_{e['date']}")

    sid_to_idx = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()

    samples = []
    for ev in catalog:
        s = build_sample(ev, dataset, baseline_daily_df, sid_to_idx, daily_totals_table, encoder)
        if s is not None:
            samples.append(s)
    return samples


def evaluate_backbone_only(ckpt_path, samples, P_f, P_b):
    """No FiLM at all (event_emb=None) -- fold-independent, computed once per
    backbone seed and reused across all 8 held-out-type folds."""
    model = EventConditionedDiffusionConvGNN.from_pretrained(
        ckpt_path, event_emb_dim=384, film_hidden=64, freeze_backbone=True, device=DEVICE,
    )
    model.eval()
    maes = {}
    with torch.no_grad():
        for s in samples:
            out = model(s["x"], P_f, P_b, event_emb=None)[s["venue_idx"]].mean(dim=0).numpy()
            maes[s["event_id"]] = float(np.abs(out - s["target"].numpy()).mean())
    return maes


def fine_tune_film(train_samples, P_f, P_b, ckpt_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EventConditionedDiffusionConvGNN.from_pretrained(
        ckpt_path, event_emb_dim=384, film_hidden=64, freeze_backbone=True, device=DEVICE,
    )
    model.train()
    optimizer = torch.optim.Adam(model.film_parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(N_EPOCHS):
        for s in train_samples:
            optimizer.zero_grad()
            out = model(s["x"], P_f, P_b, event_emb=s["event_emb"])
            pred = out[s["venue_idx"]].mean(dim=0)
            loss = loss_fn(pred, s["target"])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated subset of backbone seeds, e.g. '42' for a quick "
                             "single-seed sanity check, or '1,2,3'. Default: all 9.")
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="Soft time budget in minutes for this call before stopping "
                             "(re-run to resume). Default 30.")
    args = parser.parse_args()
    backbone_seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                      else ALL_BACKBONE_SEEDS)
    time_budget_s = args.minutes * 60

    t0 = time.time()
    print("=" * 70)
    print("Redirect check -- FiLM on DiffusionConvGNN, zero-shot LOTO-CV, 63-event catalog")
    print(f"Backbone seeds this run: {backbone_seeds}")
    print("=" * 70)

    dataset, _, baseline_daily_df, _ = load_dataset_and_model(freeze_backbone=True)
    stations = dataset.stations
    P_f, P_b = build_diffusion_matrices(stations, k=8)
    patch_msg_venue_support()
    encoder = EventEncoder(cache_path=OUT_NYC / "leakage_ablation_embeddings_cache.json")

    samples = load_samples(dataset, baseline_daily_df, encoder)
    print(f"{len(samples)} events with real clean-text embeddings")
    by_type = {}
    for s in samples:
        by_type.setdefault(s["event_type"], []).append(s)
    for t in EVAL_TYPES:
        print(f"  {t}: {len(by_type.get(t, []))} events")

    if OUT_CSV.exists():
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(r["backbone_seed"], r["held_out_type"]) for r in rows}
        print(f"Resuming: {len(done)} (seed, type) pairs already done")
    else:
        rows, done = [], set()

    for seed in backbone_seeds:
        if time.time() - t0 > time_budget_s:
            print(f"\nTime budget ({args.minutes} min) reached -- stopping. Re-run to resume.")
            break
        ckpt_path = ROOT / "outputs" / "nyc" / f"manhattan_model_dcrnn_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  MISSING checkpoint {ckpt_path.name} -- skipping seed {seed}")
            continue

        base_maes = evaluate_backbone_only(ckpt_path, samples, P_f, P_b)

        for held_out_type in EVAL_TYPES:
            if (seed, held_out_type) in done:
                continue
            if time.time() - t0 > time_budget_s:
                print(f"\nTime budget ({args.minutes} min) reached -- stopping. Re-run to resume.")
                break
            test_samples = [s for s in samples if s["event_type"] == held_out_type]
            train_samples = [s for s in samples if s["event_type"] != held_out_type]
            if not test_samples or not train_samples:
                print(f"  [seed={seed}][{held_out_type}] skipped -- no samples")
                continue

            t1 = time.time()
            film_model = fine_tune_film(train_samples, P_f, P_b, ckpt_path, seed)
            with torch.no_grad():
                for s in test_samples:
                    out_film = film_model(s["x"], P_f, P_b, event_emb=s["event_emb"])[
                        s["venue_idx"]].mean(dim=0).numpy()
                    mae_film = float(np.abs(out_film - s["target"].numpy()).mean())
                    rows.append({
                        "backbone_seed": seed, "held_out_type": held_out_type,
                        "event_id": s["event_id"],
                        "mae_backbone": base_maes[s["event_id"]],
                        "mae_film": mae_film,
                    })
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            fold_rows = [r for r in rows if r["backbone_seed"] == seed and r["held_out_type"] == held_out_type]
            mean_base = np.mean([r["mae_backbone"] for r in fold_rows])
            mean_film = np.mean([r["mae_film"] for r in fold_rows])
            imp = (mean_base - mean_film) / (mean_base + 1e-8) * 100
            print(f"  [seed={seed}][{held_out_type}] n={len(test_samples)} "
                  f"mae_backbone={mean_base:.4f} mae_film={mean_film:.4f} "
                  f"Delta_vs_backbone={imp:+.1f}% ({time.time()-t1:.1f}s)")

    df = pd.DataFrame(rows)
    n_pairs_done = df.groupby(["backbone_seed", "held_out_type"]).ngroups if len(df) else 0
    n_pairs_wanted = len(backbone_seeds) * len(EVAL_TYPES)
    print(f"\n{n_pairs_done}/{n_pairs_wanted} requested (seed, type) pairs done this run "
          f"({len(done) + n_pairs_done - len(done)} total in file).")

    if df.empty:
        return
    done_types = set(df["held_out_type"].unique())
    if len(backbone_seeds) > 1 and df.groupby("held_out_type")["backbone_seed"].nunique().min() < len(backbone_seeds):
        print("Not all requested seeds are complete for every type yet -- re-run to continue "
              "before trusting the final comparison below.")

    print("\n" + "=" * 70)
    print("COMPARISON SO FAR: RF zero-shot vs DiffusionConvGNN+FiLM zero-shot")
    print("(averaged over whatever backbone seeds are done so far per type)")
    print("=" * 70)
    per_type = df.groupby("held_out_type").agg(
        mae_film_mean=("mae_film", "mean"),
        mae_film_std=("mae_film", "std"),
        mae_backbone_mean=("mae_backbone", "mean"),
        n_seed_obs=("mae_film", "size"),
        n_backbone_seeds=("backbone_seed", "nunique"),
    ).reset_index()

    if RF_REF.exists():
        rf = pd.read_csv(RF_REF)[["held_out_type", "mae_rf_zeroshot"]]
        merged = per_type.merge(rf, on="held_out_type", how="left")
        merged["film_vs_rf_improvement_pct"] = (
            (merged["mae_rf_zeroshot"] - merged["mae_film_mean"]) / (merged["mae_rf_zeroshot"] + 1e-8) * 100
        )
        merged["film_vs_backbone_improvement_pct"] = (
            (merged["mae_backbone_mean"] - merged["mae_film_mean"]) / (merged["mae_backbone_mean"] + 1e-8) * 100
        )
    else:
        merged = per_type
        print(f"NOTE: {RF_REF.name} not found -- run classical_ml_loto_zeroshot_v3.py first "
              f"for the RF comparison column. Showing FiLM-vs-backbone only.")
        merged["film_vs_backbone_improvement_pct"] = (
            (merged["mae_backbone_mean"] - merged["mae_film_mean"]) / (merged["mae_backbone_mean"] + 1e-8) * 100
        )
    print(merged.to_string(index=False))
    merged.to_csv(COMPARISON_CSV, index=False)

    if "film_vs_rf_improvement_pct" in merged.columns:
        n_film_beats_rf = (merged["film_vs_rf_improvement_pct"] > 0).sum()
        print(f"\nFiLM-on-DiffusionConvGNN beats RF in {n_film_beats_rf}/{len(merged)} types.")
        print("If this is a clear majority (>=6/8), the redirect toward DiffusionConvGNN+FiLM "
              "as the headline result has real legs. If it is <=2/8 (matching the simple-GCN "
              "pattern), the 'RF beats semantic conditioning' finding is robust to backbone "
              "choice and the redirect does not hold up.")

    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {COMPARISON_CSV}")
    print(f"\nTotal time this run: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
