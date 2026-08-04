# SE-GNN: Sentence-Encoder-Augmented Spatial-Temporal GNN for Transit Demand Forecasting

Code repository for:

> **Anticipating Transit Demand Surges for Mega-Events: A Sentence-Encoder-Augmented Spatial-Temporal Graph Neural Network Applied to FIFA World Cup 2026**  
> J. A. Lucio-Rojas, Y. A. Ríos-Solís, F. Elizalde-Ramírez  
> *Expert Systems With Applications* (in revision)

> **Note on repository status:** this repository is updated alongside the manuscript on a best-effort basis and may lag the latest submitted revision. The "Key results" table below reflects the version currently in revision. If you need the exact code/data version behind a specific number in the paper, please open an issue.

---

## Overview

This repository contains the code for a hierarchical framework to forecast transit demand surges during mega-events, applied to two FIFA World Cup 2026 host cities: Mexico City (CDMX) and New York/New Jersey (NYC/NJ).

The framework has three tiers:

- **Tier 1** - Robust z-score shock estimation from historical GTFS and ridership data (both cities)
- **Tier 2** - Spatial-Temporal GNN that disaggregates projected daily demand into 24-hour station-level profiles (NYC) or across Metro stations (CDMX)
- **Tier 2b** - SE-GNN: Tier 2 backbone augmented with FiLM conditioning on sentence-encoder embeddings of event descriptions, enabling few-shot adaptation to unseen event types

### Key results

| Experiment | Result |
|------------|--------|
| Copa América 2024 LOO-CV (NYC, n=3) | best case: **−67.5% MAE** vs. baseline GNN; **+12.9%** vs. a day-of-week historical average (the more conservative baseline) |
| k-shot LOTO-CV, k=1 (NYC, 5 types, 21-event catalog) | **+19.1% ± 1.1%** macro-averaged improvement |
| k-shot LOTO-CV, k=1, DiffusionConvGNN backbone (NYC, competitive baseline) | consistent phase-transition pattern; see `dcrnn_film_loocv_multiseed_b4.py` |
| CDMX spatial LOTO-CV (196 stations, 246 shock days, 12-seed mean) | **+16.8% MAE** improvement, cosine similarity 0.971 |
| CDMX real-world validation vs. observed WC2026 ridership (4 played matches) | model underperforms a naive historical baseline by **−24.0%** on average; see paper §9.6 for the operational explanation (security/access changes, not a modeling failure of demand semantics) |

The CDMX row above supersedes an earlier single-seed report (+20.0%, cosine 0.972) after correcting a coarse same-Metro-line indicator feature to a distance-decay-weighted one (`cdmx_gnn_loto_catchment_fix.py`); see that script's docstring and paper §9.3/§11.2 for detail.

---

## Installation

```bash
conda env create -f environment.yml
conda activate se-gnn-wc2026
```

---

## Data setup

Raw ridership data are not included in this repository due to file size. See individual README files for download instructions:

- [`data/nyc/README.md`](data/nyc/README.md) - MTA Subway Hourly Ridership (data.ny.gov)
- [`data/cdmx/README.md`](data/cdmx/README.md) - SEMOVI Metro CDMX ridership (datos.cdmx.gob.mx)

Event catalogs are included and require no download:

- `data/nyc/event_catalog_expanded_v2.json` - 21 event-days, 5 types (used by the original LOTO-CV/k-shot scripts below)
- `data/nyc/event_catalog_expanded_v3.json` - 63 event-days, 9 types (adds 42 Madison Square Garden events: NBA, NHL, concerts; used by the DCRNN zero-shot LOTO-CV comparison)
- `data/cdmx/event_catalog_2016_2025.csv` - 246 CDMX shock-day events (2016–2025)

---

## Reproducing paper results

All scripts use paths relative to the repository root. Run from the repo root directory.

### Step 0 - Prepare data
```bash
# NYC step 0a: convert raw MTA CSV to station-level parquet (requires ~17 GB raw CSV)
python src/demand/prepare_data.py --input data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv

# NYC step 0b: build normalized daily profile dataset for GNN training
python src/demand/build_dataset.py

# CDMX: preprocess multimodal ridership
python src/paper/tier1/cdmx_shocks.py --preprocess
```

### Step 1 - Tier 1: Shock estimation (Table 2 and Table 3 in paper)
```bash
python src/paper/tier1/nyc_shocks.py      # NYC Penn corridor uplifts
python src/paper/tier1/cdmx_shocks.py     # CDMX multimodal uplifts
```

### Step 2 - Train GNN backbone (required before Steps 3–5)
```bash
python src/ml/disaggregation/train.py
# Saves checkpoint to: outputs/nyc/manhattan_model.pt
# Estimated time: ~30 min on CPU, ~8 min on GPU
```

### Step 3 - Copa América LOO-CV (Table 4, best-case −67.5% / +12.9% vs. DoW baseline)
```bash
python src/paper/tier2/llm_augmented_gnn.py --phase loocv
# Output: outputs/paper/nyc/loocv_results.csv
```

### Step 4 - k-Shot LOTO-CV ablation (Table 11, +19.1% at k=1)
```bash
python src/paper/tier2/run_kshot_ablation_v2.py
# Output: outputs/paper/nyc/kshot_ablation_v2_aggregate.csv
# Estimated time: ~45 min on GPU (3 seeds × 5 types × 4 k-values)
```

### Step 5 - Bootstrap confidence intervals
```bash
python src/paper/tier2/run_bootstrap_ci.py
# Output: outputs/paper/nyc/bootstrap_ci.csv
```

### Step 6 - CDMX spatial LOTO-CV, corrected feature (Table 13, +16.8% result)
```bash
# Fork of cdmx_gnn_loto.py with a swappable node feature (baseline / catchment / distance_decay);
# the paper reports the distance_decay variant, 12-seed mean.
python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --prepare
python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --train --budget_s 260   # resumable; repeat until all jobs done
python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --summarize
# Original single-seed script (pre-correction) still available for reference:
python src/paper/tier2/cdmx_gnn_loto.py
```

### Step 7 - CDMX real-world validation against observed WC2026 ridership (Table 15)
```bash
python src/ml/disaggregation/cdmx_wc2026_real_validation_distance_decay.py
# Trains a RandomForestRegressor on the same 6-feature vector (no GNN/FiLM) and
# evaluates against real, published per-station ridership for the four played
# WC2026 matches at Estadio Ciudad de México.
```

### Step 8 - DiffusionConvGNN competitive baseline
```bash
python src/ml/disaggregation/train_dcrnn.py
# Saves checkpoint to: outputs/nyc/manhattan_model_dcrnn.pt

python src/ml/disaggregation/eval_dcrnn_vs_backbone.py
# Compares DiffusionConvGNN vs. the GNN backbone on the same NYC data.

python src/paper/tier2/dcrnn_film_loocv_multiseed_b4.py
# FiLM conditioning on the DiffusionConvGNN backbone, multi-seed LOOCV
# (uses model_dcrnn_v2.py's EventConditionedDiffusionConvGNN).
```

### Step 9 - Adjacency sensitivity check
```bash
python src/ml/disaggregation/topology_adjacency.py   # builds adjacency from GTFS stop_sequence topology
python src/ml/disaggregation/compare_adjacency.py    # retrains the backbone under both adjacency graphs for comparison
# Paper reports the full-budget run: 60 epochs, all ~700 train days, 3 seeds.
```

---

## Repository structure

```
se-gnn-transit-wc2026/
├── environment.yml
├── data/
│   ├── nyc/
│   │   ├── event_catalog_expanded_v2.json   # 21 event-days, 5 types
│   │   ├── event_catalog_expanded_v3.json   # 63 event-days, 9 types (MSG expansion)
│   │   └── README.md                        # data download instructions
│   └── cdmx/
│       ├── event_catalog_2016_2025.csv      # 246 CDMX shock-day events
│       └── README.md
├── src/
│   ├── demand/
│   │   ├── prepare_data.py        # Step 0a: convert raw MTA CSV to parquet
│   │   └── build_dataset.py       # Step 0b: build normalized daily profile dataset
│   ├── ml/
│   │   └── disaggregation/
│   │       ├── model.py                                    # GNN backbone (DisaggregationGNN)
│   │       ├── model_v2.py                                 # SE-GNN with FiLM conditioning
│   │       ├── model_dcrnn.py                              # DiffusionConvGNN backbone (competitive baseline)
│   │       ├── model_dcrnn_v2.py                           # EventConditionedDiffusionConvGNN (FiLM on DCRNN backbone)
│   │       ├── train.py                                    # Backbone training script
│   │       ├── train_dcrnn.py                              # DiffusionConvGNN training script
│   │       ├── eval_dcrnn_vs_backbone.py                   # DiffusionConvGNN vs. GNN backbone comparison
│   │       ├── cdmx_wc2026_real_validation_distance_decay.py # CDMX RF validation vs. real WC2026 ridership
│   │       ├── topology_adjacency.py                       # GTFS-topology adjacency construction
│   │       ├── compare_adjacency.py                        # Geographic vs. topology adjacency comparison
│   │       └── event_encoder.py                            # Sentence encoder wrapper (all-MiniLM-L6-v2)
│   └── paper/
│       ├── config.py              # Central path and parameter configuration
│       ├── utils.py               # Shared utilities (z-score, uplift, projection)
│       ├── data/
│       │   └── event_catalog.py   # Event descriptions for sentence encoding
│       ├── tier1/
│       │   ├── cdmx_shocks.py     # CDMX Tier 1 shock estimation
│       │   └── nyc_shocks.py      # NYC Tier 1 shock estimation
│       └── tier2/
│           ├── llm_augmented_gnn.py             # Copa América LOO-CV pipeline
│           ├── multi_venue_loocv.py             # Multi-venue LOTO-CV with baselines
│           ├── run_kshot_ablation_v2.py         # k-shot ablation (main result)
│           ├── run_bootstrap_ci.py              # Bootstrap CI computation
│           ├── cdmx_gnn_loto.py                 # CDMX spatial disaggregation LOTO-CV (original feature)
│           ├── cdmx_gnn_loto_catchment_fix.py   # Fork: corrected node feature (distance-decay), 12-seed
│           └── dcrnn_film_loocv_multiseed_b4.py # DCRNN+FiLM multi-seed LOOCV baseline
└── outputs/                       # Generated outputs (not tracked by git)
```

---

## Citation

```bibtex
@article{lucio2026segnn,
  title   = {Anticipating Transit Demand Surges for Mega-Events:
             A Sentence-Encoder-Augmented Spatial-Temporal Graph Neural Network
             Applied to {FIFA} {World Cup} 2026},
  author  = {Lucio-Rojas, J.~A. and R\'ios-Sol\'is, Y.~A. and Elizalde-Ram\'irez, F.},
  journal = {Expert Systems With Applications},
  year    = {2026},
  note    = {In revision}
}
```

---

## License

Code: MIT License. See `LICENSE`.  
Data: Subject to original data providers' terms (MTA Open Data, SEMOVI CDMX).
