# SE-GNN: Sentence-Encoder-Augmented Spatial-Temporal GNN for Transit Demand Forecasting

Code repository for:

> **Anticipating Transit Demand Surges for Mega-Events: A Sentence-Encoder-Augmented Spatial-Temporal Graph Neural Network Applied to FIFA World Cup 2026**  
> J. A. Lucio-Rojas, Y. A. Ríos-Solís, F. Elizalde-Ramírez  
> *Transportation Research Part C: Emerging Technologies* (under review)

---

## Overview

This repository contains the code for a hierarchical framework to forecast transit demand surges during mega-events, applied to two FIFA World Cup 2026 host cities: Mexico City (CDMX) and New York/New Jersey (NYC/NJ).

The framework has three tiers:

- **Tier 1** — Robust z-score shock estimation from historical GTFS and ridership data (both cities)
- **Tier 2** — Spatial-Temporal GNN that disaggregates projected daily demand into 24-hour station-level profiles (NYC only)
- **Tier 2b** — SE-GNN: Tier 2 backbone augmented with FiLM conditioning on sentence-encoder embeddings of event descriptions, enabling few-shot adaptation to unseen event types

### Key results

| Experiment | Result |
|------------|--------|
| Copa América 2024 LOO-CV (NYC) | **−67.5% MAE** vs. baseline GNN |
| k-shot LOTO-CV, k=1 (NYC, 5 types) | **+19.1% ± 1.1%** macro-averaged improvement |
| CDMX spatial LOTO-CV (196 stations) | **+20.0% MAE** improvement, cosine similarity 0.972 |

---

## Installation

```bash
conda env create -f environment.yml
conda activate se-gnn-wc2026
```

---

## Data setup

Raw ridership data are not included in this repository due to file size. See individual README files for download instructions:

- [`data/nyc/README.md`](data/nyc/README.md) — MTA Subway Hourly Ridership (data.ny.gov)
- [`data/cdmx/README.md`](data/cdmx/README.md) — SEMOVI Metro CDMX ridership (datos.cdmx.gob.mx)

The event catalog (`data/nyc/event_catalog_expanded_v2.json`, 21 events) is included and requires no download.

---

## Reproducing paper results

All scripts use paths relative to the repository root. Run from the repo root directory.

### Step 0 — Prepare data
```bash
# NYC step 0a: convert raw MTA CSV to station-level parquet (requires ~17 GB raw CSV)
python src/demand/prepare_data.py --input data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv

# NYC step 0b: build normalized daily profile dataset for GNN training
python src/demand/build_dataset.py

# CDMX: preprocess multimodal ridership
python src/paper/tier1/cdmx_shocks.py --preprocess
```

### Step 1 — Tier 1: Shock estimation (Table 2 and Table 3 in paper)
```bash
python src/paper/tier1/nyc_shocks.py      # NYC Penn corridor uplifts
python src/paper/tier1/cdmx_shocks.py     # CDMX multimodal uplifts
```

### Step 2 — Train GNN backbone (required before Steps 3–5)
```bash
python src/ml/disaggregation/train.py
# Saves checkpoint to: outputs/nyc/manhattan_model.pt
# Estimated time: ~30 min on CPU, ~8 min on GPU
```

### Step 3 — Copa América LOO-CV (Table 4, −67.5% result)
```bash
python src/paper/tier2/llm_augmented_gnn.py --phase loocv
# Output: outputs/paper/nyc/loocv_results.csv
```

### Step 4 — k-Shot LOTO-CV ablation (Table 11, +19.1% at k=1)
```bash
python src/paper/tier2/run_kshot_ablation_v2.py
# Output: outputs/paper/nyc/kshot_ablation_v2_aggregate.csv
# Estimated time: ~45 min on GPU (3 seeds × 5 types × 4 k-values)
```

### Step 5 — Bootstrap confidence intervals
```bash
python src/paper/tier2/run_bootstrap_ci.py
# Output: outputs/paper/nyc/bootstrap_ci.csv
```

### Step 6 — CDMX spatial LOTO-CV (Table 13, +20.0% result)
```bash
python src/paper/tier2/cdmx_gnn_loto.py
# Output: outputs/paper/cdmx/cdmx_loto_results.csv
```

---

## Repository structure

```
se-gnn-transit-wc2026/
├── environment.yml
├── data/
│   ├── nyc/
│   │   ├── event_catalog_expanded_v2.json   # 21 event-days, 5 types
│   │   └── README.md                        # data download instructions
│   └── cdmx/
│       └── README.md
├── src/
│   ├── demand/
│   │   ├── prepare_data.py        # Step 0a: convert raw MTA CSV to parquet
│   │   └── build_dataset.py       # Step 0b: build normalized daily profile dataset
│   ├── ml/
│   │   └── disaggregation/
│   │       ├── model.py           # GNN backbone (DisaggregationGNN)
│   │       ├── model_v2.py        # SE-GNN with FiLM conditioning
│   │       ├── train.py           # Backbone training script
│   │       └── event_encoder.py   # Sentence encoder wrapper (all-MiniLM-L6-v2)
│   └── paper/
│       ├── config.py              # Central path and parameter configuration
│       ├── utils.py               # Shared utilities (z-score, uplift, projection)
│       ├── data/
│       │   └── event_catalog.py   # Event descriptions for sentence encoding
│       ├── tier1/
│       │   ├── cdmx_shocks.py     # CDMX Tier 1 shock estimation
│       │   └── nyc_shocks.py      # NYC Tier 1 shock estimation
│       └── tier2/
│           ├── llm_augmented_gnn.py      # Copa América LOO-CV pipeline
│           ├── multi_venue_loocv.py      # Multi-venue LOTO-CV with baselines
│           ├── run_kshot_ablation_v2.py  # k-shot ablation (main result)
│           ├── run_bootstrap_ci.py       # Bootstrap CI computation
│           └── cdmx_gnn_loto.py          # CDMX spatial disaggregation LOTO-CV
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
  journal = {Transportation Research Part C: Emerging Technologies},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code: MIT License. See `LICENSE`.  
Data: Subject to original data providers' terms (MTA Open Data, SEMOVI CDMX).
