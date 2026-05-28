# NYC Data

This directory should contain the following files before running any Tier 1 or Tier 2 scripts.

## Required files

### 1. MTA Subway Hourly Ridership (Tier 2 — GNN training)
**File generated:** `raw/mta_manhattan_2022_2024.parquet`

Download the raw CSV from the MTA Open Data portal:
- URL: https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-2022/wujg-7c2s
- Filter: January 2022 – December 2024
- Save to: `data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv`

Then run the two-step preparation pipeline:
```bash
# Step 0a: filter to Manhattan stations and convert to parquet (~17 GB CSV → ~26 MB parquet)
python src/demand/prepare_data.py --input data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv

# Step 0b: build normalized daily profile dataset for GNN training
python src/demand/build_dataset.py
```
Step 0b generates `manhattan/daily_profiles.parquet`, `manhattan/baseline_profile.parquet`,
`manhattan/baseline_daily_totals.parquet`, `manhattan/station_lookup.csv`, and `manhattan/event_labels.csv`.

### 2. Station lookup and Penn corridor spikes
**Files:** `mta_manhattan_lookup.csv`, `penn_corridor_spikes_2022_2024.csv`  
Both files are included in this repository — no download required.

### 3. Baseline profile
**File:** `manhattan/baseline_profile.parquet`  
Generated automatically by `build_dataset.py` (Step 0b above).

## Included files

- `event_catalog_expanded_v2.json` — 21 confirmed NYC/NJ event-days used in Tier 2b training and LOTO-CV (Copa América 2024, Taylor Swift Eras Tour, NFL, NYC Pride, NYC Marathon). No raw ridership data; safe to distribute.
- `mta_manhattan_lookup.csv` — station metadata (sid, name, lat, lon) for all Manhattan subway stations.
- `penn_corridor_spikes_2022_2024.csv` — pre-computed Penn corridor ridership ratios used for event labeling in Tier 1 and GNN training.
