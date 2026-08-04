# NYC Data

This directory should contain the following files before running any Tier 1 or Tier 2 scripts.

## Data source and licensing

Ridership figures in this section originate from the MTA's own Open Data
portal, not from this project. As of this writing, the dataset's own metadata
page lists its license as "unspecified" (see
[data.ny.gov, Licensing and Attribution](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s/about_data)),
so no third-party-derived ridership file is committed to this repository.
Only the pipeline needed to regenerate them locally is provided.

## Required files

### 1. MTA Subway Hourly Ridership (Tier 2 - GNN training)
**Not included.** Download the raw CSV from the MTA Open Data portal:
- URL: https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s
- Filter: January 2022 – December 2024
- Save to: `data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv`

Then run the two-step preparation pipeline:
```bash
# Step 0a: filter to Manhattan stations and convert to parquet (~17 GB CSV → ~26 MB parquet)
python src/demand/prepare_data.py --input data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv
# Writes: raw/mta_manhattan_2022_2024.parquet (not committed; regenerate locally)

# Step 0b: build normalized daily profile dataset for GNN training
python src/demand/build_dataset.py
```
Step 0b generates `manhattan/daily_profiles.parquet`, `manhattan/baseline_profile.parquet`,
`manhattan/baseline_daily_totals.parquet`, `manhattan/station_lookup.csv`, and
`manhattan/event_labels.csv` - none of these are committed to this repository, since they are
all derived directly from the MTA ridership figures above. Station metadata equivalent to
`manhattan/station_lookup.csv` is available without regenerating anything, see
`mta_manhattan_lookup.csv` below.

## Included files

These files are safe to distribute: they contain no raw or derived ridership figures, only
station metadata, pre-computed ratios used as model inputs, or catalogs curated by this project.

- `event_catalog_expanded_v2.json` - 21 confirmed NYC/NJ event-days used in Tier 2b training and LOTO-CV (Copa América 2024, Taylor Swift Eras Tour, NFL, NYC Pride, NYC Marathon). No raw ridership data; safe to distribute.
- `mta_manhattan_lookup.csv` - station metadata (sid, name, lat, lon) for all Manhattan subway stations.
- `penn_corridor_spikes_2022_2024.csv` - pre-computed Penn corridor ridership ratios used for event labeling in Tier 1 and GNN training.
