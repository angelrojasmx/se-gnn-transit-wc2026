"""
prepare_data.py — Step 0: Convert raw MTA CSV to filtered Manhattan parquet
============================================================================
Run this ONCE after downloading the raw MTA Subway Hourly Ridership CSV.

Usage:
    python src/demand/prepare_data.py --input data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv

The raw CSV (~17 GB) can be downloaded from:
    https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-2022/wujg-7c2s
Filter the download to: transit_timestamp between 2022-01-01 and 2024-12-31.

Outputs:
    data/nyc/raw/mta_manhattan_2022_2024.parquet   (~26 MB, Manhattan only)
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Manhattan station IDs — loaded from the lookup table included in the repo
LKP_PATH = ROOT / "data/nyc/mta_manhattan_lookup.csv"

DATE_START = "2022-01-01"
DATE_END   = "2024-12-31"

# Expected column names in the raw MTA CSV (as of 2024 download)
# Adjust if MTA changes their schema
COL_TS        = "transit_timestamp"
COL_STATION   = "station_complex_id"
COL_RIDERSHIP = "ridership"


def main(input_csv: Path):
    out_dir = ROOT / "data/nyc/raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mta_manhattan_2022_2024.parquet"

    if not LKP_PATH.exists():
        raise FileNotFoundError(
            f"Station lookup not found at {LKP_PATH}. "
            "This file should be included in the repository."
        )

    print(f"Loading station lookup from {LKP_PATH}...")
    lkp = pd.read_csv(LKP_PATH)
    lkp["sid"] = lkp["sid"].astype(int)
    manhattan_sids = set(lkp["sid"].tolist())
    print(f"  Manhattan stations: {len(manhattan_sids)}")

    print(f"\nReading raw CSV: {input_csv}")
    print("  (This may take several minutes for a ~17 GB file...)")

    # Read in chunks to avoid loading 17 GB at once
    chunk_size = 500_000
    records = []
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(
        input_csv,
        usecols=[COL_TS, COL_STATION, COL_RIDERSHIP],
        parse_dates=[COL_TS],
        chunksize=chunk_size,
        low_memory=False,
    ):
        total_rows += len(chunk)

        # Rename columns to internal schema
        chunk = chunk.rename(columns={
            COL_TS:        "ts",
            COL_STATION:   "sid",
            COL_RIDERSHIP: "ridership",
        })

        chunk["sid"] = pd.to_numeric(chunk["sid"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["sid", "ts", "ridership"])
        chunk["sid"] = chunk["sid"].astype(int)

        # Filter: Manhattan stations only
        chunk = chunk[chunk["sid"].isin(manhattan_sids)]

        # Filter: date range
        chunk = chunk[
            (chunk["ts"] >= DATE_START) &
            (chunk["ts"] <= DATE_END + " 23:59:59")
        ]

        # Aggregate to hourly if needed (MTA sometimes publishes sub-hourly)
        chunk["ts"] = chunk["ts"].dt.floor("h")
        chunk = (
            chunk.groupby(["ts", "sid"])["ridership"]
                 .sum()
                 .reset_index()
        )

        kept_rows += len(chunk)
        records.append(chunk)

        if total_rows % 5_000_000 < chunk_size:
            print(f"  Processed {total_rows:,} rows — kept {kept_rows:,} Manhattan rows")

    print(f"\nTotal rows read:  {total_rows:,}")
    print(f"Manhattan rows:   {kept_rows:,}")

    print("\nCombining and saving...")
    df = pd.concat(records, ignore_index=True)
    df["ridership"] = df["ridership"].astype("float32")
    df = df.sort_values(["ts", "sid"]).reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print(f"Shape: {df.shape}  |  Stations: {df['sid'].nunique()}")
    print(f"Date range: {df['ts'].min()} → {df['ts'].max()}")
    print("\nNext step:")
    print("  python src/demand/build_dataset.py --source manhattan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw MTA CSV to Manhattan parquet.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/nyc/raw/MTA_Subway_Hourly_Ridership_2022_2024.csv",
        help="Path to raw MTA Subway Hourly Ridership CSV",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("\nDownload the raw CSV from:")
        print("  https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-2022/wujg-7c2s")
        print(f"\nThen place it at: {args.input}")
        sys.exit(1)

    main(args.input)
