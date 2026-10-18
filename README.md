# Personalized Treatment Effects

Estimating individual treatment effects of aspirin and heparin in acute ischaemic stroke,
using patient-level data from the International Stroke Trial (IST).

## Data
Download version 2 of the IST database from https://datashare.ed.ac.uk/handle/10283/124
and save `IST_corrected.csv` in `data/raw/`.

Note: this project uses `IST_corrected.csv` (version 2, corrected), not the file attached
to the original 2011 paper, which contains errors in some date variables.

The raw file is encoded in Windows-1250 (month abbreviations are in Polish, e.g. `paź`);
it is converted to UTF-8 in `data/interim/` before loading.

## Data pipeline
1. `notebooks/01_data_loading.ipynb`: converts the raw file to UTF-8 (`data/interim/`)
2. `notebooks/02_data_checking.ipynb`: data quality checks on the raw data
3. `uv run python scripts/clean.py`: cleans the data (`data/processed/ist_clean.parquet`)
4. `notebooks/03_cleaned_data_check.ipynb`: checks the cleaned data
5. `uv run python scripts/build_features.py`: builds the model features (`data/processed/ist_features.parquet`)

## Setup
uv sync