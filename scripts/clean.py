from pathlib import Path

import pandas as pd

from ml_med.cleaning import clean

ORIGINAL_DATASET_PATH = Path("data/interim/IST_corrected_utf8.csv")

CLEANED_DATASET_PATH = Path("data/processed/ist_clean.parquet")


def main() -> None:
    CLEANED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ORIGINAL_DATASET_PATH)
    df = clean(df)
    df.to_parquet(CLEANED_DATASET_PATH, index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
