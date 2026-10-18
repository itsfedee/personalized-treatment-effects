from pathlib import Path

import pandas as pd

from ml_med.features import build_features

CLEANED_DATASET_PATH = Path("data/processed/ist_clean.parquet")
FEATURES_DATASET_PATH = Path("data/processed/ist_features.parquet")


def main() -> None:
    df = pd.read_parquet(CLEANED_DATASET_PATH)
    df = build_features(df)
    df.to_parquet(FEATURES_DATASET_PATH, index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
