from pathlib import Path

import joblib
import pandas as pd

from ml_med.modeling import average_effects, fit_s_learner, prepare_data

FEATURES_DATASET_PATH = Path("data/processed/ist_features.parquet")
MODELS_DIR = Path("models")


def main() -> None:
    df = pd.read_parquet(FEATURES_DATASET_PATH)

    X, y, t = prepare_data(df, "RXASP")
    model = fit_s_learner(X, y, t, "RXASP")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "risk_model.joblib")
    for treatment in ("RXASP", "RXHEP"):
        average_effects(df, treatment).to_csv(MODELS_DIR / f"effects_{treatment}.csv")

    pd.crosstab(df["STYPE"], df["n_deficits"]).to_csv(
        MODELS_DIR / "stroke_type_by_deficits.csv"
    )
    print("saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
