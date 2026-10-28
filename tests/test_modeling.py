import pandas as pd
import pytest

from ml_med import modeling
from ml_med.features import DEFICIT_COLUMNS, FEATURE_COLUMNS


def test_column_groups_cover_all_features():
    grouped = (
        set(modeling.NUMERIC_COLUMNS)
        | set(modeling.BINARY_COLUMNS)
        | set(modeling.CATEGORICAL_COLUMNS)
        | set(modeling.BOOLEAN_COLUMNS)
    )
    assert grouped == set(FEATURE_COLUMNS)


def make_df() -> pd.DataFrame:
    """Two patients with every feature, treatments and outcome."""
    df = pd.DataFrame(
        {
            "AGE": [70, 82],
            "RDELAY": [5, 30],
            "RSBP": [150, 180],
            "n_deficits": [2, 5],
            "pilot_phase": [False, True],
            "rhep24_missing": [False, True],
            "deficit_not_assessable": [False, True],
            "SEX": ["M", "F"],
            "RCONSC": ["F", "D"],
            "STYPE": ["PACS", "TACS"],
            "ct_status": ["infarct_visible", "no_ct"],
            "RSLEEP": pd.Series([False, True], dtype="boolean"),
            "RATRIAL": pd.Series([False, None], dtype="boolean"),
            "RASP3": pd.Series([True, None], dtype="boolean"),
            "RHEP24": pd.Series([False, None], dtype="boolean"),
            "RXASP": pd.Series([True, False], dtype="boolean"),
            "RXHEP": ["N", "M"],
            "death_or_dependency_6m": pd.Series([0, 1], dtype="Int64"),
        }
    )
    for col in DEFICIT_COLUMNS:
        df[col] = ["Y", "C"]
    return df


def test_make_preprocessor_output_width():
    df = make_df()
    X = df[FEATURE_COLUMNS + ["RXHEP"]]
    X_prep = modeling.make_preprocessor("RXASP").fit_transform(X)
    assert X_prep.shape == (2, 37)


def test_make_preprocessor_rejects_unknown_treatment():
    with pytest.raises(ValueError):
        modeling.make_preprocessor("AGE")


def test_prepare_data_drops_missing_outcome_and_adds_other_treatment():
    df = make_df()
    df.loc[1, "death_or_dependency_6m"] = None
    X, y, t = modeling.prepare_data(df, "RXASP")
    assert len(X) == len(y) == len(t) == 1
    assert y.isna().sum() == 0
    assert "RXHEP" in X.columns
    assert "RXASP" not in X.columns


def test_prepare_data_rejects_unknown_treatment():
    with pytest.raises(ValueError):
        modeling.prepare_data(make_df(), "AGE")
