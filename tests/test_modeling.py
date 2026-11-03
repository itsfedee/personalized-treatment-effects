import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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


def make_split_df() -> pd.DataFrame:
    """Twenty patients: make_df() repeated, so both strata have 10 rows."""
    return pd.concat([make_df()] * 10, ignore_index=True)


def make_learner_df() -> pd.DataFrame:
    """Twenty patients with both outcomes in each arm, enough to fit a T-learner."""
    df = make_split_df()
    df["death_or_dependency_6m"] = pd.Series([0, 1, 1, 0] * 5, dtype="Int64")
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


def test_split_train_test_sizes():
    """make_split_df() gives a dataset with 20 patients, so I expect the train to be 16 and test 4"""
    X, y, t = modeling.prepare_data(make_split_df(), "RXASP")
    X_train, X_test, y_train, y_test, t_train, t_test = modeling.split_train_test(
        X, y, t
    )
    assert len(X_test) == len(y_test) == len(t_test) == 4
    assert len(X_train) == len(y_train) == len(t_train) == 16
    assert X_test.index.tolist() == y_test.index.tolist() == t_test.index.tolist()


def test_split_train_test_is_reproducible():
    """test if split_train_test gives always the same result"""
    X, y, t = modeling.prepare_data(make_split_df(), "RXASP")
    first = modeling.split_train_test(X, y, t)
    second = modeling.split_train_test(X, y, t)
    assert first[1].index.tolist() == second[1].index.tolist()


def test_make_model_default_is_logistic():
    model = modeling.make_model("RXASP")
    assert isinstance(model["clf"], LogisticRegression)


def test_make_model_accepts_custom_classifier():
    model = modeling.make_model("RXHEP", clf=RandomForestClassifier())
    assert isinstance(model["clf"], RandomForestClassifier)


def test_fit_t_learner_returns_two_fitted_models():
    X, y, t = modeling.prepare_data(make_learner_df(), "RXASP")
    models = modeling.fit_t_learner(X, y, t, "RXASP")
    assert set(models) == {0, 1}
    assert models[0].predict_proba(X).shape == (20, 2)
    assert models[1].predict_proba(X).shape == (20, 2)


def test_fit_t_learner_models_are_independent():
    X, y, t = modeling.prepare_data(make_learner_df(), "RXASP")
    models = modeling.fit_t_learner(X, y, t, "RXASP")
    assert models[0]["clf"] is not models[1]["clf"]


def test_predict_cate_shape():
    X, y, t = modeling.prepare_data(make_learner_df(), "RXASP")
    models = modeling.fit_t_learner(X, y, t, "RXASP")
    cate = modeling.predict_cate(models, X)
    assert cate.shape == (20,)
    assert ((cate >= -1) & (cate <= 1)).all()


def test_cross_fit_cate_covers_every_row():
    X, y, t = modeling.prepare_data(make_learner_df(), "RXASP")
    cate = modeling.cross_fit_cate(X, y, t, "RXASP", n_splits=2)
    assert cate.index.equals(X.index)
    assert cate.notna().all()
    assert cate.between(-1, 1).all()


def test_cross_fit_cate_is_reproducible():
    X, y, t = modeling.prepare_data(make_learner_df(), "RXASP")
    first = modeling.cross_fit_cate(X, y, t, "RXASP", n_splits=2, random_state=0)
    second = modeling.cross_fit_cate(X, y, t, "RXASP", n_splits=2, random_state=0)
    assert first.equals(second)


def make_uplift_inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    """100 patients: cate increasing, treatment alternating, outcome equal to treatment."""
    cate = pd.Series(range(100)) / 100 - 0.5
    t = pd.Series([0, 1] * 50)
    y = t.copy()
    return cate, y, t


def test_uplift_by_decile_shape():
    cate, y, t = make_uplift_inputs()
    table = modeling.uplift_by_decile(cate, y, t)
    assert list(table.index) == list(range(1, 11))
    assert table["n"].sum() == 100
    assert (table["uplift"] == table["rate_treated"] - table["rate_control"]).all()


def test_uplift_by_decile_detects_perfect_model():
    cate, y, t = make_uplift_inputs()
    # decile 1: treated have outcome 0, controls 1 -> uplift -1
    y.iloc[:10] = 1 - t.iloc[:10]
    table = modeling.uplift_by_decile(cate, y, t)
    assert table.loc[1, "uplift"] == -1
    assert table.loc[10, "uplift"] == 1
