import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_med.features import DEFICIT_COLUMNS, FEATURE_COLUMNS

# standardised (StandardScaler)
NUMERIC_COLUMNS = ["AGE", "RDELAY", "RSBP", "n_deficits"]

# already 0/1, no missing: passed through unchanged
BINARY_COLUMNS = ["pilot_phase", "rhep24_missing", "deficit_not_assessable"]

# one-hot encoded
CATEGORICAL_COLUMNS = ["SEX", "RCONSC", "STYPE", *DEFICIT_COLUMNS, "ct_status"]

# nullable booleans: missing values imputed with False
BOOLEAN_COLUMNS = ["RSLEEP", "RATRIAL", "RASP3", "RHEP24"]

OTHER_TREATMENT = {"RXASP": "RXHEP", "RXHEP": "RXASP"}


def make_preprocessor(treatment: str, include_treatment=False) -> ColumnTransformer:
    """
    Build the ColumnTransformer that prepares the features for a model.

    treatment is the treatment under study ("RXASP" or "RXHEP"); the other
    one is added to the features, because the two were randomised
    independently. Numeric columns are
    standardized, binary ones are already ok, categorical ones one-hot
    encoded, nullable booleans imputed with False. These steps learn
    from the data (means, categories), so they live in the model
    Pipeline and are refitted on each training fold, not in features.py.

    With include_treatment=True the treatment itself becomes a feature, as the S-learner needs.
    """
    if treatment not in OTHER_TREATMENT:
        raise ValueError(
            f"treatment must be one of {list(OTHER_TREATMENT)}, got {treatment!r}"
        )

    if treatment == "RXASP":
        binary = BINARY_COLUMNS
        categorical = CATEGORICAL_COLUMNS + ["RXHEP"]
    else:
        binary = BINARY_COLUMNS + ["RXASP"]
        categorical = CATEGORICAL_COLUMNS

    if include_treatment:
        if treatment == "RXASP":
            binary = binary + [treatment]
        else:
            categorical = categorical + [treatment]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "bool",
                SimpleImputer(
                    strategy="constant", fill_value=False, keep_empty_features=True
                ),
                BOOLEAN_COLUMNS,
            ),
        ]
    )


def prepare_data(
    df: pd.DataFrame, treatment: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the features dataset into X, y, t for the study of one treatment.

    Rows with a missing primary outcome (150 patients...<1%) are dropped:
    a model learns nothing from them. X holds FEATURE_COLUMNS plus the
    other treatment, which is a legitimate feature because the two
    treatments were randomised independently. y is the outcome as int,
    t is the treatment under study, used to split patients, not as a feature.
    """

    if treatment not in OTHER_TREATMENT:
        raise ValueError(
            f"treatment must be one of {list(OTHER_TREATMENT)}, got {treatment!r}"
        )

    has_outcome = df["death_or_dependency_6m"].notna()
    df = df[has_outcome]
    other = OTHER_TREATMENT[treatment]
    X = df[FEATURE_COLUMNS + [other]]
    y = df["death_or_dependency_6m"].astype(int)
    t = df[treatment]

    if treatment == "RXHEP":
        t = t != "N"

    return X, y, t


def split_train_test(
    X: pd.DataFrame, y: pd.Series, t: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the features X, labels y and (presence of) treatment t in train/test"""
    strata = y.astype(str) + "_" + t.astype(str)
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=0, stratify=strata
    )
    return X_train, X_test, y_train, y_test, t_train, t_test


def make_model(treatment: str, clf=None, include_treatment: bool = False) -> Pipeline:
    """Return a Pipeline object with architecture clf.
    clf is any sklearn classifier; the default is logistic regression, chosen in notebook 05 as the best of four candidates on 5-fold CV.
    """
    if clf is None:
        clf = LogisticRegression(max_iter=1000)

    return Pipeline(
        [("prep", make_preprocessor(treatment, include_treatment)), ("clf", clf)]
    )


def fit_t_learner(
    X: pd.DataFrame, y: pd.Series, t: pd.Series, treatment: str, clf=None
) -> dict[int, Pipeline]:
    """
    Fit a T-learner: one risk model per arm, same classifier.

    Returns {0: model fitted on controls (not treated), 1: model fitted on treated}.
    clf is cloned for each arm so the two models are independent.
    """

    if clf is None:
        clf = LogisticRegression(max_iter=1000)

    models = {}
    for arm in (0, 1):
        mask = t == arm  # mask is true when t = arm
        model = make_model(treatment, clone(clf))
        models[arm] = model.fit(X[mask], y[mask])

    return models


def fit_s_learner(X, y, t, treatment, clf=None) -> Pipeline:

    if clf is None:
        clf = LogisticRegression(max_iter=1000)

    X_s = X.assign(**{treatment: t.astype(int)})
    model = make_model(treatment, clf, include_treatment=True).fit(X_s, y)

    return model


def fit_causal_forest(X, y, t, treatment, min_samples_leaf=100, random_state=0) -> dict:

    prep = make_preprocessor(treatment)
    X_num = prep.fit_transform(X)
    cf = CausalForestDML(
        model_y=LogisticRegression(max_iter=1000),
        model_t=DummyClassifier(strategy="prior"),
        discrete_outcome=True,
        discrete_treatment=True,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    cf.fit(y, t, X=X_num)

    return {"prep": prep, "forest": cf}


def predict_cate(models: dict[int, Pipeline], X: pd.DataFrame) -> np.ndarray:
    p1 = models[1].predict_proba(X)[:, 1]  # risk if treated
    p0 = models[0].predict_proba(X)[:, 1]  # risk if not treated
    return p1 - p0


def predict_cate_s(model, X, treatment) -> np.ndarray:
    X1 = X.assign(**{treatment: 1})
    X0 = X.assign(**{treatment: 0})
    return model.predict_proba(X1)[:, 1] - model.predict_proba(X0)[:, 1]


def predict_cate_cf(model: dict, X: pd.DataFrame) -> np.ndarray:

    X_num = model["prep"].transform(X)
    cate = model["forest"].effect(X_num).ravel()

    return cate


def cross_fit_cate(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    treatment: str,
    clf=None,
    learner: str = "t",
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.Series:
    """Compute CATE for each patient using a t-learneror s-learner that didn't see it in the training.
    Use k-fold.
    """
    if learner not in ("t", "s", "cf"):
        raise ValueError(f"learner must be 't', 's' or 'cf', got {learner!r}")

    cate = np.empty(len(X))  # initialize empty list of cate
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kfold.split(X):
        if learner == "t":
            models = fit_t_learner(
                X.iloc[train_idx], y.iloc[train_idx], t.iloc[train_idx], treatment, clf
            )
            cate[test_idx] = predict_cate(models, X.iloc[test_idx])
        elif learner == "s":
            model = fit_s_learner(
                X.iloc[train_idx], y.iloc[train_idx], t.iloc[train_idx], treatment, clf
            )
            cate[test_idx] = predict_cate_s(model, X.iloc[test_idx], treatment)

        elif learner == "cf":
            model = fit_causal_forest(
                X.iloc[train_idx], y.iloc[train_idx], t.iloc[train_idx], treatment
            )
            cate[test_idx] = predict_cate_cf(model, X.iloc[test_idx])
    return pd.Series(cate, index=X.index)


def fold_ids(X: pd.DataFrame, n_splits: int = 5, random_state: int = 0) -> pd.Series:
    """Fold number of each patient, with the same split used by cross_fit_cate."""
    folds = np.empty(len(X), dtype=int)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for k, (_, test_idx) in enumerate(kfold.split(X)):
        folds[test_idx] = k

    return pd.Series(folds, index=X.index)


def uplift_by_decile(
    cate: pd.Series, y: pd.Series, t: pd.Series, n_bins: int = 10, fold=None
) -> pd.DataFrame:
    """
    Observed outcome rates by decile of predicted CATE.

    Patients are ranked by cate and cut into n_bins groups of equal size
    (ties broken by position, so a learner predicting many identical
    values still gets n_bins groups). For each group: size, mean predicted
    cate, outcome rate in controls and treated, and their difference
    (uplift), to be compared with the predicted cate_mean.
    """

    df = pd.DataFrame({"cate": cate, "y": y, "t": t.astype(int)})
    if fold is None:
        ranks = df["cate"].rank(method="first")
    else:
        ranks = df["cate"].groupby(fold).rank(method="first", pct=True)
        ranks = ranks.rank(method="first")

    df["decile"] = pd.qcut(ranks, n_bins, labels=False) + 1

    rates = df.groupby(["decile", "t"])["y"].mean().unstack("t")

    by_decile = df.groupby("decile")
    return pd.DataFrame(
        {
            "n": by_decile.size(),
            "cate_mean": by_decile["cate"].mean(),
            "rate_control": rates[0],
            "rate_treated": rates[1],
            "uplift": rates[1] - rates[0],
        }
    )


def bootstrap_uplift(
    cate, y, t, n_bins=10, n_boot=1000, random_state=0, fold=None
) -> pd.DataFrame:
    df = pd.DataFrame({"cate": cate, "y": y, "t": t})
    if fold is not None:  # the fold travels with the patient
        df["fold"] = fold
    rng = np.random.default_rng(random_state)

    rows = []
    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True, random_state=rng).reset_index(
            drop=True
        )
        sample_fold = None if fold is None else sample["fold"]
        table = uplift_by_decile(
            sample["cate"], sample["y"], sample["t"], n_bins, sample_fold
        )
        rows.append(table["uplift"])

    return pd.DataFrame(rows).reset_index(drop=True)
