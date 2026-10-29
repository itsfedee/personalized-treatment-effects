import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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


def make_preprocessor(treatment: str) -> ColumnTransformer:
    """
    Build the ColumnTransformer that prepares the features for a model.

    treatment is the treatment under study ("RXASP" or "RXHEP"); the other
    one is added to the features, because the two were randomised
    independently. Numeric columns are
    standardised, binary ones are already ok, categorical ones one-hot
    encoded, nullable booleans imputed with False. These steps learn
    from the data (means, categories), so they live in the model
    Pipeline and are refitted on each training fold, not in features.py.
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

    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(), categorical),
            (
                "bool",
                SimpleImputer(strategy="constant", fill_value=False),
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
