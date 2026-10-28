from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_med.features import DEFICIT_COLUMNS

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
