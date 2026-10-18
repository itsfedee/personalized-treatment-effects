import numpy as np
import pandas as pd

CT_STATUS_VALUES = ["no_ct", "no_infarct_visible", "infarct_visible"]
DEFICIT_COLUMNS = [f"RDEF{i}" for i in range(1, 9)]


def add_ct_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ct_status, a single CT variable built from RCT and RVISINF.

    RVISINF is ambiguous: it is False both when a CT scan was done and
    showed no infarct, and when no CT scan was done at all. ct_status
    separates the two cases:
      - RCT False                 -> "no_ct"
      - RCT True, RVISINF False   -> "no_infarct_visible"
      - RCT True, RVISINF True    -> "infarct_visible"
    RCT is checked first, so the one patient with RVISINF missing
    (set to missing during cleaning because RCT was False) is classified
    as "no_ct": we trust RCT, as we did in cleaning. Raises ValueError if
    any row matches none of the three cases.
    """
    df = df.copy()
    ct_done = df["RCT"].fillna(False)
    infarct = df["RVISINF"].fillna(False)
    conditions = [
        ~ct_done,
        ct_done & ~infarct,
        ct_done & infarct,
    ]
    ct_status = np.select(conditions, CT_STATUS_VALUES, default=None)
    df["ct_status"] = pd.Series(ct_status, index=df.index, dtype="str")
    if df["ct_status"].isna().any():
        raise ValueError("ct_status undefined for some rows")
    return df


def add_n_deficits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_deficits, the number of neurological deficits present.

    Counts how many of RDEF1-RDEF8 are "Y". "C" (can't assess) is not counted: it is
    neither present nor absent, and is tracked separately by
    any_deficit_not_assessable. Values range from 0 to 8, no missing.
    """
    df = df.copy()
    df["n_deficits"] = (df[DEFICIT_COLUMNS] == "Y").sum(axis=1)
    return df


def add_deficit_not_assessable(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a new column "deficit_not_assessable" that specifies
    whether the patient could not be fully assessed, usually because too severely impaired
    """
    df = df.copy()
    df["deficit_not_assessable"] = (df[DEFICIT_COLUMNS] == "C").any(axis=1)
    return df


def add_rhep24_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rhep24_missing: True where RHEP24 is missing.

    RHEP24 is missing for 344 pilot-phase patients randomised before the
    question was added to the form (Feb 1992). Imputation in the modelling
    pipeline will fill these values, so this flag keeps track of them.
    No indicators are built for RATRIAL and RASP3: their missing values
    coincide exactly with pilot_phase.
    """
    df = df.copy()
    df["rhep24_missing"] = df["RHEP24"].isna()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature functions to the cleaned dataset.

    """
    df = add_ct_status(df)
    df = add_n_deficits(df)
    df = add_deficit_not_assessable(df)
    df = add_rhep24_missing(df)
    return df


# Model inputs (X) for the causal models of phase 5.
# Treatment (RXASP, RXHEP) and outcomes are not features: they enter the
# models as T and Y. RCT and RVISINF are replaced by ct_status.
# COUNTRY is a candidate to evaluate separately in phase 5.
FEATURE_COLUMNS = [
    # demographics
    "AGE",
    "SEX",
    # clinical presentation at randomisation
    "RDELAY",
    "RCONSC",
    "RSLEEP",
    "RSBP",
    "STYPE",
    # history and pre-randomisation drugs
    "RATRIAL",
    "RASP3",
    "RHEP24",
    "rhep24_missing",
    # neurological deficits
    *DEFICIT_COLUMNS,
    "n_deficits",
    "deficit_not_assessable",
    # CT scan
    "ct_status",
    # trial phase
    "pilot_phase",
]
