import pandas as pd


def recode_heparin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode the pilot-phase heparin code H as M
    In the pilot phase the medium dose was coded H (High);
    in the main ohase it is coded M; they are the same dose.
    """
    df = df.copy()
    df["RXHEP"] = df["RXHEP"].replace({"H": "M"})
    return df


def set_occode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set 0,8,9 values of OCCODE as missing values.
    8,9 mean missing according to documentation
    0 is not documented but all patients with 0 have FDEAD missing
    """
    df = df.copy()
    df["OCCODE"] = df["OCCODE"].astype("Int64").mask(df["OCCODE"].isin([0, 8, 9]))
    return df


def fix_rvisinf_without_ct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set RVISINF as missing when RCT = N and RVISINF = Y.
    An infarct cannot be visible on a CT that was not done.
    This impossible combination occurs for 1 patient: since we cannot
    tell which of the two values is wrong, RVISINF is set to missing.
    """
    df = df.copy()
    df["RVISINF"] = df["RVISINF"].mask((df["RCT"] == "N") & (df["RVISINF"] == "Y"))
    return df


def set_fdead_unknown_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set FDEAD as missing value when it is U (Unknown).
    U is not documented for FDEAD, but the dictionary uses U for
    "unknown" in other variables (DPLACE, FPLACE), and 53 of the 57
    patients with FDEAD = U have a missing 6-month outcome (OCCODE = 9).
    """
    df = df.copy()
    df["FDEAD"] = df["FDEAD"].mask(df["FDEAD"] == "U")
    return df


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a patient_id column as the first column.
    The dataset has no patient identifier: each row is a patient.
    patient_id is the row number in the original file, starting at 1.
    Apply this function first, before any step that removes or
    reorders rows, so that the id always matches the original file.
    """
    df = df.copy()
    id_values = range(1, len(df) + 1)
    df.insert(0, "patient_id", id_values)
    return df


def add_pilot_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pilot_phase"] = df["RATRIAL"].isna()
    return df
