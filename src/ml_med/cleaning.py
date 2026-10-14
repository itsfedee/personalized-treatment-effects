import pandas as pd

POLISH_MONTHS = {
    "sty": 1,  # styczeń, January
    "lut": 2,  # luty, February
    "mar": 3,  # marzec, March
    "kwi": 4,  # kwiecień, April
    "maj": 5,  # maj, May
    "cze": 6,  # czerwiec, June
    "lip": 7,  # lipiec, July
    "sie": 8,  # sierpień, August
    "wrz": 9,  # wrzesień, September
    "paź": 10,  # październik, October
    "lis": 11,  # listopad, November
    "gru": 12,  # grudzień, December
}

OCCODE_TO_OUTCOME = {
    1: 1,
    2: 1,
    3: 0,
    4: 0,
}

MAP_Y_N = {"Y": True, "N": False}

COLUMNS_TO_KEEP = [
    # identifier and auxiliary
    "patient_id",
    "HOSPNUM",
    "COUNTRY",
    "RDATE",
    "rdate_year",
    "rdate_month",
    "pilot_phase",
    # baseline covariates
    "RDELAY",
    "RCONSC",
    "SEX",
    "AGE",
    "RSLEEP",
    "RATRIAL",
    "RCT",
    "RVISINF",
    "RHEP24",
    "RASP3",
    "RSBP",
    "RDEF1",
    "RDEF2",
    "RDEF3",
    "RDEF4",
    "RDEF5",
    "RDEF6",
    "RDEF7",
    "RDEF8",
    "STYPE",
    # treatment
    "RXASP",
    "RXHEP",
    # outcomes
    "death_or_dependency_6m",
    "OCCODE",
    "ID14",
    "SET14D",
    "FDEAD",
    "ISC14",
    "PE14",
    "H14",
    "TRAN14",
]


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


def add_rdate_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rdate_year and rdate_month, derived from RDATE.
    RDATE is month-year text with Polish month abbreviations,
    e.g. "paź-91" = October 1991. Raises ValueError if a month
    abbreviation is not in POLISH_MONTHS."""
    df = df.copy()
    df["rdate_year"] = df["RDATE"].str[-2:].astype(int) + 1900
    df["rdate_month"] = df["RDATE"].str[:3].map(POLISH_MONTHS)
    if df["rdate_month"].isna().any():
        raise ValueError("unknown month abbreviation in RDATE")
    return df


def add_primary_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Build primary outcome column in the dataset: death or dependency of the patient 6 months later"""
    df = df.copy()
    df["death_or_dependency_6m"] = df["OCCODE"].map(OCCODE_TO_OUTCOME).astype("Int64")
    return df


YES_NO_COLUMNS = [
    "RSLEEP",  # symptoms noted on waking
    "RATRIAL",  # atrial fibrillation
    "RCT",  # CT before randomisation
    "RVISINF",  # infarct visible on CT
    "RHEP24",  # heparin in the 24 h before randomisation
    "RASP3",  # aspirin in the 3 days before randomisation
    "RXASP",  # trial aspirin allocated
    "FDEAD",  # dead at the 6-month follow-up contact
]


def convert_yes_no_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Y/N to True/False booleans. Columns of YES_NO_COLUMNS not present in df are skipped"""
    df = df.copy()
    for var in YES_NO_COLUMNS:
        if var not in df.columns:
            continue
        invalid = df[var].notna() & ~df[var].isin(["Y", "N"])
        if invalid.any():
            raise ValueError(f"unexpected values in {var}")
        df[var] = df[var].map(MAP_Y_N).astype("boolean")

    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep in the dataframe only the columns in COLUMNS_TO_KEEP"""
    missing = []
    for col in COLUMNS_TO_KEEP:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[COLUMNS_TO_KEEP].copy()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the previous functions to the dataset.
    Constraints:
    -add_patient_id is the first one and select_columns the last
    -set_fdead_unknown_missing before convert_yes_no_to_bool (U value would break)
    -fix_rvisinf_without_ct before convert_yes_no_to_bool
    -add_pilot_phase uses a missing RATRIAL"""

    df = add_patient_id(df)
    df = add_pilot_phase(df)
    df = add_rdate_year_month(df)
    df = recode_heparin(df)
    df = set_occode_missing(df)
    df = fix_rvisinf_without_ct(df)
    df = set_fdead_unknown_missing(df)
    df = add_primary_outcome(df)
    df = convert_yes_no_to_bool(df)
    df = select_columns(df)

    return df
