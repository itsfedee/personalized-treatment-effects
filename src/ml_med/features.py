import numpy as np
import pandas as pd

CT_STATUS_VALUES = ["no_ct", "no_infarct_visible", "infarct_visible"]


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
