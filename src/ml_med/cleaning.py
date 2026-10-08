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
