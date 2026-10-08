import pandas as pd

from ml_med.cleaning import recode_heparin


def test_recode_heparin_replaces_h_with_m():
    df = pd.DataFrame({"RXHEP": ["H", "M", "L", "N"]})
    result = recode_heparin(df)
    assert result["RXHEP"].tolist() == ["M", "M", "L", "N"]


def test_recode_heparin_doesnt_modify_input():
    df = pd.DataFrame({"RXHEP": ["H", "L"]})
    recode_heparin(df)
    assert df["RXHEP"].tolist() == ["H", "L"]
