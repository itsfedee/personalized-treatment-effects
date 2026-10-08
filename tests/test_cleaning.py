import pandas as pd
import pytest

from ml_med import cleaning


@pytest.mark.parametrize(
    "function, df",
    [
        (cleaning.recode_heparin, pd.DataFrame({"RXHEP": ["H", "L"]})),
        (cleaning.set_occode_missing, pd.DataFrame({"OCCODE": [0, 1, 9]})),
        (
            cleaning.fix_rvisinf_without_ct,
            pd.DataFrame({"RCT": ["N"], "RVISINF": ["Y"]}),
        ),
        (
            cleaning.set_fdead_unknown_missing,
            pd.DataFrame({"FDEAD": ["Y", "U", None]}),
        ),
        (cleaning.add_patient_id, pd.DataFrame({"AGE": [69, 76]})),
        (cleaning.add_pilot_phase, pd.DataFrame({"RATRIAL": ["Y", None]})),
    ],
    ids=["heparin", "occode", "rvisinf", "fdead", "patient_id", "pilot_phase"],
)
def test_does_not_modify_input(function, df):
    original = df.copy()
    function(df)
    pd.testing.assert_frame_equal(df, original)


def test_recode_heparin_replaces_h_with_m():
    df = pd.DataFrame({"RXHEP": ["H", "M", "L", "N"]})
    result = cleaning.recode_heparin(df)
    assert result["RXHEP"].tolist() == ["M", "M", "L", "N"]


def test_set_occode_missing_replaces_missing_codes():
    df = pd.DataFrame({"OCCODE": [0, 1, 2, 3, 4, 8, 9]})
    result = cleaning.set_occode_missing(df)
    expected_missing = [True, False, False, False, False, True, True]
    assert result["OCCODE"].isna().tolist() == expected_missing
    assert result["OCCODE"].dropna().tolist() == [1, 2, 3, 4]


def test_fix_rvisinf_without_ct_sets_missing():
    df = pd.DataFrame(
        {
            "RCT": ["Y", "Y", "N", "N"],
            "RVISINF": ["Y", "N", "N", "Y"],
        }
    )
    result = cleaning.fix_rvisinf_without_ct(df)
    assert result["RVISINF"].isna().tolist() == [False, False, False, True]
    assert result["RVISINF"].dropna().tolist() == ["Y", "N", "N"]


def test_set_fdead_unknown_missing_replaces_u():
    df = pd.DataFrame({"FDEAD": ["Y", "N", "U", None]})
    result = cleaning.set_fdead_unknown_missing(df)
    assert result["FDEAD"].isna().tolist() == [False, False, True, True]
    assert result["FDEAD"].dropna().tolist() == ["Y", "N"]


def test_add_patient_id_numbers_rows_from_1():
    df = pd.DataFrame({"AGE": [69, 76, 81]})
    result = cleaning.add_patient_id(df)
    assert result["patient_id"].tolist() == [1, 2, 3]
    assert result.columns.tolist() == ["patient_id", "AGE"]


def test_add_pilot_phase_flags_missing_ratrial():
    df = pd.DataFrame({"RATRIAL": ["Y", "N", None]})
    result = cleaning.add_pilot_phase(df)
    assert result["pilot_phase"].tolist() == [False, False, True]
    assert result.columns.tolist() == ["RATRIAL", "pilot_phase"]
