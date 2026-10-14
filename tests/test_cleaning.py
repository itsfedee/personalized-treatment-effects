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
        (cleaning.add_rdate_year_month, pd.DataFrame({"RDATE": ["sty-91", "paź-92"]})),
        (
            cleaning.add_primary_outcome,
            pd.DataFrame({"OCCODE": pd.Series([1, 3, None], dtype="Int64")}),
        ),
        (
            cleaning.convert_yes_no_to_bool,
            pd.DataFrame({"RCT": ["Y", "N", None]}),
        ),
        (
            cleaning.select_columns,
            pd.DataFrame({col: [1] for col in cleaning.COLUMNS_TO_KEEP}),
        ),
    ],
    ids=[
        "heparin",
        "occode",
        "rvisinf",
        "fdead",
        "patient_id",
        "pilot_phase",
        "rdate_year_month",
        "death_or_dependency_6m",
        "convert_yes_no_to_bool",
        "select_columns",
    ],
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


def test_add_rdate_year_month():
    df = pd.DataFrame({"RDATE": ["sty-91", "paź-92", "gru-96"]})
    result = cleaning.add_rdate_year_month(df)
    assert result.columns.tolist() == ["RDATE", "rdate_year", "rdate_month"]
    assert result["rdate_year"].tolist() == [1991, 1992, 1996]
    assert result["rdate_month"].tolist() == [1, 10, 12]


def test_add_rdate_year_month_rejects_unknown_month():
    df = pd.DataFrame({"RDATE": ["sty-91", "xyz-91"]})
    with pytest.raises(ValueError):
        cleaning.add_rdate_year_month(df)


def test_add_primary_outcome_maps_occode():
    df = pd.DataFrame({"OCCODE": pd.Series([1, 2, 3, 4, None], dtype="Int64")})
    result = cleaning.add_primary_outcome(df)
    assert result["death_or_dependency_6m"].isna().tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert result["death_or_dependency_6m"].dropna().tolist() == [1, 1, 0, 0]


def test_convert_yes_no_to_bool_converts_values():
    df = pd.DataFrame({"RCT": ["Y", "N", None]})
    result = cleaning.convert_yes_no_to_bool(df)
    assert result["RCT"].isna().tolist() == [False, False, True]
    assert result["RCT"].dropna().tolist() == [True, False]
    assert result["RCT"].dtype == "boolean"


def test_convert_yes_no_to_bool_rejects_unexpected_values():
    df = pd.DataFrame({"RCT": ["Y", "C"]})
    with pytest.raises(ValueError):
        cleaning.convert_yes_no_to_bool(df)


def test_convert_yes_no_other_columns():
    df = pd.DataFrame({"RCT": ["Y", "N"], "RDEF1": ["Y", "C"]})
    result = cleaning.convert_yes_no_to_bool(df)
    assert result["RDEF1"].tolist() == ["Y", "C"]
    assert result["RCT"].tolist() == [True, False]


def test_select_columns_keeps_only_listed_columns():
    df = pd.DataFrame({col: [1] for col in cleaning.COLUMNS_TO_KEEP})
    df["EXPDD"] = [0.5]
    result = cleaning.select_columns(df)
    assert result.columns.tolist() == cleaning.COLUMNS_TO_KEEP


def test_select_columns_missing_columns():
    df = pd.DataFrame({col: [1] for col in cleaning.COLUMNS_TO_KEEP})
    df = df.drop(columns=["AGE"])
    with pytest.raises(ValueError):
        cleaning.select_columns(df)
