import pandas as pd
import pytest

from ml_med import features


def test_add_ct_status_maps_combinations():
    df = pd.DataFrame(
        {
            "RCT": pd.Series([False, True, True, False], dtype="boolean"),
            "RVISINF": pd.Series([False, False, True, None], dtype="boolean"),
        }
    )
    result = features.add_ct_status(df)
    assert result["ct_status"].tolist() == [
        "no_ct",
        "no_infarct_visible",
        "infarct_visible",
        "no_ct",
    ]


def test_add_ct_status_adds_string_column():
    df = pd.DataFrame(
        {
            "RCT": pd.Series([True], dtype="boolean"),
            "RVISINF": pd.Series([True], dtype="boolean"),
        }
    )
    result = features.add_ct_status(df)
    assert result.columns.tolist() == ["RCT", "RVISINF", "ct_status"]
    assert result["ct_status"].dtype == "str"


@pytest.mark.parametrize(
    "function, df",
    [
        (
            features.add_ct_status,
            pd.DataFrame(
                {
                    "RCT": pd.Series([True, False], dtype="boolean"),
                    "RVISINF": pd.Series([True, None], dtype="boolean"),
                }
            ),
        ),
        (
            features.add_n_deficits,
            pd.DataFrame({col: ["Y", "C"] for col in features.DEFICIT_COLUMNS}),
        ),
        (
            features.add_deficit_not_assessable,
            pd.DataFrame({col: ["Y", "C"] for col in features.DEFICIT_COLUMNS}),
        ),
        (
            features.add_rhep24_missing,
            pd.DataFrame({"RHEP24": pd.Series([True, None], dtype="boolean")}),
        ),
    ],
    ids=["ct_status", "n_deficits", "deficit_not_assessable", "rhep24_missing"],
)
def test_does_not_modify_input(function, df):
    original = df.copy()
    function(df)
    pd.testing.assert_frame_equal(df, original)


def test_n_deficits():
    df = pd.DataFrame({col: ["N", "N", "C"] for col in features.DEFICIT_COLUMNS})
    df["RDEF1"] = ["Y", "N", "C"]
    df["RDEF2"] = ["Y", "N", "C"]
    df["RDEF3"] = ["N", "N", "Y"]

    result = features.add_n_deficits(df)
    assert result["n_deficits"].tolist() == [2, 0, 1]


def test_add_deficit_not_assessable_flags_any_c():
    df = pd.DataFrame({col: ["N", "N", "C"] for col in features.DEFICIT_COLUMNS})
    df["RDEF4"] = ["N", "C", "C"]

    result = features.add_deficit_not_assessable(df)
    assert result["deficit_not_assessable"].tolist() == [False, True, True]


def test_add_rhep24_missing_flags_missing_values():
    df = pd.DataFrame({"RHEP24": pd.Series([True, False, None], dtype="boolean")})

    result = features.add_rhep24_missing(df)
    assert result["rhep24_missing"].tolist() == [False, False, True]


def test_build_features_adds_all_feature_columns():
    df = pd.DataFrame({col: ["Y"] for col in features.DEFICIT_COLUMNS})
    df["RCT"] = pd.Series([True], dtype="boolean")
    df["RVISINF"] = pd.Series([True], dtype="boolean")
    df["RHEP24"] = pd.Series([None], dtype="boolean")
    for col in [
        "AGE",
        "SEX",
        "RDELAY",
        "RCONSC",
        "RSLEEP",
        "RSBP",
        "STYPE",
        "RATRIAL",
        "RASP3",
        "pilot_phase",
    ]:
        df[col] = [0]

    result = features.build_features(df)
    new_columns = [
        "ct_status",
        "n_deficits",
        "deficit_not_assessable",
        "rhep24_missing",
    ]
    assert all(col in result.columns for col in new_columns)
    assert len(result) == 1
    assert all(col in result.columns for col in features.FEATURE_COLUMNS)
