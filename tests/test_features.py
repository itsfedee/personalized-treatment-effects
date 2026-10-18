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
    ],
    ids=["ct_status"],
)
def test_does_not_modify_input(function, df):
    original = df.copy()
    function(df)
    pd.testing.assert_frame_equal(df, original)
