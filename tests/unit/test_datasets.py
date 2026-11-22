import numpy as np
import pandas as pd
import pytest

from carepilot.ml.datasets import DIABETES, HEART, clean, get_spec


def test_pima_impossible_zeros_become_missing_but_pregnancies_zero_stays():
    df = pd.DataFrame({c: [0, 5] for c in DIABETES.numeric} | {"Outcome": [0, 1]})
    out = clean(df, DIABETES)
    for col in DIABETES.zero_as_missing:
        assert np.isnan(out.loc[0, col])
    assert out.loc[0, "Pregnancies"] == 0


def test_heart_target_binarized():
    df = pd.DataFrame({c: [1, 2, 3] for c in HEART.feature_order} | {"num": [0, 2, 4]})
    df["age"] = [40, 50, 60]  # avoid duplicate-row removal
    out = clean(df, HEART)
    assert out["num"].tolist() == [0, 1, 1]


def test_clean_does_not_mutate_input():
    df = pd.DataFrame({c: [0] for c in DIABETES.numeric} | {"Outcome": [1]})
    clean(df, DIABETES)
    assert df.loc[0, "Glucose"] == 0


def test_unknown_disease_raises():
    with pytest.raises(ValueError, match="unknown disease"):
        get_spec("cancer")
