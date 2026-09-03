import pandas as pd

from src.utils import first_present


def test_first_present_skips_null_alias_value():
    row = {"old_name": pd.NA, "new_name": 2024}

    assert first_present(row, ["old_name", "new_name"]) == 2024


def test_first_present_returns_default_when_all_aliases_are_null():
    row = {"old_name": None, "new_name": float("nan")}

    assert first_present(row, ["old_name", "new_name"], default="missing") == "missing"
