import os
import pytest
import pandas as pd
import numpy as np
from src.data.download import download_and_clean_data, COLUMNS
from src.data.preprocess import load_and_split_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def test_download_and_clean_data(tmp_path):
    raw_file = str(tmp_path / "raw.csv")
    clean_file = str(tmp_path / "clean.csv")

    df = download_and_clean_data(raw_path=raw_file, processed_path=clean_file)

    assert os.path.exists(raw_file), "Raw data file was not created"
    assert os.path.exists(clean_file), "Cleaned data file was not created"
    assert not df.empty, "Cleaned dataframe is empty"
    assert set(df["target"].unique()).issubset({0, 1}), "Target contains non-binary values"

    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns, f"Missing expected feature {col}"

def test_load_and_split_data(tmp_path):
    clean_file = str(tmp_path / "clean.csv")
    download_and_clean_data(raw_path=str(tmp_path / "raw.csv"), processed_path=clean_file)

    X_train, X_test, y_train, y_test = load_and_split_data(clean_file, test_size=0.2, random_state=42)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(0.2, abs=0.05)
