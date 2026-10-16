import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.download import download_and_clean_data
from src.data.preprocess import get_preprocessing_pipeline, load_and_split_data
from src.models.predict import HeartDiseasePredictor

def test_preprocessing_pipeline(tmp_path):
    clean_file = str(tmp_path / "clean.csv")
    download_and_clean_data(raw_path=str(tmp_path / "raw.csv"), processed_path=clean_file)
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file)

    preprocessor = get_preprocessing_pipeline()
    transformed_X = preprocessor.fit_transform(X_train)

    assert isinstance(transformed_X, np.ndarray)
    assert transformed_X.shape[0] == len(X_train)
    assert not np.isnan(transformed_X).any(), "Transformed data contains NaNs"

def test_model_pipeline_training_and_prediction(tmp_path):
    clean_file = str(tmp_path / "clean.csv")
    download_and_clean_data(raw_path=str(tmp_path / "raw.csv"), processed_path=clean_file)
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessing_pipeline()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]

    assert set(preds).issubset({0, 1})
    assert (probas >= 0.0).all() and (probas <= 1.0).all()

def test_heart_disease_predictor_class(tmp_path):
    clean_file = str(tmp_path / "clean.csv")
    model_file = str(tmp_path / "test_model.joblib")
    download_and_clean_data(raw_path=str(tmp_path / "raw.csv"), processed_path=clean_file)
    X_train, X_test, y_train, y_test = load_and_split_data(clean_file)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessing_pipeline()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_file)

    predictor = HeartDiseasePredictor(model_path=model_file)
    sample = {
        "age": 60, "sex": 1, "cp": 4, "trestbps": 140.0, "chol": 260.0,
        "fbs": 0, "restecg": 1, "thalach": 140.0, "exang": 1, "oldpeak": 2.0,
        "slope": 2, "ca": 1.0, "thal": 3.0
    }
    result = predictor.predict_single(sample)

    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ("Low", "Moderate", "High")
