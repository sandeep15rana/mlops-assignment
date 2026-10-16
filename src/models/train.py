import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

from src.data.download import download_and_clean_data
from src.data.preprocess import get_preprocessing_pipeline, load_and_split_data

ARTIFACTS_DIR = "report/screenshots"
MODELS_DIR = "models"

def evaluate_and_log_model(
    model_name: str,
    estimator,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor
) -> tuple:
    """Train a pipeline, run cross-validation, evaluate on test set, and log everything to MLflow."""
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    with mlflow.start_run(run_name=model_name):
        print(f"\n--- Training {model_name} ---")
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        for metric in scoring:
            mean_val = float(np.mean(cv_results[f"test_{metric}"]))
            std_val = float(np.std(cv_results[f"test_{metric}"]))
            mlflow.log_metric(f"cv_{metric}_mean", mean_val)
            mlflow.log_metric(f"cv_{metric}_std", std_val)
            print(f" CV {metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate on Test Set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, zero_division=0))
        test_rec = float(recall_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred))
        test_roc_auc = float(roc_auc_score(y_test, y_proba))

        metrics = {
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc
        }

        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)
            print(f" Test {m_name}: {m_val:.4f}")

        # Artifact: Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name.lower().replace(' ', '_')}_cm.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Artifact: ROC Curve Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
        ax.set_title(f"ROC Curve - {model_name}")
        roc_path = os.path.join(ARTIFACTS_DIR, f"{model_name.lower().replace(' ', '_')}_roc.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        mlflow.log_artifact(roc_path)

        # Log scikit-learn model pipeline in MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return pipeline, metrics

def train_and_select_best_model():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    data_file = "data/processed/heart_cleaned.csv"
    if not os.path.exists(data_file):
        download_and_clean_data()

    X_train, X_test, y_train, y_test = load_and_split_data(data_file)
    print(f"Data split: Train={len(X_train)} samples, Test={len(X_test)} samples")

    mlflow.set_experiment("Heart_Disease_Classification")

    models = [
        (
            "Logistic Regression",
            LogisticRegression(C=1.0, solver="liblinear", random_state=42),
            {"C": 1.0, "solver": "liblinear", "penalty": "l2"}
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42),
            {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4}
        )
    ]

    best_model_name = None
    best_pipeline = None
    best_metrics = None
    best_score = -1.0

    for name, estimator, params in models:
        preprocessor = get_preprocessing_pipeline()
        pipeline, metrics = evaluate_and_log_model(
            name, estimator, params, X_train, y_train, X_test, y_test, preprocessor
        )

        # Selection criterion: ROC-AUC score
        if metrics["test_roc_auc"] > best_score:
            best_score = metrics["test_roc_auc"]
            best_model_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    # Save best model to disk
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n==========================================")
    print(f"Best Model Selected: {best_model_name} (ROC-AUC: {best_score:.4f})")
    print(f"Saved to: {best_model_path}")
    print(f"==========================================")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "selection_metric": "test_roc_auc",
        "score": best_score,
        "metrics": best_metrics
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    train_and_select_best_model()
