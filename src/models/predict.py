import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
from typing import Dict, Any

MODEL_PATH = "models/best_model.joblib"

class HeartDiseasePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
        self.model = joblib.load(model_path)

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict heart disease presence and risk probability for a single patient record."""
        df = pd.DataFrame([data])
        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0][1])

        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "diagnosis": "Heart Disease Present" if prediction == 1 else "Heart Disease Absent"
        }

if __name__ == "__main__":
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    predictor = HeartDiseasePredictor()
    result = predictor.predict_single(sample)
    print("Inference Result:", result)
