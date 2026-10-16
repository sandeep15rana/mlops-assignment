from pydantic import BaseModel, Field, ConfigDict

class PatientData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0.0,
                "thal": 1.0
            }
        }
    )

    age: int = Field(..., ge=18, le=110, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)")
    trestbps: float = Field(..., ge=70, le=250, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., ge=80, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG (0, 1, or 2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=3, description="Slope of the peak exercise ST segment (1, 2, or 3)")
    ca: float = Field(..., ge=0.0, le=4.0, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., ge=0.0, le=7.0, description="Thal (3 = normal, 6 = fixed defect, 7 = reversible defect)")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0 = Absence of heart disease, 1 = Presence")
    probability: float = Field(..., description="Risk probability score between 0.0 and 1.0")
    risk_level: str = Field(..., description="Clinical risk category: Low, Moderate, High")
    diagnosis: str = Field(..., description="Human-readable diagnosis statement")
    model_version: str = Field(default="1.0.0", description="Serving model version")
