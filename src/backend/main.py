from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Annotated, Literal, List, Dict
import shap
import os

app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")

try:
    model        = joblib.load(MODEL_PATH)
    preprocessor = model[:-1]
    classifier   = model[-1]
    explainer    = shap.TreeExplainer(classifier)
    feature_names = preprocessor.get_feature_names_out().tolist()
except Exception as e:
    print(f"Startup error: {e}")
    model = preprocessor = classifier = explainer = None
    feature_names = []


class CustomerData(BaseModel):
    gender: Annotated[int, Field(ge=0, le=1)]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: Annotated[int, Field(ge=0, le=1)]
    Dependents: Annotated[int, Field(ge=0, le=1)]
    PhoneService: Annotated[int, Field(ge=0, le=1)]
    PaperlessBilling: Annotated[int, Field(ge=0, le=1)]
    tenure: Annotated[int, Field(ge=0, le=72)]
    MonthlyCharges: Annotated[float, Field(gt=0)]
    TotalCharges: Annotated[float, Field(ge=0)]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]


class PredictResponse(BaseModel):
    churn: int
    churn_probability: float
    result: str
    shap_values: List[Dict]


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df            = pd.DataFrame([customer.model_dump()])
        probability   = float(model.predict_proba(df)[0][1])
        prediction    = 1 if probability >= 0.3 else 0

        X_transformed = preprocessor.transform(df)
        sv            = explainer.shap_values(X_transformed)
        values        = sv[1][0] if isinstance(sv, list) else sv[0]

        aggregated = {}
        for fname, v in zip(feature_names, values):
            raw = fname.split("__", 1)[-1].split("_")[0]
            aggregated[raw] = aggregated.get(raw, 0.0) + float(v)

        shap_output = sorted(
            [{"feature": f, "impact": round(v, 4)} for f, v in aggregated.items()],
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        return {
            "churn": prediction,
            "churn_probability": round(probability, 4),
            "result": "Customer will churn" if prediction == 1 else "Customer will stay",
            "shap_values": shap_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))