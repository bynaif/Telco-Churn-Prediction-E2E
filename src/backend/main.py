from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import shap
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict Customer Churn"
)

# Loading model
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/mdnaif/Desktop/E2E Project Practice/Telco-Churn-Prediction/model/model.pkl")  
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None


# Input Schema 
class CustomerData(BaseModel):
    gender: Annotated[int, Field(ge=0, le=1, description="Gender: 0 (female), 1 (male)")]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1, description="Senior citizen: 0 (no), 1 (yes)")]
    Partner: Annotated[int, Field(ge=0, le=1, description="Has partner: 0 (no), 1 (yes)")]
    Dependents: Annotated[int, Field(ge=0, le=1, description="Has dependents: 0 (no), 1 (yes)")]
    PhoneService: Annotated[int, Field(ge=0, le=1, description="Phone service: 0 (no), 1 (yes)")]
    PaperlessBilling: Annotated[int, Field(ge=0, le=1, description="Paperless billing: 0 (no), 1 (yes)")]

    tenure: Annotated[int, Field(ge=0, le=72, description="Tenure in months (0–72)")]
    MonthlyCharges: Annotated[float, Field(gt=0, description="Monthly charges (positive value)")]
    TotalCharges: Annotated[float, Field(ge=0, description="Total charges (non-negative)")]

    MultipleLines: Annotated[
        Literal["No", "Yes", "No phone service"],
        Field(description="Multiple lines subscription")
    ]

    InternetService: Annotated[
        Literal["DSL", "Fiber optic", "No"],
        Field(description="Type of internet service")
    ]

    OnlineSecurity: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Online security service")
    ]

    OnlineBackup: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Online backup service")
    ]

    DeviceProtection: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Device protection service")
    ]

    TechSupport: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Technical support service")
    ]

    StreamingTV: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Streaming TV service")
    ]

    StreamingMovies: Annotated[
        Literal["Yes", "No", "No internet service"],
        Field(description="Streaming movies service")
    ]

    Contract: Annotated[
        Literal["Month-to-month", "One year", "Two year"],
        Field(description="Contract type")
    ]

    PaymentMethod: Annotated[
        Literal[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        Field(description="Payment method")
    ]


# Output Schema
class PredictResponse(BaseModel):
    churn: int
    churn_probability: float
    result: str
    recommendation: list


# Helper function to generate recommendations
def generate_recommendation(prob, data):
    """Generate actionable recommendations based on churn probability and customer data"""
    recs = []
    
    if prob > 0.3:
        recs.append("Offer discount or loyalty plan")
        recs.append("Provide better customer service")
        
    if data["Contract"] == "Month-to-month":
        recs.append("Encourage long-term contract")
        
    if data["TechSupport"] == "No":
        recs.append("Offer free tech support trial")
    
    if data["InternetService"] == "Fiber optic":
        recs.append("Check pricing competitiveness")
    
    return recs


# Landing Page
@app.get("/")
def greet():
    """Welcome endpoint with links to documentation"""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


# Health Check
@app.get("/health")
def health():
    """Check if model is loaded and API is healthy"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ok",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerData):
    """
    Predict customer churn and provide recommendations
    
    Returns:
    - churn: 1 (will churn) or 0 (will stay)
    - churn_probability: Probability score (0-1)
    - result: Human-readable prediction
    - recommendation: List of actionable recommendations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        df = pd.DataFrame([customer.model_dump()])
        
        threshold = 0.3
        probability = model.predict_proba(df)[0][1]
        prediction = 1 if probability >= threshold else 0
        
        result = "Customer will churn" if prediction == 1 else "Customer will stay"
        
        recommendation = generate_recommendation(probability, customer.model_dump())
        
        return {
            "churn": prediction,
            "churn_probability": round(probability, 3),
            "result": result,
            "recommendation": recommendation
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")