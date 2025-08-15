from fastapi import FastAPI
import joblib
import pandas as pd
from .schemas import HeartData

app = FastAPI()

# Load model
model = joblib.load('model/heart_model.joblib')

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.get("/info")
def model_info():
    return {
        "model_type": "Random Forest",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
    }

@app.post("/predict")
def predict(data: HeartData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    return {"heart_disease": bool(prediction)}