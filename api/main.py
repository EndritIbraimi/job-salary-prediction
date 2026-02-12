from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import re
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Job Salary Range Predictor",
    description="Predicts salary range (Low/Medium/High) from job title and description.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Could not load ML model: {e}")

class PredictionRequest(BaseModel):
    job_title: str = Field(..., min_length=2, max_length=200, example="Senior Data Scientist")
    job_description: str = Field(..., min_length=10, max_length=5000, example="Looking for an experienced data scientist with Python and ML skills.")

class PredictionResponse(BaseModel):
    salary_range: str
    confidence: dict
    input_received: dict

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Job Salary Prediction API is live!",
        "usage": "Send a POST request to /predict"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_salary(request: PredictionRequest):
    try:
        combined = clean_text(request.job_title) + " " + clean_text(request.job_description)
        features = vectorizer.transform([combined])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = {
            cls: round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }

        short_desc = request.job_description[:100] + "..." if len(request.job_description) > 100 else request.job_description

        return PredictionResponse(
            salary_range=prediction,
            confidence=confidence,
            input_received={
                "job_title": request.job_title,
                "job_description": short_desc
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")