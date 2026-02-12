
# Job Salary Prediction API

A minimal REST API built with FastAPI that exposes a trained ML model for predicting job salary ranges.

## Project Structure
```
api/
├── main.py          # FastAPI application
├── models/
│   ├── model.pkl        # Trained Naive Bayes classifier
│   └── vectorizer.pkl   # TF-IDF vectorizer
├── requirements.txt # Dependencies
└── README.md        # This file
```

## How It Works

1. The trained model and vectorizer are loaded once at startup
2. Each request sends a job title + description
3. Text is cleaned and vectorized using TF-IDF
4. The model predicts Low / Medium / High salary range
5. Confidence scores are returned alongside the prediction

## Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload

# API will be available at:
# http://127.0.0.1:8000
```

## Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "job_title": "Senior Data Scientist",
       "job_description": "Looking for an experienced data scientist with Python, ML, and TensorFlow skills."
     }'
```

## Example Response
```json
{
  "salary_range": "High",
  "confidence": {
    "High": 0.6523,
    "Medium": 0.2241,
    "Low": 0.1236
  },
  "input_received": {
    "job_title": "Senior Data Scientist",
    "job_description": "Looking for an experienced data scientist with Python, ML..."
  }
}
```

## API Endpoints

| Method | Endpoint  | Description              |
|--------|-----------|--------------------------|
| GET    | /         | Health check             |
| POST   | /predict  | Predict salary range     |
| GET    | /docs     | Interactive API docs     |
